import lzma
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import ray
import torch


_WORKER_RUNTIME = None


def _metric_cache_path(cache_root, metadata_path):
    return Path(cache_root).joinpath(*Path(metadata_path).parts[-4:])


def _prepend_python_paths(paths):
    for path in reversed(paths):
        if path not in sys.path:
            sys.path.insert(0, path)


def _get_worker_runtime(item):
    global _WORKER_RUNTIME
    if _WORKER_RUNTIME is not None:
        return _WORKER_RUNTIME

    _prepend_python_paths((item["drivor_root"], item["nuplan_devkit_root"]))
    sys.dont_write_bytecode = True

    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.agents.drivoR.score_module.train_pdm_scorer import PDMScorer, PDMScorerConfig
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
    from navsim.common.dataclasses import Trajectory
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
        PDMScorer as StandardPDMScorer,
        PDMScorerConfig as StandardPDMScorerConfig,
    )
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
        MultiMetricIndex,
        WeightedMetricIndex,
    )

    simulation_sampling = TrajectorySampling(
        num_poses=item["simulation_num_poses"],
        interval_length=item["simulation_interval"],
    )
    _WORKER_RUNTIME = {
        "Trajectory": Trajectory,
        "TrajectorySampling": TrajectorySampling,
        "get_trajectory_as_array": get_trajectory_as_array,
        "transform_trajectory": transform_trajectory,
        "simulator": PDMSimulator(simulation_sampling),
        "scorer": PDMScorer(simulation_sampling, PDMScorerConfig()),
        "standard_scorer": StandardPDMScorer(
            simulation_sampling, StandardPDMScorerConfig()
        ),
        "MultiMetricIndex": MultiMetricIndex,
        "WeightedMetricIndex": WeightedMetricIndex,
    }
    return _WORKER_RUNTIME


def _compute_pdm_progress(
    scorer,
    simulated_baseline,
    metric_cache,
    centerline=None,
    *,
    calculate_ego_area=False,
):
    if simulated_baseline.shape[0] != 1:
        raise ValueError("PDM reference progress requires exactly one trajectory.")
    scorer._reset(
        simulated_baseline,
        metric_cache.observation,
        metric_cache.centerline if centerline is None else centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
    )
    if calculate_ego_area:
        scorer._calculate_ego_area()
    scorer._calculate_no_at_fault_collision()
    scorer._calculate_drivable_area_compliance()
    scorer._calculate_progress()
    multiplicative_scores = scorer._multi_metrics.prod(axis=0).cumprod(axis=-1)
    return scorer._progress_raw * multiplicative_scores


def _augmented_centerline(route_points, route_mask, ego_state, line_string):
    points = np.asarray(route_points, dtype=np.float64)
    mask = np.asarray(route_mask, dtype=np.bool_)
    if points.ndim != 2 or points.shape[1] != 2 or mask.shape != points.shape[:1]:
        raise ValueError(
            "Augmented PDM route must have shapes [P,2] and [P], got "
            f"{points.shape} and {mask.shape}."
        )
    points = points[mask]
    if len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("Augmented PDM route must contain at least two finite points.")
    keep = np.concatenate(
        [np.ones(1, dtype=np.bool_), np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-6]
    )
    points = points[keep]
    if len(points) < 2:
        raise ValueError("Augmented PDM route collapses to a single point.")

    origin = ego_state.rear_axle
    cosine, sine = np.cos(origin.heading), np.sin(origin.heading)
    global_points = np.stack(
        (
            origin.x + points[:, 0] * cosine - points[:, 1] * sine,
            origin.y + points[:, 0] * sine + points[:, 1] * cosine,
        ),
        axis=-1,
    )
    centerline = line_string(global_points)
    if not np.isfinite(centerline.length) or centerline.length <= 1e-6:
        raise ValueError("Augmented PDM centerline has zero or invalid length.")
    return centerline


def _score_batch(items):
    scores = []
    validities = []
    for item in items:
        runtime = _get_worker_runtime(item)
        with lzma.open(item["cache_path"], "rb") as stream:
            metric_cache = pickle.load(stream)

        trajectory_sampling = runtime["TrajectorySampling"](
            num_poses=item["poses"].shape[1],
            interval_length=item["trajectory_interval"],
        )
        proposal_states = []
        for poses in item["poses"]:
            trajectory = runtime["Trajectory"](
                poses.astype(np.float32), trajectory_sampling
            )
            transformed = runtime["transform_trajectory"](
                trajectory, metric_cache.ego_state
            )
            proposal_states.append(
                runtime["get_trajectory_as_array"](
                    transformed,
                    runtime["simulator"].proposal_sampling,
                    metric_cache.ego_state.time_point,
                )
            )

        augmented = bool(item.get("route_augmented", False))
        if augmented:
            from shapely.geometry import LineString

            centerline = _augmented_centerline(
                item["route_points"],
                item["route_mask"],
                metric_cache.ego_state,
                LineString,
            )
            reference_poses = np.asarray(item["reference_poses"], dtype=np.float32)
            if reference_poses.shape != item["poses"].shape[1:]:
                raise ValueError(
                    "V10 reference trajectory must match proposal horizon, got "
                    f"{reference_poses.shape} and {item['poses'].shape[1:]}."
                )
            reference_trajectory = runtime["Trajectory"](
                reference_poses, trajectory_sampling
            )
            transformed_reference = runtime["transform_trajectory"](
                reference_trajectory, metric_cache.ego_state
            )
            reference_states = runtime["get_trajectory_as_array"](
                transformed_reference,
                runtime["simulator"].proposal_sampling,
                metric_cache.ego_state.time_point,
            )
            simulated = runtime["simulator"].simulate_proposals(
                np.concatenate(
                    [reference_states[None], np.stack(proposal_states)], axis=0
                ),
                metric_cache.ego_state,
            )
            pdm_progress = _compute_pdm_progress(
                runtime["standard_scorer"],
                simulated[:1],
                metric_cache,
                centerline=centerline,
                calculate_ego_area=True,
            )
            simulated = simulated[1:]
        elif hasattr(metric_cache, "pdm_progress"):
            simulated = runtime["simulator"].simulate_proposals(
                np.stack(proposal_states), metric_cache.ego_state
            )
            pdm_progress = metric_cache.pdm_progress
            centerline = metric_cache.centerline
        else:
            pdm_states = runtime["get_trajectory_as_array"](
                metric_cache.trajectory,
                runtime["simulator"].proposal_sampling,
                metric_cache.ego_state.time_point,
            )
            simulated = runtime["simulator"].simulate_proposals(
                np.concatenate([pdm_states[None], np.stack(proposal_states)]),
                metric_cache.ego_state,
            )
            pdm_progress = _compute_pdm_progress(
                runtime["standard_scorer"], simulated[:1], metric_cache
            )
            simulated = simulated[1:]
            centerline = metric_cache.centerline

        scorer = runtime["scorer"]
        scorer.score_proposals(
            simulated,
            metric_cache.observation,
            centerline,
            metric_cache.route_lane_ids,
            metric_cache.drivable_area_map,
            pdm_progress,
        )
        multi = runtime["MultiMetricIndex"]
        weighted = runtime["WeightedMetricIndex"]
        scores.append(
            np.stack(
                [
                    scorer._multi_metrics[multi.NO_COLLISION],
                    scorer._multi_metrics[multi.DRIVABLE_AREA],
                    scorer._weighted_metrics[weighted.PROGRESS],
                    scorer._weighted_metrics[weighted.TTC],
                    scorer._weighted_metrics[weighted.COMFORTABLE],
                    scorer._weighted_metrics[weighted.DRIVING_DIRECTION],
                ],
                axis=-1,
            ).astype(np.float32)
        )
        validity = np.ones(scores[-1].shape, dtype=np.bool_)
        if augmented:
            validity[..., -1] = False
        validities.append(validity)
    return scores, validities


@ray.remote(num_cpus=1)
class _PDMScoreActor:
    def __init__(self, runtime_config):
        _get_worker_runtime(runtime_config)

    def ready(self):
        return None

    def score(self, items):
        return _score_batch(items)


class PDMOracle:
    def __init__(self, config):
        self.drivor_root = str(Path(config["drivor_root"]).resolve())
        self.nuplan_devkit_root = str(Path(config["nuplan_devkit_root"]).resolve())
        self.trajectory_interval = float(config["trajectory_interval"])
        self.simulation_num_poses = int(config["simulation_num_poses"])
        self.simulation_interval = float(config["simulation_interval"])

        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = os.pathsep.join(
            path for path in (
                self.drivor_root,
                self.nuplan_devkit_root,
                existing_pythonpath,
            ) if path
        )
        _prepend_python_paths((self.drivor_root, self.nuplan_devkit_root))
        sys.dont_write_bytecode = True

        from navsim.common.dataloader import MetricCacheLoader
        from navsim.planning.utils.multithreading.worker_ray_no_torch import RayDistributedNoTorch
        from nuplan.planning.utils.multithreading.worker_utils import chunk_list

        self.metric_cache_paths = {}
        for cache_root in config["metric_cache_paths"]:
            loader = MetricCacheLoader(Path(cache_root))
            overlap = self.metric_cache_paths.keys() & loader.metric_cache_paths.keys()
            if overlap:
                raise ValueError(
                    "Duplicate PDM metric-cache tokens: "
                    f"{sorted(overlap)[:5]}"
                )
            for token, cache_path in loader.metric_cache_paths.items():
                self.metric_cache_paths[token] = _metric_cache_path(
                    cache_root, cache_path
                )
        self.chunk_list = chunk_list
        actor_count = int(config["threads_per_node"])
        self.worker = RayDistributedNoTorch(
            threads_per_node=actor_count
        )
        runtime_config = {
            "drivor_root": self.drivor_root,
            "nuplan_devkit_root": self.nuplan_devkit_root,
            "simulation_num_poses": self.simulation_num_poses,
            "simulation_interval": self.simulation_interval,
        }
        self.actors = [
            _PDMScoreActor.remote(runtime_config) for _ in range(actor_count)
        ]
        ray.get([actor.ready.remote() for actor in self.actors])

    def score(
        self,
        trajectories,
        metadata,
        *,
        route_points,
        route_masks,
        route_augmented,
        reference_trajectories,
    ):
        proposal_array = trajectories.detach().float().cpu().numpy()
        flags = route_augmented.detach().bool().cpu().numpy()
        if flags.any():
            route_array = route_points.detach().float().cpu().numpy()
            route_mask_array = route_masks.detach().bool().cpu().numpy()
            reference_array = reference_trajectories.detach().float().cpu().numpy()

        items = []
        for index, (poses, sample_metadata) in enumerate(
            zip(proposal_array, metadata)
        ):
            item = {
                "cache_path": self.metric_cache_paths[sample_metadata["anchor_token"]],
                "poses": poses,
                "drivor_root": self.drivor_root,
                "nuplan_devkit_root": self.nuplan_devkit_root,
                "trajectory_interval": self.trajectory_interval,
                "simulation_num_poses": self.simulation_num_poses,
                "simulation_interval": self.simulation_interval,
            }
            item["route_augmented"] = bool(flags[index])
            if flags[index]:
                item.update(
                    route_points=route_array[index],
                    route_mask=route_mask_array[index],
                    reference_poses=reference_array[index],
                )
            items.append(item)
        chunks = self.chunk_list(items, len(self.actors))
        score_batches = ray.get([
            actor.score.remote(chunk)
            for actor, chunk in zip(self.actors, chunks)
        ])
        scores = np.stack(
            [score for score_batch, _ in score_batches for score in score_batch]
        )
        validities = np.stack(
            [validity for _, validity_batch in score_batches for validity in validity_batch]
        )
        return (
            torch.from_numpy(scores).to(trajectories.device),
            torch.from_numpy(validities).to(trajectories.device),
        )
