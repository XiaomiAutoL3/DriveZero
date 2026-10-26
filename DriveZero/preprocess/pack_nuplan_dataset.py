"""Attach synchronized nuPlan camera views and OpenScene commands to Hermes shards."""
import argparse
import bisect
import io
import json
import pickle
import queue
import sqlite3
import tarfile
import numpy as np
import torch.multiprocessing as mp
import webdataset as wds

from pathlib import Path
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from loaders.common import CAMERA_MODEL_PINHOLE
from preprocess.camera_image import CAMERA_IMAGE_HEIGHT, CAMERA_IMAGE_WIDTH


CAMERAS = ("CAM_F0", "CAM_L0", "CAM_L1", "CAM_L2", "CAM_R0", "CAM_R1", "CAM_R2", "CAM_B0")
SAMPLES_PER_SHARD = 1024
COMMAND_MAX_DELTA_US = 500_000


def qmat(q):
    w, x, y, z = (np.asarray(q, dtype=np.float64) / np.linalg.norm(q)).tolist()
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def load_scene(db_path: Path):
    placeholders = ",".join("?" for _ in CAMERAS)
    frames = {camera: [] for camera in CAMERAS}
    calib = {}

    with sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True) as conn:
        for ch, ts, fn in conn.execute(
            f"""
            select camera.channel, image.timestamp, image.filename_jpg
            from image join camera on image.camera_token = camera.token
            where camera.channel in ({placeholders})
            order by camera.channel, image.timestamp
            """,
            CAMERAS,
        ):
            frames[str(ch)].append((int(ts), str(fn)))

        for row in conn.execute(
            f"""
            select channel, translation, rotation, intrinsic, distortion, width, height
            from camera
            where channel in ({placeholders})
            """,
            CAMERAS,
        ):
            calib[str(row[0])] = row[1:]

    intrinsics, distortions, sensor2vehicles = [], [], []
    for camera in CAMERAS:
        translation, rotation, intrinsic, distortion, width, height = calib[camera]
        intrinsic = np.asarray(pickle.loads(intrinsic), dtype=np.float32)
        intrinsic[0] *= CAMERA_IMAGE_WIDTH / int(width)
        intrinsic[1] *= CAMERA_IMAGE_HEIGHT / int(height)
        dist = np.zeros(18, dtype=np.float32)
        dist[0] = CAMERA_MODEL_PINHOLE
        dist[1:6] = np.asarray(pickle.loads(distortion), dtype=np.float32)
        s2v = np.eye(4, dtype=np.float32)
        s2v[:3, :3] = qmat(pickle.loads(rotation))
        s2v[:3, 3] = np.asarray(pickle.loads(translation), dtype=np.float32)
        intrinsics.append(intrinsic)
        distortions.append(dist)
        sensor2vehicles.append(s2v)
    
    return {
        "frames": frames,
        "calib_intrinsic": np.stack(intrinsics).astype(np.float32),
        "calib_distortion": np.stack(distortions).astype(np.float32),
        "calib_sensor2vehicle": np.stack(sensor2vehicles).astype(np.float32),
    }


def nearest(frames, target):
    timestamps = [item[0] for item in frames]
    pos = bisect.bisect_left(timestamps, target)
    indexes = [i for i in (pos - 1, pos) if 0 <= i < len(frames)]
    index = min(indexes, key=lambda i: abs(timestamps[i] - target))
    return frames[index], abs(timestamps[index] - target)


def load_command_timeline(path: Path):
    with path.open("rb") as handle:
        frames = pickle.load(handle)
    timestamps = np.asarray([frame["timestamp"] for frame in frames], dtype=np.int64)
    commands = np.asarray([frame["driving_command"] for frame in frames], dtype=np.float32)
    return timestamps, commands


def match_command(timeline, target, max_delta_us=COMMAND_MAX_DELTA_US):
    timestamps, commands = timeline
    if target < timestamps[0] or target > timestamps[-1]:
        return None
    pos = bisect.bisect_left(timestamps, target)
    indexes = [i for i in (pos - 1, pos) if 0 <= i < len(timestamps)]
    index = min(indexes, key=lambda i: abs(timestamps[i] - target))
    if abs(timestamps[index] - target) > max_delta_us:
        return None
    return commands[index]


def iter_samples(tar_path: Path):
    pending = {}
    with tarfile.open(tar_path, "r:*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            if member.name.endswith(".data.npz"):
                key, field = member.name[: -len(".data.npz")], "npz"
            elif member.name.endswith(".meta.json"):
                key, field = member.name[: -len(".meta.json")], "meta"
            else:
                continue
            pending.setdefault(key, {})[field] = tar.extractfile(member).read()
            if {"npz", "meta"} <= pending[key].keys():
                item = pending.pop(key)
                yield key, item["npz"], item["meta"], json.loads(item["meta"])


def build_arrays(npz_bytes, meta, scene_data, command_timeline, sensor_root, max_delta_us=50*1000):
    import cv2

    command = match_command(command_timeline, int(meta["anchor_timestamp_us"]))
    if command is None:
        return None, "bad_command_timestamp"

    jpegs, timestamps = [], []
    for camera in CAMERAS:
        frame, delta = nearest(scene_data["frames"][camera], int(meta["anchor_timestamp_us"]))
        if delta > max_delta_us:
            return None, "bad_camera_timestamp"
        path = sensor_root / frame[1]
        if not path.is_file():
            return None, "missing_image_file"
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (CAMERA_IMAGE_WIDTH, CAMERA_IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
        _, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        jpegs.append(encoded)
        timestamps.append(frame[0])

    with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as npz:
        arrays = {name: npz[name] for name in npz.files}

    arrays["cam_timestamps"] = np.asarray(timestamps, dtype=np.int64)
    arrays["calib_intrinsic"] = scene_data["calib_intrinsic"]
    arrays["calib_distortion"] = scene_data["calib_distortion"]
    arrays["calib_sensor2vehicle"] = scene_data["calib_sensor2vehicle"]
    arrays["command"] = command
    for i, jpeg in enumerate(jpegs):
        arrays[f"cam_jpeg_{i}"] = jpeg

    return arrays, None


class NuplanSampleDataset(IterableDataset):
    def __init__(self, shard_queue, command_paths, db_dir, sensor_root):
        self.db_dir = db_dir
        self.sensor_root = sensor_root
        self.shard_queue = shard_queue
        self.command_paths = command_paths

    def __iter__(self):
        current_scene, scene_data = None, None
        command_cache = {}
        while True:
            try:
                src_shard = Path(self.shard_queue.get_nowait())
            except queue.Empty:
                return
            for key, npz_bytes, meta_bytes, meta in iter_samples(src_shard):
                scene = Path(meta["source_db"]).stem
                command_path = self.command_paths.get(scene)
                if command_path is None:
                    continue
                if scene not in command_cache:
                    command_cache[scene] = load_command_timeline(command_path)
                command_timeline = command_cache[scene]
                if scene != current_scene:
                    current_scene = scene
                    scene_data = load_scene(self.db_dir / f"{scene}.db")
                arrays, reason = build_arrays(npz_bytes, meta, scene_data, command_timeline, self.sensor_root)
                if reason:
                    continue
                buf = io.BytesIO()
                np.savez_compressed(buf, **arrays)
                safe_key = key.replace(".", "_")
                yield {"__key__": safe_key, "data.npz": buf.getvalue(), "meta.json": meta_bytes}


def pack_split(sources, out_dir, command_paths, workers, db_dir, sensor_root, frame_time_interval):
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = [shard for src_dir in sources for shard in sorted(src_dir.glob("*.tar"))]
    shard_queue = mp.Queue()
    for shard in shards:
        shard_queue.put(str(shard))
    print(
        f"packing: {len(shards)} input shards with {workers} workers -> {out_dir}",
        flush=True,
    )

    dataset = NuplanSampleDataset(shard_queue, command_paths, db_dir, sensor_root)
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=workers,
        prefetch_factor=2 if workers else None,
        collate_fn=lambda record: record,
    )

    writer = tmp = final = None
    out_index = current = written = 0
    for record in tqdm(loader, desc="packing", smoothing=0.01):
        if writer is None:
            final = out_dir / f"shard_{out_index:05d}.tar"
            tmp = final.with_suffix(".tar.tmp")
            writer = wds.TarWriter(str(tmp))
        writer.write(record)
        current += 1
        written += 1
        if current == SAMPLES_PER_SHARD:
            writer.close()
            tmp.replace(final)
            out_index += 1
            current = 0
            writer = tmp = final = None
    if writer is not None:
        writer.close()
        tmp.replace(final)

    num_output_shards = out_index + (1 if current else 0)
    print(
        f"DONE: {written} samples in {num_output_shards} shards -> {out_dir}",
        flush=True,
    )

    (out_dir / "manifest.json").write_text(json.dumps({
        "frame_time_interval": frame_time_interval,
        "samples": written,
        "samples_per_shard": SAMPLES_PER_SHARD,
        "camera_sensors": CAMERAS,
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--db-dir", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--command-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--frame-time-interval", type=float, required=True,
                        help="Seconds between source states; packing does not resample.")
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    available = {path.stem for path in args.db_dir.glob("*.db")}
    command_paths = {
        path.stem: path for path in args.command_root.glob("*.pkl") if path.stem in available
    }
    pack_split(args.sources, args.out_dir, command_paths, args.workers,
               args.db_dir.resolve(), args.sensor_root, args.frame_time_interval)


if __name__ == "__main__":
    main()
