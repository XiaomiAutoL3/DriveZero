"""Reward components used by TTS value decomposition."""

from collections.abc import Mapping

import torch

VALUE_COMPONENT_NAMES: tuple[str, ...] = (
    "hard",
    "goal",
    "soft_cross_lane",
    "soft_centerline",
    "soft_curb_clearance",
    "soft_comfort",
    "soft_ttc",
    "soft_overspeed",
    "soft_other",
)
SOFT_VALUE_COMPONENT_NAMES: tuple[str, ...] = VALUE_COMPONENT_NAMES[2:]
DECOMPOSED_VALUE_DIM = len(VALUE_COMPONENT_NAMES)

_SOFT_VALUE_COMPONENT_INDEX = {
    name: i for i, name in enumerate(SOFT_VALUE_COMPONENT_NAMES)
}

_SOFT_SCORE_ALIASES = {
    "crosslane": "soft_cross_lane",
    "cross_lane": "soft_cross_lane",
    "centerline": "soft_centerline",
    "center_line": "soft_centerline",
    "curbclearance": "soft_curb_clearance",
    "curb_clearance": "soft_curb_clearance",
    "comfort": "soft_comfort",
    "ttc": "soft_ttc",
    "overspeed": "soft_overspeed",
    "over_speed": "soft_overspeed",
}


def _canonical_soft_component_name(name: str) -> str:
    return _SOFT_SCORE_ALIASES.get(name.lower(), "soft_other")


def split_soft_product_reward(
    soft_scores: Mapping[str, torch.Tensor],
    soft_product_reward: torch.Tensor,
) -> torch.Tensor:
    """Split a multiplicative soft reward into additive conserved parts.

    The split attributes the already-computed product reward to each active
    soft score according to how far that score is below 1.0. If all active
    scores are perfect, the product reward is split evenly across them.
    """
    parts = soft_product_reward.new_zeros(
        (*soft_product_reward.shape, len(SOFT_VALUE_COMPONENT_NAMES))
    )
    if not soft_scores:
        parts[..., -1] = soft_product_reward
        return parts

    badness_by_component: dict[str, torch.Tensor] = {}
    for name, score in soft_scores.items():
        component_name = _canonical_soft_component_name(name)
        badness = (1.0 - score).clamp(min=0.0)
        if component_name in badness_by_component:
            badness_by_component[component_name] = (
                badness_by_component[component_name] + badness
            )
        else:
            badness_by_component[component_name] = badness

    active_names = list(badness_by_component)
    active_badness = torch.stack(
        [badness_by_component[name] for name in active_names], dim=-1
    )
    total_badness = active_badness.sum(dim=-1, keepdim=True)
    even_weights = torch.full_like(active_badness, 1.0 / len(active_names))
    weights = torch.where(
        total_badness > 1e-8,
        active_badness / total_badness.clamp(min=1e-8),
        even_weights,
    )

    for i, name in enumerate(active_names):
        component_idx = _SOFT_VALUE_COMPONENT_INDEX[name]
        parts[..., component_idx] = weights[..., i] * soft_product_reward
    return parts
