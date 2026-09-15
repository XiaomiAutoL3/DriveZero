from typing import Optional, Tuple

import torch


def pairwise_graph(
    positions: torch.Tensor,  # [B, A, 2]
    agent_mask: torch.Tensor,  # [B, A]
    topk: Optional[int] = None,
    radius: Optional[float] = None,
    self_loop: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, A, _ = positions.shape
    pos_i = positions.unsqueeze(2)  # [B, A, 1, 2]
    pos_j = positions.unsqueeze(1)  # [B, 1, A, 2]
    rel = pos_j - pos_i  # [B, A, A, 2]
    dist = torch.linalg.norm(rel, dim=-1)  # [B, A, A]

    agent_mask = agent_mask.bool()
    valid_ij = agent_mask.unsqueeze(2) & agent_mask.unsqueeze(1)  # [B, A, A]
    not_self = ~torch.eye(A, dtype=torch.bool, device=positions.device).unsqueeze(
        0
    )  # [1, A, A]
    A_mask = valid_ij & not_self

    if radius is not None:
        A_mask = A_mask & (dist <= radius)

    if topk is not None:
        dist_masked = dist.masked_fill(~A_mask, float("inf"))
        k = min(topk, A - 1)
        _, topk_idx = torch.topk(-dist_masked, k, dim=-1)
        new_mask = torch.zeros_like(A_mask)
        new_mask.scatter_(-1, topk_idx, True)
        A_mask = new_mask & A_mask

    eyeAA = torch.eye(A, dtype=torch.bool, device=positions.device)  # [A, A]
    valid_ij = agent_mask.unsqueeze(2) & agent_mask.unsqueeze(1)  # [B, A, A]

    if self_loop:
        A_mask = A_mask | (eyeAA & valid_ij)
    else:
        row_has_neighbor = A_mask.any(dim=-1, keepdim=True)  # [B, A, 1]
        A_mask = A_mask | (~row_has_neighbor & valid_ij & eyeAA)

    return A_mask, rel, dist


def pairwise_graph_clip(
    positions: torch.Tensor,  # [B, A, 2]
    agent_mask: torch.Tensor,  # [B, A]
    topk: int = None,
    radius: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, A, _ = positions.shape
    device = positions.device
    K = topk

    pos_i = positions.unsqueeze(2)  # [B, A, 1, 2]
    pos_j = positions.unsqueeze(1)  # [B, 1, A, 2]
    rel = pos_j - pos_i  # [B, A, A, 2]
    dist = torch.linalg.norm(rel, dim=-1)  # [B, A, A]

    agent_mask = agent_mask.bool()
    valid_ij = agent_mask.unsqueeze(2) & agent_mask.unsqueeze(1)  # [B, A, A]
    not_self = ~torch.eye(A, dtype=torch.bool, device=device).unsqueeze(0)  # [1, A, A]
    A_mask_full = valid_ij & not_self

    if radius is not None:
        A_mask_full = A_mask_full & (dist <= radius)

    dist_masked = dist.masked_fill(~A_mask_full, float("inf"))
    k_sel = min(A, K)
    topk_vals, topk_idx = torch.topk(
        dist_masked, k=k_sel, dim=-1, largest=False, sorted=True
    )  # [B, A, k_sel]

    if k_sel < K:
        pad_n = K - k_sel
        self_idx_pad = (
            torch.arange(A, device=device).view(1, A, 1).expand(B, A, pad_n)
        )  # [B, A, pad]
        pad_vals = torch.full((B, A, pad_n), float("inf"), device=device)
        topk_idx = torch.cat([topk_idx, self_idx_pad], dim=-1)  # [B, A, K]
        topk_vals = torch.cat([topk_vals, pad_vals], dim=-1)  # [B, A, K]

    self_idx_full = (
        torch.arange(A, device=device).view(1, A, 1).expand(B, A, K)
    )  # [B, A, K]
    is_pad = torch.isinf(topk_vals)  # [B, A, K]
    nbr_idx = torch.where(is_pad, self_idx_full, topk_idx)  # [B, A, K]

    dist_k = torch.gather(dist, 2, nbr_idx)  # [B, A, K]
    rel_k = torch.gather(
        rel, 2, nbr_idx.unsqueeze(-1).expand(-1, -1, -1, 2)
    )  # [B, A, K, 2]
    mask_k = torch.gather(A_mask_full, 2, nbr_idx) & (~is_pad)  # [B, A, K]

    row_has_neighbor = mask_k.any(dim=-1, keepdim=True)  # [B, A, 1]
    no_neighbor = ~row_has_neighbor  # [B, A, 1]

    if no_neighbor.any():
        self_idx0 = torch.arange(A, device=device).view(1, A).expand(B, A)  # [B, A]
        nbr_idx[..., 0] = torch.where(
            no_neighbor.squeeze(-1), self_idx0, nbr_idx[..., 0]
        )
        mask_k[..., 0] = torch.where(
            no_neighbor.squeeze(-1), torch.ones_like(mask_k[..., 0]), mask_k[..., 0]
        )
        dist_k[..., 0] = torch.where(
            no_neighbor.squeeze(-1), torch.zeros_like(dist_k[..., 0]), dist_k[..., 0]
        )

        zeros_rel = torch.zeros((B, A, 2), device=device)
        rel_k[:, :, 0, :] = torch.where(
            no_neighbor.expand(-1, -1, 2), zeros_rel, rel_k[:, :, 0, :]
        )

    return mask_k, rel_k, dist_k, nbr_idx
