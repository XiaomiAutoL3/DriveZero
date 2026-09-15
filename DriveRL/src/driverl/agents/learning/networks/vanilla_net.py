"""
This module defines a simple neural network with attention mechanisms.
"""

from typing import Optional

import torch
import torch.nn as nn

from driverl.agents.learning.networks.base_network import NETWORK_REGISTER, BaseNetwork
from driverl.agents.learning.networks.graph import pairwise_graph
from driverl.agents.learning.networks.layers.attention import (
    gen_sineembed_for_position,
    mha_sdpa,
)
from driverl.env.engine.reward_decomposition import DECOMPOSED_VALUE_DIM


@NETWORK_REGISTER.register_module
class VanillaNet(BaseNetwork):
    """
    A neural network for processing agent features with attention mechanisms and lane context.
    Optimized for single ego agent control (agent at index 0).

    The network consists of the following components:
    1.  MLPs to create initial embeddings for agents and lane-boundary polylines.
    2.  A cross-attention block where the ego agent attends to all agents (O(n) instead of O(n²)).
    3.  A final MLP to produce action logits from the attended ego agent embedding.
    4.  A value head to estimate agent values.
    """

    def __init__(
        self,
        agent_feature_size: int,
        goal_feature_size: int,
        kinematics_feature_size: int,
        lane_bound_feature_size: int,
        output_size: int,
        embed_dim: int,
        num_heads: int,
        no_goal_allowed: bool,
        max_agents: Optional[int] = None,
        history_steps: int = 1,
        future_steps: int = 0,
        enable_agent_frame_masking: bool = False,
        enable_occupancy_grid: bool = True,
        is_continuous: bool = False,
        enable_value_decomposition: bool = False,
        num_goal_positions: int = 1,
        enable_ordered_goal_encoding: bool = False,
        # --- optional capacity / structure knobs (all default to legacy behavior) ---
        # Total ego→agent cross-attention layers (>=1). 1 = legacy single layer.
        num_agent_attention_layers: int = 1,
        # Ego→lane cross-attention (replaces masked.amax pooled lane token).
        enable_lane_attention: bool = False,
        num_lane_attention_layers: int = 1,
        num_heads_lane: Optional[int] = None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.no_goal_allowed = no_goal_allowed
        self.max_agents = max_agents
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.enable_agent_frame_masking = enable_agent_frame_masking
        self.enable_occupancy_grid = enable_occupancy_grid
        self.is_continuous = is_continuous
        self.enable_value_decomposition = enable_value_decomposition
        self.value_dim = DECOMPOSED_VALUE_DIM if enable_value_decomposition else 1
        self.num_goal_positions = max(1, int(num_goal_positions))
        self.enable_ordered_goal_encoding = bool(enable_ordered_goal_encoding)
        if num_agent_attention_layers < 1:
            raise ValueError(
                f"num_agent_attention_layers must be >= 1, got {num_agent_attention_layers}"
            )
        # The first layer reuses the legacy agent_self_attention / agent_ffn
        # module names so old checkpoints still match. Any additional layers
        # go into the "extra_*" ModuleLists below.
        self.num_agent_attention_layers = num_agent_attention_layers
        self.enable_lane_attention = enable_lane_attention
        self.num_lane_attention_layers = (
            num_lane_attention_layers if enable_lane_attention else 0
        )
        self.num_heads_lane = (
            num_heads_lane if num_heads_lane is not None else num_heads
        )

        # MLPs for initial feature embedding
        if self.history_steps > 1:
            self.agent_mlp = nn.Sequential(
                self.layer_init(nn.Linear(agent_feature_size + 1, embed_dim // 4)),
                nn.ReLU(),
                self.layer_init(nn.Linear(embed_dim // 4, embed_dim)),
            )
        else:
            self.agent_mlp = nn.Sequential(
                self.layer_init(nn.Linear(agent_feature_size, embed_dim)),
                nn.ReLU(),
                self.layer_init(nn.Linear(embed_dim, embed_dim)),
            )
        self.kinematics_mlp = nn.Sequential(
            self.layer_init(nn.Linear(kinematics_feature_size, embed_dim)),
            nn.ReLU(),
            self.layer_init(nn.Linear(embed_dim, embed_dim)),
        )
        self.lane_bound_mlp = nn.Sequential(
            self.layer_init(nn.Linear(lane_bound_feature_size, embed_dim // 4)),
            nn.ReLU(),
            self.layer_init(nn.Linear(embed_dim // 4, embed_dim)),
        )
        if self.enable_occupancy_grid:
            self.occ_conv = nn.Sequential(
                self.layer_init(
                    nn.Conv1d(2, embed_dim // 8, kernel_size=5, stride=4, padding=2)
                ),
                nn.ReLU(),
                self.layer_init(
                    nn.Conv1d(
                        embed_dim // 8, embed_dim, kernel_size=3, stride=2, padding=1
                    )
                ),
            )
            self.occ_pool = nn.AdaptiveMaxPool1d(1)
        else:
            self.occ_conv = None
            self.occ_pool = None

        # Agent cross-attention block (ego to all agents). First layer keeps
        # legacy flat names so old checkpoints load without key remapping.
        self.agent_ln1 = nn.LayerNorm(embed_dim)
        self.agent_self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.agent_ln2 = nn.LayerNorm(embed_dim)
        self.agent_ffn = nn.Sequential(
            self.layer_init(nn.Linear(embed_dim, embed_dim * 4)),
            nn.ReLU(),
            self.layer_init(nn.Linear(embed_dim * 4, embed_dim)),
        )

        # Optional extra ego→agent cross-attention layers. Grouped per-layer
        # under distinct names; legacy checkpoints don't contain these keys
        # and default-initialize through the flexible loader.
        self.extra_agent_layers = nn.ModuleList(
            nn.ModuleDict(
                {
                    "ln1": nn.LayerNorm(embed_dim),
                    "attn": nn.MultiheadAttention(
                        embed_dim=embed_dim, num_heads=num_heads, batch_first=True
                    ),
                    "ln2": nn.LayerNorm(embed_dim),
                    "ffn": nn.Sequential(
                        self.layer_init(nn.Linear(embed_dim, embed_dim * 4)),
                        nn.ReLU(),
                        self.layer_init(nn.Linear(embed_dim * 4, embed_dim)),
                    ),
                }
            )
            for _ in range(self.num_agent_attention_layers - 1)
        )

        # Optional ego→lane cross-attention. When disabled we keep the legacy
        # masked-amax global lane token and default_lane_embedding.
        if self.enable_lane_attention:
            self.lane_attn_ln_q = nn.ModuleList(
                [nn.LayerNorm(embed_dim) for _ in range(self.num_lane_attention_layers)]
            )
            self.lane_attn_ln_kv = nn.ModuleList(
                [nn.LayerNorm(embed_dim) for _ in range(self.num_lane_attention_layers)]
            )
            self.lane_cross_attn = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        embed_dim=embed_dim,
                        num_heads=self.num_heads_lane,
                        batch_first=True,
                    )
                    for _ in range(self.num_lane_attention_layers)
                ]
            )
            self.lane_attn_ln2 = nn.ModuleList(
                [nn.LayerNorm(embed_dim) for _ in range(self.num_lane_attention_layers)]
            )
            self.lane_attn_ffn = nn.ModuleList(
                [
                    nn.Sequential(
                        self.layer_init(nn.Linear(embed_dim, embed_dim * 4)),
                        nn.ReLU(),
                        self.layer_init(nn.Linear(embed_dim * 4, embed_dim)),
                    )
                    for _ in range(self.num_lane_attention_layers)
                ]
            )

        self.default_lane_embedding = nn.Parameter(torch.randn(1, 1, embed_dim))

        goal_mlp_input_dim = (
            embed_dim * self.num_goal_positions
            if self.enable_ordered_goal_encoding
            else embed_dim
        )
        self.goal_position_mlp = (
            nn.Sequential(
                self.layer_init(nn.Linear(goal_mlp_input_dim, embed_dim)),
                nn.ReLU(),
            )
            if self.num_goal_positions > 1
            else None
        )

        no_goal_label_dim = 1 if no_goal_allowed else 0
        self.combiner_mlp = nn.Sequential(
            self.layer_init(
                nn.Linear(embed_dim * 4 + no_goal_label_dim, embed_dim * 4)
            ),
            nn.ReLU(),
        )

        # Action heads
        if self.is_continuous:
            # Beta distribution policy: two heads output raw parameters that are
            # passed through softplus(x) + 1 to produce α, β > 1.
            # With α = β ≈ 2 the initial distribution is a symmetric bell
            # centred at 0.5 (the midpoint of the action range).
            # softplus(0.54) + 1 ≈ 2.0, so we use bias_const=0.54.
            _beta_init_bias = 0.54
            self.final_mlp = nn.Sequential(
                self.layer_init(nn.Linear(embed_dim * 4, embed_dim * 4)),
                nn.ReLU(),
                self.layer_init(
                    nn.Linear(embed_dim * 4, output_size * 2),
                    std=0.01,
                    bias_const=_beta_init_bias,
                ),
            )
        else:
            # Discrete policy: single head outputs logits.
            self.final_mlp = nn.Sequential(
                self.layer_init(nn.Linear(embed_dim * 4, embed_dim * 4)),
                nn.ReLU(),
                self.layer_init(nn.Linear(embed_dim * 4, output_size), std=0.01),
            )

        # Value head for each agent
        self.value_head = nn.Sequential(
            self.layer_init(nn.Linear(embed_dim * 4, embed_dim * 4)),
            nn.ReLU(),
            self.layer_init(nn.Linear(embed_dim * 4, self.value_dim), std=1.0),
        )

    def _iter_agent_attn_layers(self):
        """Yield (ln1, attn, ln2, ffn) per layer.

        First layer uses legacy flat module names (agent_ln1/...) so old
        checkpoints load without key remapping; extras come from
        extra_agent_layers. Isolating this here keeps forward uniform.
        """
        yield (
            self.agent_ln1,
            self.agent_self_attention,
            self.agent_ln2,
            self.agent_ffn,
        )
        for layer in self.extra_agent_layers:
            yield layer["ln1"], layer["attn"], layer["ln2"], layer["ffn"]

    def forward(
        self,
        agent_features: torch.Tensor,
        goal_features: torch.Tensor,
        kinematics_features: torch.Tensor,
        agent_mask: torch.Tensor,
        lane_bound_features: torch.Tensor,
        lane_bound_mask: torch.Tensor,
        occ_points: torch.Tensor,
        occ_mask: torch.Tensor,
        visible_mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Forward pass for the VanillaNet model.

        Args:
            agent_features: Tensor of shape [B, A, history_steps, agent_feature_size]
            goal_features: Tensor of shape [B, A, goal_feature_size]. Goal
                coordinates are stored as flattened ``(x, y)`` pairs, followed
                by an optional goal-reached flag when no-goal mode is enabled.
            kinematics_features: Tensor of shape [B, A, kinematics_feature_size]
            agent_mask: Tensor of shape [B, A, history_steps] where 1 is valid, 0 is masked.
            lane_bound_features: Tensor of shape [B, M, C]
            lane_bound_mask: Tensor of shape [B, M] where 1 is valid, 0 is masked.
            occ_points: Tensor [B, R, 2] with per-ray hit points in ego frame.
            occ_mask: Tensor [B, R] with valid hits (True = hit).
            visible_mask: Optional tensor [B, A, history_steps, A] with truthy entries marking visible edges.

        Returns:
            Tuple of (logits, values) where:
            - logits: Tensor of shape [B, A, output_size] representing action logits
            - values: Tensor of shape [B, 1, value_dim] representing ego values
        """
        B, A, _, _ = agent_features.shape

        # Create boolean masks for attention (True means ignore)
        agent_padding_mask = ~(agent_mask.any(dim=-1))  # [B, A]

        # 1. Initial Embeddings
        # Agent features: learned MLP embedding
        if self.history_steps > 1:
            current_step = self.history_steps - self.future_steps - 1
            time_info = torch.arange(
                self.history_steps, device=agent_features.device
            ) / max(current_step, 1)
            time_info = time_info.view(1, 1, self.history_steps, 1).expand(
                B, A, self.history_steps, 1
            )
            time_info = time_info * agent_mask.unsqueeze(-1)

            agent_features_with_time_info = torch.cat(
                [
                    agent_features,  # [B, A, history_steps, agent_feature_size]
                    time_info,  # [B, A, history_steps, 1]
                ],
                dim=-1,
            )

            frame_embeddings = self.agent_mlp(
                agent_features_with_time_info
            )  # [B, A, history_steps, D]
            if self.enable_agent_frame_masking:
                valid_frames = agent_mask.bool().unsqueeze(-1)
                frame_embeddings = frame_embeddings.masked_fill(
                    ~valid_frames, -torch.inf
                )
            pooled_frame_embeddings, _ = torch.max(frame_embeddings, dim=2)  # [B, A, D]
            if self.enable_agent_frame_masking:
                agent_embeddings = pooled_frame_embeddings.masked_fill(
                    ~valid_frames.any(dim=2), 0
                )
            else:
                agent_embeddings = pooled_frame_embeddings
        else:
            agent_features = agent_features.squeeze(2)
            agent_embeddings = self.agent_mlp(agent_features)

        # Goal features. For the legacy single-goal case, preserve the original
        # pure sinusoidal embedding. For multiple goals, embed each goal with
        # the same MLP and mean-pool so the model is invariant to goal order and
        # its parameter shapes do not depend on the goal count.
        # We only use the first agent from A dimension, which is ego agent.
        ego_goal_features = goal_features[:, :1, :]
        goal_label_dim = 1 if self.no_goal_allowed else 0
        goal_coord_width = ego_goal_features.shape[-1] - goal_label_dim
        if goal_coord_width < 2 or goal_coord_width % 2 != 0:
            raise ValueError(
                "goal_features must contain flattened (x, y) goal coordinates."
            )
        goal_count = goal_coord_width // 2
        goal_positions = ego_goal_features[..., :goal_coord_width].reshape(
            B, 1, goal_count, 2
        )
        goal_reached = (
            ego_goal_features[..., goal_coord_width:]
            if goal_label_dim
            else ego_goal_features[..., goal_coord_width:goal_coord_width]
        )
        if goal_count == 1 or self.goal_position_mlp is None:
            goal_embeddings = gen_sineembed_for_position(
                goal_positions[:, :, 0, :], self.embed_dim
            )
        else:
            goal_pos_embeddings = gen_sineembed_for_position(
                goal_positions, self.embed_dim
            )
            if self.enable_ordered_goal_encoding:
                goal_embeddings = self.goal_position_mlp(
                    goal_pos_embeddings.flatten(start_dim=2)
                )
            else:
                goal_pos_embeddings = self.goal_position_mlp(goal_pos_embeddings)
                goal_embeddings = goal_pos_embeddings.mean(dim=2)

        # Kinematics are learned
        # We only use the first agent from A dimension, which is ego agent.
        kinematics_embeddings = self.kinematics_mlp(kinematics_features[:, :1, :])

        # lane boundaries: learned MLP embedding + positional embeddings for start/end
        lane_bound_start = lane_bound_features[..., :2]
        lane_bound_end = lane_bound_features[..., 2:4]
        lane_bound_pos_embed = gen_sineembed_for_position(
            lane_bound_start, self.embed_dim
        ) + gen_sineembed_for_position(lane_bound_end, self.embed_dim)
        lane_bound_embeddings = (
            self.lane_bound_mlp(lane_bound_features) + lane_bound_pos_embed
        )

        lane_bound_mask_bool = lane_bound_mask.bool().unsqueeze(-1)  # [B, M, 1]
        B = lane_bound_embeddings.shape[0]
        default_embedding = self.default_lane_embedding.expand(B, 1, -1)

        if self.enable_lane_attention:
            # Lane cross-attention below will produce the lane token; skip
            # the masked-amax pooling entirely to save work.
            global_lane_embedding = default_embedding
        else:
            # Pool over all polygons to get a global lane feature using masked amax
            pooled_lane_by_polygon = torch.masked.amax(
                lane_bound_embeddings, dim=1, keepdim=True, mask=lane_bound_mask_bool
            )  # [B, 1, D]
            has_valid_lanes = lane_bound_mask_bool.any(dim=1, keepdim=True)  # [B, 1, 1]
            global_lane_embedding = torch.where(
                has_valid_lanes, pooled_lane_by_polygon, default_embedding
            )  # [B, 1, D]

        if (
            self.enable_occupancy_grid
            and self.occ_conv is not None
            and self.occ_pool is not None
        ):
            occ_points_filled = torch.where(
                occ_mask.unsqueeze(-1), occ_points, torch.full_like(occ_points, -1.0)
            )
            occ_signal = occ_points_filled.transpose(1, 2)  # [B, 2, R]
            occ_features = self.occ_pool(self.occ_conv(occ_signal)).transpose(
                1, 2
            )  # [B, 1, D]
            has_occ = occ_mask.any(dim=1, keepdim=True).unsqueeze(-1)
            global_occ_embedding = torch.where(
                has_occ, occ_features, torch.zeros_like(occ_features)
            )
        else:
            global_occ_embedding = torch.zeros_like(global_lane_embedding)

        # 2. Agent Cross-Attention Block (ego attends to all agents)
        attn_mask = None
        attn_masks = []

        if isinstance(self.max_agents, int) and self.max_agents > 0:
            # For cross-attention, we only need the first row of the adjacency matrix
            # since ego (index 0) is the only query
            graph_positions, graph_agent_mask = _latest_valid_agent_positions(
                agent_features[..., :2],
                agent_mask,
            )
            adj = pairwise_graph(
                graph_positions,
                graph_agent_mask,
                topk=self.max_agents,
            )[0].bool()
            ego_adj = adj[:, :1, :]  # [B, 1, A] - ego's connections to all agents
            per_batch_mask = ~ego_adj
            attn_masks.append(per_batch_mask)

        # TODO: Adapt for multi-frame
        if visible_mask is not None:
            visible_mask = visible_mask.bool()
            attn_masks.append(~visible_mask[:, :1, :])

        if attn_masks:
            combined_mask = attn_masks[0]
            for extra_mask in attn_masks[1:]:
                combined_mask = combined_mask | extra_mask
            attn_mask = combined_mask.repeat_interleave(self.num_heads, dim=0)
            agent_padding_mask = None

        # Ego→agent cross-attention layers. Cost per layer is O(A) since ego
        # is a single query.
        ego_embedding = agent_embeddings[:, :1, :]
        for ln1, attn, ln2, ffn in self._iter_agent_attn_layers():
            q_norm = ln1(ego_embedding)
            kv_norm = ln1(agent_embeddings)
            attn_out = mha_sdpa(
                attn,
                query=q_norm,
                key_value=kv_norm,
                num_heads=self.num_heads,
                key_padding_mask=agent_padding_mask,
                attn_mask=attn_mask,
            )
            ego_embedding = ego_embedding + attn_out
            ego_embedding = ego_embedding + ffn(ln2(ego_embedding))

        # Optional: ego→lane cross-attention. Replaces the masked.amax pooled
        # lane token. Key length M is small (topk-filtered earlier), so cost
        # per layer is O(M) with a single query.
        if self.enable_lane_attention:
            lane_kpm = ~lane_bound_mask.bool()  # [B, M] True = ignore
            # Rows with all lanes masked would make softmax NaN (and taint
            # backward). Force-unmask index 0 there so attention is always
            # well-defined; the output for these rows is overwritten with
            # default_lane_embedding after the loop.
            fully_masked = lane_kpm.all(dim=1)  # [B]
            lane_kpm = lane_kpm.clone()
            lane_kpm[:, 0] = lane_kpm[:, 0] & ~fully_masked
            lane_query = ego_embedding  # [B, 1, D]
            lane_kv = lane_bound_embeddings  # [B, M, D]
            for i in range(self.num_lane_attention_layers):
                q_norm = self.lane_attn_ln_q[i](lane_query)
                kv_norm = self.lane_attn_ln_kv[i](lane_kv)
                attn_out = mha_sdpa(
                    self.lane_cross_attn[i],
                    query=q_norm,
                    key_value=kv_norm,
                    num_heads=self.num_heads_lane,
                    key_padding_mask=lane_kpm,
                )
                lane_query = lane_query + attn_out
                ffn_in = self.lane_attn_ln2[i](lane_query)
                lane_query = lane_query + self.lane_attn_ffn[i](ffn_in)
            # Use default_lane_embedding as fallback for batches with no lanes.
            default_embedding = self.default_lane_embedding.expand(
                lane_query.shape[0], 1, -1
            )
            global_lane_embedding = torch.where(
                fully_masked.view(-1, 1, 1), default_embedding, lane_query
            )

        # 3. Combine features for EGO agent
        B, A, D = agent_embeddings.shape

        combined_features = torch.cat(
            [
                ego_embedding,
                goal_embeddings,
                kinematics_embeddings,
                global_lane_embedding + global_occ_embedding,
            ],
            dim=-1,
        )
        if self.no_goal_allowed:
            combined_features = torch.cat(
                [
                    combined_features,
                    goal_reached,
                ],
                dim=-1,
            )

        agents_with_context = self.combiner_mlp(combined_features)

        # 4. Action head(s)
        if self.is_continuous:
            # Beta distribution: softplus(raw) + 1 ensures α, β > 1 (unimodal).
            raw = self.final_mlp(agents_with_context)  # [B, 1, action_dim * 2]
            raw_alpha, raw_beta = raw.chunk(2, dim=-1)  # each [B, 1, action_dim]
            alpha = torch.nn.functional.softplus(raw_alpha) + 1.0
            beta = torch.nn.functional.softplus(raw_beta) + 1.0
            action_params = torch.cat([alpha, beta], dim=-1)  # [B, 1, action_dim * 2]
        else:
            action_params = self.final_mlp(agents_with_context)  # [B, 1, num_actions]

        # 5. Ego value head: scalar or decomposed reward-component values.
        values = self.value_head(agents_with_context)

        # Return ego-only outputs: [B, 1, output_size] and [B, 1, value_dim]
        return action_params, values


def _latest_valid_agent_positions(
    positions: torch.Tensor,
    agent_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse multi-frame agent poses to newest valid pose for graph edges."""
    if positions.dim() == 3:
        return positions, agent_mask.bool()
    if positions.dim() != 4:
        raise ValueError(
            f"positions must be [B,A,2] or [B,A,T,2], got {tuple(positions.shape)}"
        )
    valid = agent_mask.bool()
    B, A, T, _ = positions.shape
    time_idx = torch.arange(T, device=positions.device).view(1, 1, T)
    latest_idx = torch.where(valid, time_idx, torch.zeros_like(time_idx)).amax(dim=-1)
    gather_idx = latest_idx.view(B, A, 1, 1).expand(-1, -1, 1, positions.shape[-1])
    latest_positions = positions.gather(dim=2, index=gather_idx).squeeze(2)
    return latest_positions, valid.any(dim=-1)
