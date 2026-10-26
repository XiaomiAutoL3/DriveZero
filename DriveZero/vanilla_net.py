"""Four-camera DriveVFM student with LoRA and direct scene trajectory decoding."""
import importlib
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from camera_augment import GridMask


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class LoRAQKV(nn.Module):
    """DrivoR-style rank-r adapters on the Q and V slices of fused QKV."""

    def __init__(self, qkv: nn.Linear, rank: int):
        super().__init__()
        self.qkv = qkv
        dim = qkv.in_features
        self.q_a = nn.Linear(dim, rank, bias=False)
        self.q_b = nn.Linear(rank, dim, bias=False)
        self.v_a = nn.Linear(dim, rank, bias=False)
        self.v_b = nn.Linear(rank, dim, bias=False)
        nn.init.kaiming_uniform_(self.q_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.q_b.weight)
        nn.init.kaiming_uniform_(self.v_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.v_b.weight)

    @property
    def in_features(self) -> int:
        return self.qkv.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        qkv[..., : self.in_features] += self.q_b(self.q_a(x))
        qkv[..., -self.in_features :] += self.v_b(self.v_a(x))
        return qkv


def inject_visual_lora(model: nn.Module, rank: int = 32) -> int:
    """Inject Q/V LoRA into every fused attention qkv projection."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    injected = 0
    for name, module in list(model.named_modules()):
        if not name.endswith(".attn.qkv") or not isinstance(module, nn.Linear):
            continue
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        adapter = LoRAQKV(module, rank)
        setattr(parent, child_name, adapter)
        for parameter in adapter.q_a.parameters():
            parameter.requires_grad_(True)
        for parameter in adapter.q_b.parameters():
            parameter.requires_grad_(True)
        for parameter in adapter.v_a.parameters():
            parameter.requires_grad_(True)
        for parameter in adapter.v_b.parameters():
            parameter.requires_grad_(True)
        injected += 1

    if injected == 0:
        raise ValueError("visual_lora requires at least one fused attention qkv layer.")
    return injected


class FrozenVisualEncoder(nn.Module):
    def __init__(self, weights: str, source: str):
        super().__init__()
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        source_path = str(Path(source).resolve())
        sys.path.insert(0, source_path)
        try:
            create_student_module = importlib.import_module("create_student")
            expected_source = Path(source_path, "create_student.py").resolve()
            actual_source = Path(create_student_module.__file__).resolve()
            if actual_source != expected_source:
                raise ImportError(
                    f"Loaded create_student from {actual_source}, "
                    f"expected {expected_source}."
                )

            original_create_model = create_student_module.create_model

            def create_model_without_pretrained(*args, **kwargs):
                kwargs["pretrained"] = False
                return original_create_model(*args, **kwargs)

            create_student_module.create_model = create_model_without_pretrained
            try:
                self.model = create_student_module.create_student(
                    checkpoint["config"]
                )
            finally:
                create_student_module.create_model = original_create_model
        finally:
            sys.path.pop(0)

        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        inject_visual_lora(self.model)
        self.model.eval()
        self.grid_mask = GridMask()
        torch.set_float32_matmul_precision("high")

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, camera_images: torch.Tensor) -> torch.Tensor:
        batch_size, num_cameras, height, width, _ = camera_images.shape
        images = camera_images.flatten(0, 1).permute(0, 3, 1, 2).contiguous().float() / 255.0
        images = self.grid_mask(images)
        with torch.set_grad_enabled(True):
            with torch.amp.autocast("cuda", torch.bfloat16, enabled=images.is_cuda):
                features = self.model(images)["backbone"].features
        return features.reshape(
            batch_size, num_cameras, (height // 16) * (width // 16), features.shape[-1]
        )


def _drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class _DrivoRDropPath(nn.Module):
    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self.drop_prob, self.training)


class DrivoRDecoderBlock(nn.Module):
    """Decoder block matching DrivoR's proposal/scorer refiner."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        projection_dropout: float,
        drop_path: float,
    ):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.self_attn_projection_dropout = nn.Dropout(projection_dropout)
        self.self_attn_drop_path = _DrivoRDropPath(drop_path)

        self.cross_attn_norm_q = nn.LayerNorm(hidden_dim)
        self.cross_attn_norm_kv = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.cross_attn_projection_dropout = nn.Dropout(projection_dropout)
        self.cross_attn_drop_path = _DrivoRDropPath(drop_path)

        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(projection_dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(projection_dropout),
        )
        self.mlp_drop_path = _DrivoRDropPath(drop_path)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        self_attn_input = self.self_attn_norm(x)
        self_attn_output = self.self_attn(
            self_attn_input,
            self_attn_input,
            self_attn_input,
            need_weights=False,
        )[0]
        x = x + self.self_attn_drop_path(
            self.self_attn_projection_dropout(self_attn_output)
        )

        normalized_memory = self.cross_attn_norm_kv(memory)
        cross_attn_output = self.cross_attn(
            self.cross_attn_norm_q(x),
            normalized_memory,
            normalized_memory,
            need_weights=False,
        )[0]
        x = x + self.cross_attn_drop_path(
            self.cross_attn_projection_dropout(cross_attn_output)
        )
        return x + self.mlp_drop_path(self.mlp(self.mlp_norm(x)))


class DrivoRTransformerDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        projection_dropout: float,
        drop_path: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            DrivoRDecoderBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                projection_dropout=projection_dropout,
                drop_path=drop_path,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            target = layer(target, memory)
        return target


class PerCameraVisualRegisterCompressor(nn.Module):
    """Compress each camera's patch tokens into camera-specific registers."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        num_cameras: int,
        num_register_tokens: int,
        projection_dropout: float,
        drop_path: float,
    ):
        super().__init__()
        self.num_cameras = int(num_cameras)
        self.num_register_tokens = int(num_register_tokens)
        self.register_tokens = nn.Parameter(
            torch.randn(
                1, self.num_cameras, self.num_register_tokens, hidden_dim
            ) * 1e-6
        )
        self.decoder = DrivoRTransformerDecoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            projection_dropout=projection_dropout,
            drop_path=drop_path,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        batch, cameras, patches, hidden_dim = patch_tokens.shape
        if cameras != self.num_cameras:
            raise ValueError(
                f"Expected {self.num_cameras} cameras, got {cameras}."
            )
        camera_memory = patch_tokens.reshape(
            batch * cameras, patches, hidden_dim
        )
        registers = self.register_tokens.expand(batch, -1, -1, -1).reshape(
            batch * cameras, self.num_register_tokens, hidden_dim
        )
        registers = self.decoder(registers, camera_memory)
        registers = self.output_norm(registers.float())
        return registers.reshape(
            batch, cameras * self.num_register_tokens, hidden_dim
        )


class DirectScenePlanningHead(nn.Module):
    def __init__(
        self,
        *,
        horizon: int,
        num_proposals: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        trajectory_head_hidden_dim: int,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.num_proposals = int(num_proposals)
        self.proposal_queries = nn.Parameter(
            torch.randn(self.num_proposals, hidden_dim)
        )

        def build_decoder():
            layer = nn.TransformerDecoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
                batch_first=True, norm_first=True,
            )
            return nn.TransformerDecoder(layer, num_layers)

        self.trajectory_decoder = build_decoder()
        self.score_decoder = build_decoder()
        self.trajectory_head = nn.Sequential(
            nn.Linear(hidden_dim, trajectory_head_hidden_dim),
            nn.LayerNorm(trajectory_head_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(trajectory_head_hidden_dim, trajectory_head_hidden_dim),
            nn.LayerNorm(trajectory_head_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(trajectory_head_hidden_dim, self.horizon * 3),
        )
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(self.horizon * 3, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.score_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Linear(hidden_dim * 4, 1),
            )
            for _ in range(6)
        ])

    def forward(
        self,
        ego_token: torch.Tensor,
        scene_memory: torch.Tensor,
        score_memory: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        queries = self.proposal_queries.unsqueeze(0).expand(
            ego_token.shape[0], -1, -1
        )
        queries = queries + ego_token.float()
        with torch.amp.autocast(
            "cuda",
            torch.bfloat16,
            enabled=False,
        ):
            proposal_features = self.trajectory_decoder(
                queries, scene_memory.float()
            )
        trajectories = self.trajectory_head(proposal_features.float()).reshape(
            ego_token.shape[0], self.num_proposals, self.horizon, 3
        )

        score_queries = self.trajectory_encoder(
            trajectories.detach().flatten(-2)
        )
        with torch.amp.autocast(
            "cuda",
            torch.bfloat16,
            enabled=False,
        ):
            score_features = self.score_decoder(
                score_queries.float(), score_memory.float()
            )
        score_features = score_features.float() + ego_token.float()
        return {
            "trajectories": trajectories,
            "proposal_score": torch.cat(
                [head(score_features) for head in self.score_heads], dim=-1
            ),
        }


class VanillaNetE2E(nn.Module):
    def __init__(self, *, planning_config: dict, camera_depth_num: int, chunk_horizon_k: int):
        super().__init__()
        embed_dim = int(planning_config["embed_dim"])
        self.visual_encoder = FrozenVisualEncoder(
            planning_config["visual_backbone_weights"],
            planning_config["visual_backbone_source"],
        )
        self.visual_proj = layer_init(nn.Linear(384, embed_dim))
        self.position_encoder = nn.Sequential(
            layer_init(nn.Linear(camera_depth_num * 3, embed_dim * 4)),
            nn.ReLU(),
            layer_init(nn.Linear(embed_dim * 4, embed_dim)),
        )
        self.scene_memory_norm = nn.LayerNorm(embed_dim)
        self.visual_register_compressor = PerCameraVisualRegisterCompressor(
            hidden_dim=embed_dim,
            num_heads=int(planning_config["planning_num_heads"]),
            num_layers=int(planning_config["num_visual_register_attention_layers"]),
            num_cameras=int(planning_config["num_visual_register_cameras"]),
            num_register_tokens=int(planning_config["num_visual_register_tokens_per_camera"]),
            projection_dropout=0.1,
            drop_path=0.2,
        )
        self.kinematics_mlp = nn.Sequential(
            layer_init(nn.Linear(9, embed_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(embed_dim, embed_dim)),
        )
        self.command_mlp = layer_init(nn.Linear(4, embed_dim))
        self.ego_fuse_mlp = nn.Sequential(
            layer_init(nn.Linear(embed_dim * 2, embed_dim * 4)),
            nn.ReLU(),
            layer_init(nn.Linear(embed_dim * 4, embed_dim)),
        )
        self.final_mlp = DirectScenePlanningHead(
            horizon=chunk_horizon_k,
            num_proposals=int(planning_config["num_trajectory_proposals"]),
            hidden_dim=embed_dim,
            num_heads=int(planning_config["planning_num_heads"]),
            num_layers=int(planning_config["planning_num_layers"]),
            trajectory_head_hidden_dim=int(planning_config["trajectory_head_hidden_dim"]),
        )

    def forward(self, camera_images, camera_geometry, navigation_features, kinematics_features):
        with torch.amp.autocast("cuda", torch.bfloat16, enabled=camera_images.is_cuda):
            navigation_embeddings = self.command_mlp(navigation_features)[:, None]
            kinematics_embeddings = self.kinematics_mlp(kinematics_features[:, :1, :])
            visual_tokens = self.visual_proj(self.visual_encoder(camera_images))
            camera_geometry = camera_geometry.to(dtype=visual_tokens.dtype) / 100.0
            geometry_tokens = camera_geometry.reshape(
                camera_geometry.shape[0], camera_geometry.shape[1],
                camera_geometry.shape[2] * camera_geometry.shape[3],
                camera_geometry.shape[4] * camera_geometry.shape[5],
            )
            visual_tokens = visual_tokens + self.position_encoder(geometry_tokens)
            ego_token = self.ego_fuse_mlp(
                torch.cat([kinematics_embeddings, navigation_embeddings], dim=-1)
            )
        patch_tokens_per_camera = self.scene_memory_norm(visual_tokens.float())
        proposal_memory = self.visual_register_compressor(patch_tokens_per_camera)
        return self.final_mlp(
            ego_token, proposal_memory, patch_tokens_per_camera.flatten(1, -2)
        ), None


def build_student(config: dict, device):
    return VanillaNetE2E(
        planning_config=config["student"]["model"],
        camera_depth_num=int(config["dataset"]["camera_depth_num"]),
        chunk_horizon_k=int(config["chunk"]["horizon_k"]),
    ).to(device)
