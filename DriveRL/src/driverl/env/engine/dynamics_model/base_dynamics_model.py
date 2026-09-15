import abc
import math

import torch

from driverl.datatypes.scenario_data import ScenarioData
from driverl.env.config import EngineConfig
from driverl.env.constants import MAX_STEERING_ANGLE
from driverl.utils.registry import Registry

__all__ = ["BaseDynamicsModel", "DYNAMICS_MODEL_REGISTER"]


DYNAMICS_MODEL_REGISTER = Registry("dynamics_model")


class BaseDynamicsModel(abc.ABC):
    """Base class for dynamics model."""

    def __init__(self, config: EngineConfig | None = None):
        """Initialize the dynamic model."""
        super().__init__()
        self.config = config
        self.action_keys_tensor = None
        self.action_key_to_values = None
        if self.config is None:
            self.frame_time_interval = 0.2
        else:
            self.frame_time_interval = self.config.frame_time_interval

    @staticmethod
    def dynamics_model_factory(model_name: str, *args, **kwargs) -> "BaseDynamicsModel":
        """Factory method for dynamic model."""
        dynamics_model_name = Registry.get_model_instance_name(model_name)
        supported_dynamics_model_names = DYNAMICS_MODEL_REGISTER.module_keys
        assert dynamics_model_name in supported_dynamics_model_names, (
            f"Current only support {supported_dynamics_model_names}, but got {dynamics_model_name}."
        )
        return DYNAMICS_MODEL_REGISTER.get(dynamics_model_name)(*args, **kwargs)

    def forward(self, scenario_data: "ScenarioData", actions: torch.Tensor):
        raise NotImplementedError

    def preview(
        self,
        scenario_data: "ScenarioData",
        actions: torch.Tensor,
        *,
        duration_seconds: float,
    ):
        """Predict a state from the current frame without changing rollout state.

        The random-number state is restored after the prediction so enabling
        dynamics noise does not change the subsequent rollout result.
        """
        duration_seconds = float(duration_seconds)
        if not math.isfinite(duration_seconds) or duration_seconds <= 0.0:
            raise ValueError("duration_seconds must be a positive finite value")

        cuda_devices: list[int] = []
        if actions.is_cuda:
            device_index = actions.device.index
            cuda_devices.append(
                torch.cuda.current_device() if device_index is None else device_index
            )
        with torch.random.fork_rng(devices=cuda_devices):
            return self._forward_for_duration(
                scenario_data,
                actions,
                duration_seconds=duration_seconds,
            )

    def _forward_for_duration(
        self,
        scenario_data: "ScenarioData",
        actions: torch.Tensor,
        *,
        duration_seconds: float,
    ):
        raise NotImplementedError(
            f"{type(self).__name__} does not support dynamics preview"
        )

    def inverse(self, scenario_data: "ScenarioData"):
        """
        Infers actions from a sequence of states.

        Args:
            scenario_data (ScenarioData): Dataclass containing the full scenario data.

        Returns:
            torch.Tensor: Inferred actions.
        """
        raise NotImplementedError

    def _add_gaussian_noise(self, tensor: torch.Tensor, std: float) -> torch.Tensor:
        """Add zero-mean Gaussian noise to a tensor."""
        return tensor + torch.randn_like(tensor) * std

    def _add_beta_noise(
        self,
        tensor: torch.Tensor,
        std: float,
        alpha: float = 5.0,
    ) -> torch.Tensor:
        """
        Add zero-mean symmetric Beta noise to a tensor.

        Args:
            tensor: input tensor
            std: noise scale (similar role to Gaussian std)
            alpha: Beta(alpha, alpha), controls shape
                larger -> more concentrated near 0
        """
        # Sample Beta(alpha, alpha) in [0, 1]
        alpha_t = torch.tensor(alpha, device=tensor.device, dtype=tensor.dtype)
        beta_dist = torch.distributions.Beta(alpha_t, alpha_t)
        u = beta_dist.sample(tensor.shape)

        # Map to [-1, 1], zero-mean, symmetric
        noise = (2.0 * u - 1.0) * std
        return tensor + noise

    def apply_dynamics_noise(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        yaws: torch.Tensor,
        accelerations: torch.Tensor,
        steerings: torch.Tensor,
        yaw_rates: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Apply small Gaussian noise to dynamics outputs to simulate sensor/model noise.
        """
        # Sample bounded dynamics noise from the configured scale.
        positions = self._add_gaussian_noise(positions, std=0.005)
        # velocities = self._add_gaussian_noise(velocities, std=MAX_SPEED / 1000)
        yaws = self._add_gaussian_noise(yaws, std=0.001)
        accelerations = self._add_gaussian_noise(
            accelerations, std=self.config.max_a_long / 1000
        )
        # Steering noise scale: sigma = MAX_STEERING_ANGLE/1500.
        # Equivalent steering-wheel angle sigma ~= 432/1500 = 0.288 deg,
        # so 68%/95%/99.7% of samples lie within +-0.288/0.576/0.864 deg.
        steerings = self._add_gaussian_noise(steerings, std=MAX_STEERING_ANGLE / 2500)
        yaw_rates = self._add_gaussian_noise(yaw_rates, std=0.0005)
        return positions, velocities, yaws, accelerations, steerings, yaw_rates

    @staticmethod
    def _gather_delayed_control(
        control_all: torch.Tensor, delay: torch.Tensor | int
    ) -> torch.Tensor:
        """Return a [N,A,1] slice of control from `delay` frames ago."""
        if control_all.numel() == 0:
            raise ValueError("control_all is empty; expected a time-series tensor")
        T = control_all.shape[-1]

        if isinstance(delay, int):
            idx = max(T - delay, 0)
            return control_all[..., idx : idx + 1]

        idx = (T - delay).clamp(min=0).long().unsqueeze(-1)
        return torch.gather(control_all, -1, idx)

    def _apply_control_delay(
        self,
        control_all: torch.Tensor,
        delay: int,
        fallback_state: torch.Tensor,
        new_control: torch.Tensor,
        smoothing: float = 0.0,
    ) -> torch.Tensor:
        """Apply actuator delay with optional smoothing toward the latest control."""
        if delay == 0:
            delayed = new_control
        elif control_all.numel() == 0:
            delayed = fallback_state
        else:
            delays = torch.randint(
                low=1,
                high=delay + 1,
                size=control_all.shape[:2],
                device=control_all.device,
            )
            delayed = self._gather_delayed_control(control_all, delays)

        if smoothing <= 0.0:
            return delayed

        smoothing = max(0.0, min(1.0, float(smoothing)))
        return (1.0 - smoothing) * delayed + smoothing * new_control
