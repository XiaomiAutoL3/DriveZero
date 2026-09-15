from itertools import product

import torch

from driverl.datatypes.scenario_data import ScenarioData
from driverl.env.constants import (
    MAX_SPEED,
    MAX_STEERING_ANGLE,
    MAX_STEERING_RATE,
    MIN_SPEED,
    MIN_STEERING_ANGLE,
    MIN_STEERING_RATE,
)
from driverl.env.engine.dynamics_model.base_dynamics_model import (
    DYNAMICS_MODEL_REGISTER,
    BaseDynamicsModel,
)
from driverl.utils.geometry import (
    angle_difference,
    get_wheelbase_from_length,
    wrap_angle,
)


def apply_jerk_and_clamp_acceleration(
    current_acceleration: torch.Tensor,
    jerk: torch.Tensor,
    time_interval: float,
    min_acceleration: float,
    max_acceleration: float,
    positive_jerk_limit_when_negative_acc: float | None = None,
    positive_jerk_limit_when_nonnegative_acc: float | None = None,
) -> torch.Tensor:
    """Applies jerk to the current acceleration and clamps the result.

    This function implements a key feature of the jerk-actuated model:
    if the acceleration changes sign after applying jerk, it is set to zero.
    This helps the agent to maintain a constant velocity or to wait in place,
    resulting in smoother trajectories.

    Args:
        current_acceleration: The current acceleration.
        jerk: The jerk to apply.
        time_interval: The time interval over which to apply the jerk.
        min_acceleration: The minimum allowed acceleration.
        max_acceleration: The maximum allowed acceleration.
        positive_jerk_limit_when_negative_acc: Optional cap for positive jerk
            while current acceleration is negative.
        positive_jerk_limit_when_nonnegative_acc: Optional cap for positive jerk
            while current acceleration is zero or positive.

    Returns:
        The new acceleration.
    """
    effective_jerk = jerk
    if (
        positive_jerk_limit_when_negative_acc is not None
        and positive_jerk_limit_when_nonnegative_acc is not None
    ):
        positive_jerk_limit = torch.where(
            current_acceleration < 0,
            torch.full_like(
                jerk,
                float(positive_jerk_limit_when_negative_acc),
            ),
            torch.full_like(
                jerk,
                float(positive_jerk_limit_when_nonnegative_acc),
            ),
        )
        effective_jerk = torch.where(
            jerk > 0,
            torch.minimum(jerk, positive_jerk_limit),
            jerk,
        )

    new_acceleration = current_acceleration + effective_jerk * time_interval
    new_acceleration.clamp_(min=min_acceleration, max=max_acceleration)

    # If the acceleration changes sign, set it to zero.
    sign_product = torch.sign(current_acceleration) * torch.sign(new_acceleration)
    mask = sign_product < -1e9
    new_acceleration[mask] = 0

    return new_acceleration


@DYNAMICS_MODEL_REGISTER.register_module
class JerkBicycleModel(BaseDynamicsModel):
    def __init__(self, config=None):
        super().__init__(config)
        self._use_velocity_direction = getattr(config, "use_velocity_direction", False)
        self._setup_discrete_action_space()

    def _setup_discrete_action_space(self):
        """Configure the discrete action space for the delta model."""
        device = getattr(self.config, "device", "cpu")

        # Keep the action vocabulary symmetric around the neutral command.
        self.jerk_long = torch.linspace(
            start=self.config.min_jerk_long,
            end=self.config.max_jerk_long,
            steps=self.config.num_jerk_long_actions,
            device=device,
        )
        self.jerk_lat = torch.linspace(
            start=self.config.min_jerk_lat,
            end=self.config.max_jerk_lat,
            steps=self.config.num_jerk_lat_actions,
            device=device,
        )

        # Ensure that 0 is in the action space
        if 0 not in self.jerk_long:
            self.jerk_long = torch.cat(
                [self.jerk_long, torch.tensor([0.0], device=device)]
            )
            self.jerk_long, _ = torch.sort(self.jerk_long)
        if 0 not in self.jerk_lat:
            self.jerk_lat = torch.cat(
                [self.jerk_lat, torch.tensor([0.0], device=device)]
            )
            self.jerk_lat, _ = torch.sort(self.jerk_lat)

        products = product(self.jerk_long, self.jerk_lat)

        self.action_key_to_values = {}
        self.values_to_action_key = {}
        for action_idx, (action_1, action_2) in enumerate(products):
            self.action_key_to_values[action_idx] = [
                action_1.item(),
                action_2.item(),
            ]
            self.values_to_action_key[
                round(action_1.item(), 5),
                round(action_2.item(), 5),
            ] = action_idx

        self.action_keys_tensor = torch.tensor(
            [
                self.action_key_to_values[key]
                for key in sorted(self.action_key_to_values.keys())
            ],
            device=device,
        )
        return

    def forward(self, scenario_data: "ScenarioData", actions: torch.Tensor):
        """Applies a bicycle action to a state to produce the next state.

        Args:
            scenario_data (ScenarioData): Dataclass containing the full scenario data.
            actions: Action indices, shape [N, A, T] / [K, N, A]

        Returns:
            Tuple of (new_positions, new_velocities, new_yaws) with updated states
        """
        # Get the most recent state from the scenario data, keeping the time dimension.
        positions = scenario_data.agent_positions_all[:, :, -1:, :]
        velocities = scenario_data.agent_velocity_all[:, :, -1:, :]
        yaws = scenario_data.agent_orientation_all[:, :, -1:]
        wheelbases = get_wheelbase_from_length(
            scenario_data.agent_size_all[:, :, -1:, 0]
        )
        accelerations = scenario_data.agent_acceleration_state_all[:, :, -1:]
        steerings = scenario_data.agent_steering_state_all[:, :, -1:]

        return self._forward_internal(
            positions, velocities, accelerations, yaws, actions, wheelbases, steerings
        )

    def _forward_internal(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        accelerations: torch.Tensor,
        yaws: torch.Tensor,
        actions: torch.Tensor,
        wheelbases: torch.Tensor,
        steerings: torch.Tensor,
    ):
        """Applies the jerk-actuated bicycle model to a state to produce the next state.

        This model uses jerk (the rate of change of acceleration) as the action input.
        Reference to: https://arxiv.org/abs/2502.03349 Appendix B.2

        Args:
            positions: Agent positions, shape [N, A, T, 2] / [K, N, A, 2]
            velocities: Agent velocities, shape [N, A, T, 2] / [K, N, A, 2]
            accelerations: Agent longitudinal accelerations, shape [N, A, T]
            yaws: Agent yaws, shape [N, A, T] / [K, N, A]
            actions: Action indices, shape [N, A, T] / [K, N, A]
            wheelbases: [N, A, T] / [K, N, A]
            steerings: Agent steering angles, shape [N, A, T] / [K, N, A]

        Returns:
            Tuple of (new_positions, new_velocities, new_yaws, new_accelerations, new_steering) with updated states
        """
        t = self.frame_time_interval
        action_values = self.action_keys_tensor[actions.long()]

        jerk_long = action_values[..., 0]
        jerk_lat = action_values[..., 1]
        a_long = accelerations

        speed = torch.linalg.norm(velocities, dim=-1)

        # Recalculate a_lat from steering
        curvature = torch.tan(steerings) / wheelbases
        a_lat = speed**2 * curvature

        new_accelerations = apply_jerk_and_clamp_acceleration(
            current_acceleration=a_long,
            jerk=jerk_long,
            time_interval=t,
            min_acceleration=self.config.min_a_long,
            max_acceleration=self.config.max_a_long,
            positive_jerk_limit_when_negative_acc=self.config.max_jerk_long,
            positive_jerk_limit_when_nonnegative_acc=1.0,
        )
        new_a_lat = apply_jerk_and_clamp_acceleration(
            current_acceleration=a_lat,
            jerk=jerk_lat,
            time_interval=t,
            min_acceleration=self.config.min_a_lat,
            max_acceleration=self.config.max_a_lat,
        )

        # At standstill, clamp acc to a small negative floor (real chassis
        # reports up to ~-0.6 m/s², use -1.0 for margin).
        STANDSTILL_MIN_ACC = -1.0
        stopped = speed <= MIN_SPEED
        new_accelerations = torch.where(
            stopped & (new_accelerations < STANDSTILL_MIN_ACC),
            torch.full_like(new_accelerations, STANDSTILL_MIN_ACC),
            new_accelerations,
        )

        new_speed = (speed + 0.5 * (a_long + new_accelerations) * t).clamp(
            MIN_SPEED, MAX_SPEED
        )

        # --- Curvature and Steering --- #

        # Curvature is the rate of change of the vehicle's heading with respect to the distance traveled.
        # It is calculated as the lateral acceleration divided by the square of the speed.
        curvature = new_a_lat / ((new_speed * new_speed).clamp(min=1e-5))

        # The required steering angle is calculated from the curvature and the wheelbase.
        new_steering = torch.arctan(curvature * wheelbases)

        # The change in steering is clamped to the maximum steering rate.
        delta_steering = (new_steering - steerings).clamp(
            t * MIN_STEERING_RATE, t * MAX_STEERING_RATE
        )

        # The new steering angle is clamped to the maximum steering angle.
        new_steering = (steerings + delta_steering).clamp(
            MIN_STEERING_ANGLE, MAX_STEERING_ANGLE
        )

        # The curvature is recalculated from the new steering angle.
        # This is necessary because the steering angle may have been clamped.
        curvature = torch.tan(new_steering) / wheelbases
        new_a_lat = new_speed * new_speed * curvature

        # --- Yaw and Position --- #
        delta = 0.5 * (speed + new_speed) * t

        delta_yaws = delta * curvature

        new_yaws = wrap_angle(yaws + delta_yaws)

        mean_yaws = wrap_angle(yaws + delta_yaws * 0.5)

        x_new = positions[..., 0] + delta * torch.cos(mean_yaws)
        y_new = positions[..., 1] + delta * torch.sin(mean_yaws)
        new_positions = torch.stack([x_new, y_new], dim=-1)

        # --- Velocity --- #
        new_vx = new_speed * torch.cos(new_yaws)
        new_vy = new_speed * torch.sin(new_yaws)
        new_velocities = torch.stack([new_vx, new_vy], dim=-1)

        new_yaw_rates = delta_yaws / t

        if self.config and self.config.enable_dynamics_noise:
            (
                new_positions,
                new_velocities,
                new_yaws,
                new_accelerations,
                new_steering,
                new_yaw_rates,
            ) = self.apply_dynamics_noise(
                new_positions,
                new_velocities,
                new_yaws,
                new_accelerations,
                new_steering,
                new_yaw_rates,
            )

        return (
            new_positions,
            new_velocities,
            new_yaws,
            new_accelerations,
            new_accelerations,
            new_steering,
            new_steering,
            new_yaw_rates,
            jerk_lat,
            jerk_long,
        )

    def inverse(self, scenario_data: "ScenarioData"):
        """
        Infers bicycle actions from a sequence of states using forward model.

        Args:
            scenario_data (ScenarioData): Dataclass containing the full scenario data.

        Returns:
            torch.Tensor: Inferred action tokens, shape [N, A, T].
        """

        positions = scenario_data.agent_positions_all
        velocities = scenario_data.agent_velocity_all
        accelerations = scenario_data.agent_acceleration_state_all
        steerings = scenario_data.agent_steering_state_all
        yaws = scenario_data.agent_orientation_all
        wheelbases = get_wheelbase_from_length(scenario_data.agent_size_all[..., 0])
        npc_mask = scenario_data.npc_mask_all

        N, A, T = positions.shape[0], positions.shape[1], positions.shape[2]
        device = positions.device
        num_actions = len(self.action_keys_tensor)

        action_tokens = torch.zeros(N, A, T - 1, dtype=torch.long, device=device)
        inferred_pos = positions[..., 0, :].clone()  # [N, A, 2]
        inferred_vel = velocities[..., 0, :].clone()  # [N, A, 2]
        inferred_yaw = yaws[..., 0].clone()  # [N, A]
        inferred_a = accelerations[..., 0].clone()
        inferred_steering = steerings[..., 0].clone()

        all_actions = torch.arange(num_actions, device=device).view(
            -1, 1, 1
        )  # [K, 1, 1]
        all_actions = all_actions.expand(num_actions, N, A)

        for t in range(T - 1):
            target_pos = positions[..., t + 1, :]  # [N, A, 2]
            target_vel = velocities[..., t + 1, :]  # [N, A, 2]
            target_yaw = yaws[..., t + 1]  # [N, A]
            target_a = accelerations[..., t + 1]  # [N, A]

            current_pos_batch = inferred_pos.unsqueeze(0)  # [1, N, A, 2]
            current_vel_batch = inferred_vel.unsqueeze(0)  # [1, N, A, 2]
            current_yaw_batch = inferred_yaw.unsqueeze(0)  # [1, N, A]
            current_wheelbases = wheelbases[..., t].unsqueeze(0)  # [1, N, A]
            current_a_batch = inferred_a.unsqueeze(0)
            current_steering_batch = inferred_steering.unsqueeze(0)

            # try all actions
            (
                pos_pred,
                vel_pred,
                yaw_pred,
                a_pred,
                _,
                steering_pred,
                _,
                yaw_rate_pred,
                _,
                _,
            ) = self._forward_internal(
                positions=current_pos_batch.expand(
                    num_actions, -1, -1, -1
                ),  # [K, N, A, 2]
                velocities=current_vel_batch.expand(
                    num_actions, -1, -1, -1
                ),  # [K, N, A, 2]
                accelerations=current_a_batch.expand(num_actions, -1, -1),
                yaws=current_yaw_batch.expand(num_actions, -1, -1),  # [K, N, A]
                actions=all_actions,  # [K, 1, 1] →  [K, N, A]
                wheelbases=current_wheelbases.expand(num_actions, -1, -1),  # [k, N, A]
                steerings=current_steering_batch.expand(num_actions, -1, -1),
            )

            # print ("initial ", (current_wheelbases.expand(num_actions, -1, -1))[:, 1, 0])

            # calc error with normalized yaw difference
            K_POS, K_VEL, K_YAW, K_A = 0, 1, 0, 0
            pos_error = torch.mean(
                (pos_pred - target_pos.unsqueeze(0)) ** 2, dim=-1
            )  # [K, N, A]

            vel_error = torch.mean(
                (vel_pred - target_vel.unsqueeze(0)) ** 2, dim=-1
            )  # [K, N, A]
            a_error = (a_pred - target_a.unsqueeze(0)) ** 2  # [K, N, A]
            yaw_diff = angle_difference(yaw_pred, target_yaw.unsqueeze(0))
            yaw_error = yaw_diff**2  # [K, N, A]
            total_error = (
                K_POS * pos_error
                + K_VEL * vel_error
                + K_YAW * yaw_error
                + K_A * a_error
            )  # [K, N, A]

            best_actions = torch.argmin(total_error, dim=0)  # [N, A]

            # if npc is not valid, use previous action
            target_npc_mask = npc_mask[..., t + 1]  # [N, A]
            if t > 0:
                prev_actions = action_tokens[..., t - 1]
            else:
                prev_actions = torch.zeros_like(best_actions)
            best_actions = torch.where(target_npc_mask, best_actions, prev_actions)

            # Update inferred state for the next iteration by applying the best action
            (
                inferred_pos,
                inferred_vel,
                inferred_yaw,
                inferred_a,
                _,
                inferred_steering,
                _,
                inferred_yaw_rate,
                _,
                _,
            ) = self._forward_internal(
                positions=inferred_pos,
                velocities=inferred_vel,
                accelerations=inferred_a,
                yaws=inferred_yaw,
                actions=best_actions,
                wheelbases=wheelbases[..., t],
                steerings=inferred_steering,
            )
            action_tokens[..., t] = best_actions

        # the last step
        padding = torch.zeros(N, A, 1, device=device, dtype=torch.long)
        action_tokens = torch.cat([action_tokens, padding], dim=2)

        return action_tokens
