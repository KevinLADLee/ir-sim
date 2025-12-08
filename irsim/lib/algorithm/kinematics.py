"""
This file is the implementation of the kinematics for different robots.

reference: Lynch, Kevin M., and Frank C. Park. Modern Robotics: Mechanics, Planning, and Control. 1st ed. Cambridge, MA: Cambridge University Press, 2017.
"""

from math import cos, sin, tan
from typing import Optional

import numpy as np

from irsim.util.random import rng
from irsim.util.util import WrapToPi


def differential_kinematics(
    state: np.ndarray,
    velocity: np.ndarray,
    step_time: float,
    noise: bool = False,
    alpha: Optional[list[float]] = None,
) -> np.ndarray:
    """
    Calculate the next state for a differential wheel robot.

    Args:
        state: A 3x1 vector [x, y, theta] representing the current position and orientation.
        velocity: A 2x1 vector [linear, angular] representing the current velocities.
        step_time: The time step for the simulation.
        noise: Boolean indicating whether to add noise to the velocity (default False).
        alpha: List of noise parameters for the velocity model (default [0.03, 0, 0, 0.03]). alpha[0] and alpha[1] are for linear velocity, alpha[2] and alpha[3] are for angular velocity.

    Returns:
        next_state: A 3x1 vector [x, y, theta] representing the next state.
    """
    if alpha is None:
        alpha = [0.03, 0, 0, 0.03]

    assert state.shape[0] >= 3
    assert velocity.shape[0] >= 2

    if noise:
        assert len(alpha) >= 4
        std_linear = np.sqrt(
            alpha[0] * (velocity[0, 0] ** 2) + alpha[1] * (velocity[1, 0] ** 2)
        )
        std_angular = np.sqrt(
            alpha[2] * (velocity[0, 0] ** 2) + alpha[3] * (velocity[1, 0] ** 2)
        )
        real_velocity = velocity + rng.normal(
            [[0], [0]], scale=[[std_linear], [std_angular]]
        )
    else:
        real_velocity = velocity

    phi = state[2, 0]
    co_matrix = np.array([[cos(phi), 0], [sin(phi), 0], [0, 1]])
    next_state = state[0:3] + co_matrix @ real_velocity * step_time
    next_state[2, 0] = WrapToPi(next_state[2, 0])

    return next_state


def ackermann_kinematics(
    state: np.ndarray,
    velocity: np.ndarray,
    step_time: float,
    noise: bool = False,
    alpha: Optional[list[float]] = None,
    mode: str = "steer",
    wheelbase: float = 1,
) -> np.ndarray:
    """
    Calculate the next state for an Ackermann steering vehicle.

    Args:
        state: A 4x1 vector [x, y, theta, steer_angle] representing the current state.
        velocity: A 2x1 vector representing the current velocities, format depends on mode.
            For "steer" mode, [linear, steer_angle] is expected.
            For "angular" mode, [linear, angular] is expected.

        step_time: The time step for the simulation.
        noise: Boolean indicating whether to add noise to the velocity (default False).
        alpha: List of noise parameters for the velocity model (default [0.03, 0, 0, 0.03]). alpha[0] and alpha[1] are for linear velocity, alpha[2] and alpha[3] are for angular velocity.
        mode: The kinematic mode, either "steer" or "angular" (default "steer").
        wheelbase: The distance between the front and rear axles (default 1).

    Returns:
        new_state: A 4x1 vector representing the next state.
    """
    if alpha is None:
        alpha = [0.03, 0, 0, 0.03]

    assert state.shape[0] >= 4
    assert velocity.shape[0] >= 2

    phi = state[2, 0]
    psi = state[3, 0]

    if noise:
        assert len(alpha) >= 4
        std_linear = np.sqrt(
            alpha[0] * (velocity[0, 0] ** 2) + alpha[1] * (velocity[1, 0] ** 2)
        )
        std_angular = np.sqrt(
            alpha[2] * (velocity[0, 0] ** 2) + alpha[3] * (velocity[1, 0] ** 2)
        )
        real_velocity = velocity + rng.normal(
            [[0], [0]], scale=[[std_linear], [std_angular]]
        )
    else:
        real_velocity = velocity

    if mode == "steer" or mode == "angular":
        co_matrix = np.array(
            [[cos(phi), 0], [sin(phi), 0], [tan(psi) / wheelbase, 0], [0, 1]]
        )

    d_state = co_matrix @ real_velocity
    new_state = state + d_state * step_time

    if mode == "steer":
        new_state[3, 0] = real_velocity[1, 0]

    new_state[2, 0] = WrapToPi(new_state[2, 0])

    return new_state


def omni_kinematics(
    state: np.ndarray,
    velocity: np.ndarray,
    step_time: float,
    noise: bool = False,
    alpha: Optional[list[float]] = None,
    frame: str = "robot",
) -> np.ndarray:
    """
    Calculate the next state for an omnidirectional robot with optional yaw rate.

    Args:
        state: 2x1 (x, y) or 3x1 (x, y, theta) vector representing the current pose.
        velocity: 2x1 [vx, vy] or 3x1 [vx, vy, w]. vx, vy are in `frame` (default robot).
        step_time: Simulation time step.
        noise: Whether to add noise to the velocity commands.
        alpha: Noise parameters. If length >= 3, alpha[2] is used for angular noise.
        frame: ``"robot"`` (default) interprets (vx, vy) in robot frame; ``"world"`` uses world frame.

    Returns:
        np.ndarray: Next state with the same dimension as input state.
    """
    if alpha is None:
        alpha = [0.03, 0, 0, 0.03, 0.01]

    assert velocity.shape[0] >= 2
    assert state.shape[0] >= 2

    # Pad velocity to [vx, vy, w]
    if velocity.shape[0] < 3:
        velocity = np.vstack((velocity, [[0.0]]))

    if noise:
        assert len(alpha) >= 2
        std_vx = np.sqrt(alpha[0])
        std_vy = np.sqrt(alpha[1] if len(alpha) > 1 else alpha[-1])
        std_w = np.sqrt(alpha[2] if len(alpha) > 2 else 0.0)
        noise_vec = rng.normal([[0], [0], [0]], scale=[[std_vx], [std_vy], [std_w]])
        real_velocity = velocity + noise_vec
    else:
        real_velocity = velocity

    vx, vy, w = real_velocity[0, 0], real_velocity[1, 0], real_velocity[2, 0]
    theta = state[2, 0] if state.shape[0] >= 3 else 0.0

    if frame == "robot":
        # Rotate body-frame velocities to world frame using current heading
        cos_th, sin_th = cos(theta), sin(theta)
        world_vx = cos_th * vx - sin_th * vy
        world_vy = sin_th * vx + cos_th * vy
    else:
        world_vx, world_vy = vx, vy

    next_state = state.copy()
    next_state[0, 0] = state[0, 0] + world_vx * step_time
    next_state[1, 0] = state[1, 0] + world_vy * step_time

    if state.shape[0] >= 3:
        next_state[2, 0] = WrapToPi(theta + w * step_time)

    return next_state
