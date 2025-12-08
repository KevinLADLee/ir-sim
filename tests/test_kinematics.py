import numpy as np
import pytest

from irsim.lib.algorithm.kinematics import (
    ackermann_kinematics,
    differential_kinematics,
    omni_kinematics,
)
from irsim.lib.behavior.behavior_methods import OmniDash


def test_differential_kinematics():
    """Test differential drive robot kinematics"""
    # Test basic movement
    state = np.array([[0], [0], [0]])  # x, y, theta
    velocity = np.array([[1], [0]])  # linear, angular
    next_state = differential_kinematics(state, velocity, 1.0)
    assert np.allclose(next_state, np.array([[1], [0], [0]]))

    # Test rotation
    velocity = np.array([[0], [1]])  # linear, angular
    next_state = differential_kinematics(state, velocity, 1.0)
    assert np.allclose(next_state, np.array([[0], [0], [1]]))

    # Test with noise
    next_state_noisy = differential_kinematics(state, velocity, 1.0, noise=True)
    assert next_state_noisy.shape == (3, 1)

    # Test angle wrapping
    state = np.array([[0], [0], [np.pi]])
    velocity = np.array([[0], [np.pi]])
    next_state = differential_kinematics(state, velocity, 1.0)
    assert np.allclose(next_state[2], 0)


def test_ackermann_kinematics():
    """Test Ackermann steering vehicle kinematics"""
    # Test steer mode
    state = np.array([[0], [0], [0], [0]])  # x, y, theta, steer_angle
    velocity = np.array([[1], [0]])  # linear, steer_angle
    next_state = ackermann_kinematics(state, velocity, 1.0, mode="steer")
    assert np.allclose(next_state[:3], np.array([[1], [0], [0]]))

    # Test angular mode
    velocity = np.array([[1], [0.1]])  # linear, angular
    next_state = ackermann_kinematics(state, velocity, 1.0, mode="angular")
    assert next_state.shape == (4, 1)

    # Test with noise
    next_state_noisy = ackermann_kinematics(state, velocity, 1.0, noise=True)
    assert next_state_noisy.shape == (4, 1)

    # Test angle wrapping
    state = np.array([[0], [0], [np.pi], [0]])
    velocity = np.array([[0], [np.pi]])
    next_state = ackermann_kinematics(state, velocity, 1.0)
    assert np.allclose(next_state[2], np.pi)


def test_omni_kinematics():
    """Test omnidirectional robot kinematics"""
    # Test basic movement
    state = np.array([[0], [0]])  # x, y
    velocity = np.array([[1], [0]])  # vx, vy
    next_state = omni_kinematics(state, velocity, 1.0)
    assert np.allclose(next_state, np.array([[1], [0]]))

    # Test diagonal movement
    velocity = np.array([[1], [1]])  # vx, vy
    next_state = omni_kinematics(state, velocity, 1.0)
    assert np.allclose(next_state, np.array([[1], [1]]))

    # Test with noise
    next_state_noisy = omni_kinematics(state, velocity, 1.0, noise=True)
    assert next_state_noisy.shape == (2, 1)

    # Test body-frame rotation and yaw update (3D velocity)
    state3 = np.array([[0], [0], [np.pi / 2]])  # facing +y
    velocity3 = np.array([[1], [0], [np.pi / 2]])  # vx in body, w
    next_state3 = omni_kinematics(state3, velocity3, 1.0)
    assert np.allclose(next_state3[0:2], np.array([[0], [1]]), atol=1e-6)
    assert np.isclose(next_state3[2, 0], 0.0)  # pi/2 + pi/2 wraps to 0

    # Backward compatibility: 2D velocity pads w=0 and rotates to world
    velocity2d = np.array([[1], [0]])  # vx only
    next_state2d = omni_kinematics(state3, velocity2d, 1.0)
    assert np.allclose(next_state2d[0:2], np.array([[0], [1]]), atol=1e-6)
    assert np.isclose(next_state2d[2, 0], state3[2, 0])


def test_omni_behavior_output_shape_and_bounds():
    """Omni behaviors should emit 3D twist with bounded yaw rate."""
    state = np.array([[0], [0], [0]])
    goal = np.array([[1], [0], [0]])
    max_vel = np.array([[1], [1], [1.5]])
    out = OmniDash(state, goal, max_vel, yaw_rate_limit=1.0, guarantee_time=0.5)
    assert out.shape == (3, 1)
    assert np.isclose(out[2, 0], np.clip(out[2, 0], -1.0, 1.0))


def test_kinematics_error_handling():
    """Test error handling in kinematics functions"""
    # Test invalid state dimensions
    state = np.array([[0], [0]])  # Too few dimensions
    velocity = np.array([[1], [0]])
    with pytest.raises(AssertionError):
        differential_kinematics(state, velocity, 1.0)

    # Test invalid velocity dimensions
    state = np.array([[0], [0], [0]])
    velocity = np.array([[1]])  # Too few dimensions
    with pytest.raises(AssertionError):
        differential_kinematics(state, velocity, 1.0)

    # Test invalid noise parameters
    state = np.array([[0], [0], [0]])
    velocity = np.array([[1], [0]])
    with pytest.raises(AssertionError):
        differential_kinematics(
            state, velocity, 1.0, noise=True, alpha=[0.03]
        )  # Too few parameters


if __name__ == "__main__":
    test_differential_kinematics()
    test_ackermann_kinematics()
    test_omni_kinematics()
    test_kinematics_error_handling()
