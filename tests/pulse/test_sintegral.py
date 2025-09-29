import numpy as np
import pytest

import qubex as qx
from qubex.pulse import Pulse, Sintegral

dt = qx.pulse.get_sampling_period()


def test_inheritance():
    """Sintegral should inherit from Pulse."""
    assert issubclass(Sintegral, Pulse)


def test_empty_init():
    """Sintegrak should raise a TypeError if no duration is provided."""
    with pytest.raises(TypeError):
        Sintegral()  # type: ignore


def test_init():
    """Sintegral should be initialized with valid parameters."""
    pulse = Sintegral(duration=5 * dt, amplitude=1, power=2, beta=1)
    assert pulse.name == "Sintegral"
    assert pulse.length == 5
    assert pulse.duration == 5 * dt
    assert pulse.amplitude == 1
    assert pulse.power == 2
    assert pulse.drag_coef == {1: 1}
    assert pulse.values == pytest.approx(
        [
            0.04863465 + 1.38196601e-01j,
            0.69354893 + 3.61803399e-01j,
            1.0,
            0.69354893 - 3.61803399e-01j,
            0.04863465 - 1.38196601e-01j,
        ]
    )


def test_zero_duration():
    """Sintegral should be initialized with zero duration."""
    pulse = Sintegral(duration=0, amplitude=1, power=2, beta=1)
    assert pulse.duration == 0
    assert pulse.values == pytest.approx([])


def test_invalid_parameter():
    """Sintegral should raise a ValueError if power is less than 1."""
    with pytest.raises(ValueError):
        Sintegral(duration=5 * dt, amplitude=1, power=-1, beta=1)


def test_beta_and_drag_coef():
    """Sintegral should prioritize explicit drag coefficients over beta."""

    neither = Sintegral(
        duration=5 * dt,
        amplitude=1,
        power=2,
    )
    beta_only = Sintegral(
        duration=5 * dt,
        amplitude=1,
        power=2,
        beta=0.2,
    )
    drag_only = Sintegral(
        duration=5 * dt,
        amplitude=1,
        power=2,
        drag_coef={1: 0.2},
    )
    both = Sintegral(
        duration=5 * dt,
        amplitude=1,
        power=2,
        beta=0.2,
        drag_coef={1: 2.0},
    )
    assert neither.drag_coef == {1: 0.0}
    assert beta_only.drag_coef == {1: 0.2}
    assert drag_only.drag_coef == {1: 0.2}
    assert both.drag_coef == {1: 2.0}

    assert drag_only.values == pytest.approx(beta_only.values)
    assert drag_only.values != pytest.approx(both.values)
    assert neither.values != pytest.approx(both.values)
