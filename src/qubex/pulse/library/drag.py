from __future__ import annotations

from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..pulse import Pulse
from .bump import Bump
from .gaussian import Gaussian
from .raised_cosine import RaisedCosine
from .sintegral import Sintegral


class Drag(Pulse):
    """
    A class to represent a DRAG pulse.

    Parameters
    ----------
    duration : float
        Duration of the DRAG pulse in ns.
    amplitude : float
        Amplitude of the DRAG pulse.
    beta : float
        DRAG correction coefficient.
    type : Literal["Gaussian", "RaisedCosine", "Sintegral", "Bump"], optional
        Type of the pulse. Default is "Gaussian".

    Examples
    --------
    >>> pulse = Drag(
    ...     duration=100,
    ...     amplitude=1.0,
    ...     beta=1.0,
    ... )
    """

    def __init__(
        self,
        *,
        duration: float,
        amplitude: float,
        beta: float,
        type: Literal["Gaussian", "RaisedCosine", "Sintegral", "Bump"] = "Gaussian",
        drag_coef: dict[int, float] | None = None,
        sintegral_power: int | None = None,
        **kwargs,
    ):
        self.amplitude: Final = amplitude
        self.beta: Final = beta
        self.type: Final = type
        self.drag_coef: Final = drag_coef
        self.sintegral_power: Final = sintegral_power

        if duration == 0:
            values = np.array([], dtype=np.complex128)
        else:
            values = self.func(
                t=self._sampling_points(duration),
                duration=duration,
                amplitude=amplitude,
                beta=beta,
                type=type,
                drag_coef=drag_coef,
                sintegral_power=sintegral_power,
            )

        super().__init__(values, **kwargs)

    @staticmethod
    def func(
        t: ArrayLike,
        *,
        duration: float,
        amplitude: float,
        beta: float,
        type: Literal["Gaussian", "RaisedCosine", "Sintegral", "Bump"] = "Gaussian",
        drag_coef: dict[int, float] | None = None,
        sintegral_power: int | None = None,
    ) -> NDArray:
        """
        DRAG pulse function.

        Parameters
        ----------
        t : ArrayLike
            Time points at which to evaluate the pulse.
        duration : float
            Duration of the DRAG pulse in ns.
        amplitude : float
            Amplitude of the DRAG pulse.
        beta : float
            DRAG correction coefficient.
        type : Literal["Gaussian", "RaisedCosine", "Sintegral", "Bump"]
            Type of the pulse. Default is "gaussian".
        """
        if type == "Gaussian":
            return Gaussian.func(
                t=t,
                duration=duration,
                amplitude=amplitude,
                sigma=duration / 4,
                zero_bounds=True,
                beta=beta,
            )
        elif type == "RaisedCosine":
            return RaisedCosine.func(
                t=t,
                duration=duration,
                amplitude=amplitude,
                beta=beta,
            )
        elif type == "Sintegral":
            power = sintegral_power or 2
            return Sintegral.func(
                t=t,
                duration=duration,
                amplitude=amplitude,
                power=power,
                beta=beta,
                drag_coef=drag_coef or {1: beta or 0.0},
            )
        elif type == "Bump":
            return Bump.func(
                t=t,
                duration=duration,
                amplitude=amplitude,
                beta=beta,
            )
        else:
            raise ValueError(f"Unknown pulse type: {type}")
