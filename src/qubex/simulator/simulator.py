from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Final, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import qctrlvisualizer as qv
import qutip as qt

from .system import StateAlias, System
from .pulsesequence import Sequence, SAMPLING_PERIOD

SAMPLING_PERIOD: float = 2.0  # ns
STEP_PER_SAMPLE: int = 4


class Control:
    def __init__(
        self,
        target: str,
        frequency: float,
        waveform: list | npt.NDArray,
        sampling_period: float = SAMPLING_PERIOD,
        steps_per_sample: int = STEP_PER_SAMPLE,
    ):
        """
        A control signal for a single qubit.

        Parameters
        ----------
        target : str
            The label of the qubit to control.
        frequency : float
            The frequency of the control signal.
        waveform : list | npt.NDArray
            The waveform of the control signal.
        sampling_period : float, optional
            The sampling period of the control signal, by default 2.0 ns.
        steps_per_sample : int, optional
            The number of steps per sample, by default 4.
        """
        self.target = target
        self.frequency = frequency
        self.waveform = waveform
        self.sampling_period = sampling_period
        self.steps_per_sample = steps_per_sample

    @property
    def values(self) -> npt.NDArray[np.complex128]:
        """
        The piecewise constant values of the control signal.

        Returns
        -------
        npt.NDArray[np.complex128]
            The values of the control signal.

        Notes
        -----
        The values are constant during each sample period and repeat `steps_per_sample` times.

        Examples
        --------
        >>> control = Control(
        ...     target="Q01",
        ...     frequency=5.0e9,
        ...     waveform=[1, 2, 3],
        ...     steps_per_sample=4,
        ... )
        >>> control.values
        array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
        """
        return np.repeat(self.waveform, self.steps_per_sample).astype(np.complex128)

    @property
    def times(self) -> npt.NDArray[np.float64]:
        """
        The time points of the control signal.

        Returns
        -------
        npt.NDArray[np.float64]
            The time points of the control signal.
        """
        length = len(self.values)
        return np.linspace(
            0.0,
            length * SAMPLING_PERIOD / self.steps_per_sample,
            length,
        )

    @property
    def sampling_rate(self) -> float:
        """
        The sampling rate of the control signal.

        Returns
        -------
        float
            The sampling rate of the control signal in GHz.
        """
        return self.steps_per_sample / self.sampling_period

    def plot(
        self,
        polar: bool = False,
    ) -> None:
        """
        Plot the control signal.

        Parameters
        ----------
        polar : bool, optional
            Whether to plot the control signal in polar coordinates, by default False.
        """
        durations = [self.sampling_period * 1e-9] * len(self.waveform)
        values = np.array(self.waveform, dtype=np.complex128) * 1e9
        qv.plot_controls(
            controls={
                self.target: {"durations": durations, "values": values},
            },
            polar=polar,
            figure=plt.figure(),
        )


class MultiControl:
    def __init__(
        self,
        frequencies: dict[str, float],
        waveforms: dict[str, list] | dict[str, npt.NDArray],
        sampling_period: float = SAMPLING_PERIOD,
        steps_per_sample: int = STEP_PER_SAMPLE,
    ):
        """
        Parameters
        ----------
        frequencies : dict[str, float]
            The frequencies of the control signals.
        waveforms : dict[str, list | npt.NDArray]
            The waveforms of the control signals.
        sampling_period : float, optional
            The sampling period of the control signals, by default 2.0 ns.
        steps_per_sample : int, optional
            The number of steps per sample, by default 4.

        Raises
        ------
        ValueError
            If the keys of frequencies and waveforms do not match.
            If the waveforms have different lengths.

        Examples
        --------
        >>> control = MultiControl(
        ...     frequencies={"Q01": 5.0e9, "Q02": 6.0e9},
        ...     waveforms={"Q01": [0.5, 0.5], "Q02": [0.5, -0.5]},
        ... )

        Notes
        -----
        Specify `step_per_sample` to sufficiently resolve the frequency difference of the control pulses.
        """
        if set(frequencies.keys()) != set(waveforms.keys()):
            raise ValueError("The keys of frequencies and waveforms must match.")

        if len(set(len(waveform) for waveform in waveforms.values())) > 1:
            raise ValueError("All waveforms must have the same length.")

        self.frequencies = frequencies
        self.waveforms = waveforms
        self.sampling_period = sampling_period
        self.steps_per_sample = steps_per_sample

    @property
    def frequency(self) -> float:
        """
        The average frequency of the control signals.

        Returns
        -------
        float
            The average frequency of the control signals in GHz.

        Notes
        -----
        This frequency is used to calculate the rotating frame of the control signals.
        """
        return sum(self.frequencies.values()) / len(self.frequencies)

    @property
    def values(self) -> dict[str, npt.NDArray[np.complex128]]:
        """
        The piecewise constant values of the control signals.

        Returns
        -------
        dict[str, npt.NDArray[np.complex128]]
            The values of the control signals.

        Notes
        -----
        The values are constant during each sample period and repeat `steps_per_sample` times.

        Examples
        --------
        >>> control = MultiControl(
        ...     frequencies={"Q01": 5.0e9, "Q02": 6.0e9},
        ...     waveforms={"Q01": [1, 2], "Q02": [3, 4]},
        ...     steps_per_sample=6,
        ... )
        >>> control.values
        {
            'Q01': array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]),
            'Q02': array([3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4]),
        }
        """
        waveforms = {
            target: np.repeat(waveform, self.steps_per_sample).astype(np.complex128)
            for target, waveform in self.waveforms.items()
        }
        return waveforms

    @property
    def length(self) -> int:
        """
        The length of the control signals.

        Returns
        -------
        int
            The length of the control signals.
        """
        return len(next(iter(self.values.values())))

    @property
    def times(self) -> npt.NDArray[np.float64]:
        """
        The time points of the control signals.

        Returns
        -------
        npt.NDArray[np.float64]
            The time points of the control signals.
        """
        return np.linspace(
            0.0,
            self.length * self.sampling_period / self.steps_per_sample,
            self.length,
        )

    def plot(
        self,
        polar: bool = False,
    ) -> None:
        """
        Plot the control signals.

        Parameters
        ----------
        polar : bool, optional
            Whether to plot the control signals in polar coordinates, by default False.
        """
        controls = {}
        for target, waveform in self.waveforms.items():
            durations = [self.sampling_period * 1e-9] * len(waveform)
            values = np.array(waveform, dtype=np.complex128) * 1e9
            controls[target] = {"durations": durations, "values": values}
        qv.plot_controls(
            controls=controls,
            polar=polar,
            figure=plt.figure(),
        )


@dataclass
class Result:
    """
    The result of a simulation.

    Attributes
    ----------
    system : System
        The quantum system.
    control : Control | MultiControl
        The control signal.
    states : list[qt.Qobj]
        The states of the quantum system at each time point.
    """

    system: System
    control: Control | MultiControl
    states: list[qt.Qobj]

    def substates(
        self,
        label: str,
    ) -> list[qt.Qobj]:
        """
        Extract the substates of a qubit from the states.

        Parameters
        ----------
        label : str
            The label of the qubit.
        frame : Literal["qubit", "drive"], optional
            The frame of the substates, by default "qubit".

        Returns
        -------
        list[qt.Qobj]
            The substates of the qubit.
        """
        index = self.system.index(label)
        substates = [state.ptrace(index) for state in self.states]
        return substates

    def display_bloch_sphere(
        self,
        label: str,
        frame: Literal["qubit", "drive"] = "qubit",
    ) -> None:
        """
        Display the Bloch sphere of a qubit.

        Parameters
        ----------
        label : str
            The label of the qubit.
        frame : Literal["qubit", "drive"], optional
            The frame of the Bloch sphere, by default "qubit".
        """
        substates = self.substates(label, frame)
        rho = np.array([substate.full() for substate in substates])[:, :2, :2]
        print(f"{label} in the {frame} frame")
        qv.display_bloch_sphere_from_density_matrices(rho)

    def show_last_population(
        self,
        label: Optional[str] = None,
    ) -> None:
        """
        Show the population of the last state.

        Parameters
        ----------
        label : Optional[str], optional
            The label of the qubit, by default
        """
        states = self.states if label is None else self.substates(label)
        population = states[-1].diag()
        for idx, prob in enumerate(population):
            basis = self.system.basis_labels[idx] if label is None else str(idx)
            print(f"|{basis}⟩: {prob:.3f}")

    def plot_population_dynamics(
        self,
        label: Optional[str] = None,
    ) -> None:
        """
        Plot the population dynamics of the states.

        Parameters
        ----------
        label : Optional[str], optional
            The label of the qubit, by default
        """
        states = self.states if label is None else self.substates(label)
        populations = defaultdict(list)
        for state in states:
            population = state.diag()
            population = np.clip(population, 0, 1)
            for idx, prob in enumerate(population):
                basis = self.system.basis_labels[idx] if label is None else str(idx)
                populations[rf"$|{basis}\rangle$"].append(prob)

        figure = plt.figure()
        figure.suptitle(f"Population dynamics of {label}")
        qv.plot_population_dynamics(
            self.control.times * 1e-9,
            populations,
            figure=figure,
        )

    def _sigmax(self, label: str) -> qt.Qobj:
        if label not in self.system.graph.nodes:
            raise ValueError(f"Transmon {label} does not exist.")
        index = self.system.index(label)
        ket0bra1 = (
            qt.basis(self.system.transmons[index].dimension, 0)
            * qt.basis(self.system.transmons[index].dimension, 1).dag()
        )
        ket1bra0 = (
            qt.basis(self.system.transmons[index].dimension, 1)
            * qt.basis(self.system.transmons[index].dimension, 0).dag()
        )
        return ket0bra1 + ket1bra0

    def _sigmay(self, label: str) -> qt.Qobj:
        if label not in self.system.graph.nodes:
            raise ValueError(f"Transmon {label} does not exist.")
        index = self.system.index(label)
        ket0bra1 = (
            qt.basis(self.system.transmons[index].dimension, 0)
            * qt.basis(self.system.transmons[index].dimension, 1).dag()
        )
        ket1bra0 = (
            qt.basis(self.system.transmons[index].dimension, 1)
            * qt.basis(self.system.transmons[index].dimension, 0).dag()
        )
        return 1j * ket0bra1 - 1j * ket1bra0

    def _sigmaz(self, label: str) -> qt.Qobj:
        if label not in self.system.graph.nodes:
            raise ValueError(f"Transmon {label} does not exist.")
        index = self.system.index(label)
        ket0bra0 = (
            qt.basis(self.system.transmons[index].dimension, 0)
            * qt.basis(self.system.transmons[index].dimension, 0).dag()
        )
        ket1bra1 = (
            qt.basis(self.system.transmons[index].dimension, 1)
            * qt.basis(self.system.transmons[index].dimension, 1).dag()
        )
        return ket0bra0 - ket1bra1

    def expectation_values(
        self,
        label: str,
    ) -> dict[str, list[float]]:
        index = self.system.index(label)
        substates = self.substates(label)
        sigmax = self._sigmax(label)
        sigmay = self._sigmay(label)
        sigmaz = self._sigmaz(label)

        expectation_values = {
            "x": [],
            "y": [],
            "z": [],
        }
        for _, state in enumerate(substates):
            expectation_values["x"].append(qt.expect(sigmax, state))
            expectation_values["y"].append(qt.expect(sigmay, state))
            expectation_values["z"].append(qt.expect(sigmaz, state))

        return expectation_values


class Simulator:
    def __init__(
        self,
        system: System,
    ):
        """
        A quantum simulator to simulate the dynamics of the quantum system.

        Parameters
        ----------
        system : System
            The quantum system.
        """
        self.system: Final = system

    def simulate(
        self,
        control: Control | MultiControl,
        initial_state: qt.Qobj | StateAlias | dict[str, StateAlias] = "0",
    ):
        """
        Simulate the dynamics of the quantum system.

        Parameters
        ----------
        control : Control | MultiControl
            The control signal.

        Returns
        -------
        Result
            The result of the simulation.
        """
        # convert the initial state to a Qobj
        if not isinstance(initial_state, qt.Qobj):
            initial_state = self.system.state(initial_state)

        static_hamiltonian = self.system.hamiltonian
        dynamic_hamiltonian: list = []
        collapse_operators: list = []

        for transmon in self.system.transmons:
            label = transmon.label
            a = self.system.lowering_operator(label)
            ad = a.dag()

            # rotating frame of the control frequency
            frame_frequency = control.frequency
            static_hamiltonian -= 2 * np.pi * frame_frequency * ad * a

            if isinstance(control, Control) and label == control.target:
                dynamic_hamiltonian.append([0.5 * ad, control.values])
                dynamic_hamiltonian.append([0.5 * a, np.conj(control.values)])
            elif isinstance(control, MultiControl) and label in control.frequencies:
                control_frequency = control.frequencies[label]
                delta = 2 * np.pi * (control_frequency - frame_frequency)
                values = control.values[label] * np.exp(-1j * delta * control.times)
                dynamic_hamiltonian.append([0.5 * ad, values])
                dynamic_hamiltonian.append([0.5 * a, np.conj(values)])

            decay_operator = np.sqrt(transmon.decay_rate) * a
            dephasing_operator = np.sqrt(transmon.dephasing_rate) * ad * a
            collapse_operators.append(decay_operator)
            collapse_operators.append(dephasing_operator)

        for _, coupling in enumerate(self.system.couplings):
            static_hamiltonian -= self.system.coupling_hamiltonian(coupling)

        dynamic_hamiltonian = self._drive_hamiltonian(
            control
        ) + self._coupling_hamiltonian(control)
        total_hamiltonian = [static_hamiltonian] + dynamic_hamiltonian

        result = qt.mesolve(
            H=total_hamiltonian,
            rho0=initial_state,
            tlist=control.times,
            c_ops=collapse_operators,
        )

        return Result(
            system=self.system,
            control=control,
            states=result.states,
        )

    def _drive_hamiltonian(self, control):
        drive_hamiltonian: list = []
        for (
            _,
            label,
        ) in enumerate(control.labels()):
            if "-" in label:
                b = self.system.lowering_operator(
                    control.sequence.control_qubit_2Qgate(label)
                )
                index_b = list(self.system.graph.nodes).index(
                    control.sequence.control_qubit_2Qgate(label)
                )

                delta = (
                    2
                    * np.pi
                    * (
                        control.sequence.channels[label]
                        - self.system.transmons[index_b].rotating_frame
                    )
                )
                drive_hamiltonian.append(
                    [
                        b,
                        np.exp(1j * delta * control.times)
                        * np.conj(control.values(label)),
                    ]
                )
                drive_hamiltonian.append(
                    [
                        b.dag(),
                        np.exp(-1j * delta * control.times) * control.values(label),
                    ]
                )
            else:
                a = self.system.lowering_operator(label)
                index_a = list(self.system.graph.nodes).index(label)
                delta = (
                    2
                    * np.pi
                    * (
                        control.sequence.channels[label]
                        - self.system.transmons[index_a].rotating_frame
                    )
                )
                drive_hamiltonian.append(
                    [
                        a,
                        np.exp(1j * delta * control.times)
                        * np.conj(control.values(label)),
                    ]
                )
                drive_hamiltonian.append(
                    [
                        a.dag(),
                        np.exp(-1j * delta * control.times) * control.values(label),
                    ]
                )

        return drive_hamiltonian

    def _coupling_hamiltonian(self, control):
        coupling_hamiltonian: list = []
        for _, coupling in enumerate(self.system.couplings):
            if "-" in coupling.label:
                indexa = list(self.system.graph.nodes).index(coupling.pair[0])
                a = self.system.lowering_operator(coupling.pair[0])
                indexb = list(self.system.graph.nodes).index(coupling.pair[1])
                b = self.system.lowering_operator(coupling.pair[1])

                delta = (
                    2
                    * np.pi
                    * (
                        self.system.transmons[indexa].rotating_frame
                        - self.system.transmons[indexb].rotating_frame
                    )
                )
                coupling_hamiltonian.append(
                    [
                        a.dag() * b,
                        coupling.strength * np.exp(1j * delta * control.times),
                    ]
                )
                coupling_hamiltonian.append(
                    [
                        a * b.dag(),
                        coupling.strength * np.exp(-1j * delta * control.times),
                    ]
                )
            else:
                pass
        return coupling_hamiltonian
