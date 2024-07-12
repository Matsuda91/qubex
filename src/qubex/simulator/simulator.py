from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Final, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import qctrlvisualizer as qv
import qutip as qt

from .system import StateAlias,System
from .pulsesequence import Sequence, SAMPLING_PERIOD

# from .system import StateAlias, System

STEPS_PER_SAMPLE: int = 1

@dataclass
class Control:
    sequence: Sequence
    sampling_period: float = SAMPLING_PERIOD
    steps_per_sample: int = STEPS_PER_SAMPLE

    def values(self,label) -> npt.NDArray[np.complex128]:
        sequence = np.array(self.sequence.sequences[label], dtype=np.complex128)
        return np.repeat(sequence, self.steps_per_sample)

    @property
    def times(self) -> npt.NDArray[np.float64]:
        labels = list(self.sequence.sequences.keys())
        lengthes = [len(self.sequence.sequences[label]) for label in labels]

        if all(length==lengthes[0] for length in lengthes):
            length = len(self.values(labels[-1]))
        else:
            raise ValueError("All the sequences must have the same length")
        
        return np.linspace(
            0.0,
            length * self.sampling_period / self.steps_per_sample,
            length,
        )

    def labels(self) -> list[str]:
        return list(self.sequence.sequences.keys())

    def plot(self, polar: bool = False) -> None:
        for target, sequence in self.sequence.sequences.items():
            durations = [self.sampling_period * 1e-9] * len(sequence)
            values = np.array(sequence, dtype=np.complex128) * 1e9
            qv.plot_controls(
                controls={
                    target: {"durations": durations, "values": values},
                },
                polar=polar,
                figure=plt.figure(),
            )


@dataclass
class Result:
    system: System
    control: Control
    states: list[qt.Qobj]

    def substates(
        self,
        label: str,
    ) -> list[qt.Qobj]:
        index = self.system.index(label)
        substates = [state.ptrace(index) for state in self.states]
        return substates

    def display_bloch_sphere(
        self,
        label: str,
    ) -> None:
        substates = self.substates(label)
        rho = np.array(substates).squeeze()[:, :2, :2]
        print(f"{label}")
        qv.display_bloch_sphere_from_density_matrices(rho)

    def show_last_population(
        self,
        label: Optional[str] = None,
    ) -> None:
        states = self.states if label is None else self.substates(label)
        population = states[-1].diag()
        for idx, prob in enumerate(population):
            basis = self.system.basis_labels[idx] if label is None else str(idx)
            print(f"|{basis}⟩: {prob:.3f}")

    def plot_population_dynamics(
        self,
        label: Optional[str] = None,
    ) -> None:
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


class Simulator:
    def __init__(
        self,
        system: System,
    ):
        self.system: Final = system

    def simulate(
        self,
        control: Control,
        initial_state: qt.Qobj | StateAlias | dict[str, StateAlias] = "0",
    ):
        # convert the initial state to a Qobj
        if not isinstance(initial_state, qt.Qobj):
            initial_state = self.system.state(initial_state)

        static_hamiltonian = self.system.hamiltonian
        dynamic_hamiltonian: list = []
        collapse_operators: list = []

        for transmon in self.system.transmons:
            a = self.system.lowering_operator(transmon.label)
            ad = a.dag()

            # rotating frame of the control frequency
            static_hamiltonian -= 2 * np.pi * transmon.rotating_frame * ad * a
            
            decay_operator = np.sqrt(transmon.decay_rate) * a
            dephasing_operator = np.sqrt(transmon.dephasing_rate) * ad * a
            collapse_operators.append(decay_operator)
            collapse_operators.append(dephasing_operator)

        for _, coupling in enumerate(self.system.couplings):
            static_hamiltonian -= self.system.coupling_hamiltonian(coupling)
        
        dynamic_hamiltonian =  self._drive_hamiltonian(control) + self._coupling_hamiltonian(control)
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
    
    def _drive_hamiltonian(self,control):
        drive_hamiltonian: list = []
        for _, label,in enumerate(control.labels()):
            if  "-" in label:
                b = self.system.lowering_operator(control.sequence.control_qubit_2Qgate(label))
                index_b = list(self.system.graph.nodes).index(control.sequence.control_qubit_2Qgate(label))

                delta = 2*np.pi*(control.sequence.channels[label]-self.system.transmons[index_b].rotating_frame)
                drive_hamiltonian.append([b, np.exp(1j*delta*control.times)*np.conj(control.values(label))])
                drive_hamiltonian.append([b.dag(), np.exp(-1j*delta*control.times)*control.values(label)])
            else:
                a = self.system.lowering_operator(label)
                drive_hamiltonian.append([a, np.conj(control.values(label))])
                drive_hamiltonian.append([a.dag(), control.values(label)])
                
            
        return drive_hamiltonian
    
    def _coupling_hamiltonian(self,control):
        coupling_hamiltonian: list = []
        for _, coupling in enumerate(self.system.couplings):
            if  "-" in coupling.label:
                indexa = list(self.system.graph.nodes).index(coupling.pair[0])
                a = self.system.lowering_operator(coupling.pair[0])
                indexb = list(self.system.graph.nodes).index(coupling.pair[1])
                b = self.system.lowering_operator(coupling.pair[1])

                delta = 2*np.pi*(self.system.transmons[indexa].rotating_frame-self.system.transmons[indexb].rotating_frame)
                coupling_hamiltonian.append([a.dag()*b, coupling.strength*np.exp(1j*delta*control.times)])
                coupling_hamiltonian.append([a*b.dag(), coupling.strength*np.exp(-1j*delta*control.times)])
            else:
                pass
        return coupling_hamiltonian
