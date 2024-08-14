from __future__ import annotations

from typing import Final

from pydantic.dataclasses import dataclass

from .control_system import (
    Box,
    CapChannel,
    CapPort,
    ControlSystem,
    GenChannel,
    GenPort,
    PortType,
)
from .model import Model
from .quantum_system import Chip, Mux, QuantumSystem, Qubit, Resonator
from .target import Target

DEFAULT_CONTROL_AMPLITUDE: Final = 0.03
DEFAULT_READOUT_AMPLITUDE: Final = 0.01
DEFAULT_CONTROL_VATT: Final = 3072
DEFAULT_READOUT_VATT: Final = 2048
DEFAULT_CONTROL_FSC: Final = 40527
DEFAULT_READOUT_FSC: Final = 40527
DEFAULT_CAPTURE_DELAY: Final = 7


@dataclass
class WiringInfo(Model):
    ctrl: list[tuple[Qubit, GenPort]]
    read_out: list[tuple[Mux, GenPort]]
    read_in: list[tuple[Mux, CapPort]]


@dataclass
class QubitPortSet(Model):
    ctrl_port: GenPort
    read_out_port: GenPort
    read_in_port: CapPort


@dataclass
class ControlParams(Model):
    control_amplitude: dict[str, float]
    readout_amplitude: dict[str, float]
    control_vatt: dict[str, int]
    readout_vatt: dict[int, int]
    control_fsc: dict[str, int]
    readout_fsc: dict[int, int]
    capture_delay: dict[int, int]

    def get_control_amplitude(self, qubit: str) -> float:
        return self.control_amplitude.get(qubit, DEFAULT_CONTROL_AMPLITUDE)

    def get_readout_amplitude(self, qubit: str) -> float:
        return self.readout_amplitude.get(qubit, DEFAULT_READOUT_AMPLITUDE)

    def get_control_vatt(self, qubit: str) -> int:
        return self.control_vatt.get(qubit, DEFAULT_CONTROL_VATT)

    def get_readout_vatt(self, mux: int) -> int:
        return self.readout_vatt.get(mux, DEFAULT_READOUT_VATT)

    def get_control_fsc(self, qubit: str) -> int:
        return self.control_fsc.get(qubit, DEFAULT_CONTROL_FSC)

    def get_readout_fsc(self, mux: int) -> int:
        return self.readout_fsc.get(mux, DEFAULT_READOUT_FSC)

    def get_capture_delay(self, mux: int) -> int:
        return self.capture_delay.get(mux, DEFAULT_CAPTURE_DELAY)


@dataclass
class SystemState(Model):
    quantum_system: int
    control_system: int
    wiring_info: int
    control_params: int


class ExperimentSystem:
    def __init__(
        self,
        quantum_system: QuantumSystem,
        control_system: ControlSystem,
        wiring_info: WiringInfo,
        control_params: ControlParams,
    ):
        self._quantum_system: Final = quantum_system
        self._control_system: Final = control_system
        self._wiring_info: Final = wiring_info
        self._control_params: Final = control_params
        self._system_state: Final = SystemState(
            quantum_system=quantum_system.hash,
            control_system=control_system.hash,
            wiring_info=wiring_info.hash,
            control_params=control_params.hash,
        )
        self._ge_target_dict: Final = self._create_ge_target_dict()
        self._ef_target_dict: Final = self._create_ef_target_dict()
        self._cr_target_dict: Final = self._create_cr_target_dict()
        self._readout_target_dict: Final = self._create_readout_target_dict()
        self._target_dict: Final = (
            self._ge_target_dict
            | self._ef_target_dict
            | self._cr_target_dict
            | self._readout_target_dict
        )
        self._target_gen_channel_map: Final = self._create_target_gen_channel_map()
        self._target_cap_channel_map: Final = self._create_target_cap_channel_map()
        self._qubit_port_set_map: Final = self._create_qubit_port_set_map()
        self._configure_control_system()

    @property
    def state(self) -> SystemState:
        return self._system_state

    @property
    def quantum_system(self) -> QuantumSystem:
        return self._quantum_system

    @property
    def control_system(self) -> ControlSystem:
        return self._control_system

    @property
    def wiring_info(self) -> WiringInfo:
        return self._wiring_info

    @property
    def control_params(self) -> ControlParams:
        return self._control_params

    @property
    def chip(self) -> Chip:
        return self.quantum_system.chip

    @property
    def qubits(self) -> list[Qubit]:
        return self.quantum_system.qubits

    @property
    def resonators(self) -> list[Resonator]:
        return self.quantum_system.resonators

    @property
    def boxes(self) -> list[Box]:
        return self.control_system.boxes

    @property
    def ge_targets(self) -> list[Target]:
        return list(self._ge_target_dict.values())

    @property
    def ef_targets(self) -> list[Target]:
        return list(self._ef_target_dict.values())

    @property
    def cr_targets(self) -> list[Target]:
        return list(self._cr_target_dict.values())

    @property
    def control_targets(self) -> list[Target]:
        return self.ge_targets + self.ef_targets + self.cr_targets

    @property
    def readout_targets(self) -> list[Target]:
        return list(self._readout_target_dict.values())

    @property
    def targets(self) -> list[Target]:
        return (
            self.ge_targets + self.ef_targets + self.cr_targets + self.readout_targets
        )

    @property
    def target_gen_channel_map(self) -> dict[Target, GenChannel]:
        return self._target_gen_channel_map

    @property
    def target_cap_channel_map(self) -> dict[Target, CapChannel]:
        return self._target_cap_channel_map

    def get_spectator_qubits(self, qubit: int | str) -> list[Qubit]:
        return self.quantum_system.get_spectator_qubits(qubit)

    def get_box(self, label: str) -> Box:
        return self.control_system.get_box(label)

    def get_target(self, label: str) -> Target:
        try:
            return self._target_dict[label]
        except KeyError:
            raise KeyError(f"Target `{label}` not found.")

    def get_ge_target(self, label: str) -> Target:
        label = Target.ge_label(label)
        return self.get_target(label)

    def get_ef_target(self, label: str) -> Target:
        label = Target.ef_label(label)
        return self.get_target(label)

    def get_cr_target(self, label: str) -> Target:
        label = Target.cr_label(label)
        return self.get_target(label)

    def get_readout_target(self, label: str) -> Target:
        label = Target.readout_label(label)
        return self.get_target(label)

    def get_qubit_port_set(self, qubit: int | str) -> QubitPortSet:
        if isinstance(qubit, int):
            qubit = self.qubits[qubit].label
        return self._qubit_port_set_map[qubit]

    def get_base_frequency(self, label: str) -> int:
        target = self.get_target(label)
        channel = self.target_gen_channel_map[target]
        port = self.control_system.get_port_by_id(channel.port_id)
        if isinstance(port, GenPort):
            return port.base_frequencies[channel.number]
        raise ValueError("Port is not a GenPort.")

    def get_mux_by_readout_port(self, port: GenPort | CapPort) -> Mux | None:
        if isinstance(port, CapPort):
            for mux, cap_port in self.wiring_info.read_in:
                if cap_port == port:
                    return mux
        elif isinstance(port, GenPort):
            for mux, gen_port in self.wiring_info.read_out:
                if gen_port == port:
                    return mux
        return None

    def get_qubit_by_control_port(self, port: GenPort) -> Qubit | None:
        for qubit, gen_port in self.wiring_info.ctrl:
            if gen_port == port:
                return qubit
        return None

    def get_readout_pair(self, port: CapPort) -> GenPort:
        cap_mux = self.get_mux_by_readout_port(port)
        if cap_mux is None:
            raise ValueError(f"No mux found for port: {port}")
        for gen_mux, gen_port in self.wiring_info.read_out:
            if gen_mux.index == cap_mux.index:
                return gen_port
        raise ValueError(f"No readout pair found for port: {port}")

    def find_readout_lo_nco(
        self,
        mux: Mux,
        *,
        lo_min: int = 8_000_000_000,
        lo_max: int = 11_000_000_000,
        lo_step: int = 500_000_000,
        nco_step: int = 23_437_500,
        cnco: int = 1_500_000_000,
        fnco_min: int = -234_375_000,
        fnco_max: int = +234_375_000,
    ) -> tuple[int, int, int]:
        """
        Finds the (lo, cnco, fnco) values for the readout mux.

        Parameters
        ----------
        mux : Mux
            The readout mux.
        lo_min : int, optional
            The minimum LO frequency, by default 8_000_000_000.
        lo_max : int, optional
            The maximum LO frequency, by default 11_000_000_000.
        lo_step : int, optional
            The LO frequency step, by default 500_000_000.
        nco_step : int, optional
            The NCO frequency step, by default 23_437_500.
        cnco : int, optional
            The CNCO frequency, by default 2_250_000_000.
        fnco_min : int, optional
            The minimum FNCO frequency, by default -750_000_000.
        fnco_max : int, optional
            The maximum FNCO frequency, by default +750_000_000.

        Returns
        -------
        tuple[int, int, int]
            The tuple (lo, cnco, fnco) for the readout mux.
        """
        frequencies = [resonator.frequency * 1e9 for resonator in mux.resonators]
        target_frequency = (max(frequencies) + min(frequencies)) / 2

        min_diff = float("inf")
        best_lo = None
        best_fnco = None

        for lo in range(lo_min, lo_max + 1, lo_step):
            for fnco in range(fnco_min, fnco_max + 1, nco_step):
                current_value = lo + cnco + fnco
                current_diff = abs(current_value - target_frequency)
                if current_diff < min_diff:
                    min_diff = current_diff
                    best_lo = lo
                    best_fnco = fnco
        if best_lo is None or best_fnco is None:
            raise ValueError("No valid (lo, fnco) pair found.")
        return best_lo, cnco, best_fnco

    def find_control_lo_nco(
        self,
        qubit: Qubit,
        n_channels: int,
        *,
        lo_min: int = 8_000_000_000,
        lo_max: int = 11_000_000_000,
        lo_step: int = 500_000_000,
        nco_step: int = 23_437_500,
        cnco: int = 2_250_000_000,
        fnco_min: int = -750_000_000,
        fnco_max: int = +750_000_000,
        max_diff: int = 2_000_000_000,
    ) -> tuple[int, int, tuple[int, int, int]]:
        """
        Finds the (lo, cnco, (fnco_ge, fnco_ef, fnco_cr)) values for the control qubit.

        Parameters
        ----------
        qubit : Qubit
            The control qubit.
        n_channels : int
            The number of channels.
        lo_min : int, optional
            The minimum LO frequency, by default 8_000_000_000.
        lo_max : int, optional
            The maximum LO frequency, by default 11_000_000_000.
        lo_step : int, optional
            The LO frequency step, by default 500_000_000.
        nco_step : int, optional
            The NCO frequency step, by default 23_437_500.
        cnco : int, optional
            The CNCO frequency, by default 2_250_000_000.
        fnco_min : int, optional
            The minimum FNCO frequency, by default -750_000_000.
        fnco_max : int, optional
            The maximum FNCO frequency, by default +750_000_000.
        max_diff : int, optional
            The maximum difference between CR and ge frequencies, by default 1_500_000_000.

        Returns
        -------
        tuple[int, int, tuple[int, int, int]]
            The tuple (lo, cnco, (fnco_ge, fnco_ef, fnco_cr)) for the control qubit.
        """
        ge_target = self.get_ge_target(qubit.label)
        ef_target = self.get_ef_target(qubit.label)
        cr_target = self.get_cr_target(qubit.label)

        f_ge = ge_target.frequency * 1e9
        f_ef = ef_target.frequency * 1e9
        f_cr = cr_target.frequency * 1e9

        if n_channels == 1:
            if f_ge > 0:
                f_target = f_ge
            else:
                f_target = f_cr
        elif n_channels == 3:
            targets = [
                target
                for target in [ge_target, ef_target, cr_target]
                if target.frequency > 0
            ]
            if len(targets) > 0:
                target_max = max(targets, key=lambda target: target.frequency)
                target_min = min(targets, key=lambda target: target.frequency)
                f_max = target_max.frequency * 1e9
                f_min = target_min.frequency * 1e9
                if f_max - f_min > max_diff:
                    if f_ge < f_cr < f_ge + max_diff:
                        f_target = (f_ge + f_cr) / 2
                        # print(f"{ge_target.label}    : {f_ge * 1e-9:.3f} GHz")
                        # print(f"{ef_target.label} : {f_ef * 1e-9:.3f} GHz <- ignored")
                        # print(f"{cr_target.label} : {f_cr * 1e-9:.3f} GHz")
                    else:
                        f_target = (f_ge + f_ef) / 2
                        # print(f"{ge_target.label}    : {f_ge * 1e-9:.3f} GHz")
                        # print(f"{ef_target.label} : {f_ef * 1e-9:.3f} GHz")
                        # print(f"{cr_target.label} : {f_cr * 1e-9:.3f} GHz <- ignored")

                else:
                    f_target = (f_max + f_min) / 2
            else:
                f_target = 0
        else:
            raise ValueError("Invalid number of channels: ", n_channels)

        min_diff = float("inf")
        best_lo = None
        for lo in range(lo_min, lo_max + 1, lo_step):
            current_value = lo - cnco
            current_diff = abs(current_value - f_target)
            if current_diff < min_diff:
                min_diff = current_diff
                best_lo = lo
        if best_lo is None:
            raise ValueError("No valid lo value found for: ", f_target)

        def find_fnco(target_frequency: float):
            min_diff = float("inf")
            best_fnco = None
            for fnco in range(fnco_min, fnco_max + 1, nco_step):
                current_value = abs(best_lo - cnco - fnco)
                current_diff = abs(current_value - target_frequency)
                if current_diff < min_diff:
                    min_diff = current_diff
                    best_fnco = fnco
            if best_fnco is None:
                raise ValueError("No valid fnco value found for: ", target_frequency)
            return best_fnco

        fnco_ge = find_fnco(f_ge)

        if n_channels == 1:
            return best_lo, cnco, (fnco_ge, 0, 0)

        fnco_ef = find_fnco(f_ef)
        fnco_cr = find_fnco(f_cr)

        return best_lo, cnco, (fnco_ge, fnco_ef, fnco_cr)

    def _create_ge_target_dict(self) -> dict[str, Target]:
        targets = [
            Target.ge_target(
                label=qubit.label,
                frequency=qubit.ge_frequency,
            )
            for qubit in self.qubits
        ]
        return {target.label: target for target in targets}

    def _create_ef_target_dict(self) -> dict[str, Target]:
        targets = [
            Target.ef_target(
                label=qubit.label,
                frequency=qubit.ef_frequency,
            )
            for qubit in self.qubits
        ]
        return {target.label: target for target in targets}

    def _calc_cr_target_frequency(self, qubit: Qubit) -> float:
        spectator_qubits = self.get_spectator_qubits(qubit.label)
        frequencies = [
            spectator.ge_frequency
            for spectator in spectator_qubits
            if spectator.ge_frequency > 0
        ]
        if not frequencies:
            return qubit.ge_frequency
        return sum(frequencies) / len(frequencies)

    def _create_cr_target_dict(self) -> dict[str, Target]:
        targets = [
            Target.cr_target(
                label=qubit.label,
                frequency=self._calc_cr_target_frequency(qubit),
            )
            for qubit in self.qubits
        ]
        return {target.label: target for target in targets}

    def _create_readout_target_dict(self) -> dict[str, Target]:
        targets = [
            Target.readout_target(
                label=resonator.label,
                frequency=resonator.frequency,
            )
            for resonator in self.resonators
        ]
        return {target.label: target for target in targets}

    def _create_target_gen_channel_map(self) -> dict[Target, GenChannel]:
        target_gen_channel_map = {}
        for box in self.boxes:
            for port in box.ports:
                if isinstance(port, GenPort):
                    if port.type == PortType.CTRL:
                        qubit = self.get_qubit_by_control_port(port)
                        if qubit is None:
                            continue
                        targets = [
                            self.get_ge_target(qubit.label),
                            self.get_ef_target(qubit.label),
                            self.get_cr_target(qubit.label),
                        ]
                        for target, channel in zip(targets, port.channels):
                            target_gen_channel_map[target] = channel
                    elif port.type == PortType.READ_OUT:
                        mux = self.get_mux_by_readout_port(port)
                        if mux is None:
                            continue
                        targets = [
                            self.get_readout_target(resonator.label)
                            for resonator in mux.resonators
                        ]
                        for target in targets:
                            target_gen_channel_map[target] = port.channels[0]
        return dict(
            sorted(target_gen_channel_map.items(), key=lambda target: target[0].label)
        )

    def _create_target_cap_channel_map(self) -> dict[Target, CapChannel]:
        target_cap_channel_map = {}
        for box in self.boxes:
            for port in box.ports:
                if isinstance(port, CapPort):
                    if port.type == PortType.READ_IN:
                        mux = self.get_mux_by_readout_port(port)
                        if mux is None:
                            continue
                        targets = [
                            self.get_readout_target(resonator.label)
                            for resonator in mux.resonators
                        ]
                        for target, channel in zip(targets, port.channels):
                            target_cap_channel_map[target] = channel
        return dict(
            sorted(target_cap_channel_map.items(), key=lambda target: target[0].label)
        )

    def _create_qubit_port_set_map(self) -> dict[str, QubitPortSet]:
        ctrl_port_map: dict[str, GenPort] = {}
        read_out_port_map: dict[str, GenPort] = {}
        read_in_port_map: dict[str, CapPort] = {}
        for qubit, gen_port in self.wiring_info.ctrl:
            ctrl_port_map[qubit.label] = gen_port
        for mux, gen_port in self.wiring_info.read_out:
            for resonator in mux.resonators:
                read_out_port_map[resonator.qubit] = gen_port
        for mux, cap_port in self.wiring_info.read_in:
            for resonator in mux.resonators:
                read_in_port_map[resonator.qubit] = cap_port
        return {
            qubit: QubitPortSet(
                ctrl_port=ctrl_port_map[qubit],
                read_out_port=read_out_port_map[qubit],
                read_in_port=read_in_port_map[qubit],
            )
            for qubit in ctrl_port_map
        }

    def _configure_control_system(self):
        params = self.control_params

        for box in self.boxes:
            for port in box.ports:
                if isinstance(port, GenPort):
                    port.rfswitch = "pass"
                    if port.type == PortType.READ_OUT:
                        mux = self.get_mux_by_readout_port(port)
                        if mux is None:
                            continue
                        lo, cnco, fnco = self.find_readout_lo_nco(mux=mux)
                        port.lo_freq = lo
                        port.cnco_freq = cnco
                        port.sideband = "U"
                        port.vatt = params.get_readout_vatt(mux.index)
                        port.fullscale_current = params.get_readout_fsc(mux.index)
                        port.channels[0].fnco_freq = fnco
                    elif port.type == PortType.CTRL:
                        qubit = self.get_qubit_by_control_port(port)
                        if qubit is None:
                            continue
                        lo, cnco, fncos = self.find_control_lo_nco(
                            qubit=qubit,
                            n_channels=port.n_channels,
                        )
                        port.lo_freq = lo
                        port.cnco_freq = cnco
                        port.sideband = "L"
                        port.vatt = params.get_control_vatt(qubit.label)
                        port.fullscale_current = params.get_control_fsc(qubit.label)
                        for idx, gen_channel in enumerate(port.channels):
                            gen_channel.fnco_freq = fncos[idx]
                elif isinstance(port, CapPort):
                    port.rfswitch = "open"
                    if port.type == PortType.READ_IN:
                        mux = self.get_mux_by_readout_port(port)
                        if mux is None:
                            continue
                        lo, cnco, fnco = self.find_readout_lo_nco(mux=mux)
                        port.lo_freq = lo
                        port.cnco_freq = cnco
                        for cap_channel in port.channels:
                            cap_channel.fnco_freq = fnco
                            cap_channel.ndelay = params.get_capture_delay(mux.index)
