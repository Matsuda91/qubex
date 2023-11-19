from typing import Final, Optional

import matplotlib.pyplot as plt

import qubecalib as qc
from qubecalib.pulse import Schedule, Channel, Blank, Arbitrary, Read
from qubecalib.setupqube import run

from .pulse import Rect, Waveform
from .params import Params
from .typing import (
    QubitKey,
    QubitDict,
    IQValue,
    IQArray,
    IntArray,
    ReadoutPorts,
)
from .consts import (
    MUX,
    SAMPLING_PERIOD,
    T_CONTROL,
    T_READOUT,
    T_MARGIN,
)

CONTROL_PORTS: Final = ["port5", "port6", "port7", "port8"]

CONTROL_HIGH: Final = "_hi"
CONTROL_LOW: Final = "_lo"
READOUT_TX: Final = "TX_"
READOUT_RX: Final = "RX_"


class QubeManager:
    def __init__(
        self,
        mux_number: int,
        params: Params,
        readout_ports: ReadoutPorts,
        control_duration: int = T_CONTROL,
        readout_duration: int = T_READOUT,
    ):
        self.qube_id: Optional[str] = None
        self.qube: Optional[qc.qube.Qube] = None
        self.qubits: Final = MUX[mux_number]
        self.params: Final = params
        self.readout_ports: Final = readout_ports
        self.control_duration: Final = control_duration
        self.readout_duration: Final = readout_duration
        self.schedule: Final = Schedule()
        qc.ui.MATPLOTLIB_PYPLOT = plt  # type: ignore
        self._init_channels()

    def connect(self, qube_id: str):
        self.qube_id = qube_id
        self.qube = qc.ui.QubeControl(f"{qube_id}.yml").qube
        self._init_qube()

    def _ctrl_channel(self, qubit: QubitKey) -> Channel:
        return self.schedule[qubit]

    def _ctrl_lo_channel(self, qubit: QubitKey) -> Channel:
        return self.schedule[qubit + CONTROL_LOW]

    def _ctrl_hi_channel(self, qubit: QubitKey) -> Channel:
        return self.schedule[qubit + CONTROL_HIGH]

    def _read_tx_channel(self, qubit: QubitKey) -> Channel:
        return self.schedule[READOUT_TX + qubit]

    def _read_rx_channel(self, qubit: QubitKey) -> Channel:
        return self.schedule[READOUT_RX + qubit]

    def _ctrl_channels(self) -> list[Channel]:
        return [self._ctrl_channel(qubit) for qubit in self.qubits]

    def _ctrl_lo_channels(self) -> list[Channel]:
        return [self._ctrl_lo_channel(qubit) for qubit in self.qubits]

    def _ctrl_hi_channels(self) -> list[Channel]:
        return [self._ctrl_hi_channel(qubit) for qubit in self.qubits]

    def _read_tx_channels(self) -> list[Channel]:
        return [self._read_tx_channel(qubit) for qubit in self.qubits]

    def _read_rx_channels(self) -> list[Channel]:
        return [self._read_rx_channel(qubit) for qubit in self.qubits]

    def _ctrl_slot(self, qubit: QubitKey) -> Arbitrary:
        return self._ctrl_channel(qubit).findall(Arbitrary)[0]

    def _read_tx_slot(self, qubit: QubitKey) -> Arbitrary:
        return self._read_tx_channel(qubit).findall(Arbitrary)[0]

    def _read_rx_slot(self, qubit: QubitKey) -> Read:
        return self._read_rx_channel(qubit).findall(Read)[0]

    def _ctrl_times(self, qubit: QubitKey) -> IntArray:
        channel = self._ctrl_channel(qubit)
        slot = self._ctrl_slot(qubit)
        local_times = channel.get_timestamp(slot)
        global_times = local_times - self.schedule.offset
        return global_times

    def _read_tx_times(self, qubit: QubitKey) -> IntArray:
        channel = self._read_tx_channel(qubit)
        slot = self._read_tx_slot(qubit)
        local_times = channel.get_timestamp(slot)
        global_times = local_times - self.schedule.offset
        return global_times

    def _init_channels(self):
        for qubit in self.qubits:
            control_frequency = self.params.qubit_dressed_frequency[qubit]
            anharmonicity = self.params.anharmonicity[qubit]
            self.schedule[qubit] = Channel(
                center_frequency=control_frequency,
            )
            self.schedule[qubit + CONTROL_LOW] = Channel(
                center_frequency=control_frequency + anharmonicity,
            )
            self.schedule[qubit + CONTROL_HIGH] = Channel(
                center_frequency=control_frequency - anharmonicity,
            )
        for qubit in self.qubits:
            readout_frequency = self.params.cavity_frequency[qubit]
            self.schedule[READOUT_TX + qubit] = Channel(
                center_frequency=readout_frequency,
            )
            self.schedule[READOUT_RX + qubit] = Channel(
                center_frequency=readout_frequency,
            )

    def _init_qube(self):
        ports = self.qube.ports

        tx = self.readout_ports[0]
        rx = self.readout_ports[1]

        config_tx = self.params.port_config[tx]
        ports[tx].lo.mhz = config_tx["lo"]
        ports[tx].nco.mhz = config_tx["nco"]
        ports[tx].mix.ssb = qc.qube.SSB.LSB
        ports[tx].awg0.nco.mhz = config_tx["awg0"]
        ports[tx].mix.vatt = config_tx["vatt"]

        ports[rx].nco.mhz = ports[tx].nco.mhz
        ports[rx].adc.capt0.ssb = qc.qube.SSB.LSB
        ports[rx].delay = 128 + 6 * 128  # [ns]

        for control_port in CONTROL_PORTS:
            config = self.params.port_config[control_port]
            port = ports[control_port]
            port.lo.mhz = config["lo"]
            port.nco.mhz = config["nco"]
            port.awg0.nco.mhz = config["awg0"]
            port.awg1.nco.mhz = config["awg1"]
            port.awg2.nco.mhz = config["awg2"]
            port.mix.vatt = config["vatt"]

        self.qube.gpio.write_value(0x0000)  # loopback off

        self.triggers = [ports[tx].dac.awg0]

        adda_to_channels = {
            ports[tx].dac.awg0: self._read_tx_channels(),
            ports[rx].adc.capt0: self._read_rx_channels(),
        }
        for index, control_port in enumerate(CONTROL_PORTS):
            port = ports[control_port]
            qubit = self.qubits[index]
            adda_to_channels[port.dac.awg0] = [self._ctrl_lo_channel(qubit)]
            adda_to_channels[port.dac.awg1] = [self._ctrl_channel(qubit)]
            adda_to_channels[port.dac.awg2] = [self._ctrl_hi_channel(qubit)]
        self.adda_to_channels = adda_to_channels

    def _init_slots(self):
        self.schedule.offset = self.control_duration

        for ch in self._ctrl_channels():
            ch.clear()
            ch.append(Arbitrary(duration=self.control_duration, amplitude=1))
            ch.append(Blank(duration=self.readout_duration + 4 * T_MARGIN))

        for ch in self._ctrl_lo_channels():
            ch.clear()
            ch.append(Arbitrary(duration=self.control_duration, amplitude=1))
            ch.append(Blank(duration=self.readout_duration + 4 * T_MARGIN))

        for ch in self._ctrl_hi_channels():
            ch.clear()
            ch.append(Arbitrary(duration=self.control_duration, amplitude=1))
            ch.append(Blank(duration=self.readout_duration + 4 * T_MARGIN))

        for ch in self._read_tx_channels():
            ch.clear()
            ch.append(Blank(duration=self.control_duration))
            ch.append(Arbitrary(duration=self.readout_duration, amplitude=1))
            ch.append(Blank(duration=4 * T_MARGIN))

        for ch in self._read_rx_channels():
            ch.clear()
            ch.append(Blank(duration=self.control_duration - T_MARGIN))
            ch.append(Read(duration=self.readout_duration + 5 * T_MARGIN))

        durations = [channel.duration for channel in self.schedule.values()]
        assert len(set(durations)) == 1, "All channels must have the same duration."

    def _set_waveforms(
        self,
        control_qubits: list[QubitKey],
        readout_qubits: list[QubitKey],
        control_waveforms: QubitDict[Waveform],
    ):
        self._init_slots()

        self._set_control_waveforms(control_qubits, control_waveforms)

        readout_amplitude = self.params.readout_amplitude
        tau = 50
        readout_waveforms = {
            qubit: Rect(
                duration=T_READOUT - tau,
                amplitude=readout_amplitude[qubit],
                tau=tau,
            )
            for qubit in readout_qubits
        }
        self._set_readout_waveforms(readout_qubits, readout_waveforms)

    def _set_control_waveforms(
        self,
        qubits: list[QubitKey],
        waveforms: QubitDict[Waveform],
    ):
        for qubit in qubits:
            values = waveforms[qubit].values
            T = len(values) * SAMPLING_PERIOD
            t = self._ctrl_times(qubit)
            self._ctrl_slot(qubit).iq[(-T <= t) & (t < 0)] = values

    def _set_readout_waveforms(
        self,
        qubits: list[QubitKey],
        waveforms: QubitDict[Waveform],
    ):
        for qubit in qubits:
            values = waveforms[qubit].values
            self._read_tx_slot(qubit).iq[:] = values

    def loopback_mode(
        self,
        use_loopback: bool,
    ):
        if use_loopback:
            self.qube.gpio.write_value(0xFFFF)
        else:
            self.qube.gpio.write_value(0x0000)

    def readout_range(
        self,
    ) -> slice:
        return slice(T_MARGIN // 2, (self.readout_duration + T_MARGIN) // 2)

    def get_control_frequency(
        self,
        qubit: QubitKey,
    ) -> float:
        return self.schedule[qubit].center_frequency

    def set_control_frequency(
        self,
        qubit: QubitKey,
        frequency: float,
    ):
        self.schedule[qubit].center_frequency = frequency

    def get_control_waveforms(
        self,
        ctrl_qubits: list[QubitKey],
    ) -> QubitDict[IQArray]:
        return {qubit: self._ctrl_slot(qubit).iq for qubit in ctrl_qubits}

    def get_control_times(
        self,
        ctrl_qubits: list[QubitKey],
    ) -> QubitDict[IntArray]:
        return {qubit: self._ctrl_times(qubit) for qubit in ctrl_qubits}

    def get_readout_tx_waveforms(
        self,
        read_qubits: list[QubitKey],
    ) -> QubitDict[IQArray]:
        return {qubit: self._read_tx_slot(qubit).iq for qubit in read_qubits}

    def get_readout_tx_times(
        self,
        read_qubits: list[QubitKey],
    ) -> QubitDict[IntArray]:
        return {qubit: self._read_tx_times(qubit) for qubit in read_qubits}

    def get_readout_rx_waveforms(
        self,
        read_qubits: list[QubitKey],
    ) -> QubitDict[IQArray]:
        return {qubit: self._read_rx_slot(qubit).iq for qubit in read_qubits}  # type: ignore

    def get_readout_rx_times(
        self,
        read_qubits: list[QubitKey],
    ) -> QubitDict[IntArray]:
        return {qubit: self._read_rx_slot(qubit).timestamp for qubit in read_qubits}

    def measure(
        self,
        control_qubits: list[QubitKey],
        readout_qubits: list[QubitKey],
        control_waveforms: QubitDict[Waveform],
        repeats: int = 10_000,
        interval: int = 150_000,
    ) -> QubitDict[IQValue]:
        self._set_waveforms(
            control_qubits=control_qubits,
            readout_qubits=readout_qubits,
            control_waveforms=control_waveforms,
        )
        run(
            schedule=self.schedule,
            repeats=repeats,
            interval=interval,
            adda_to_channels=self.adda_to_channels,
            triggers=self.triggers,
        )
        rx_waveforms = self.get_readout_rx_waveforms(readout_qubits)
        readout_range = self.readout_range()
        result = {
            qubit: waveform[readout_range].mean()
            for qubit, waveform in rx_waveforms.items()
        }
        return result