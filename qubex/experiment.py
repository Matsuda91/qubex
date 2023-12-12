import datetime
from collections import defaultdict
from dataclasses import dataclass
from typing import Final, Optional

import numpy as np
from IPython.display import clear_output
from numpy.typing import NDArray

from .analysis import fit_and_rotate, fit_chevron, fit_damped_rabi, fit_rabi
from .configs import Configs
from .consts import T_CONTROL, T_READOUT
from .experiment_record import ExperimentRecord
from .pulse import Rect, Waveform
from .qube_manager import QubeManager
from .typing import (
    FloatArray,
    IntArray,
    IQArray,
    IQValue,
    ParametricWaveform,
    QubitDict,
    QubitKey,
)
from .visualization import show_measurement_results, show_pulse_sequences


@dataclass
class RabiParams:
    qubit: QubitKey
    phase_shift: float
    fluctuation: float
    amplitude: float
    omega: float
    phi: float
    offset: float


@dataclass
class SweepResult:
    qubit: QubitKey
    sweep_range: NDArray
    signals: IQArray
    created_at: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def rotated(self, rabi_params: RabiParams) -> IQArray:
        return self.signals * np.exp(-1j * rabi_params.phase_shift)

    def normalized(self, rabi_params: RabiParams) -> FloatArray:
        values = self.signals * np.exp(-1j * rabi_params.phase_shift)
        values_normalized = -(values.imag - rabi_params.offset) / rabi_params.amplitude
        return values_normalized


@dataclass
class ChevronResult:
    qubit: QubitKey
    center_frequency: float
    freq_range: FloatArray
    time_range: IntArray
    signals: list[FloatArray]
    created_at: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Experiment:
    def __init__(
        self,
        config_file: str,
        readout_window: int = T_READOUT,
        control_window: int = T_CONTROL,
        repeats: int = 10_000,
        interval: int = 150_000,
        data_dir="./data",
    ):
        self.configs: Final = Configs.load(config_file)
        self.qube_manager: Final = QubeManager(
            configs=self.configs,
            readout_window=readout_window,
            control_window=control_window,
        )
        self.qubits: Final = self.configs.qubits
        self.params: Final = self.configs.params
        self.repeats: Final = repeats
        self.interval: Final = interval
        self.data_dir: Final = data_dir

    def connect(self):
        self.qube_manager.connect()

    def loopback_mode(self, use_loopback: bool):
        self.qube_manager.loopback_mode(use_loopback)

    def get_control_frequency(self, qubit: QubitKey) -> float:
        return self.qube_manager.get_control_frequency(qubit)

    def set_control_frequency(self, qubit: QubitKey, frequency: float):
        self.qube_manager.set_control_frequency(qubit, frequency)

    def measure(
        self,
        waveforms: QubitDict[Waveform],
        repeats: Optional[int] = None,
        interval: Optional[int] = None,
    ) -> QubitDict[IQValue]:
        if repeats is None:
            repeats = self.repeats
        if interval is None:
            interval = self.interval
        qubits = list(waveforms.keys())
        result = self.qube_manager.measure(
            control_qubits=qubits,
            readout_qubits=qubits,
            control_waveforms=waveforms,
            repeats=repeats,
            interval=interval,
        )
        return result

    def rabi_experiment(
        self,
        time_range: IntArray,
        amplitudes: QubitDict[float],
        plot: bool = True,
    ) -> QubitDict[SweepResult]:
        qubits = list(amplitudes.keys())
        control_qubits = qubits
        readout_qubits = qubits
        signals: QubitDict[list[IQValue]] = defaultdict(list)

        for index, duration in enumerate(time_range):
            waveforms = {
                qubit: Rect(
                    duration=duration,
                    amplitude=amplitudes[qubit],
                )
                for qubit in qubits
            }

            measured_values = self.measure(waveforms)

            for qubit, value in measured_values.items():
                signals[qubit].append(value)

            if plot:
                self.show_experiment_results(
                    control_qubits=control_qubits,
                    readout_qubits=readout_qubits,
                    sweep_range=time_range,
                    index=index,
                    signals=signals,
                )

            print(f"{index+1}/{len(time_range)} : {duration} ns")

        result = {
            qubit: SweepResult(
                qubit=qubit,
                sweep_range=time_range,
                signals=np.array(values),
            )
            for qubit, values in signals.items()
        }
        return result

    def sweep_parameter(
        self,
        sweep_range: NDArray,
        parametric_waveforms: QubitDict[ParametricWaveform],
        pulse_count=1,
        plot: bool = True,
    ) -> QubitDict[SweepResult]:
        qubits = list(parametric_waveforms.keys())
        control_qubits = qubits
        readout_qubits = qubits
        signals: QubitDict[list[IQValue]] = defaultdict(list)

        for index, param in enumerate(sweep_range):
            waveforms = {
                qubit: waveform(param).repeated(pulse_count)
                for qubit, waveform in parametric_waveforms.items()
            }

            measured_values = self.measure(waveforms)

            for qubit, value in measured_values.items():
                signals[qubit].append(value)

            if plot:
                self.show_experiment_results(
                    control_qubits=control_qubits,
                    readout_qubits=readout_qubits,
                    sweep_range=sweep_range,
                    index=index,
                    signals=signals,
                )

            print(f"{index+1}/{len(sweep_range)}")

        result = {
            qubit: SweepResult(
                qubit=qubit,
                sweep_range=sweep_range,
                signals=np.array(values),
            )
            for qubit, values in signals.items()
        }
        return result

    def rabi_check(
        self,
        time_range=np.arange(0, 201, 10),
    ) -> QubitDict[SweepResult]:
        amplitudes = self.params.default_hpi_amplitude
        result = self.rabi_experiment(
            amplitudes=amplitudes,
            time_range=time_range,
        )
        return result

    def repeat_pulse(
        self,
        waveforms: QubitDict[Waveform],
        n: int,
    ) -> QubitDict[SweepResult]:
        parametric_waveforms = {
            qubit: lambda param, w=waveform: w.repeated(int(param))
            for qubit, waveform in waveforms.items()
        }
        result = self.sweep_parameter(
            sweep_range=np.arange(n + 1),
            parametric_waveforms=parametric_waveforms,
            pulse_count=1,
        )
        return result

    def show_experiment_results(
        self,
        control_qubits: list[QubitKey],
        readout_qubits: list[QubitKey],
        sweep_range: NDArray,
        index: int,
        signals: QubitDict[list[IQValue]],
    ):
        signals_rotated = {
            qubit: fit_and_rotate(values) for qubit, values in signals.items()
        }

        ctrl_waveforms = self.qube_manager.get_control_waveforms(control_qubits)
        ctrl_times = self.qube_manager.get_control_times(control_qubits)
        rotx_waveforms = self.qube_manager.get_readout_tx_waveforms(readout_qubits)
        rotx_times = self.qube_manager.get_readout_tx_times(readout_qubits)
        rorx_waveforms = self.qube_manager.get_readout_rx_waveforms(readout_qubits)
        rorx_times = self.qube_manager.get_readout_rx_times(readout_qubits)
        readout_range = self.qube_manager.readout_range
        control_duration = self.qube_manager.control_window
        readout_duration = self.qube_manager.readout_window

        clear_output(True)
        show_measurement_results(
            qubits=readout_qubits,
            waveforms=rorx_waveforms,
            times=rorx_times,
            sweep_range=sweep_range[: index + 1],
            signals=signals,
            signals_rotated=signals_rotated,
            readout_range=readout_range,
        )
        show_pulse_sequences(
            control_qubits=control_qubits,
            control_waveforms=ctrl_waveforms,
            control_times=ctrl_times,
            control_duration=control_duration,
            readout_qubits=readout_qubits,
            readout_waveforms=rotx_waveforms,
            readout_times=rotx_times,
            readout_duration=readout_duration,
        )

    def normalize(
        self,
        iq_value: IQValue,
        rabi_params: RabiParams,
    ) -> float:
        iq_value = iq_value * np.exp(-1j * rabi_params.phase_shift)
        value = iq_value.imag
        value = -(value - rabi_params.offset) / rabi_params.amplitude
        return value

    def fit_rabi(
        self,
        data: SweepResult,
        wave_count=2.5,
    ) -> RabiParams:
        times = data.sweep_range
        signals = data.signals

        phase_shift, fluctuation, popt = fit_rabi(
            times=times,
            signals=signals,
            wave_count=wave_count,
        )

        rabi_params = RabiParams(
            qubit=data.qubit,
            phase_shift=phase_shift,
            fluctuation=fluctuation,
            amplitude=popt[0],
            omega=popt[1],
            phi=popt[2],
            offset=popt[3],
        )
        return rabi_params

    def fit_damped_rabi(
        self,
        data: SweepResult,
        wave_count=2.5,
    ) -> RabiParams:
        times = data.sweep_range
        signals = data.signals

        phase_shift, fluctuation, popt = fit_damped_rabi(
            times=times,
            signals=signals,
            wave_count=wave_count,
        )

        rabi_params = RabiParams(
            qubit=data.qubit,
            phase_shift=phase_shift,
            fluctuation=fluctuation,
            amplitude=popt[0],
            omega=popt[2],
            phi=popt[3],
            offset=popt[4],
        )
        return rabi_params

    def chevron_experiment(
        self,
        qubits: list[QubitKey],
        freq_range: FloatArray,
        time_range: IntArray,
        rabi_params: RabiParams,
    ) -> QubitDict[ChevronResult]:
        amplitudes = self.params.default_hpi_amplitude
        frequenties = self.params.transmon_dressed_frequency_ge

        signals = defaultdict(list)

        for idx, freq in enumerate(freq_range):
            print(f"### {idx+1}/{len(freq_range)}: {freq:.2f} MHz")
            for qubit in self.qubits:
                freq_mod = frequenties[qubit] + freq * 1e6
                self.set_control_frequency(qubit, freq_mod)

            result_rabi = self.rabi_experiment(
                time_range=time_range,
                amplitudes=amplitudes,
                plot=False,
            )

            for qubit in qubits:
                signals[qubit].append(result_rabi[qubit].normalized(rabi_params))

        result = {
            qubit: ChevronResult(
                qubit=qubit,
                center_frequency=frequenties[qubit],
                freq_range=freq_range,
                time_range=time_range,
                signals=values,
            )
            for qubit, values in signals.items()
        }

        ExperimentRecord.create(
            name="result_chevron",
            description="Chevron experiment",
            data=result,
        )

        return result

    def fit_chevron(
        self,
        data: ChevronResult,
    ):
        fit_chevron(
            center_frequency=data.center_frequency,
            freq_range=data.freq_range,
            time_range=data.time_range,
            signals=data.signals,
        )
