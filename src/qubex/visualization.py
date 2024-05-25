import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
from IPython.display import clear_output, display
from matplotlib import gridspec
from numpy.typing import NDArray
from plotly.subplots import make_subplots


def scatter_iq_data(data: dict[str, list[complex]]):
    fig = make_subplots(rows=1, cols=1)
    fig.update_layout(
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        width=500,
        height=500,
        scattermode="group",
    )
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(dtick=1)

    for qubit, iq in data.items():
        scatter = go.Scatter(
            x=np.real(iq),
            y=np.imag(iq),
            mode="markers",
            name=qubit,
        )
        fig.add_trace(scatter)
    clear_output(wait=True)
    display(fig)


def show_pulse_sequences(
    control_qubits: list[str],
    control_waveforms: dict[str, NDArray[np.complex128]],
    control_times: dict[str, NDArray[np.int64]],
    control_duration: int,
    readout_qubits: list[str],
    readout_waveforms: dict[str, NDArray[np.complex128]],
    readout_times: dict[str, NDArray[np.int64]],
    readout_duration: int,
):
    """
    Shows the pulse sequences.

    Parameters
    ----------
    control_qubits : list[str]
        The list of control qubits.
    control_waveforms : dict[str, NDArray[np.complex128]]
        The dictionary of control waveforms.
    control_times : dict[str, NDArray[np.int64]]
        The dictionary of control times.
    control_duration : int
        The duration of the control pulses.
    readout_qubits : list[str]
        The list of readout qubits.
    readout_waveforms : dict[str, NDArray[np.complex128]]
        The dictionary of readout waveforms.
    readout_times : dict[str, NDArray[np.int64]]
        The dictionary of readout times.
    readout_duration : int
        The duration of the readout pulses.
    """

    # number of qubits
    N = len(control_qubits)

    # initialize the figure
    plt.figure(figsize=(15, 1.5 * (N + 1)))
    plt.subplots_adjust(hspace=0.0)  # no space between subplots

    # the grid is adjusted to the number of qubits (N)
    # the last row (N+1) is for the readout waveform
    gs = gridspec.GridSpec(N + 1, 1)

    # the x-axis range
    xlim = (
        min(-1.0, -control_duration * 1e-3),
        readout_duration * 1e-3,
    )

    # initialize the axes
    # share the x-axis with the first subplot
    axes = []
    for i in range(N):
        if i == 0:
            ax = plt.subplot(gs[i])
            ax.set_title("Pulse waveform")
            ax.set_xlim(xlim)  # μs
            ax.xaxis.set_visible(False)
            axes.append(ax)
        else:
            ax = plt.subplot(gs[i], sharex=axes[0])
            ax.xaxis.set_visible(False)
            axes.append(ax)
    ro_ax = plt.subplot(gs[N], sharex=axes[0])
    ro_ax.set_xlabel("Time (μs)")
    ro_ax.xaxis.set_visible(True)
    axes.append(ro_ax)

    # the list of the maximum amplitudes used for the ylim
    max_ampl_list = []

    # plot the control pulses
    # the real and imaginary parts are plotted in the same subplot
    for i, qubit in enumerate(control_qubits):
        times = control_times[qubit] * 1e-3  # ns -> μs
        waveform = control_waveforms[qubit]
        axes[i].plot(
            times,
            np.real(waveform),
            label=qubit + " control (real)",
        )
        axes[i].plot(
            times,
            np.imag(waveform),
            label=qubit + " control (imag)",
        )
        axes[i].legend()
        max_ampl_list.append(np.max(np.abs(waveform)))

    # plot the readout pulses
    for i, qubit in enumerate(readout_qubits):
        times = readout_times[qubit] * 1e-3  # ns -> us
        waveform = readout_waveforms[qubit]
        axes[N].plot(
            times,
            np.abs(waveform),
            label=qubit + " readout (abs)",
            linestyle="dashed",
        )
        axes[N].legend()
        max_ampl_list.append(np.max(np.abs(waveform)))

    # set the y-axis range according to the maximum amplitude
    max_ampl = np.max(max_ampl_list)
    for i in range(N + 1):
        axes[i].set_ylim(-1.1 * max_ampl, 1.1 * max_ampl)

    plt.show()


def show_measurement_results(
    qubits: list[str],
    waveforms: dict[str, NDArray[np.complex128]],
    times: dict[str, NDArray[np.int64]],
    sweep_range: NDArray,
    signals: dict[str, list[complex]],
    signals_rotated: dict[str, NDArray[np.complex128]],
    readout_range: slice,
):
    """
    Shows the measurement results.

    Parameters
    ----------
    qubits : list[str]
        The list of qubits.
    waveforms : dict[str, NDArray[np.complex128]]
        The dictionary of waveforms.
    times : dict[str, NDArray[np.int64]]
        The dictionary of times.
    sweep_range : NDArray
        The sweep range.
    signals : QubitDict[list[IQValue]]
        The dictionary of signals.
    signals_rotated : dict[str, NDArray[np.complex128]]
        The dictionary of signals after rotation.
    readout_range : slice
        The slice of the readout range.
    """
    plt.figure(figsize=(15, 6 * len(qubits)))
    gs = gridspec.GridSpec(2 * len(qubits), 2, wspace=0.3, hspace=0.5)

    ax = {}
    for i, qubit in enumerate(qubits):
        ax[qubit] = [
            plt.subplot(gs[i * 2, 0]),
            plt.subplot(gs[i * 2 + 1, 0]),
            plt.subplot(gs[i * 2 : i * 2 + 2, 1]),
        ]

    for qubit in qubits:
        avg_num = 50  # 平均化する個数

        mov_avg_readout_iq = (
            np.convolve(waveforms[qubit], np.ones(avg_num), mode="valid") / avg_num
        )  # 移動平均
        mov_avg_readout_iq = np.append(mov_avg_readout_iq, np.zeros(avg_num - 1))

        ax[qubit][0].plot(
            times[qubit] * 1e-3,
            np.real(mov_avg_readout_iq),
            label="I",
        )
        ax[qubit][0].plot(
            times[qubit] * 1e-3,
            np.imag(mov_avg_readout_iq),
            label="Q",
        )

        ax[qubit][0].plot(
            times[qubit][readout_range] * 1e-3,
            np.real(mov_avg_readout_iq)[readout_range],
            lw=5,
        )
        ax[qubit][0].plot(
            times[qubit][readout_range] * 1e-3,
            np.imag(mov_avg_readout_iq)[readout_range],
            lw=5,
        )

        ax[qubit][0].set_xlabel("Time (μs)")
        ax[qubit][0].set_xlim(0, 2.0)
        ax[qubit][0].set_title("Detected readout pulse waveform " + qubit)
        ax[qubit][0].legend()
        ax[qubit][0].grid(color="gray", linestyle="--", alpha=0.2)

        ax[qubit][1].plot(sweep_range, np.real(signals[qubit]), "o-", label="I")
        ax[qubit][1].plot(sweep_range, np.imag(signals[qubit]), "o-", label="Q")
        ax[qubit][1].set_xlabel("Sweep index")
        ax[qubit][1].set_title("Detected signal " + qubit)
        ax[qubit][1].legend()
        ax[qubit][1].grid(color="gray", linestyle="--", alpha=0.2)

        ax[qubit][2].plot(
            np.real(mov_avg_readout_iq), np.imag(mov_avg_readout_iq), lw=0.2
        )

        width = max(np.abs(signals[qubit]))
        ax[qubit][2].set_xlim(-width, width)
        ax[qubit][2].set_ylim(-width, width)
        ax[qubit][2].plot(
            np.linspace(-width, width, 2), np.zeros(2), linewidth=1, color="black"
        )
        ax[qubit][2].plot(
            np.zeros(2), np.linspace(-width, width, 2), linewidth=1, color="black"
        )
        ax[qubit][2].set_xlabel("I")
        ax[qubit][2].set_ylabel("Q")
        ax[qubit][2].set_title("Complex amplitude on IQ plane " + qubit)

        ax[qubit][2].scatter(
            np.real(signals[qubit]),
            np.imag(signals[qubit]),
            label="Before rotation",
        )
        ax[qubit][2].scatter(
            np.real(signals[qubit])[0],
            np.imag(signals[qubit])[0],
            color="blue",
        )

        ax[qubit][2].scatter(
            np.real(signals_rotated[qubit]),
            np.imag(signals_rotated[qubit]),
            label="After rotation",
        )
        ax[qubit][2].scatter(
            np.real(signals_rotated[qubit][0]),
            np.imag(signals_rotated[qubit][0]),
            color="red",
        )
        ax[qubit][2].legend()
        ax[qubit][2].grid(color="gray", linestyle="--", alpha=0.2)
    plt.show()
