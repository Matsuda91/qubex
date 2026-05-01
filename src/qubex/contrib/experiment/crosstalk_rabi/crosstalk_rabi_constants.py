"""Constants for the crosstalk Rabi experiment."""

DEFAULT_DATA_DIR = "./data/"
DEFAULT_IMAGES_DIR = "./images/"
DEFAULT_CONFIG_DIR = "./crosstalk_rabi_config"
SAVE_FILENAME = "CrosstalkRabiExperiment"

LOW_INDEX = (0, 3)
HIGH_INDEX = (1, 2)

SAVE_DESCRIPTION_TEMPLATE = lambda drive_target, measure_target: (
    f"Rabi data for drive_target={drive_target} and measure_target={measure_target}"
)

DEFAULT_CROSSTALK_RABI_TIME_RANGE = range(0, 40001, 1000)
DEFAULT_CROSSTALK_RATIO = 0.01
DEFAULT_SAMPLING_PERIOD = 2
QUBITS_WITH_CONTROL_LINE_AMP_IN_144Q = ["Q031", "Q036", "Q039", "Q056"]
QUBITS_WITH_CONTROL_LINE_AMP_IN_64Q = []
