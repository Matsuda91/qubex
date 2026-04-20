DEFAULT_DATA_DIR = "./data/"
DEFAULT_CONFIG_DIR = "./crosstalk_rabi_config"
SAVE_FILENAME = "CrosstalkRabiExperiment"

LOW_INDEX = (0, 3)
HIGH_INDEX = (1, 2)

SAVE_DESCRIPTION_TEMPLATE = lambda drive_target, measure_target: (
    f"Rabi data for drive_target={drive_target} and measure_target={measure_target}"
)
