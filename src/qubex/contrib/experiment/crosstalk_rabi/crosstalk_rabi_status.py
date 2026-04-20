"""Crosstalk Rabi status codes and labels."""

STATUS_NOT_CROSSTALK_PAIR = 0
STATUS_NOT_MEASURED = 1
STATUS_MEASURED = 2
STATUS_FIT_FAILED = 3
STATUS_SKIPPED = 4

STATUS_LABELS = {
    STATUS_NOT_CROSSTALK_PAIR: "not_crosstalk_pair",
    STATUS_NOT_MEASURED: "not_measured",
    STATUS_MEASURED: "measured",
    STATUS_FIT_FAILED: "fit_failed",
    STATUS_SKIPPED: "skipped",
}

STATUS_BY_SUMMARY = {
    "not_crosstalk_pair": STATUS_NOT_CROSSTALK_PAIR,
    "not_measured": STATUS_NOT_MEASURED,
    "measured": STATUS_MEASURED,
    "fit_failed": STATUS_FIT_FAILED,
    "skipped": STATUS_SKIPPED,
}
