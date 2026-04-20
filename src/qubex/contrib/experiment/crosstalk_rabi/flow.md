# Crosstalk Rabi Flow

## Measurement Flow

```mermaid
flowchart TD
    A[measure_crosstalk_rabi_experiment] --> B{_validate_qubit_pairs}
    B -->|invalid| C[Return Result\nstatus=not_crosstalk_pair\nwarning]
    B -->|valid| D{_validate_frequency_group}
    D -->|invalid| E[Return Result\nstatus=skipped\nwarning]
    D -->|valid| F{_validate_target_frequency_difference}
    F -->|invalid| E
    F -->|valid| G[_measure_crosstalk_rabi_experiment]

    G --> H[_crosstalk_rabi_experiment]
    H --> I[Resolve defaults\ntime range and amplitude]
    I --> J[obtain_reference_points]
    J --> K[obtain_rabi_params for JJ]
    K --> L[Build CrosstalkRabiData for JJ]
    L --> M[sweep_parameter for KJ]
    M --> N[Build CrosstalkRabiData for KJ]
    N --> O[Return raw Result\nreference_points, rabi_data_jj, rabi_data_kj]

    O --> P[Fit JJ data]
    P --> Q[Fit KJ data]
    Q --> R[build_pair_summary]

    R --> S{JJ fit success}
    S -->|yes| T[Save ExperimentResult for JJ]
    S -->|no| U[Save failed-fit plot for JJ]

    T --> V{KJ fit success}
    U --> V
    V -->|yes| W[Save ExperimentResult for KJ]
    V -->|no| X[Save failed-fit plot for KJ]

    W --> Y[Build CrosstalkRabiRecord\npair_summary + saved result files]
    X --> Y
    Y --> Z[Save CrosstalkRabiRecord]
    Z --> AA[Return Result\ndrive_target, measure_target, pair_summary, pair_record]
```

## Data and Artifact Flow

```mermaid
flowchart LR
    A[measure_crosstalk_rabi_experiment] --> B[pair_summary]
    A --> C[CrosstalkRabiRecord.save]
    C --> D[data/*CrosstalkRabiExperimentRecord*.json]
    A --> E[ExperimentResult.save for JJ or KJ]
    E --> F[data/*CrosstalkRabiExperiment*.json]
    D --> G[CrosstalkRabiRecord.list]
    G --> H[CrosstalkRabiMatrix.from_records]
    H --> I[plot_ratio_matrix]
    H --> J[plot_status_matrix]
    G --> K[CrosstalkRabiCollection.load_records]
    K --> L[CrosstalkRabiCollection.rebuild_matrix]
    H --> P[optional matrix cache]
    P --> Q[_save_matrix_cache]
    Q --> R[CrosstalkRabiCollection._load_cache]
    D --> M[latest pair record for one cell]
    M --> N[kj_result_file ref]
    N --> O[CrosstalkRabiCollection.find_result]
    O --> P[CrosstalkRabiCollection.plot_rabi]
```

## Responsibility Split in Current Code

- `crosstalk_rabi_experiment.py`: validation, measurement execution, fit, `ExperimentResult` save, pair-level record save.
- `crosstalk_rabi_record.py`: internal pair-level persisted summary with JJ/KJ saved-file references.
- `crosstalk_rabi_result.py`: pair summary construction from fit outputs.
- `crosstalk_rabi_matrix.py`: matrix storage, jsonpickle persistence, record-to-matrix projection, status and ratio plotting.
- `crosstalk_rabi_collection.py`: matrix facade plus saved record loading, matrix rebuild, and `ExperimentResult` discovery.
- `crosstalk_rabi_status.py`: status code and label mapping.

## Current Coupling Points

- Validation failures are converted directly into public `Result` payloads in `measure_crosstalk_rabi_experiment`.
- `_measure_crosstalk_rabi_experiment` mixes orchestration with persistence side effects.
- matrix cache helpers are internal.
- `load_records()` is the primary collection entry point.
- `CrosstalkRabiCollection.find_result` now requires a record-backed collection so the returned result matches the matrix projection state.
- `CrosstalkRabiMatrix._update` depends on string status values produced by `CrosstalkRabiPairSummary`.