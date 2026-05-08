# Research Project Conventions

## Folder Structure
- `src/core/`           : Orekit initialization and coordinate conversions.
- `src/core/setup.py`   : Orekit VM initialization and Orekit-data loading.
- `src/core/frames.py`  : Frame and Time transformations.
- `src/models/`         : Propagators (J2, J3, SGP4).
- `src/algorithms/`     : Optimization logic (Gradient, Genetic).
- `src/utils/`          : Plotting and metrics (No side effects in logic files).
- `data/`               : Config and TLE files.
- `outputs/`            : Results (CSV, JSON) and saved Figure files.
- `notebooks/`          : Jupyter notebooks for interactive exploratory research

## Guidelines
- Use the `config.yaml` for all hyperparameters and paths.
- Separation of Concerns: Plotting functions must be isolated in `src/utils/plotting.py`.

