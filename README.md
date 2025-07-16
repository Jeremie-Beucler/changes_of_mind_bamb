# Changes-of-mind in controllable and uncontrollable environments [Python translation]


This ia a Python implementation of code from https://github.com/marionrouault/actobscom/tree/main and contains analyis code and data for the paper:

Rouault M., Weiss A., Lee J. K., Drugowitsch J., Chambon V.* and Wyart V.* Controllability boosts neural and cognitive correlates of changes-of-mind in uncertain environments. BioRxiv (2022) https://doi.org/10.7554/eLife.75038


## Overview

This pipeline implements a Bayesian computational model that estimates how human observers:
- Update beliefs about environmental states based on sensory evidence
- Make decisions about when to sample new information
- Express confidence in their choices

The main fitting procedure uses Bayesian Adaptive Direct Search (BADS) optimization to estimate model parameters from behavioral data.

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd actobscom/SCRIPTS/python
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import numpy, scipy, pybads; print('Installation successful!')"
   ```

## Project Structure

```
python/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── run_fit_actobs_sim.py    # Main pipeline script
├── fit_actobs_sim.py        # Bayesian fitting module
├── getc.py                  # Behavioral curve computation
├── DATA/                    # Experimental data files (.mat format)
├── FITS/                    # Output directory for fitted results
└── tests/
    └── test_complete_module.py  # Unit tests
```

## Usage

### Quick Start

To run the complete fitting pipeline on all subjects and tasks:

```bash
python run_fit_actobs_sim.py
```

This will:
1. Load experimental data from the `DATA/` directory
2. Fit the computational model to each subject's data
3. Save results to the `FITS/` directory

### Configuration

Edit the configuration variables at the top of `run_fit_actobs_sim.py`:

```python
# === Config ===
expename = 'Experiment_1'  # or 'Experiment_2A'
data_dir = './DATA'        # Input data directory
output_dir = './FITS'      # Output directory
```

### Programmatic Usage

```python
from run_fit_actobs_sim import process_subject
from fit_actobs_sim import fit_actobs_sim
from getc import getc

# Process a single subject
output_file = process_subject('Experiment_1', subject_id=1, task_id=1)

# Or use the modules directly
import numpy as np

# Example configuration for fitting
cfg = {
    'seqind': np.array([1, 2, 3, 4, 5]),
    'seqpos': np.array([1, 2, 3, 4, 5]), 
    'seqdir': np.array([1, 1, 2, 2, 1]),
    'seqllr': np.array([0.5, -0.3, 0.8, -0.2, 0.6]),
    'seqlen': np.array([1, 1, 1, 1, 1]),
    'rbef': np.array([1, 1, 1, 2, 2]),
    'raft': np.array([1, 1, 2, 2, 1]),
    'nsmp': 1000,  # Number of simulation samples
    'nrun': 5      # Number of optimization runs
}

# Fit the model
result = fit_actobs_sim(cfg)
print(f"Fitted hazard rate: {result['h']:.3f}")
```

## Data Format

The pipeline expects MATLAB data files in the `DATA/` directory with the following structure:

**Required fields:**
- `seqind`: Sequence indices within current block
- `seqpos`: Position within current episode  
- `seqdir`: True direction for each trial (1 or 2)
- `seqllr`: Log-likelihood ratio (positive favors direction 1)
- `seqlen`: Length of each sequence
- `rbef`: Response before each sequence (1 or 2)
- `raft`: Response after each sequence (1 or 2)

**Optional fields (for confidence analysis):**
- `cbef`: Confidence before each trial (1=low, 2=high)
- `caft`: Confidence after each trial (1=low, 2=high)

## Model Parameters

The computational model estimates the following parameters:

- **h**: Hazard rate (probability of environmental change)
- **sinf**: Inference noise (uncertainty in belief updates)
- **ssel**: Selection noise (uncertainty in decision making) 
- **scnf**: Confidence noise (uncertainty in confidence reports)*
- **tcnf**: Confidence threshold (bias in confidence reports)*
- **gcnf**: Confidence gain (scaling during switches)*

*Only fitted when confidence data is available

## Output

Results are saved as MATLAB files in the `FITS/` directory containing:

- **Fitted parameters**: h, sinf, ssel, scnf, tcnf, gcnf
- **Optimization results**: Log-likelihood, AIC, BIC
- **Behavioral curves**: Subject and fitted curves
- **Model diagnostics**: Number of parameters, trials, etc.

## Testing

Run the unit tests to verify the installation:

```bash
python -m pytest tests/ -v
```

Or using unittest:

```bash
python -m unittest tests.test_complete_module -v
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`

2. **MATLAB file errors**: Verify that data files are in the correct `DATA/` directory and follow the expected format

3. **Optimization failures**: The pipeline uses multiple random starts. If optimization fails, try increasing `nrun` in the configuration

### Performance Tips

- Use fewer optimization runs (`nrun=3`) for faster fitting during development
- Reduce simulation samples (`nsmp=500`) for faster computation
- Run on subsets of subjects first to verify the pipeline

## Dependencies

- **numpy** (≥1.21.0): Numerical computations
- **scipy** (≥1.7.0): Scientific computing and optimization
- **pybads** (≥1.0.0): Bayesian Adaptive Direct Search optimization

