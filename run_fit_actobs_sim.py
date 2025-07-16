"""Main pipeline for fitting the Bayesian observer models to experimental data.

The pipeline performs the following steps for each subject/task combination:
1. Load experimental data from MATLAB files
2. Reconstruct trial metadata (episode numbers, sequence positions)
3. Compute behavioral curves using getc
4. Configure model fitting parameters
5. Fit the computational model using fit_actobs_sim
6. Save results to MATLAB format

The script supports two experimental paradigms:
- Experiment_1: ACTOBS_C data files
- Experiment_2A: ACTOBS_D_rule1 data files

Configuration can be modified by editing the global variables at the top of the script.

Usage:
    Simply run the script to process all subjects and tasks:
    
    $ python run_fit_actobs_sim.py
    
    Or import and use the functions programmatically:
    
    >>> from run_fit_actobs_sim import matlab_struct_to_dict, process_subject
    >>> data = matlab_struct_to_dict(matlab_struct)

Author: [Diksha Gupta]
Date: [15 July 2025]
"""

import numpy as np
import scipy.io as sio
import os
from getc import getc
from fit_actobs_sim import fit_actobs_sim 
from scipy.io.matlab import mat_struct
from typing import Dict, Any, List, Union

# === Config ===
expename = 'Experiment_1'  # or 'Experiment_2A'
data_dir = './DATA'
output_dir = './FITS'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if expename == 'Experiment_1':
    subjlist = [i for i in range(1, 18) if i not in [4, 16]]
elif expename == 'Experiment_2A':
    subjlist = [i for i in range(1, 21) if i not in [6, 15]]
else:
    raise ValueError("Undefined experiment name.")



def matlab_struct_to_dict(obj: mat_struct) -> Dict[str, Any]:
    """Convert MATLAB structure to Python dictionary.
    
    This function recursively converts mat_struct objects (returned by
    scipy.io.loadmat with struct_as_record=False) to standard Python
    dictionaries for easier manipulation.
    
    Parameters
    ----------
    obj : mat_struct
        MATLAB structure object to convert
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing all fields from the MATLAB structure
        
    Notes
    -----
    This function performs a shallow conversion - nested structures
    are not recursively converted. The data
    structures are typically flat so this is sufficient.
    
    Examples
    --------
    >>> import scipy.io as sio
    >>> mat_data = sio.loadmat('data.mat', struct_as_record=False)
    >>> raw_struct = mat_data['dat']
    >>> data_dict = matlab_struct_to_dict(raw_struct)
    >>> print(data_dict.keys())
    """
    result: Dict[str, Any] = {}
    for fieldname in obj._fieldnames:
        value = getattr(obj, fieldname)
        result[fieldname] = value
    return result


def load_experimental_data(experiment_name: str, subject_id: int, task_id: int, 
                          data_directory: str = './DATA') -> Dict[str, Any]:
    """Load experimental data for a specific subject and task.
    
    Parameters
    ----------
    experiment_name : str
        Name of the experiment ('Experiment_1' or 'Experiment_2A')
    subject_id : int
        Subject ID number
    task_id : int
        Task ID number (1 or 2)
    data_directory : str, optional
        Directory containing data files (default: './DATA')
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing experimental data
        
    Raises
    ------
    ValueError
        If experiment name is not recognized
    FileNotFoundError
        If data file is not found
    """
    if experiment_name == 'Experiment_1':
        fname = f'{data_directory}/ACTOBS_C_S{subject_id:02d}_task{task_id}_expdata.mat'
    elif experiment_name == 'Experiment_2A':
        fname = f'{data_directory}/ACTOBS_D_rule1_S{subject_id:02d}_task{task_id}_expdata.mat'
    else:
        raise ValueError(f"Unknown experiment name: {experiment_name}")
    
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Data file not found: {fname}")
    
    mat = sio.loadmat(fname, struct_as_record=False, squeeze_me=True)
    raw_dat = mat['dat']
    return matlab_struct_to_dict(raw_dat)


def reconstruct_trial_metadata(dat: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstruct trial metadata from raw experimental data.
    
    This function adds episode numbers and sequence positions to the data
    by analyzing block indices and sequence directions.
    
    Parameters
    ----------
    dat : Dict[str, Any]
        Dictionary containing raw experimental data
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with added metadata fields:
        - 'epinum': Episode number for each trial
        - 'seqpos': Position within each sequence
        - 'seqlen': Length of each sequence
        - 'seqllr': Total log-likelihood ratio for each sequence
        
    Notes
    -----
    Episodes are defined as contiguous sequences of trials with the same
    direction within a block. Episode numbers restart at 1 for each new
    block, and increment whenever the direction changes.
    """
    # Initialize metadata arrays
    dat['epinum'] = np.full_like(dat['seqdir'], np.nan, dtype=float)
    dat['seqpos'] = np.full_like(dat['seqdir'], np.nan, dtype=float)
    
    # Track state across trials
    blkind = 0
    seqdir = 0
    epi = 0
    pos = 0
    
    for i in range(len(dat['seqdir'])):
        if dat['blkind'][i] != blkind:
            # New block - reset episode counter
            blkind = dat['blkind'][i]
            seqdir = dat['seqdir'][i]
            epi = 1
            pos = 1
        elif dat['seqdir'][i] != seqdir:
            # New episode within block
            seqdir = dat['seqdir'][i]
            epi += 1
            pos = 1
        
        dat['epinum'][i] = epi
        dat['seqpos'][i] = pos
        pos += 1
    
    # Compute sequence lengths and LLRs
    dat['seqlen'] = [len(x) for x in dat['smpang']]
    dat['seqllr'] = [sum(x) for x in dat['smpllr']]
    
    return dat


def create_fitting_config(dat: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Create configuration dictionary for model fitting.
    
    Parameters
    ----------
    dat : Dict[str, Any]
        Dictionary containing experimental data
    **kwargs
        Additional configuration parameters to override defaults
        
    Returns
    -------
    Dict[str, Any]
        Configuration dictionary for fit_actobs_sim
        
    Notes
    -----
    Default configuration:
    - nsmp: 1000 simulation samples
    - nres: 1000 bootstrap samples
    - nval: 100 validation samples
    - nrun: 5 optimization runs
    - verbose: 1 (moderate verbosity)
    - ssel: 0 (no selection noise)
    - fitrev: False (don't fit reversal curves)
    - fitrep: True (fit repetition curves)
    - fitcnf: True (fit confidence data)
    """
    cfg = {
        'seqind': np.array(dat['seqind']),
        'seqpos': np.array(dat['seqpos']),
        'seqdir': np.array(dat['seqdir']),
        'seqllr': np.array(dat['seqllr']),
        'seqlen': np.array(dat['seqlen']),
        'rbef': np.array(dat['rbef']),
        'raft': np.array(dat['raft']),
        'cbef': np.array(dat['cbef']),
        'caft': np.array(dat['caft']),
        'nsmp': 1000,
        'nres': 1000,
        'nval': 100,
        'nrun': 5,
        'verbose': 1,
        'ssel': 0,  # no selection noise
        'fitrev': False,
        'fitrep': True,
        'fitcnf': True,
    }
    
    # Override defaults with any provided kwargs
    cfg.update(kwargs)
    return cfg


def process_subject(experiment_name: str, subject_id: int, task_id: int,
                   data_directory: str = './DATA', output_directory: str = './FITS',
                   **fit_kwargs) -> str:
    """Process a single subject and task.
    
    Parameters
    ----------
    experiment_name : str
        Name of the experiment
    subject_id : int
        Subject ID number
    task_id : int
        Task ID number
    data_directory : str, optional
        Directory containing data files
    output_directory : str, optional
        Directory for output files
    **fit_kwargs
        Additional parameters for model fitting
        
    Returns
    -------
    str
        Path to the saved output file
        
    Notes
    -----
    This function performs the complete processing pipeline:
    1. Load experimental data
    2. Reconstruct trial metadata
    3. Compute behavioral curves
    4. Fit computational model
    5. Save results
    """
    print(f"\nFitting subj {subject_id:02d} task {task_id}")
    
    # Load and process data
    dat = load_experimental_data(experiment_name, subject_id, task_id, data_directory)
    dat = reconstruct_trial_metadata(dat)
    
    # Compute behavioral curves
    c = getc(
        np.array(dat['seqind']),
        np.array(dat['seqpos']),
        np.array(dat['seqdir']),
        np.array(dat['seqllr']),
        np.array(dat['rbef']),
        np.array(dat['raft']),
        np.array(dat['cbef']),
        np.array(dat['caft'])
    )
    
    # Configure and run fitting
    cfg = create_fitting_config(dat, **fit_kwargs)
    out_fit = fit_actobs_sim(cfg)
    out_fit['c'] = c
    
    # Save results
    if experiment_name == 'Experiment_1':
        outname = f'{output_directory}/{experiment_name}_sub{subject_id:02d}_task{task_id}_fit_bads.mat'
    elif experiment_name == 'Experiment_2A':
        outname = f'{output_directory}/{experiment_name}_rule1_sub{subject_id:02d}_task{task_id}_fit_bads.mat'
    else:
        raise ValueError(f"Unknown experiment name: {experiment_name}")
    
    sio.savemat(outname, {'out_fit': out_fit})
    print(f"Saved to {outname}")
    
    return outname


def main():
    """Main processing function."""
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process all subjects and tasks
    for isubj in subjlist:
        for itask in [1, 2]:
            try:
                process_subject(expename, isubj, itask, data_dir, output_dir)
            except Exception as e:
                print(f"Error processing subject {isubj:02d} task {itask}: {e}")
                continue


if __name__ == "__main__":
    main()
