"""Behavioral curve computation.

This module computes behavioral curves (reversal and repetition curves) from
the experimental data. These curves characterize how
observers' choices and confidence depend on the task history and current
evidence.

The main function `getc` computes:
- Reversal curves: How choice depends on trial position relative to direction reversals
- Repetition curves: How choice repetition depends on log-likelihood ratio bins
- Confidence curves: How confidence varies with the same factors (if confidence data provided)

These curves are used to characterize behavioral patterns and for model fitting.

Example:
    >>> import numpy as np
    >>> seqind = np.array([1, 2, 3, 4, 5])
    >>> seqpos = np.array([1, 2, 3, 4, 5])
    >>> seqdir = np.array([1, 1, 2, 2, 1])
    >>> seqllr = np.array([0.5, -0.3, 0.8, -0.2, 0.6])
    >>> rbef = np.array([1, 1, 1, 2, 2])
    >>> raft = np.array([1, 1, 2, 2, 1])
    >>> curves = getc(seqind, seqpos, seqdir, seqllr, rbef, raft)
    >>> print(f"Reversal curve shape: {curves['rrev'].shape}")
"""

import numpy as np
from typing import Optional, Dict, Any, Union


def getc(seqind: np.ndarray, seqpos: np.ndarray, seqdir: np.ndarray, 
         seqllr: np.ndarray, rbef: np.ndarray, raft: np.ndarray, 
         cbef: Optional[np.ndarray] = None, caft: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Compute behavioral curves from the experimental data.
    
    This function computes reversal and repetition curves that characterize
    how observers' choices and confidence depend on task history and
    current evidence. 
    
    Parameters
    ----------
    seqind : np.ndarray
        Index within the block each trial belongs to.
        Shape: (n_trials,)
    seqpos : np.ndarray
        Position within current episode (1- indexed)
        Shape: (n_trials,)
    seqdir : np.ndarray
        True direction for each trial (1 or 2).
        Shape: (n_trials,)
    seqllr : np.ndarray
        Log-likelihood ratio for trial (positive favors direction 1).
        Shape: (n_trials,)
    rbef : np.ndarray
        Observer response before each trial (1 or 2).
        Shape: (n_trials,) or (n_trials, n_datasets)
    raft : np.ndarray
        Observer response after each trial (1 or 2).
        Shape: (n_trials,) or (n_trials, n_datasets)
    cbef : np.ndarray, optional
        Confidence before each trial (1=low, 2=high).
        Shape: (n_trials,) or (n_trials, n_datasets)
    caft : np.ndarray, optional
        Confidence after each trial (1=low, 2=high).
        Shape: (n_trials,) or (n_trials, n_datasets)
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing behavioral curves:
        
        Reversal curves (trial positions relative to direction reversals):
        - 'nrev': Number of trials at each position. Shape: (8,) or (8, n_datasets)
        - 'rrev': Fraction correct at each position. Shape: (8,) or (8, n_datasets)
        - 'crev': Fraction high confidence at each position (if confidence data provided)
        
        Repetition curves (binned by log-likelihood ratio):
        - 'nrep': Number of trials in each bin. Shape: (8,) or (8, n_datasets)
        - 'rrep': Fraction choice repetition in each bin. Shape: (8,) or (8, n_datasets)
        - 'crep': Fraction high confidence in each bin (if confidence data provided)
        
        Overall confidence statistics:
        - 'pconf': Overall fraction of high confidence responses (if confidence data provided)
        
    Notes
    -----
    Reversal curves are computed for trial positions -4 to +3 relative to direction
    reversals (seqpos=1 trials where seqind>1). Position 0 corresponds to the
    reversal trial itself.
    
    Repetition curves are computed by binning the signed log-likelihood ratios
    (seqllr * (3 - 2*rbef)) into 8 bins with boundaries at [-inf, -3, -2, -1, 0, +1, +2, +3, +inf].
    
    For multiple datasets, input arrays should have shape (n_trials, n_datasets)
    and output arrays will have shape (n_positions/bins, n_datasets).
    
    Examples
    --------
    Basic usage with single dataset:
    
    >>> seqind = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    >>> seqpos = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    >>> seqdir = np.array([1, 1, 2, 2, 1, 2, 2, 1, 1, 2])
    >>> seqllr = np.array([0.5, -0.3, 0.8, -0.2, 0.6, -0.4, 0.7, -0.1, 0.9, -0.5])
    >>> rbef = np.array([1, 1, 1, 2, 2, 2, 2, 1, 1, 1])
    >>> raft = np.array([1, 1, 2, 2, 1, 2, 1, 1, 2, 2])
    >>> curves = getc(seqind, seqpos, seqdir, seqllr, rbef, raft)
    >>> print(f"Reversal curve: {curves['rrev']}")
    
    With confidence data:
    
    >>> cbef = np.array([1, 1, 2, 2, 1, 2, 1, 1, 2, 2])
    >>> caft = np.array([1, 2, 2, 1, 1, 2, 2, 1, 1, 2])
    >>> curves = getc(seqind, seqpos, seqdir, seqllr, rbef, raft, cbef, caft)
    >>> print(f"Confidence reversal curve: {curves['crev']}")
    >>> print(f"Overall confidence: {curves['pconf']:.3f}")
    
    Multiple datasets:
    
    >>> rbef_multi = np.column_stack([rbef, rbef])
    >>> raft_multi = np.column_stack([raft, raft])
    >>> curves = getc(seqind, seqpos, seqdir, seqllr, rbef_multi, raft_multi)
    >>> print(f"Multi-dataset reversal curve shape: {curves['rrev'].shape}")
    """
    
    # Validate input arguments
    required_args = {
        'seqind': seqind, 'seqpos': seqpos, 'seqdir': seqdir, 
        'seqllr': seqllr, 'rbef': rbef, 'raft': raft
    }
    
    for arg_name, arg_value in required_args.items():
        if arg_value is None:
            raise ValueError(f'Missing required argument: {arg_name}')
        if not isinstance(arg_value, np.ndarray):
            raise TypeError(f'{arg_name} must be a numpy array, got {type(arg_value)}')
    
    # Check array shapes are consistent
    n_trials = len(seqind)
    for arg_name, arg_value in required_args.items():
        if len(arg_value) != n_trials:
            raise ValueError(f'{arg_name} must have same length as seqind ({n_trials}), got {len(arg_value)}')
    
    # Check if confidence data is provided and validate
    has_confidence = cbef is not None and caft is not None
    if has_confidence:
        if not isinstance(cbef, np.ndarray) or not isinstance(caft, np.ndarray):
            raise TypeError('Confidence arrays must be numpy arrays')
        if len(cbef) != n_trials or len(caft) != n_trials:
            raise ValueError('Confidence arrays must have same length as other arrays')
    
    # Constants for curve computation
    REVERSAL_OFFSETS = np.arange(-4, 4)  # -4 to +3 inclusive
    N_REVERSAL_POSITIONS = len(REVERSAL_OFFSETS)
    
    # Bin limits for repetition curves
    REPETITION_BIN_LIMITS = np.array([
        [-np.inf, -3, -2, -1, 0, +1, +2, +3],
        [-3, -2, -1, 0, +1, +2, +3, +np.inf]
    ])
    N_REPETITION_BINS = REPETITION_BIN_LIMITS.shape[1]
    
    # Initialize output dictionary
    curves = {}
    
    # Determine dataset dimensions
    is_single_dataset = raft.ndim == 1
    n_datasets = 1 if is_single_dataset else raft.shape[1]

    # Get reversal curves
    reversal_trials = np.where((seqind > 1) & (seqpos == 1))[0]
    n_reversal_trials = np.full((N_REVERSAL_POSITIONS, n_datasets), np.nan)
    reversal_accuracy = np.full((N_REVERSAL_POSITIONS, n_datasets), np.nan)
    
    for dataset_idx in range(n_datasets):
        for pos_idx in range(N_REVERSAL_POSITIONS):
            n_reversal_trials[pos_idx, dataset_idx] = len(reversal_trials)
            # Calculate offset indices exactly like MATLAB
            offset_indices = reversal_trials + REVERSAL_OFFSETS[pos_idx]
            # Only use indices that are within bounds
            valid_mask = (offset_indices >= 0) & (offset_indices < len(raft))
            if np.any(valid_mask):
                valid_indices = offset_indices[valid_mask]
                # Get corresponding reversal trial indices for seqdir lookup
                valid_reversal_trials = reversal_trials[valid_mask]
                if raft.ndim > 1:
                    reversal_accuracy[pos_idx, dataset_idx] = np.mean(
                        raft[valid_indices, dataset_idx] == seqdir[valid_reversal_trials]
                    )
                else:
                    reversal_accuracy[pos_idx, dataset_idx] = np.mean(
                        raft[valid_indices] == seqdir[valid_reversal_trials]
                    )

    
    curves['nrev'] = n_reversal_trials.squeeze() if is_single_dataset else n_reversal_trials
    curves['rrev'] = reversal_accuracy.squeeze() if is_single_dataset else reversal_accuracy
        
    if has_confidence:
        reversal_confidence = np.full((N_REVERSAL_POSITIONS, n_datasets), np.nan)
        for dataset_idx in range(n_datasets):
            for pos_idx in range(N_REVERSAL_POSITIONS):
                offset_indices = reversal_trials + REVERSAL_OFFSETS[pos_idx]
                valid_mask = (offset_indices >= 0) & (offset_indices < len(caft))
                if np.any(valid_mask):
                    valid_indices = offset_indices[valid_mask]
                    if caft.ndim > 1:
                        reversal_confidence[pos_idx, dataset_idx] = np.mean(
                            caft[valid_indices, dataset_idx] == 2
                        )
                    else:
                        reversal_confidence[pos_idx, dataset_idx] = np.mean(
                            caft[valid_indices] == 2
                        )
        curves['crev'] = reversal_confidence.squeeze() if is_single_dataset else reversal_confidence
    
    # Get repetition curves
    continuation_trials = np.where(seqind > 1)[0]
    n_repetition_trials = np.full((N_REPETITION_BINS, n_datasets), np.nan)
    repetition_rate = np.full((N_REPETITION_BINS, n_datasets), np.nan)

    for dataset_idx in range(n_datasets):
        if rbef.ndim > 1:
            signed_llr = seqllr[continuation_trials] * (3 - 2 * rbef[continuation_trials, dataset_idx])
            is_repetition = raft[continuation_trials, dataset_idx] == rbef[continuation_trials, dataset_idx]
        else:
            signed_llr = seqllr[continuation_trials] * (3 - 2 * rbef[continuation_trials].astype(float))
            is_repetition = raft[continuation_trials] == rbef[continuation_trials]
        
        for bin_idx in range(N_REPETITION_BINS):
            in_bin = ((signed_llr >= REPETITION_BIN_LIMITS[0, bin_idx]) & 
                     (signed_llr < REPETITION_BIN_LIMITS[1, bin_idx]))
            n_repetition_trials[bin_idx, dataset_idx] = np.sum(in_bin)
            if np.sum(in_bin) > 0:
                repetition_rate[bin_idx, dataset_idx] = np.mean(is_repetition[in_bin])

    curves['nrep'] = (n_repetition_trials.astype(int).squeeze() if is_single_dataset 
                     else n_repetition_trials.astype(int))
    curves['rrep'] = repetition_rate.squeeze() if is_single_dataset else repetition_rate
        
        
    if has_confidence:
        repetition_confidence = np.full((N_REPETITION_BINS, n_datasets), np.nan)
        for dataset_idx in range(n_datasets):
            if rbef.ndim > 1:
                signed_llr = seqllr[continuation_trials] * (3 - 2 * rbef[continuation_trials, dataset_idx])
                is_high_confidence = caft[continuation_trials, dataset_idx] == 2
            else:
                signed_llr = seqllr[continuation_trials] * (3 - 2 * rbef[continuation_trials].astype(float))
                is_high_confidence = caft[continuation_trials] == 2
            
            for bin_idx in range(N_REPETITION_BINS):
                in_bin = ((signed_llr >= REPETITION_BIN_LIMITS[0, bin_idx]) & 
                         (signed_llr < REPETITION_BIN_LIMITS[1, bin_idx]))
                if np.sum(in_bin) > 0:
                    repetition_confidence[bin_idx, dataset_idx] = np.mean(is_high_confidence[in_bin])
        
        curves['crep'] = repetition_confidence.squeeze() if is_single_dataset else repetition_confidence
        
        # Fraction confident for within subject error bars
        curves['pconf'] = np.mean(caft == 2)
   
    return curves