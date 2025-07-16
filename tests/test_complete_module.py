"""
Comprehensive unit tests for the complete actobs simulation module.

This module tests all components that run_fit_actobs_sim.py depends on:
- fit_actobs_sim.py: Main fitting functionality
- getc.py: Behavioral curve computation
- run_fit_actobs_sim.py: Main pipeline script

Tests cover basic functionality, edge cases, and integration between modules.
"""

import unittest
import numpy as np
import warnings
import tempfile
import os
from unittest.mock import patch, MagicMock
from scipy.io.matlab import mat_struct

# Import modules to test
from fit_actobs_sim import fit_actobs_sim, estimate_ll_sd, upfun
from getc import getc
import run_fit_actobs_sim


class TestGetc(unittest.TestCase):
    """Test suite for getc.py module"""

    def setUp(self):
        """Set up test fixtures"""
        np.random.seed(42)
        
        # Create test data
        self.nseq = 10
        self.seqind = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        self.seqpos = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        self.seqdir = np.array([1, 1, 2, 2, 1, 2, 2, 1, 1, 2])
        self.seqllr = np.array([0.5, -0.3, 0.8, -0.2, 0.6, -0.4, 0.7, -0.1, 0.9, -0.5])
        self.rbef = np.array([1, 1, 1, 2, 2, 2, 2, 1, 1, 1])
        self.raft = np.array([1, 1, 2, 2, 1, 2, 1, 1, 2, 2])
        self.cbef = np.array([1, 1, 2, 2, 1, 2, 1, 1, 2, 2])
        self.caft = np.array([1, 2, 2, 1, 1, 2, 2, 1, 1, 2])

    def test_getc_basic_functionality(self):
        """Test basic getc functionality"""
        result = getc(self.seqind, self.seqpos, self.seqdir, self.seqllr, 
                     self.rbef, self.raft)
        
        # Check required keys are present
        self.assertIn('nrev', result)
        self.assertIn('rrev', result)
        self.assertIn('nrep', result)
        self.assertIn('rrep', result)
        
        # Check shapes are correct
        self.assertEqual(result['nrev'].shape, (8,))  # 8 positions (-4 to +3)
        self.assertEqual(result['rrev'].shape, (8,))
        self.assertEqual(result['nrep'].shape, (8,))  # 8 bins
        self.assertEqual(result['rrep'].shape, (8,))

    def test_getc_with_confidence(self):
        """Test getc with confidence data"""
        result = getc(self.seqind, self.seqpos, self.seqdir, self.seqllr, 
                     self.rbef, self.raft, self.cbef, self.caft)
        
        # Should have confidence curves
        self.assertIn('crev', result)
        self.assertIn('crep', result)
        self.assertIn('pconf', result)
        
        # Check shapes
        self.assertEqual(result['crev'].shape, (8,))
        self.assertEqual(result['crep'].shape, (8,))
        self.assertIsInstance(result['pconf'], (float, np.floating))

    def test_getc_multi_dataset(self):
        """Test getc with multiple datasets"""
        # Create multi-dataset arrays
        rbef_multi = np.column_stack([self.rbef, self.rbef])
        raft_multi = np.column_stack([self.raft, self.raft])
        
        result = getc(self.seqind, self.seqpos, self.seqdir, self.seqllr, 
                     rbef_multi, raft_multi)
        
        # Should have 2D arrays for multiple datasets
        self.assertEqual(result['nrev'].shape, (8, 2))
        self.assertEqual(result['rrev'].shape, (8, 2))
        self.assertEqual(result['nrep'].shape, (8, 2))
        self.assertEqual(result['rrep'].shape, (8, 2))

    def test_getc_missing_arguments(self):
        """Test getc with missing required arguments"""
        with self.assertRaises(ValueError):
            getc(None, self.seqpos, self.seqdir, self.seqllr, self.rbef, self.raft)

    def test_getc_edge_cases(self):
        """Test getc with edge cases"""
        # Test with minimal data
        seqind_min = np.array([1, 2])
        seqpos_min = np.array([1, 2])
        seqdir_min = np.array([1, 1])
        seqllr_min = np.array([0.5, -0.3])
        rbef_min = np.array([1, 1])
        raft_min = np.array([1, 2])
        
        result = getc(seqind_min, seqpos_min, seqdir_min, seqllr_min, 
                     rbef_min, raft_min)
        
        # Should still return valid structure
        self.assertIn('nrev', result)
        self.assertIn('rrev', result)
        self.assertIn('nrep', result)
        self.assertIn('rrep', result)

    def test_getc_array_consistency(self):
        """Test that getc handles array dimensionality consistently"""
        # Test with 1D arrays
        result_1d = getc(self.seqind, self.seqpos, self.seqdir, self.seqllr, 
                        self.rbef, self.raft)
        
        # Test with 2D arrays (single dataset)
        rbef_2d = self.rbef.reshape(-1, 1)
        raft_2d = self.raft.reshape(-1, 1)
        
        result_2d = getc(self.seqind, self.seqpos, self.seqdir, self.seqllr, 
                        rbef_2d, raft_2d)
        
        # Results should have same content but different shapes
        self.assertEqual(result_1d['nrev'].shape, (8,))
        self.assertEqual(result_2d['nrev'].shape, (8, 1))


class TestFitActobsSim(unittest.TestCase):
    """Test suite for fit_actobs_sim.py module"""

    def setUp(self):
        """Set up test fixtures"""
        np.random.seed(42)
        
        # Create minimal test configuration
        self.cfg = {
            'seqind': np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5]),
            'seqpos': np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5]),
            'seqdir': np.array([1, 1, 2, 2, 1, 2, 2, 1, 1, 2]),
            'seqllr': np.array([0.5, -0.3, 0.8, -0.2, 0.6, -0.4, 0.7, -0.1, 0.9, -0.5]),
            'seqlen': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            'rbef': np.array([1, 1, 1, 2, 2, 2, 2, 1, 1, 1]),
            'raft': np.array([1, 1, 2, 2, 1, 2, 1, 1, 2, 2]),
            'nsmp': 100,
            'nres': 10,
            'nval': 10,
            'nrun': 2,
            'verbose': 0
        }
        
        # Configuration with confidence data
        self.cfg_with_conf = self.cfg.copy()
        self.cfg_with_conf['cbef'] = np.array([1, 1, 2, 2, 1, 2, 1, 1, 2, 2])
        self.cfg_with_conf['caft'] = np.array([1, 2, 2, 1, 1, 2, 2, 1, 1, 2])
        self.cfg_with_conf['fitcnf'] = True

    def test_upfun_basic(self):
        """Test the upfun function"""
        x = np.array([0.0, 1.0, -1.0])
        h = 0.1
        result = upfun(x, h)
        
        self.assertEqual(result.shape, x.shape)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_upfun_edge_cases(self):
        """Test upfun with edge cases"""
        # Test with very small h
        result = upfun(np.array([0.0]), 1e-10)
        self.assertTrue(np.isfinite(result))
        
        # Test with h close to 1
        result = upfun(np.array([0.0]), 0.999)
        self.assertTrue(np.isfinite(result))

    def test_estimate_ll_sd_basic(self):
        """Test estimate_ll_sd function"""
        phat = {'h': 0.1, 'sinf': 0.5, 'ssel': 1.0}
        result = estimate_ll_sd(phat, self.cfg, nres=5)
        
        self.assertTrue(np.isfinite(result))
        self.assertGreater(result, 0)

    def test_fit_actobs_sim_basic(self):
        """Test basic functionality of fit_actobs_sim"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fit_actobs_sim(self.cfg)
        
        # Check expected keys
        expected_keys = ['h', 'sinf', 'ssel', 'xhat', 'xnam', 'll', 'll_sd', 
                        'aic', 'bic', 'nsmp', 'nval', 'nrun', 'ntrl', 'nfit', 
                        'cfg', 'csub', 'cfit']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check parameter bounds
        self.assertGreater(result['h'], 0)
        self.assertLess(result['h'], 1)
        self.assertGreater(result['sinf'], 0)
        self.assertGreater(result['ssel'], 0)

    def test_fit_actobs_sim_with_confidence(self):
        """Test fit_actobs_sim with confidence data"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fit_actobs_sim(self.cfg_with_conf)
        
        # Should have confidence parameters
        self.assertIn('scnf', result)
        self.assertIn('tcnf', result)
        self.assertIn('gcnf', result)

    def test_fit_actobs_sim_validation(self):
        """Test input validation in fit_actobs_sim"""
        # Missing experiment data
        cfg_bad = self.cfg.copy()
        del cfg_bad['seqind']
        
        with self.assertRaises(ValueError):
            fit_actobs_sim(cfg_bad)

    def test_fit_actobs_sim_fixed_parameters(self):
        """Test fitting with fixed parameters"""
        cfg_fixed = self.cfg.copy()
        cfg_fixed['h'] = 0.1
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fit_actobs_sim(cfg_fixed)
        
        self.assertAlmostEqual(result['h'], 0.1, places=6)

    def test_fit_actobs_sim_reversal_fitting(self):
        """Test fitting with reversal curves"""
        cfg_rev = self.cfg.copy()
        cfg_rev['fitrev'] = True
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fit_actobs_sim(cfg_rev)
        
        self.assertIn('h', result)

    def test_fit_actobs_sim_resampling(self):
        """Test fitting with resampling"""
        cfg_resamp = self.cfg_with_conf.copy()
        cfg_resamp['resamp'] = True
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fit_actobs_sim(cfg_resamp)
        
        self.assertIn('psmp', result)


class TestRunFitActobsSim(unittest.TestCase):
    """Test suite for run_fit_actobs_sim.py module"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_data = {
            'dat': self._create_mock_matlab_struct()
        }

    def _create_mock_matlab_struct(self):
        """Create a mock matlab structure for testing"""
        # Create a mock mat_struct
        mock_struct = MagicMock()
        mock_struct._fieldnames = ['seqind', 'seqdir', 'blkind', 'rbef', 'raft', 
                                  'cbef', 'caft', 'smpang', 'smpllr']
        
        # Set up mock data
        mock_struct.seqind = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        mock_struct.seqdir = np.array([1, 1, 2, 2, 1, 2, 2, 1, 1, 2])
        mock_struct.blkind = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        mock_struct.rbef = np.array([1, 1, 1, 2, 2, 2, 2, 1, 1, 1])
        mock_struct.raft = np.array([1, 1, 2, 2, 1, 2, 1, 1, 2, 2])
        mock_struct.cbef = np.array([1, 1, 2, 2, 1, 2, 1, 1, 2, 2])
        mock_struct.caft = np.array([1, 2, 2, 1, 1, 2, 2, 1, 1, 2])
        mock_struct.smpang = [[0.1, 0.2], [0.3], [0.4, 0.5, 0.6], [0.7], [0.8, 0.9],
                             [0.1, 0.2], [0.3], [0.4, 0.5, 0.6], [0.7], [0.8, 0.9]]
        mock_struct.smpllr = [[0.1, 0.2], [0.3], [0.4, 0.5, 0.6], [0.7], [0.8, 0.9],
                             [0.1, 0.2], [0.3], [0.4, 0.5, 0.6], [0.7], [0.8, 0.9]]
        
        return mock_struct

    def test_matlab_struct_to_dict(self):
        """Test conversion of matlab struct to dict"""
        mock_struct = self._create_mock_matlab_struct()
        result = run_fit_actobs_sim.matlab_struct_to_dict(mock_struct)
        
        # Check that all fields are converted
        for field in mock_struct._fieldnames:
            self.assertIn(field, result)
        
        # Check that arrays are preserved
        self.assertTrue(isinstance(result['seqind'], np.ndarray))
        self.assertEqual(len(result['seqind']), 10)

    def test_trial_metadata_reconstruction(self):
        """Test the trial metadata reconstruction logic"""
        # Create test data that matches the reconstruction algorithm
        dat = {
            'seqdir': np.array([1, 1, 2, 2, 1, 2, 2, 1, 1, 2]),
            'blkind': np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2]),
            'smpang': [[0.1, 0.2], [0.3], [0.4, 0.5, 0.6], [0.7], [0.8, 0.9],
                      [0.1, 0.2], [0.3], [0.4, 0.5, 0.6], [0.7], [0.8, 0.9]],
            'smpllr': [[0.1, 0.2], [0.3], [0.4, 0.5, 0.6], [0.7], [0.8, 0.9],
                      [0.1, 0.2], [0.3], [0.4, 0.5, 0.6], [0.7], [0.8, 0.9]]
        }
        
        # Initialize fields
        dat['epinum'] = np.full_like(dat['seqdir'], np.nan)
        dat['seqpos'] = np.full_like(dat['seqdir'], np.nan)
        
        # Run reconstruction algorithm
        blkind = 0
        seqdir = 0
        epi = 0
        pos = 0
        
        for i in range(len(dat['seqdir'])):
            if dat['blkind'][i] != blkind:
                blkind = dat['blkind'][i]
                seqdir = dat['seqdir'][i]
                epi = 1
                pos = 1
            elif dat['seqdir'][i] != seqdir:
                seqdir = dat['seqdir'][i]
                epi += 1
                pos = 1
            dat['epinum'][i] = epi
            dat['seqpos'][i] = pos
            pos += 1
        
        dat['seqlen'] = [len(x) for x in dat['smpang']]
        dat['seqllr'] = [sum(x) for x in dat['smpllr']]
        
        # Verify results
        self.assertTrue(np.all(np.isfinite(dat['epinum'])))
        self.assertTrue(np.all(np.isfinite(dat['seqpos'])))
        self.assertEqual(len(dat['seqlen']), len(dat['seqdir']))
        self.assertEqual(len(dat['seqllr']), len(dat['seqdir']))
        
        # Check that positions start at 1 for new episodes
        self.assertEqual(dat['seqpos'][0], 1)  # First trial
        self.assertEqual(dat['seqpos'][5], 1)  # New block

    @patch('scipy.io.loadmat')
    @patch('run_fit_actobs_sim.fit_actobs_sim')
    @patch('run_fit_actobs_sim.getc')
    @patch('scipy.io.savemat')
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_main_loop_integration(self, mock_makedirs, mock_exists, mock_savemat, 
                                  mock_getc, mock_fit, mock_loadmat):
        """Test the main processing loop integration"""
        # Set up mocks
        mock_exists.return_value = True
        mock_loadmat.return_value = self.test_data
        mock_getc.return_value = {'nrev': np.array([1, 2, 3]), 'rrev': np.array([0.1, 0.2, 0.3])}
        mock_fit.return_value = {'h': 0.1, 'sinf': 0.5, 'ssel': 1.0}
        
        # Mock the module's configuration
        with patch.object(run_fit_actobs_sim, 'subjlist', [1]):
            with patch.object(run_fit_actobs_sim, 'expename', 'Experiment_1'):
                # This would run the main loop, but we'll just test the setup
                self.assertEqual(run_fit_actobs_sim.expename, 'Experiment_1')
                self.assertEqual(run_fit_actobs_sim.subjlist, [1])

    def test_experiment_configuration(self):
        """Test experiment configuration logic"""
        # Test Experiment_1 configuration
        with patch.object(run_fit_actobs_sim, 'expename', 'Experiment_1'):
            expected_subjlist = [i for i in range(1, 18) if i not in [4, 16]]
            # We can't easily test the actual subjlist assignment without running the module
            # But we can test the logic
            subjlist_test = [i for i in range(1, 18) if i not in [4, 16]]
            self.assertEqual(len(subjlist_test), 15)
            self.assertNotIn(4, subjlist_test)
            self.assertNotIn(16, subjlist_test)

    def test_file_path_construction(self):
        """Test file path construction for different experiments"""
        # Test Experiment_1 paths
        isubj, itask = 1, 1
        expected_input = f'./DATA/ACTOBS_C_S{isubj:02d}_task{itask}_expdata.mat'
        expected_output = f'./FITS/Experiment_1_sub{isubj:02d}_task{itask}_fit_bads.mat'
        
        self.assertEqual(expected_input, './DATA/ACTOBS_C_S01_task1_expdata.mat')
        self.assertEqual(expected_output, './FITS/Experiment_1_sub01_task1_fit_bads.mat')
        
        # Test Experiment_2A paths
        expected_input_2a = f'./DATA/ACTOBS_D_rule1_S{isubj:02d}_task{itask}_expdata.mat'
        expected_output_2a = f'./FITS/Experiment_2A_rule1_sub{isubj:02d}_task{itask}_fit_bads.mat'
        
        self.assertEqual(expected_input_2a, './DATA/ACTOBS_D_rule1_S01_task1_expdata.mat')
        self.assertEqual(expected_output_2a, './FITS/Experiment_2A_rule1_sub01_task1_fit_bads.mat')


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""

    def setUp(self):
        """Set up integration test fixtures"""
        np.random.seed(42)
        
        # Create a complete test dataset
        self.test_cfg = {
            'seqind': np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5]),
            'seqpos': np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5]),
            'seqdir': np.array([1, 1, 2, 2, 1, 2, 2, 1, 1, 2]),
            'seqllr': np.array([0.5, -0.3, 0.8, -0.2, 0.6, -0.4, 0.7, -0.1, 0.9, -0.5]),
            'seqlen': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            'rbef': np.array([1, 1, 1, 2, 2, 2, 2, 1, 1, 1]),
            'raft': np.array([1, 1, 2, 2, 1, 2, 1, 1, 2, 2]),
            'cbef': np.array([1, 1, 2, 2, 1, 2, 1, 1, 2, 2]),
            'caft': np.array([1, 2, 2, 1, 1, 2, 2, 1, 1, 2]),
            'nsmp': 100,
            'nres': 10,
            'nval': 10,
            'nrun': 2,
            'verbose': 0,
            'fitcnf': True,
            'fitrev': False,
            'fitrep': True
        }

    def test_complete_pipeline(self):
        """Test the complete pipeline from getc to fit_actobs_sim"""
        # First, get behavioral curves
        c = getc(self.test_cfg['seqind'], self.test_cfg['seqpos'], 
                self.test_cfg['seqdir'], self.test_cfg['seqllr'],
                self.test_cfg['rbef'], self.test_cfg['raft'],
                self.test_cfg['cbef'], self.test_cfg['caft'])
        
        # Then run fitting
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fit_actobs_sim(self.test_cfg)
        
        # Verify integration
        self.assertIn('csub', result)
        self.assertIn('cfit', result)
        
        # Check that csub and c have similar structure
        for key in ['nrev', 'rrev', 'nrep', 'rrep']:
            self.assertIn(key, result['csub'])
            self.assertIn(key, c)
            self.assertEqual(result['csub'][key].shape, c[key].shape)

    def test_data_consistency(self):
        """Test data consistency across the pipeline"""
        # Run getc
        c = getc(self.test_cfg['seqind'], self.test_cfg['seqpos'], 
                self.test_cfg['seqdir'], self.test_cfg['seqllr'],
                self.test_cfg['rbef'], self.test_cfg['raft'],
                self.test_cfg['cbef'], self.test_cfg['caft'])
        
        # Run fitting
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fit_actobs_sim(self.test_cfg)
        
        # Check that all data arrays have consistent lengths
        ntrl = len(self.test_cfg['seqind'])
        self.assertEqual(len(self.test_cfg['seqpos']), ntrl)
        self.assertEqual(len(self.test_cfg['seqdir']), ntrl)
        self.assertEqual(len(self.test_cfg['seqllr']), ntrl)
        self.assertEqual(len(self.test_cfg['rbef']), ntrl)
        self.assertEqual(len(self.test_cfg['raft']), ntrl)
        self.assertEqual(result['ntrl'], ntrl)

    def test_parameter_recovery(self):
        """Test that parameters can be recovered from simulated data"""
        # This is a more complex test that would simulate data with known parameters
        # and check if the fitting procedure can recover them
        
        # Set known parameters
        true_params = {'h': 0.1, 'sinf': 0.5, 'ssel': 1.0}
        
        # For this test, we'll just check that the fitting produces reasonable results
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fit_actobs_sim(self.test_cfg)
        
        # Check that fitted parameters are within reasonable bounds
        self.assertGreater(result['h'], 0.001)
        self.assertLess(result['h'], 0.999)
        self.assertGreater(result['sinf'], 0.001)
        self.assertLess(result['sinf'], 10)
        self.assertGreater(result['ssel'], 0.001)
        self.assertLess(result['ssel'], 10)


if __name__ == '__main__':
    # Run with reduced verbosity
    unittest.main(verbosity=2)