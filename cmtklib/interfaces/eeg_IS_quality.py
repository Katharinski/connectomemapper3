# Copyright (C) 2009-2021, Ecole Polytechnique Federale de Lausanne (EPFL) and
# Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland, and CMP3 contributors
# All rights reserved.
#
#  This software is distributed under the open-source license Modified BSD.

#import os
import pickle
import mne
from mne.minimum_norm.resolution_matrix import make_inverse_resolution_matrix
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, traits, TraitedSpec
import pdb


class EEGQInputSpec(BaseInterfaceInputSpec):
    """Input specification for creating MNE source space."""
    
    fwd_fname = traits.File(
        desc="forward solution in fif format", mandatory=True)
    
    inv_fname = traits.File(
        desc="inverse operator in fif format", mandatory=True)
    
    epochs_fif_fname = traits.File(
        desc='eeg * epochs in .set format', mandatory=True)
    
    measures_file = traits.File(
        exists=False, desc="Quality measures dict in .pkl format")


class EEGQOutputSpec(TraitedSpec):
    """Output specification for creating MNE source space."""
    
    measures_file = traits.File(
        exists=False, desc="Quality measures dict in .pkl format")

class EEGQ(BaseInterface):
    input_spec = EEGQInputSpec
    output_spec = EEGQOutputSpec

    def _run_interface(self, runtime):
        
        fwd_fname = self.inputs.fwd_fname
        fwd = mne.read_forward_solution(fwd_fname)
        forward_1D = mne.forward.convert_forward_solution(fwd, surf_ori=True,force_fixed=True)
        inv_fname = self.inputs.inv_fname
        inverse_operator = mne.minimum_norm.read_inverse_operator(inv_fname)
        self.measures_file = self.inputs.measures_file

        measures = self._compute_measures(forward_1D, inverse_operator)
        with open(self.inputs.measures_file,'wb') as f: 
            pickle.dump(measures,f, pickle.HIGHEST_PROTOCOL)

        return runtime

    @staticmethod
    def _compute_measures(fwd, inv):
        ''' 
        Measures on the source point/dipole level are implemented using standard MNE functionality as described in 
        Hauk et al., bioRxiv (2019), https://doi.org/10.1101/672956
        
        Measures on the parcellation level are implemented following Tait, Luke, et al. "A systematic evaluation of
        source reconstruction of resting MEG of the human brain with a new high-resolution atlas: Performance,
        precision, and parcellation." Human Brain Mapping (2021). Code adapted from 
        https://github.com/lukewtait/evaluate_inverse_methods 
        '''
        method = "sLORETA" 
        snr = 3.
        lambda2 = 1. / snr ** 2

        res_matrix = make_inverse_resolution_matrix(fwd, inv, method=method, lambda2=lambda2)
        measures = dict()
        # localization error 
        # calculate difference in position between peak of point spread function and dipole position
        LE_psf = mne.minimum_norm.resolution_metrics(res_matrix, fwd['src'], function='psf', metric='peak_err')
        measures['loc_error_psf'] = LE_psf
        
        # calculate difference in position between peak of cross talk function and dipole position
        LE_ctf = mne.minimum_norm.resolution_metrics(res_matrix, fwd['src'], function='ctf', metric='peak_err')
        measures['loc_error_ctf'] = LE_ctf

        return measures

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['measures_file'] = self.measures_file
        return outputs
