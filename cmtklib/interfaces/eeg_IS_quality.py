# Copyright (C) 2009-2021, Ecole Polytechnique Federale de Lausanne (EPFL) and
# Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland, and CMP3 contributors
# All rights reserved.
#
#  This software is distributed under the open-source license Modified BSD.

import os
import pickle
import numpy as np
import mne
from mne.minimum_norm.resolution_matrix import make_inverse_resolution_matrix
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, traits, TraitedSpec
import pdb


class EEGQInputSpec(BaseInterfaceInputSpec):
    """Input specification for creating MNE source space."""

    subject = traits.Str(
        desc='subject', mandatory=True)
    
    bids_dir = traits.Str(
        desc='base directory', mandatory=True)
    
    fwd_fname = traits.File(
        desc="forward solution in fif format", mandatory=True)
    
    inv_fname = traits.File(
        desc="inverse operator in fif format", mandatory=True)

    src_file = traits.List(
        exists=True, desc='source space created with MNE', mandatory=True)
    
    epochs_fif_fname = traits.File(
        desc='eeg * epochs in .set format', mandatory=True)
    
    measures_file = traits.File(
        exists=False, desc="Quality measures dict in .pkl format")

    parcellation = traits.Str(
        desc='parcellation scheme')


class EEGQOutputSpec(TraitedSpec):
    """Output specification for creating MNE source space."""
    
    measures_file = traits.File(
        exists=False, desc="Quality measures dict in .pkl format")

class EEGQ(BaseInterface):
    input_spec = EEGQInputSpec
    output_spec = EEGQOutputSpec

    def _run_interface(self, runtime):
        
        bids_dir = self.inputs.bids_dir
        subject = self.inputs.subject
        parcellation = self.inputs.parcellation
        fwd_fname = self.inputs.fwd_fname
        fwd = mne.read_forward_solution(fwd_fname)
        forward_1D = mne.forward.convert_forward_solution(fwd, surf_ori=True,force_fixed=True)
        inv_fname = self.inputs.inv_fname
        inverse_operator = mne.minimum_norm.read_inverse_operator(inv_fname)
        src_file = self.inputs.src_file[0]
        self.measures_file = self.inputs.measures_file

        measures = self._compute_measures(forward_1D, inverse_operator, subject, bids_dir, parcellation, src_file)
        with open(self.inputs.measures_file,'wb') as f: 
            pickle.dump(measures,f, pickle.HIGHEST_PROTOCOL)

        return runtime

    @staticmethod
    def _compute_measures(fwd, inv, subject, bids_dir, parcellation, src_file):
        ''' 
        Measures on the source point/dipole level are implemented using standard MNE functionality as described in 
        Hauk et al., bioRxiv (2019), https://doi.org/10.1101/672956
        '''
        
        # Measures on the parcellation level are implemented following Tait, Luke, et al. "A systematic evaluation of
        # source reconstruction of resting MEG of the human brain with a new high-resolution atlas: Performance,
        # precision, and parcellation." Human Brain Mapping (2021). Code adapted from 
        # https://github.com/lukewtait/evaluate_inverse_methods 
        
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
        
        # calculate the parcel resolution matrix as described in Farahibozorg et al., 
        # "Adaptive cortical parcellations for source reconstructed EEG/MEG connectomes", 
        # Neuroimage (2018)
        subjects_dir = os.path.join(bids_dir,'derivatives','freesurfer','subjects')
        labels_parc = mne.read_labels_from_annot(subject, parc=parcellation, subjects_dir=subjects_dir)
        nroi = len(labels_parc)
        # determine region labels of source points and count vertices per region
        # note that some vertices are located in the medial wall and are ignored 
        nvert = np.shape(res_matrix)[0]
        labels_resmat = np.empty((nvert,))
        labels_resmat[:] = np.nan
        nvert_rois = np.zeros((nroi,))
        # look at src because not all surface vertices are used as dipole locations 
        src = mne.read_source_spaces(src_file, patch_stats=False, verbose=None)
        n = 0
        for n in range(nroi):
            # determine which hemisphere we're in 
            # pdb.set_trace()
            hem = labels_parc[n].name[-2:]
            if hem=='lh':
                src_id = 0
                add_id = 0
            else: 
                src_id = 1
                add_id = int(src[0]['nuse']) # source spaces are over separate surface meshes
            # find out which vertices that have label n are used in src 
            olap = list(set(labels_parc[n].vertices).intersection(src[src_id]['vertno']))
            nvert_rois[n] = len(olap)
            # determine position of those used vertices 
            final_id = np.where(np.isin(src[src_id]['vertno'], olap))
            labels_resmat[final_id[0]+add_id] = n
                   
        # following abovementioned paper, compute SVD eigenvectors for each parcel 
        CTFp_mat = np.zeros((nroi,nvert))
        for n in range(nroi): 
            Mp = abs(res_matrix[labels_resmat==n,:]) 
            u,s,CTFp = np.linalg.svd(Mp)
            CTFp_mat[n,:] = CTFp[:,0]
            
        PRmat = np.zeros((nroi,nroi))
        for n in range(nroi):
            for m in range(nroi):
                m_vertices = np.where(labels_resmat==m)[0]
                this_PRmat_nm = 0
                for k in m_vertices: 
                    this_PRmat_nm += CTFp_mat[n,k]/np.sum(CTFp_mat[:,k])
                    
                PRmat[n,m] = 1/(nvert_rois[n]) * this_PRmat_nm
                
        pdb.set_trace()
            
        return measures

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['measures_file'] = self.measures_file
        return outputs
