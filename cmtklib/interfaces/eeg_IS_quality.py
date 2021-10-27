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
            
        ### testing 
        # unassigned vertices
        not_assigned = np.where(np.isnan(labels_resmat))
        # left hem
        vertices_not_assigned_left = src[0]['vertno'][not_assigned[0][not_assigned[0]<4098]]
        not_found_left = []
        for v in vertices_not_assigned_left:
            not_found_this = True
            n = 0
            while not_found_this and n<nroi: 
                not_found_this = not any(v==labels_parc[n].vertices)
                n+=2
    
            not_found_left.append(not_found_this)
            
        
        # right hem
        vertices_not_assigned_right = src[1]['vertno'][not_assigned[0][not_assigned[0]>=4098]-4098] 
        not_found_right = []
        for v in vertices_not_assigned_right:
            not_found_this = True
            n = 1
            while not_found_this and n<nroi: 
                not_found_this = not any(v==labels_parc[n].vertices)
                n+=2
            not_found_right.append(not_found_this)
        
        pdb.set_trace()
        
        # check out the unassigned vertices 
        xdata=src[0]['rr'][:,0]
        ydata=src[0]['rr'][:,1]
        zdata=src[0]['rr'][:,2]
        used_and_assigned=src[0]['vertno'][~np.isnan(labels_resmat[:4098])]
        used_and_unassigned=src[0]['vertno'][np.isnan(labels_resmat[:4098])]
        import matplotlib.pyplot as plt
        from mpl_toolkits import mplot3d
        ax = plt.axes(projection='3d')
        ax.scatter3D(xdata[used_and_assigned], ydata[used_and_assigned], zdata[used_and_assigned],'.',color='C0')
        ax.scatter3D(xdata[used_and_unassigned], ydata[used_and_unassigned], zdata[used_and_unassigned],'o',color='C1')
        plt.show()
        ###
        
        # following abovementioned paper, compute SVD eigenvectors for each parcel 
        CTFp_mat = np.zeros((nroi,nvert))
        for n in range(nroi): 
            Mp = abs(res_matrix[labels_resmat==n,:]) 
            u,s,CTFp = np.linalg.svd(Mp)
            CTFp_mat[n,:] = CTFp[:,0]
            
        # PRmat = np.zeros((nroi,nroi))
        # for n in range(nroi): 
        #     for m in range(nroi): 
        #         PRmat[n,m] = 1/
            
        return measures

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['measures_file'] = self.measures_file
        return outputs
