# Copyright (C) 2009-2021, Ecole Polytechnique Federale de Lausanne (EPFL) and
# Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland, and CMP3 contributors
# All rights reserved.
#
#  This software is distributed under the open-source license Modified BSD.

#import os
#import pickle
import numpy as np
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, traits, TraitedSpec


class EEGQInputSpec(BaseInterfaceInputSpec):
    """Input specification for creating MNE source space."""
    
    inverse_matrix_file = traits.File(
        exists=True, desc='spatial filters created for computing inverse solution', mandatory=True)    
    
    leadfield_file = traits.File(
        exists=True, desc='leadfield matrix created from head model', mandatory=True)
    
    dipoles_xyz = traits.File(
        exists=True, desc='csv file containing dipole positions', mandatory=True)

class EEGQOutputSpec(TraitedSpec):
    """Output specification for creating MNE source space."""
    
    measures = traits.Dict(desc='dictionary containing quality measurements')

class EEGQ(BaseInterface):
    input_spec = EEGQInputSpec
    output_spec = EEGQOutputSpec

    def _run_interface(self, runtime):
        
        leadfield = np.load(self.inputs.leadfield_file)
        inverse_matrix = np.load(self.inputs.inverse_matrix_file)
        res_matrix = inverse_matrix*leadfield 
        pos = self.inputs.dipoles_xyz
        
        self.measures = self._compute_measures(res_matrix,pos)

        return runtime

    @staticmethod
    def _compute_measures(res_matrix,pos):
        ''' 
        measures are implemented following Tait, Luke, et al. "A systematic evaluation of source reconstruction of
        resting MEG of the human brain with a new high-resolution atlas: Performance, precision, and parcellation."
        Human Brain Mapping (2021). Code adapted from https://github.com/lukewtait/evaluate_inverse_methods 
        '''
        measures = dict()
        # localization error 
        # calculate difference in position between peak of point spread function and dipole position
        idxpeak = np.argmax(abs(res_matrix)) 
        le = np.zeros((len(idxpeak,1)))
        for i in range(len(idxpeak)): 
            le[i] = np.sqrt(np.sum((pos[i,:]-pos[idxpeak[i],:]**2))) 
                            
        measures['loc_error'] = le
        return measures

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['quality_measures'] = self.measures
        return outputs
