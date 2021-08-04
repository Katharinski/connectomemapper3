# Copyright (C) 2009-2021, Ecole Polytechnique Federale de Lausanne (EPFL) and
# Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland, and CMP3 contributors
# All rights reserved.
#
#  This software is distributed under the open-source license Modified BSD.

"""Definition of post-processing quality assessment of EEG inverse solutions."""

# General imports
import os

# Nipype imports
import nipype.pipeline.engine as pe
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, traits, TraitedSpec

# Own imports
from cmp.stages.common import Stage
from cmtklib.interfaces.eeg_IS_quality import EEGQ

class EEGQualityConfig(TraitedSpec):
    compute_measures = traits.List(['loc_error'], 
                desc='list of quality measures to be assessed for the inverse solution; default: all available measures')


class EEGQualityStage(Stage):
    def __init__(self, bids_dir, output_dir):
        """Constructor of a :class:`~cmp.stages.eeg.eeg_quality_assessment` instance."""
        self.name = 'eeg_quality_assesssment_stage'
        self.bids_dir = bids_dir
        self.output_dir = output_dir
        self.config = EEGQualityConfig()
        self.inputs = ["inverse_matrix_file","leadfield_file","dipoles_xyz"]
        self.outputs = ["measures_dict"]
        
        
        def create_workflow(self, flow, inputnode, outputnode):
            eegquality_node = pe.Node(interface=EEGQ(), name="eegquality")
            
            flow.connect([(inputnode, eegquality_node,
                   [('inverse_matrix_file','inverse_matrix_file'),
                    ('leadfield_file','leadfield_file'), 
                    ('dipoles_xyz','dipoles_xyz')
                  ]
                    )])  
            
            flow.connect([(eegquality_node, outputnode
                   [('measures','measures_dict')
                  ]
                    )])  
            
            