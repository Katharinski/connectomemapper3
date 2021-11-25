# Copyright (C) 2009-2021, Ecole Polytechnique Federale de Lausanne (EPFL) and
# Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland, and CMP3 contributors
# All rights reserved.
#
#  This software is distributed under the open-source license Modified BSD.

"""Definition of config and stage classes for computing brain parcellation."""

# General imports
import os
from traits.api import *

# Nipype imports
import nipype.pipeline.engine as pe

# Own imports
from cmp.stages.common import Stage
from cmtklib.interfaces.invsol import CartoolInverseSolutionROIExtraction
from cmtklib.interfaces.createcov import CreateCov
from cmtklib.interfaces.createfwd import CreateFwd
from cmtklib.interfaces.invsol_MNE import MNEInverseSolution


class EEGInverseSolutionConfig(HasTraits):
    invsol_format = Enum('Cartool-LAURA', 'Cartool-LORETA', 'mne-sLORETA',
                         desc='Cartool vs mne')


class EEGInverseSolutionStage(Stage):
    def __init__(self, bids_dir, output_dir):
        """Constructor of a :class:`~cmp.stages.parcellation.parcellation.ParcellationStage` instance."""
        self.name = 'eeg_inverse_solution_stage'
        self.bids_dir = bids_dir
        self.output_dir = output_dir
        self.config = EEGInverseSolutionConfig()
        self.inputs = ["bids_dir","subject","eeg_ts_file", "epochs_fif_fname", "rois_file", "src_file", "invsol_file",
                       "lamda", "svd_params", "roi_ts_file", "invsol_params", "bem_file", "noise_cov_fname",
                       "trans_fname", "fwd_fname","inv_fname","parcellation"]
        self.outputs = ["roi_ts_file", "fwd_fname"]

    def create_workflow(self, flow, inputnode, outputnode):

        if self.config.invsol_format.split('-')[0] == "Cartool":
            invsol_node = pe.Node(CartoolInverseSolutionROIExtraction(), name="invsol")
            flow.connect([(inputnode, invsol_node,
                           [('eeg_ts_file', 'eeg_ts_file'),
                            ('rois_file', 'rois_file'),
                            ('invsol_file', 'invsol_file'),
                            ('invsol_params', 'invsol_params'),
                            ('roi_ts_file', 'roi_ts_file')
                            ]
                           )])
            flow.connect([(invsol_node, outputnode,
                 [
                  ('roi_ts_file','roi_ts_file'),
                 ]
                    )])
            
        elif self.config.invsol_format.split('-')[0] == "mne":
            
            covmat_node = pe.Node(CreateCov(), name='createcov') # compute the noise covariance
            fwd_node = pe.Node(CreateFwd(), name='createfwd') # compute the forward solution 
            invsol_node = pe.Node(MNEInverseSolution(), name="invsol") # compute the inverse operator 
            
            flow.connect([(inputnode, covmat_node,
                   [('noise_cov_fname','noise_cov_fname'),
                    ('epochs_fif_fname','epochs_fif_fname')
                  ]
                    )])  
            
            flow.connect([(inputnode, fwd_node,
                   [('fwd_fname','fwd_fname'),
                    ('src_file','src'),
                    ('bem_file','bem'),
                    ('trans_fname','trans_fname'),
                    ('epochs_fif_fname','epochs_fif_fname')
                  ]
                    )]) 

            flow.connect([(inputnode, invsol_node,
                   [('subject', 'subject'),
                    ('bids_dir', 'bids_dir'),
                    ('fwd_fname','fwd_fname'),
                    ('inv_fname','inv_fname'),
                    ('src_file','src_file'),
                    ('bem_file','bem_file'),
                    ('epochs_fif_fname','epochs_fif_fname'),
                    ('parcellation','parcellation'),
                    ('roi_ts_file', 'roi_ts_file')
                   ]
                    )])      
            
            # use dummy variables "has_run" to enforce order of nodes although they don't produce outputs 
            # (outputs are files with fixed names that are defined in eeg.py)
            flow.connect([(covmat_node,invsol_node,
                           [('has_run','cov_has_run'),
                            ('noise_cov_fname','noise_cov_fname')
                            ]
                           )])
            
            flow.connect([(fwd_node,invsol_node,
                           [('has_run','fwd_has_run')
                            ]
                           )])

            flow.connect([(invsol_node, outputnode,
                  [('fwd_fname','fwd_fname'),
                   ('roi_ts_file','roi_ts_file'),
                  ]
                    )])


    def define_inspect_outputs(self):
        raise NotImplementedError

    def has_run(self):
        """Function that returns `True` if the stage has been run successfully.

        Returns
        -------
        `True` if the stage has been run successfully
        """
        if self.config.eeg_format == ".set":
            if self.config.invsol_format.split('-')[0] == "Cartool":
                return os.path.exists(self.config.roi_ts_file)
