# Copyright (C) 2009-2020, Ecole Polytechnique Federale de Lausanne (EPFL) and
# Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland
# All rights reserved.
#
#  This software is distributed under the open-source license Modified BSD.

""" CMP Stage for building connectivity matrices and resulting files."""

# Global imports
from traits.api import *
from traitsui.api import *
import subprocess

# Own imports
from cmp.stages.connectome.fmri_connectome import ConnectomeConfig, ConnectomeStage


class ConnectomeConfigUI(ConnectomeConfig):
    """Class used to represent the GUI for the configuration of the fMRI connectome stage.

    Attributes
    ----------
    output_types list<<string>>
        A list of ``output_types``. Valid ``output_types`` are
        'gPickle', 'mat', 'cff', 'graphml'

    traits_view <View>
        TraitsUI view that displays the Attributes of this class

    See also
    ---------
    cmp.stages.connectome.fmri_connectome.ConnectomeConfig
    """

    output_types = List(['gPickle'], editor=CheckListEditor(
        values=['gPickle', 'mat', 'cff', 'graphml'], cols=4))

    traits_view = View(VGroup('apply_scrubbing',
                              VGroup(Item('FD_thr', label='FD threshold'),
                                     Item('DVARS_thr', label='DVARS threshold'),
                                     visible_when="apply_scrubbing==True")),
                       Item('output_types', style='custom'))


class ConnectomeStageUI(ConnectomeStage):
    """Class used to represent the GUI for the fMRIconnectome stage.

    Attributes
    ----------
    log_visualization <Boolean>
        If True, display with a log transformation 

    circular_layout <Boolean>
        If True, display the connectivity matrix using a circular layout

    inspect_output_button <Button>
        Button that displays the selected connectivity matrix
        in the graphical component for quality inspection

    inspect_outputs_view <View>
        TraitsUI view that displays the quality inspection window of this stage

    config_view <View>
        TraitsUI view that displays the configuration window of this stage

    See also
    ---------
    cmp.stages.connectome.fmri_connectome.ConnectomeStage
    """

    log_visualization = Bool(True)
    circular_layout = Bool(False)

    inspect_output_button = Button('View')

    inspect_outputs_view = View(Group(Item('name', editor=TitleEditor(), show_label=False),
                                      Group(
                                            Item('log_visualization', label='Log scale'),
                                            Item('circular_layout', label='Circular layout'),
                                            label='Visualization', show_border=True),
                                      Group(
                                            Item('inspect_outputs_enum', show_label=False),
                                            Item('inspect_output_button',
                                                 enabled_when='inspect_outputs_enum!="Outputs not available"',
                                                 show_label=False),
                                        label='View outputs', show_border=True)),
                                scrollable=True,
                                resizable=True,
                                kind='livemodal',
                                title='Inspect stage outputs',
                                buttons=['OK', 'Cancel'])

    config_view = View(Group(
                             Item('name', editor=TitleEditor(), show_label=False),
                             Group(
                                   Item('config', style='custom', show_label=False),
                                   label='Configuration',
                                   show_border=True)),
                       scrollable=True,
                       resizable=True,
                       height=200,
                       width=408,
                       kind='livemodal',
                       title='Edit stage configuration',
                       buttons=['OK', 'Cancel'])

    def __init__(self, bids_dir, output_dir):
        """Constructor of the Functional ConnectomeStageUI class.

        Parameters
        ----------
        bids_dir <Directory>
            BIDS root directory

        output_dir <Directory>
            Output directory

        See also
        ---------
        cmp.stages.connectome.fmri_connectome.ConnectomeStage.__init__
        """
        ConnectomeStage.__init__(self, bids_dir, output_dir)
        self.config = ConnectomeConfigUI()

    def _log_visualization_changed(self, new):
        """Update the value of log_visualization in the config.

        Parameters
        ----------
        new <Boolean>
            New value
        """
        self.config.log_visualization = new
        self.define_inspect_outputs()

    def _circular_layout_changed(self, new):
        """Update the value of circular_layout in the config.

        Parameters
        ----------
        new <Boolean>
            New value
        """
        self.config.circular_layout = new
        self.define_inspect_outputs()

    def _inspect_output_button_fired(self, info):
        """Display the selected output when ``inspect_output_button`` is clicked.

        Parameters
        ----------
        info <Button>
            Button object
        """
        subprocess.Popen(self.inspect_outputs_dict[self.inspect_outputs_enum])
