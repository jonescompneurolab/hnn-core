The HNN Roadmap
===============

Project Vision
--------------
HNN was created as a resource for the MEG/EEG community to develop and
test hypotheses on the neural origin of their human data. The foundation of
HNN is a detailed cortical column model containing generalizable features of
cortical circuitry, including layer specific synaptic drive from exogenous thalamic
and cortical sources, that simulates a primary current dipole from a single
localized brain area. In addition to calculating the primary current source in
units that are directly comparable to source localized data (Am), the details in
HNN enable interpretation of multi-scale activity including layer specific and
individual cell activity. HNN was designed based on workflows to simulate the most
commonly measured signals including ERPs and low frequency brain rhythms based on
`prior studies`_.

A main goal of HNN is to create a user-friendly interactive interface and
tutorials to teach to the MEG/EEG community how to interact with the model to
study the neural origin of  these signals, without needing to access the
underlying complex neural modeling code. An equal goal is to enable the neural
modeling and coding community to participate in HNN development. We will prioritize
best practices in open-source software design and the development of a documented API
for interoperability and to facilitate integration with other relevant open-source
platforms (e.g. MNE-Python, NetPyNE). Our vision is to create a unique
transformational software specific to interpreting the neural origin of MEG/EEG.

Timeline Overview
-----------------
This roadmap timeline outlines the major short-term (1-year) and longer-term (5-year)
goals for HNNs. The 1-year goal will entail a substantial reorganization of the
HNN code and creation of an API to facilitate HNN expansions, community contribution,
and integration with other relevant open-source platforms (e.g. MNE-Python, NetPyNE).

The HNN code will be reorganized into modules distinguishing parts of the code
that require interaction with the differential equation solver software NEURON,
code for post processing data analysis and plotting, and GUI components.
Additionally, HNN will be expanded to include the ability to simulate and
visualize LFP/CSD and updated with new parameter estimation procedures, with
corresponding tutorials.

1 Year Plan 8/2020-8/2021
-------------------------

Modularize HNN code to simplify installation, development and maintenance
Over the past year, we have been cleaning up and re-organizing the
underlying code that defines the current distribution of HNN to facilitate
expansion and community engagement in its use and development. To minimize the
dependencies that are required to install before contributing to HNN development
and maintenance, HNN’s code is being organized in two repositories, one that will
contain only code necessary for the generation and maintenance of the
HNN GUI (`HNN`_), and one that will
contain the code for the neural model simulation, post-processing data analysis
and plotting (`HNN-core`_).

The functionality in HNN-core will be imported into the HNN repository
(to be renamed HNN-GUI)  and will be used by the interactive GUI.

This modularization will begin with reorganization and improvements within
the HNN-core repository and simplification of the HNN-GUI repository in the 
following steps:

-   Creation of two modules in the HNN-core/hnn_core project folder, one of which 
    will contain parts of the code that interact directly with the NEURON simulator 
    (``hnn_core.simulation``), while the other module will contain post-processing data 
    analysis and plotting functions (``hnn_core.analysis``). All post-processing data 
    analysis functions will be moved from the HNN repository into the ``hnn_core.analysis``.
    module.
-   Parallelization of the simulation in the ``hnn_core.simulation`` module via a
    combination of MPI and Joblib. xx reference github issue and milestone here XX
-   Removal of all simulation code from the HNN repository, which will be replaced 
    with by importing from hnn_core.simulator module. HNN repository will then be 
    renamed HNN-GUI.
    Milestone here: new HNN release 
-   Reorganization of the Network class within the ``hnn_core.simulator`` module 
    to separate cortical column model from exogenous drive, and optimization routines.
-   Reorganization of Param.py file within ``hnn_core.simulator`` module to multiple files that 
    contain smaller dictionaries of parameters related to different modules of the code.
-   Extending ``hnn_core.simulator`` module to run batch simulations that enable parameter
    sweeps.
-   Development of functions in HNN-GUI to enable parameter sweeps via the GUI. 
    xx reference github issue and milestone here XX -
    Milestone here: new HNN release 


LFP/CSD Simulation, Visualization and Data Comparison
-----------------------------------------------------

Essential to testing circuit-level predictions developed in HNN is the ability to 
test the predictions with invasive recordings in animals or humans.  The most fundamental 
domain over which the predictions will be tested is local field potential (LFP) recordings 
across the cortical layers and the associated current source density (CSD) profiles.  
We will develop a method to simulate and visualize LFP/CSD across the cortical layers 
and to statistically compare model simulations to recorded data. These components will 
be developed in HNN-core, and imported into the HNN-GUI repository, along with example 
tutorials, in the following steps:

- Develop code in ``hnn_core.analysis`` to simulate and visualize LFP/CSD from cellular 
  membrane potentials.
  xx reference github issue and milestone here XX
- Develop code in ``hnn_core.analysis`` to statistically compare and visualize model 
  LFP/CSD to invasive animal data.
- Develop functions in HNN-GUI to enable simulation, visualization and data comparison 
  in the GUI.

Milestone here: new HNN release 

Parameter Estimation Expansion
------------------------------
Parameter estimation is an inherent difficulty in neural model simulation. 
HNN currently enables some parameter estimation, focussing on parameters relevant
to an ERP. New methods have been recently developed that apply a machine learning
approach to parameter estimation, namely Sequential Neural Parameter Estimation (SNPE)
(Gonçalves et al BioRxiv 2019:838383). We will adapt this method for parameter 
estimation  to work with HNN-core, enabling estimation of a distribution of parameters
that could account for recorded data, and then integrate it into the HNN-GUI with 
example tutorials for the following tasks:

- Move current ERP parameter estimation code from HNN into ``hnn_core.analysis`` module.
- Develop code for SNPE parameter estimation and visualization in ``hnn_core.analysis``.
- Develop functions in HNN-GUI to enable SNPE estimation in the GUI.

Milestone here: new HNN release 

Different Cortical Model Template Choices
-----------------------------------------
HNN is distributed with a cortical column model template that represents 
generalizable features of cortical circuitry based on prior studies. Updates to 
this model are being made by the HNN team including a model with improved pyramidal
neuron calcium dynamics, and an updated inhibitory connectivity architecture. We will
expand HNN to enable a choice of template models and updated tutorials, beginning 
with those developed by the HNN team and ultimately expanding to model development
in other platforms (e.g. NetPyNE), see 5-year plan.

- Develop a new cortical column template model with improved pyramidal neuron 
  calcium dynamics as part of the  ``hnn_core.simulator`` module.
- Develop function in HNN-GUI to choose among different template models in the GUI

Milestone here: new HNN release 

API and Tutorial development
----------------------------
The ability to interpret the neural origin of macroscale MEG/EEG signals in a 
complex high-dimensional non-linear computational neural model is challenging. 
A primary goal of HNN is to facilitate this interpretation with a clear API and 
tutorials of use via the interactive GUI. The documented API will also facilitate 
the integration of HNN with other relevant open source software (e.g. MNE-python, 
NetPyNE, see 5-year plan).

The current `GUI tutorials`_ are aimed at 
teaching users the neural origin [#f1]_ of some of the most commonly measured signals, 
including ERPs and low frequency brain rhythms from a single brain area based on prior
published studies (https://hnn.brown.edu/index.php/publications/), without command 
line coding.  An interactive GUI investigation of how parameter changes map onto 
changes in the simulated current dipole signal provides the baseline intuition 
needed to examine the neural mechanisms contributing to the signal. As new 
components are developed in HNN-GUI, new tutorials will be developed to train 
the community on how to apply them in their studies.

Some of the pending API documentation and GUI tutorials based on the above  HNN Roadmap 
include,

- Running parameter sweeps
- Simulating and visualizing LFP/CSD and comparison to invasive animal recordings
- Applying updated parameter estimation methods (SNPE)
- Choosing among different HNN cortical template models

Five-Year Plan to 2025
----------------------

Develop a framework to import cortical column models developed in NetPyNE or 
other modeling platforms into HNN  The core of HNN is a cortical column model 
that simulates macroscale current dipoles. Currently, HNN is distributed with 
a template cortical column model based on generalizable features of cortical 
circuitry and as applied in `prior studies`_.
Essential to future expansion of HNN is the ability to use other cortical column 
models that include different cell types and or different network features. 
We have begun creation of a framework where models built in NetPyNE can be adapted 
to the HNN workflows of use. As a test bed, this currently entails integration of 
the HNN cortical column model and exogenous drives into the full NetPyNE 
platform (https://github.com/jonescompneurolab/hnn/tree/netpyne/netpyne). 
To limit the scope of this effort to HNN-specific goals, i.e. neural modelling 
designed for interpretation of human EEG/MEG signals, we will work to adapt 
NetPyNE developed models into the HNN framework, and to make the adaptation 
flexible enough to include models developed in other neural modeling platforms.

**Integrate HNN and MNE-Python tools:** We will work to create a framework where 
source localization using MNE-Python is seamlessly integrated with HNN  for 
circuit-level interpretation of the signal. We will begin with median-nerve 
stimulation as a test-case example.

**Convert HNN to web-based platform with dual GUI and Command Line Interface (CLI):**
We have begun working with MetaCell (metacell.org) to convert HNN to a web-based 
interactive GUI with updated graphics (https://github.com/MetaCell/HNN-UI). 
This conversion will eliminate the installation process and enhance computational 
efficiency.  Additionally, MetaCell is facilitating the transformation to a dual 
GUI and CLI interface enabled through Jupyter notebooks. There are advantages to 
both GUI and CLI in adapting HNN to user goals.  GUIs provide a framework for 
teaching the community the workflow to use such models to study the biophysical 
origin of MEG/EEG signals, like ERPs and brain rhythms. Once a meaningful 
parameter set is identified to account for the data of one subject, CLI scripts 
can be useful to investigate how well this parameter set accounts for the data 
from multiple subjects or how parameter changes impact the signal. CLIs can 
be used to generate sequences of processing steps that can then be applied 
to multiple data sets, ensuring rigor and reproducibility. Further, 
simultaneous viewing of GUI and CLI can help advanced users quickly adapt the 
code with scripting, and ultimately help create a community of HNN software 
developers. This framework will also facilitate the integration with other 
open-source platforms, including MNE-Python and NetPyNE.

**Expand HNN to include study of multi-area interactions:**
HNN is designed for detailed multi-scale interpretation of the neural origin
of macroscale current dipoles signals from a single brain area. A long term vision 
is to create a framework where multi-area interactions can be studied. We will 
begin with simulations of the interactions between sensory and motor cortices 
during median nerve stimulation.

.. _prior studies: https://hnn.brown.edu/index.php/publications/
.. _HNN-core: https://github.com/jonescompneurolab/hnn-core
.. _HNN: https://github.com/jonescompneurolab/hnn
.. _GUI tutorials: https://hnn.brown.edu/index.php/tutorials/

.. rubric:: Footnotes

.. [#f1] We do not claim all the neural mechanisms of these signals are completely understood,
         rather that there is a baseline of knowledge to build from and that HNN provides a 
         framework for further investigation.
