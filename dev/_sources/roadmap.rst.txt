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
units that are directly comparable to source localized data (Ampere-meters, Am), 
the details in
HNN enable interpretation of multi-scale activity including layer specific and
individual cell activity. HNN was designed based on workflows to simulate the most
commonly measured signals, including ERPs and low frequency brain rhythms based on
`prior studies`_.

A main goal of HNN is to create a user-friendly interactive interface and
tutorials to teach to the MEG/EEG community how to interact with the model to
study the neural origin of  these signals, without needing to access the
underlying complex neural modeling code. To this end, HNN was constructed with a 
graphical user interface (GUI) and corresponding tutorials of use for commonly measured signals, 
which are distributed on the HNN website (https:/hnn.brown.edu). 
Our philosophy is that the interactive GUI is essential for all new users of HNN to develop an intuition 
on how to interact with the large-scale computational model to study the multi-scale neural dynamics underlying 
their MEG/EEG data. Once this intuition is gained, users who chose to can dive into the computational neural 
modeling code, where further command line utily can be developed. As such, an equal goal is to enable the neural
modeling and coding community to participate in HNN development. We are prioritizing
best practices in open-source software design and the development of a documented API
for interoperability and to facilitate integration with other relevant open-source
platforms (e.g. MNE-Python, NetPyNE). Our vision is to create a unique
transformational software specific to interpreting the neural origin of MEG/EEG.

Timeline Overview
-----------------
This roadmap timeline outlines the major short-term and longer-term 
goals for HNNs. The short term goals will entail a substantial reorganization of the
HNN code and creation of an API to facilitate HNN expansions, community contribution,
and integration with other relevant open-source platforms (e.g. MNE-Python, NetPyNE). To this end, in March 2021, we released the first version of the HNN-core repository. HNN-core contains improved versions of HNN’s non-GUI components following best practices in open-source software design, with unit testing and continuous integration, along with initial API and documentation for command-line coding. We will adopt similar best practices to develop a new HNN-GUI and several new HNN features, including the ability to simulate and visualize LFP/CSD and to use improved parameter estimation procedures. Our process will be to develop all new features in HNN-core, with  API and examples of use followed, when applicable, by integration into the HNN-GUI with correspoding GUI-based tutorials on our website. Longer-term goals include integration with the related modeling software MNE-Python and NetPyNe, the development of a web-based interface with ability for simultaneous GUI and Command Line Interface (CLI), and extension to multi-area simulations. 

Short-Term Goals
--------------------------

Modularize HNN code to simplify installation, development and maintenance
-------------------------------------------------------------------------

We are working on cleaning up and re-organizing the
underlying code that defines the current distribution of HNN to facilitate
expansion and community engagement in its use and development. To minimize the
dependencies that are required to install before contributing to HNN development
and maintenance, all of the non-GUI components of HNN’s code are being organized into a new repository HNN-core (initial release March 2021).
This reorganization will entail continued improvements within the HNN-core repository, along with API development and examples of use, in the following steps:

-   Following best practices in open-source software design, including continuous integration testing, 
    to develop HNN-core. HNN-core will contain clean and reorganized code, and separate all components that 
    interact directly with the NEURON simulator (e.g. cell and network intantiation, external drives, etc..), 
    from those that pertain to post-processing data analysis and plotting functions (e.g. spectra lanalysis). 
    **COMPLETED FEB 2021** 
-   Convert installation procedures to PIP. **COMPLETED FEB 2021** 
-   Parallelization of the simulations in HNN-core via MPI or Joblib. **COMPLETED SEP 2020** 
-   Reorganization of the Network class within HNN-core module 
    to separate cortical column model from exogenous drive, and optimization routines.
    See `gh-104`_, `gh-124`_, and `gh-129`_ for related discussions.
    **COMPLETED FEB 2021** 
-   Develop initial HNN-core documentation and example simulations following those 
    detailed in the HNN-GUI tutorials https://jonescompneurolab.github.io/hnn-core/stable/index.html.
    **COMPLETED MARCH 2021** 
-   First release of HNN-Core 0.1 to the community **COMPLETED MARCH 2021** 
-   Make HNN-Core compatible for windows including installation,  testing and 
    continuous integration. 
-   Reorganization of Param.py file within HNN-core to multiple files that 
    contain smaller dictionaries of parameters related to different modules of the code.
    See `gh-104`_ for related discussions.
-   Expand details in HNN-core examples to follow HNN-GUI based tutorials.


Develop a New HNN GUI
-------------------------------------------------------------------------
A new HNN-GUI will be developed following similar best-practices in open source software design, as employed in HNN-core. 
The first step will be to ensure all of the functionality of the current GUI distribution is developed in HNN-Core, followed by
integration into a new HNN-GUI, with corresponding GUI-based tutorials on the HNN website. Once complete, the current HNN-GUI repository will be deprecated.  

-   Development of optimization routines in HNN-core that have the current functionality
    in HNN-GUI. 
-   Develop a new HNN-GUI using ipywidgets in HNN-core that has all of the functionality
    of the current HNN-GUI.
-   Rename HNN to HNN-GUI and release updated version to the community and deprecate
    original HNN repository.


LFP/CSD Simulation, Visualization and Data Comparison
-----------------------------------------------------

Essential to testing circuit-level predictions developed in HNN is the ability to 
test the predictions with invasive recordings in animals or humans.  The most fundamental 
domain over which the predictions will be tested is local field potential (LFP) recordings 
across the cortical layers and the associated current source density (CSD) profiles.  
We will develop a method to simulate and visualize LFP/CSD across the cortical layers 
and to statistically compare model simulations to recorded data. These components will 
be developed in HNN-core, with correponding API and examples of use, followed by integration 
into the HNN-GUI, with corresponding GUI based tutorials on the HNN website, in the following steps:

- Develop code in HNN-core to simulate and visualize LFP/CSD from cellular 
  membrane potentials.
- Develop code in HNN-core to statistically compare and visualize model 
  LFP/CSD to invasive animal data.
- Develop functions in HNN-GUI to enable simulation, visualization and data comparison 
  in the GUI.

Parameter Estimation Expansion
------------------------------
Parameter estimation is an inherent difficulty in neural model simulation. 
HNN currently enables some parameter estimation, focussing on parameters relevant
to an ERP. New methods have been recently developed that apply a machine learning
approach to parameter estimation, namely Sequential Neural Parameter Estimation (SNPE)
(Gonçalves et al Elife 2020: DOI: 10.7554/eLife.56261). We will adapt this method for parameter 
estimation to work with HNN-core, enabling estimation of a distribution of parameters
that could account for empirical data, and then integrate it into the HNN-GUI, with 
GUI-based tutorials, in the following steps:

- Extending HNN-core to run batch simulations that enable parameter sweeps.
- Development of functions in HNN-GUI to enable parameter sweeps via the GUI. 
- Develop code for SNPE parameter estimation and visualization in HNN-core.
- Develop functions in HNN-GUI to enable SNPE estimation in the GUI.

Different Cortical Model Template Choices
-----------------------------------------
HNN is distributed with a cortical column model template that represents 
generalizable features of cortical circuitry based on prior studies. Updates to 
this model are being made by the HNN team, including a model with alternate pyramidal
neuron calcium dynamics, and an updated inhibitory connectivity architecture. We will
expand HNN-core to enable a choice of template models, beginning 
with those developed by the HNN team and ultimately expanding to model development
in other platforms (e.g. NetPyNE), see Longer-Term goals. These models will first be 
developed in HNN-core, with corresponding API and examples of use, followed by integration 
into HNN-GUI, with GUI-based tutorials. 

- Develop new cortical column template models with pyramidal neuron 
  calcium dynamics, in HNN-core.
- Create flexibility to change local connectivity and to visualize connectivity in HNN-core.
- Create flexibility to change exogenous connectivity and to visualize connectivity in HHN-core.
- Develop functionality in HNN-GUI to chose amng different template models.
- Develop function in HNN-GUI to choose among different template models in the GUI.

See `gh-111`_ for more discussions.

API and Tutorial development
----------------------------
The ability to interpret the neural origin of macroscale MEG/EEG signals in a 
complex high-dimensional non-linear computational neural model is challenging. 
A primary goal of HNN is to facilitate this interpretation with a clear API and examples 
of use in HNN-core, and interative GUI-based tutorals for all HNN-GUI functionality on our HNN website.  
Following the process for creating new featuers in HNN, the process for documenting 
new features will be to first develop them with API and examples of use in HNN-core, followed
by integration into the HNN-GUI, with corresponding GUI-based tutorials on the HNN-website. 
Developmental goals are only complete once the corresponding documentation is available. 


Longer-Term Goals
----------------------

**Develop a framework to import cortical column models developed in NetPyNE or 
other modeling platforms into HNN:** 
The core of HNN is a cortical column model 
that simulates macroscale current dipoles. Currently, HNN is distributed with 
a template cortical column model based on generalizable features of cortical 
circuitry and as applied in `prior studies`_.
Essential to future expansion of HNN is the ability to use other cortical column 
models that include different cell types and or different network features. 
We have begun creation of a framework where models built in NetPyNE can be adapted 
to the HNN workflows of use. As a test bed, this currently entails integration of 
the HNN cortical column model and exogenous drives into the full NetPyNE 
platform (https://github.com/jonescompneurolab/hnn/tree/netpyne/netpyne). 
See also update from **MARCH 2021** https://github.com/jonescompneurolab/hnn/tree/hnn2 .

To limit the scope of this effort to HNN-specific goals, i.e. neural modeling 
designed for interpretation of human EEG/MEG signals, we will work with NetPyNE team 
to develop clean modularized framework for integrating NetPyNe developed cortical models 
that have laminar structure and multicompartment pyramidal neurons into HNN design and workflows 
of use to simulate ERPs and low frequency brain rhythms work.  

**Integrate HNN and MNE-Python tools:** We will work to create a framework where 
source localization using MNE-Python is seamlessly integrated with HNN  for 
circuit-level interpretation of the signal. We will develop workflows that enable users 
starting with sensor level signals to perform both source localization using MNE-Python 
and circuit interpretation using HNN-core. We begin with use open-source median nerve 
datasets and develop examples using three different inverse methods (Dipole, MNE, Beamformer). 

- Develop test-case example using open-source median nerve data of how to go from 
  sensor space data to source localized signal using MNE-Python, and then
  simulate the neural mechanisms of the source signal using HNN-core.  
  https://jonescompneurolab.github.io/hnn-core/stable/auto_examples/index.html
  **COMPLETED MARCH 2021 - note still needs documentation** 

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
.. _gh-104: https://github.com/jonescompneurolab/hnn-core/issues/104
.. _gh-111: https://github.com/jonescompneurolab/hnn-core/issues/111
.. _gh-124: https://github.com/jonescompneurolab/hnn-core/issues/129
.. _gh-129: https://github.com/jonescompneurolab/hnn-core/issues/124

.. rubric:: Footnotes

.. [#f1] We do not claim all the neural mechanisms of these signals are completely understood,
         rather that there is a baseline of knowledge to build from and that HNN provides a 
         framework for further investigation.
