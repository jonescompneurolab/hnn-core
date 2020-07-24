HNN Roadmap 
-----------

This roadmap outlines the major short term (1-year) and longer-term goals for HNNs. These goals will entail a substantial updating of the HNN-core code and API to facilitate HNN expansions, community contribution, and integration with other relevant open-source platforms (e.g. MNE-Python, NetPyNE).
 
2020-2021 
=========

Re-organize HNN-core code and develop API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- (2 months) Over the past year, we have been cleaning up and re-organizing the underlying NEURON-Python code that defines HNN.   This entailed separating and compartmentalizing the code for the HNN GUI, data-analysis scripts, and HNN network model, i.e., the template cortical column network model and exogenous inputs to the network.  The latter neural network model code has been cleaned and simplified and is now referred to as HNN-Core. We are currently finalizing the parallelization of HNN-core and expect to integrate this code into the HNN distribution within the next 2 months. 

- (6 months) Further re-organization of HNN-core is required for optimal functionality and to facilitate the expansion capabilities of HNN. We will re-organize the “Nnetwork class container” in HNN-core to separate the cortical column model components, from the external drive components.  We will re-organize the parameter file to contain smaller dictionaries of parameter related to different modules of the code.

- (12 months) We will create an API for HNN-core components that will be updated as HNN-core is updated. 
 
Expand utility of HNN and develop associated tutorials 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are several features that are high priority to develop in HNN-Core, integrate into the current HNN GUI, and develop tutorials of use for distribution. 

- (3 months) Essential to testing circuit-level predictions developed in HNN is the ability to test the predictions with invasive recordings in animals or humans.  The most fundamental domain over with the predictions be tested is local field potential (LFP) recordings across the cortical layers and the associated current source density (CSD) profiles.  We will develop a method to simulate and visualize LFP/CSD across the cortical layers, building from other python packages developed for this purpose (e.g. LFPy), and to statistically compare model simulations to recorded data. These components will be developed in HNN-core, and integrated into the HNN-GUI for distribution, along with tutorials of use. 

- (6 months) We will develop the framework to enable parameter sweeps and multiple-trial runs through batch simulations. These components will be developed in HNN-core, and integrated into the HNN-GUI for distribution, along with tutorials of use. 

- (12 months) Parameter estimation is an inherent difficulty in neural model simulation. New methods have been recently developed that apply a machine learning approach to parameter estimation, namely Sequential Neural Parameter Estimation  (SNPE) (XX REF XX). We will adapt this method for parameter estimation into HNN-core, enabling estimation of a distribution of parameters that could account for recorded data. This framework  will be developed in HNN-core, and integrated into the HNN-GUI for distribution, along with tutorials of use. 
 
Five-Year Plan 
--------------

- **Integrate HNN-Core and MNE-Python tools**:  We will work to create a framework where source localization using MNE-Python is seamlessly integrated with HNN-Core for circuit level interpretation of the signal. We will begin with median-nerve stimulation as a test-case example. 
 
- **Develop a framework to import cortical column models developed in NetPyNE or other modeling platforms into HNN**:  The core of HNN is cortical column model that simulates macroscale current dipoles. Currently, HNN is distributed with a template cortical column model based on generalizable features of cortical circuitry and as applied in prior studies (XXREFSXX).  Essential to future expansion of HNN is the ability to use other cortical column models that include different cell types and or different network features. We have begun creation of a framework where models built in NetPyNE can be adapted to the HNN workflows of use. As a test bed, this currently entails integration of the HNN cortical column model and exogenous drives into the full NetPyNE platform (xx link to version xx). To limit the scope of this effort to HNN specific goals, i.e. neural modelling designed for interpretation of human EEG/MEG signals, we will work to adapt NetPyNE developed models into the HNN framework, and to make the adaptation flexible enough to include models developed in other neural modeling platforms.  
 
- **Convert HNN to a web-based interactive GUI**:  We have begun working with MetaCell (metacell.org) to convert HNN to a web-based interactive GUI with updated graphics (xx link to versionxx). This conversion will eliminate the installation process and enhance computational efficiency. 
 
- **Convert HNN to dual GUI and Command Line User Interface (CLUI)**: There are advantages to both GUI and CLUI in adapting HNN to user goals.  GUIs provide a framework for teaching the community the workflow to use such models to study the biophysical origin of M/EEG signals, like ERPs and brain rhythms.5 Once a meaningful parameter set is identified to account for the data of one subject, CLI scripts can be useful to investigate how well this parameter set accounts for the data from multiple subjects or how parameter changes impact the signal. CLUIs can be used to generate sequences of processing steps that can then be applied to multiple data sets, ensuring rigor and reproducibility. Further, simultaneous viewing of GUI and CLI can help advanced users quickly adapt the code with scripting, and ultimately help create a community of HNN software developers. We will work with Metacell to create a Jupyiter Notebook framework where simultaneous GUI and CLUI programming is possible, e.g. XX see link XX . This framework will facilitate the integration with other open-source platforms, including MNE-Python and NetPyNE.
 
- **Expand HNN to include study of multi-area interactions**: HNN is designed for detailed multi-scale interpretation of the neural origin of macroscale current dipoles signals from a signal brain area. A long term vision is to create a framework were multi-area interactions can be studied. 
 

