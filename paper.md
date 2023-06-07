---
title: 'HNN-core: A Python software for cellular and circuit-level interpretation of human MEG/EEG data'
tags:
  - Python
  - neuroscience
  - EEG
  - MEG
  - modeling
  - neocortex
authors:
  - name: Mainak Jas
    orcid: 0000-0002-3199-9027
    equal-contrib: true
    affiliation: 1
  - name: Ryan Thorpe
    orcid: 0000-0003-2491-8599
    equal-contrib: true
    affiliation: "2, 3"
  - name: Nicholas Tolley
    orcid: 0000-0003-0358-0074
    equal-contrib: true
    affiliation: "2, 3"
  - name: Christopher Bailey
    orcid: 0000-0003-3318-3344
    affiliation: 4
  - name: Steven Brandt
    affiliation: "2, 3"
  - name: Blake Caldwell
    affiliation: 3
  - name: Huzi Cheng
    affiliation: 5
  - name: Dylan Daniels
    orcid: 0009-0008-1958-353X
    affiliation: 3
  - name: Carolina Fernandez
    orcid: 0009-0003-0611-1270
    affiliation: 6
  - name: Mostafa Khalil
    affiliation: 7
  - name: Samika Kanekar
    affiliation: 3
  - name: Carmen Kohl
    affiliation: 3
  - name: Orsolya Kolozsvari
    affiliation: 
  - name: Kaisu Lankinen
    orcid: 0000-0003-2210-2385
    affiliation: "1, XX"
  - name: Kenneth Loi
    affiliation: 
  - name: Sam Neymotin
    orcid: 0000-0003-3646-5195
    affiliation: "XX, XX"
  - name: Rajat Partani
    orcid: 0000-0002-6863-7046
    affiliation: 
  - name: Mattan Pelah
    affiliation: 
  - name: Alex Rockhill
    orcid: 0000-0003-3868-7453
    affiliation: 
  - name: Mohamed Sherif
    affiliation: 

  - name: Matti Hamalainen
    orcid: 0000-0001-6841-112X
    affiliation: 
  - name: Stephanie Jones
    orcid: 0000-0001-6760-5301
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: "2, 3"


affiliations:
  - name: Athinoula A. Martinos Center for Biomedical Imaging, Massachusetts General Hospital, Boston, MA, USA
    index: 1
  - name: Department of Neuroscience, Brown University, Providence, RI, USA
    index: 2
  - name: Robert J. and Nancy D. Carney Institute for Brain Science, Brown University, Providence, RI, USA
    index: 3
  - name: Department of Clinical Medicine, Aarhus University, Aarhus, Denmark # Chris Bailey
    index: 4
  - name: Department of Psychological and Brain Sciences, Indiana University Bloomington, Bloomington, IN, USA # Huzi Cheng
    index: 5
  - name: Department of Biomedical Engineering, University of Miami, Coral Gables, FL, USA # Carolina Fernandez
    index: 6
  - name: Department of Psychiatry and Behavioral Health, Penn State Milton S. Hershey Medical Center, Penn State College of Medicine, Hershey, PA, USA # Mostafa Khalil
    index: 7
  - name: Department of Radiology, Harvard Medical School, Boston, MA, USA # Kaisu Lankinen
    index: 
  - name: University of California, Berkeley, Department of Molecular and Cell Biology, Innovative Genomics Institute, Berkeley, CA, USA # Keneth Loi
    index:
  - name: Center for Biomedical Imaging and Neuromodulation, Nathan S. Kline Institute for Psychiatric Research, Orangeburg, NY, USA # Sam Neymotin
    index: 
  - name: Department of Psychiatry, New York University Grossman School of Medicine, New York, NY, USA # Sam Ney Motin
    index:
  - name: Department of Computer Science and Engineering, National Institute of Technology Karnataka, Karnataka, India # Rajat Partani
    index: 
  - name: Department of Human Physiology, University of Oregon, Eugene, OR, USA # Alex Rockhill
    index: 
date: 5 June 2023
bibliography: paper.bib

---

# Summary

HNN-core is a library for circuit and cellular interpretation of non-invasive human magneto-/electro-encephalography (MEG/EEG) data. It is based on the Human Neocortical Neurosolver (HNN) software [@neymotin2020human], a modeling tool designed to simulate multiscale neural mechanisms generating current dipoles in a localized patch of neocortex. HNN’s foundation is a biophysically detailed neural network representing a canonical neocortical column containing populations of pyramidal and inhibitory neurons together with layer specific exogenous synaptic drive. In addition to simulating network-level interactions, HNN produces the intracellular currents in the long apical dendrites of pyramidal cells across the cortical layers known to be responsible for macroscopic current dipole generation.

The original HNN software was designed monolithically with a Graphical User Interface (GUI), making expansion and maintenance difficult. HNN-core modularizes the model components and provides an interface to modify it directly from Python. This has allowed for significant expansion of the HNN functionality through scripting, including the ability to modify additional features of local network connectivity and cell properties, record voltages in extracellular arrays, and more advanced parameter optimization and batch processing. A new web-based GUI has been developed as a thin layer over the Python interface making the overall software more maintainable. HNN-core reproduces the workflows and tutorials provided in the original HNN software to generate commonly observed MEG/EEG signals including evoked response potentials (ERPs), and alpha (8-10 Hz), beta (15-30 Hz), and gamma rhythms (30-80 Hz). HNN-core enables simultaneous calculation and visualization of macro- to micro-scale dynamics including MEG/EEG current dipoles, local field potential, laminar current-source density, and cell spiking and intrinsic dynamics. Importantly, HNN-core adopts modern open source development standards including a simplified installation procedure, unit tests, automatic documentation builds, code coverage, continuous integration, and contributing guidelines, supporting community development and long-term sustainability.

# HNN-core implements a biophysically detailed model to interpret MEG/EEG primary current sources

MEG/EEG are the leading methods to non-invasively study the human brain with millisecond resolution. They have been applied as biomarkers for healthy and pathological brain processes. Yet, historically the underlying cellular and circuit level generators of MEG/EEG signals have been difficult to infer. This cell and circuit level understanding is critical to develop theories of information processing based on these signals, or to use these techniques to develop new therapeutics for neuropathology. Computational neural modeling is a powerful technique to hypothesize the neural origin of these signals and several modeling frameworks have been developed. Since MEG/EEG recordings are dominated by neocortex, all models are based on simulating neocortical activity, however they widely vary in the level of biophysical details. One class of models known as neural mass models (NMMs) uses simplified representations to simulate net population dynamics, where hypothesized connectivity among neural “nodes” can be inferred from recordings. The Virtual Brain Project [@sanz2013virtual] and Dynamic Causal Modeling from the SPM software [@friston2003dynamic; @litvak2011eeg] are prominent examples of software that implement NMMs. While NMMs are computationally tractable and advantageous for studying brain-wide interactions, they do not provide detailed interpretation of cell and circuit level phenomena underlying MEG/EEG generation. The primary electrical currents that create MEG/EEG sensor signals are known to come from the intracellular current flow and long and spatially aligned cortical pyramidal neuron dendrites [@hamalainen1993magnetoencephalography]. These currents, known as primary current dipoles, produce extracellular electrical and magnetic fields that are picked up outside of the head with MEG/EEG sensors (for a detailed discussion see @neymotin2020human). Further, source localization methods such as MNE-Python estimate the primary electrical currents. As such, models created to study the cell and circuit origin of these signals are designed with detailed pyramidal neuron morphology and physiology, and are often embedded in a full neocortical column model. HNN is one such detailed neocortical column model [@neymotin2020human], and other examples have been employed using the software LFPy [@linden2014lfpy]. A unique feature of HNN is its workflows for interacting with the template neocortical model through layer specific activations to study ERPs and low frequency brain rhythms. HNN also enables direct comparison between simulation output and source localized data in equal units of measure and supports parameter inference. HNN-core was created to maintain all of the functionality of the original HNN software with additional utility (described below) and a clean, well-tested and documented application programming interface (API). Its adoption of open source development standards, including a simplified installation procedure, unit tests, automatic documentation builds, code coverage, and continuous integration enables community development and long term sustainability.

# HNN-core facilitates reproducibility and computationally expensive workflows

The HNN GUI and its corresponding tutorials are beneficial for novice users to learn how to interact with the neocortical model to study the multiscale origin of source localized MEG/EEG signal. The interactive GUI allows users to quickly visualize how changes in parameters impact the simulated current dipole along with simultaneous changes in layer specific cell activity to test hypotheses on the mechanistic origins of recorded MEG/EEG waveforms. While the GUI is advantageous for learning how to study the multiscale origin of MEG/EEG sources, its functionality is limited as it only enables manipulation of a subset of GUI exposed parameters. Scripting in HNN-core greatly expands the software utility particularly for large research projects where reproducibility and batch processing is of key importance. The scripted interface allows multi-trial simulations enabling the use of computationally expensive parameter optimization algorithms and parameter sweeps using parallel processing on computer clusters. HNN scripting also facilitates the creation of publication-quality figures and advanced statistical analysis. Further, the software can be integrated with existing scripted workflows, such as those developed in MNE-Python [@gramfort2013meg], a well-established source localization software, enabling source localization and circuit interpretation in just a few lines of code (see tutorial in the HNN-core documentation).

# Notable features of HNN-core 

HNN-core code enables the creation of a new and improved web-based GUI based on ipywidgets and voila that can be run remotely with port forwarding. HNN-core functionality also supports advanced simulations through scripting that are not currently possible in the GUI including:

- The ability to record extracellular local field potentials from user defined positions, as well as voltages and synaptic currents from any compartment in the model
- The ability to modify all features of the morphology and biophysical properties of any cell in the network
- An API that enables complete control of cell-cell and drive-cell connectivity in the network
- An API that allows for flexibility in defining the exogenous layer specific drive to the neocortical network
- The ability to choose from multiple template models based on previous publications (`jones_2009_model()`{.python} [@jones2009quantitative], `law_2021_model()`{.python} [@law2022thalamocortical], `calcium_model()`{.python} [@kohl2022neural])
- Built-in ERP optimization functionality designed for faster convergence 
- The choice of two parallel backends for either parallelizing across cells to speed up individual simulations (MPI), or across trials to speed up batches of simulations (joblib)

All of the code associated with HNN-core has been extensively documented at multiple levels,  including an API describing basic functions/parameters and examples of  use for hypothesis generation and/or testing. Specifically, we distribute tutorials that mimic the original GUI tutorial workflows for simulating ERPs and low frequency rhythms using HNN-core functions, with commentary on the known biophysical mechanisms of these signals. We also provide short and targeted “How to” examples that describe how to use specific functionality, such as plotting firing rates, or recording extracellular LFPs. 

# Quick example code of running a simulation

HNN-core has minimal dependencies which allows for effortless installation using the pip Python installer. In addition to numpy, scipy and matplotlib common in most libraries in the scientific Python stack, HNN-core uses Neuron for the cell and circuit modeling. Here, we demonstrate how the HNN-core interface can be used to quickly simulate and plot the net cortical dipole response to a brief exogenously evoked drive representing “feedforward” thalamocortical input. This input  (referred to as ‘evprox1’) effectively targets the proximal dendrites of the pyramidal neurons in L2/3 and L5, using the template neocortical model as in @jones2009quantitative.

```python
from hnn_core import jones_2009_model, simulate_dipole
net = jones_2009_model() # Create network model
weights_ampa = {'L2_basket': 0.09, 'L2_pyramidal': 0.02, 
                'L5_basket': 0.2, 'L5_pyramidal': 8e-3}
synaptic_delays = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                   'L5_basket': 1.0, 'L5_pyramidal': 1.0}
# Add inputs to drive activity in the network
net.add_evoked_drive(name='evprox1', mu=26.61, sigma=2.47, numspikes=1,
                     weights_ampa=weights_ampa, location='proximal',
                     synaptic_delays=synaptic_delays)
# Simulate and plot electrical current dipole
dpl = simulate_dipole(net, tstop=170.0, dt=0.025)
dpl[0].plot()
```
** ADD CODE AND PLOTS HERE **

# Ongoing research using HNN-core

The scripted interface of HNN-core has enabled the development of advanced parameter inference techniques [@tolley2023methods] using Simulation Based Inference [@tejero-cantero2020sbi]. It has been used in @thorpe2021distinct to propose new mechanisms of innocuous versus noxious sensory processing in the primary somatosensory neocortex. Lankinen et al. (2023)[**NEED TO GET REFERENCE] have used HNN-core to study crossmodal interactions between auditory and visual cortices. They performed group analysis on multiple subjects along with optimization and nonparametric statistical testing. Additionally, @szul2022diverse used it for understanding features of beta bursts in motor cortex and @fernandez2023laminar to study auditory perception.

Overall, HNN-core provides an expandable and sustainable Python-based software package that can help advance understanding of the cellular and circuit mechanisms of MEG/EEG signal generation and ultimately lead to new neuroscience discoveries.

# Acknowledgements

HNN-core was supported by NIH grants R01EB022889, 2R01NS104585-05, R01MH130415, R01AG076227, and Google Summer of Code.

# References