---
title: 'HNN-core: Biophysical modeling for the cell and circuit level interpretation of human MEG and EEG signals'
tags:
  - Python
  - neuroscience
  - EEG
  - MEG
  - modeling
  - neocortex
authors:
  - name: Author1
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Author2
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Author3
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3
affiliations:
 - name: Brown University, USA
   index: 1
 - name: Institution Name, Country
   index: 2
 - name: Independent Researcher, Country
   index: 3
date: 29 July 2022
bibliography: paper.bib

---

# Summary

Magneto- and electroencephalography (MEG/EEG) are powerful techniques to non-invasively record human brain activity. Their primary utility is providing markers of healthy brain function and disease states. However, the explanatory power of MEG/EEG biomarkers is challenged by a lack of understanding of how these signals are generated at the cell and circuit level. To address this challenge, the Human Neocortical Neurosolver (HNN) neural modeling software was created [@neymotin2020human]. HNN is a biophysically detailed neocortical column model which simulates the neural activity that generates the primary electrical currents underlying MEG/EEG signals. `hnn-core` is a lightweight Pythonic Interface to the cortical column model implemented in HNN that retains all of the existing functionality. 

Activity in HNN is driven by biologically realistic layer-specific inputs. By simulating activity at the level of individual neurons, the ouputs can be directly compared to experimental recordings. The mechanistic origins of several neural phenomenon have been previously characterized using HNN, namely evoked responses [] and brain rhythyms [].

# Statement of need

`hnn-core` was created to enable the core functionality of HNN simulations in a Pythonic environment. The original implementation of HNN was with a graphical user interface (GUI) which allowed users to quickly test hypotheses on the mechanistic origins of specific current dipole activity patterns. While the GUI has made the software accessible to a wider range of neuroscientists, the lack of a low-level command line interface hampered improvements to the existing software. By recreating HNN according to modern open source development standards, the model can now be easily extended, maintained, and integrated into existing data analysis workflows. Since its creation, significant enhancements have been made on top of the existing functionality in HNN. This includes the ability to record local field potentials, modify network connectivity, and plot simulated outputs with an expanded suite of visualization functions. 

# Acknowledgements

We acknowledge support from