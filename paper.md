---
title: 'HNN-core: Biophysical modeling for the cellular and network interpretation of human MEG and EEG signals'
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

Magneto- and electroencephalography (MEG/EEG) are powerful techniques to non-invasively record human brain activity. Their primary utility is providing markers of healthy brain function and disease states. However, the explanatory power of MEG/EEG biomarkers is challenged by a lack of understanding of how these signals are generated at the cell and circuit level. To address this challenge, the Human Neocortical Neurosolver (HNN) neural modeling software was created [@neymotin:2020]. HNN is a biophysically detailed neocortical column model which simulates the neural activity that generates the primary electrical currents underlying MEG/EEG signals.

`hnn-core` is a lightweight Pythonic Interface to the cortical column model implemented in the Human Neocortical Neurosolver (HNN) software. HNN was initially introduced as a standalone software operated through a graphical user interface.

# Statement of need

`hnn-core` was created to enable the core functionality of HNN simulations in a Pythonic environment. By recreating HNN according to modern open source development standards, the software can now be easily extended, maintained, and integrated into existing data analysis workflows. Since its creation, significant enhancements have been made on top of the existing functionality in HNN. This includes the ability to record local field potentials, modify network connectivity, and an expanded suite of visualization functions. 


# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge support from

# References