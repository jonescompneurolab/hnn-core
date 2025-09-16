
<h1 align="center">
<img src="https://raw.githubusercontent.com/jonescompneurolab/jones-website/master/images/frontpage/logos/logo-hnn-medium.png" width="300">
</h1><br>

# hnn-core

[![tests](https://github.com/jonescompneurolab/hnn-core/actions/workflows/unix_unit_tests.yml/badge.svg?branch=master)](https://github.com/jonescompneurolab/hnn-core/actions/?query=branch:master+event:push)
[![CircleCI](https://circleci.com/gh/jonescompneurolab/hnn-core.svg?style=svg)](https://circleci.com/gh/jonescompneurolab/hnn-core)
[![Codecov](https://codecov.io/gh/jonescompneurolab/hnn-core/branch/master/graph/badge.svg)](https://codecov.io/gh/jonescompneurolab/hnn-core)
[![PyPI](https://img.shields.io/pypi/dm/hnn-core.svg?label=PyPI%20downloads)](https://pypi.org/project/hnn-core/)
[![Gitter](https://badges.gitter.im/jonescompneurolab/hnn_core.svg)](https://gitter.im/jonescompneurolab/hnn-core?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![JOSS](https://joss.theoj.org/papers/10.21105/joss.05848/status.svg)](https://doi.org/10.21105/joss.05848)

![HNN-GUI](https://raw.githubusercontent.com/jonescompneurolab/hnn-core/acbcc4a598610dc3be5d4b0b7c59f98251ea7690/.github/images/hnn_gui.png)

# About

The [Human Neocortical Neurosolver (HNN)](https://hnn.brown.edu) is an open-source
neural modeling tool designed to help researchers/clinicians interpret human brain
imaging data. This repository, called **HNN-core**, houses the source code for HNN.

With only a few lines of code, HNN provides a convenient way to run simulations of an
anatomically- and biophysically-detailed dynamical system model of human thalamocortical
brain circuits. Given its modular, object-oriented design, HNN makes it easy to generate
and evaluate hypotheses on the mechanistic origin of signals measured with
magnetoencephalography (MEG), electroencephalography (EEG), or intracranial
electrocorticography (ECoG). A unique feature of the HNN model is that it accounts for
the biophysics generating the primary electric currents underlying such data. Simulation
results are *directly* comparable to source-localized data (current dipoles in units of
nano-Ampere-meters), enabling precise tuning of model parameters to match
characteristics of recorded signals. Multimodal neurophysiology data such as local field
potential (LFP), current-source density (CSD), and spiking dynamics can also be
simulated simultaneously with current dipoles.

You can view [HNN's frontpage here](https://hnn.brown.edu) for an overview of all that
HNN can do. For how to use HNN, we provide scientific documentation, tutorials, and
examples aplenty on our [HNN Textbook website][]. There, we describe the use of HNN in
studying the circuit-level origin of some of the most commonly measured MEG/EEG and ECoG
signals: event related potentials (ERPs) and low-frequency rhythms (alpha/beta/gamma).

The HNN API, written in Python and built on top of
[NEURON](https://nrn.readthedocs.io), is designed to be flexible and serve
users with varying levels of coding expertise, while the [HNN
GUI](https://jonescompneurolab.github.io/textbook/content/04_using_hnn_gui/gui_quickstart.html)
is designed to be useful to researchers with no formal computational neural modeling or
coding experience.

The terms HNN, HNN-core, and `hnn-core` are effectively equivalent, as they are all
different names for the same codebase. Historically, HNN-core was developed based on
the [original, deprecated HNN repository](https://github.com/jonescompneurolab/hnn), however that
repository is **no longer supported or developed**. It is kept online only for the sake
of scientific reproducibility.

Please consider supporting HNN development efforts by voluntarily [providing your
demographic information
here](https://docs.google.com/forms/d/e/1FAIpQLSfN2F4IkGATs6cy1QBO78C6QJqvm9y14TqsCUsuR4Rrkmr1Mg/viewform)!
Note that any demographic information we collect is anonymized and aggregated for
reporting on the grants that fund the continued development of HNN. All questions are
voluntary.

# Installation

You can try HNN **in your browser for free, with no local installation required!** At
the top of our [Installation Guide][], you can find links that describe how to run HNN
online in the cloud, either using Google CoLab notebooks or using the [Neuroscience Gateway
Portal](https://www.nsgportal.org/).

To install HNN locally, see our [Installation Guide][] located at the [HNN Textbook
website][]. The easiest way to install `hnn-core` with the all its dependencies on Mac,
Linux, or Windows (using "Windows Subsystem for Linux"), is to first install the
[Anaconda Python Distribution](https://www.anaconda.com/download/success) and then run
the following commands:

```
conda create -y -q -n hnn-core-env python=3.12
conda activate hnn-core-env
conda install hnn-core-all -c jonescompneurolab -c conda-forge
```

Our Anaconda packages currently only support Python 3.12. However, installing `hnn-core`
through `pip` currently supports **Python 3.9 through 3.13**, inclusively. Please see
our [Installation Guide][] for detailed instructions on the various ways you can install
HNN.

# Usage

Once you have installed `hnn-core` and the dependencies for the features you want, you
can find tutorials, examples, and scientific documentation at our [HNN Textbook
website][].

# Problems?

You can use the [GitHub Issues
tracker](https://github.com/jonescompneurolab/hnn-core/issues) to report bugs. For user
questions, installation help, and scientific discussions, please see our [GitHub
Discussions page](https://github.com/jonescompneurolab/hnn-core/discussions).

# Interested in Contributing?

Contributors are always welcome! Please read our [Contributing Guide][] and make sure to
abide by our [Code of
Conduct](https://github.com/jonescompneurolab/hnn-core/blob/master/CODE_OF_CONDUCT.md). Our
[governance structure can be found
here](https://jonescompneurolab.github.io/hnn-core/stable/governance.html).

# Citing

If you use HNN-core in your work, please cite our [publication in
JOSS](https://doi.org/10.21105/joss.05848):

> Jas et al., (2023). HNN-core: A Python software for cellular and
> circuit-level interpretation of human MEG/EEG. *Journal of Open Source
> Software*, 8(92), 5848, <https://doi.org/10.21105/joss.05848>

[Contributing Guide]: https://jonescompneurolab.github.io/hnn-core/stable/contributing.html
[HNN Textbook website]: https://jonescompneurolab.github.io/textbook/content/preface.html
[Installation Guide]: https://jonescompneurolab.github.io/textbook/content/01_getting_started/installation.html
