
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

This is a leaner and cleaner version of the code based off the [HNN
repository](https://github.com/jonescompneurolab/hnn).

The **Human Neocortical Neurosolver (HNN)** is an open-source neural modeling
tool designed to help researchers/clinicians interpret human brain imaging
data. Based off the original [HNN
repository](https://github.com/jonescompneurolab/hnn), **HNN-core** provides a
convenient way to run simulations of an anatomically and biophysically detailed
dynamical system model of human thalamocortical brain circuits with only a few
lines of code. Given its modular, object-oriented design, HNN-core makes it
easy to generate and evaluate hypotheses on the mechanistic origin of signals
measured with magnetoencephalography (MEG), electroencephalography (EEG), or
intracranial electrocorticography (ECoG). A unique feature of the HNN model is
that it accounts for the biophysics generating the primary electric currents
underlying such data, so simulation results are directly comparable to source
localized data (current dipoles in units of nano-Ampere-meters); this enables
precise tuning of model parameters to match characteristics of recorded
signals. Multimodal neurophysiology data such as local field potential (LFP),
current-source density (CSD), and spiking dynamics can also be simulated
simultaneously with current dipoles.

While the HNN-core API is designed to be flexible and serve users with varying
levels of coding expertise, the HNN-core GUI is designed to be useful to
researchers with no formal computational neural modeling or coding experience.

For more information visit <https://hnn.brown.edu>. There, we describe the use
of HNN in studying the circuit-level origin of some of the most commonly
measured MEG/EEG and ECoG signals: event related potentials (ERPs) and low
frequency rhythms (alpha/beta/gamma).

Contributors are very welcome! Please read our [Contributing Guide][] if you are interested.

Please consider supporting HNN development efforts by voluntarily [providing your demographic information here](https://docs.google.com/forms/d/e/1FAIpQLSfN2F4IkGATs6cy1QBO78C6QJqvm9y14TqsCUsuR4Rrkmr1Mg/viewform)! Note that any demographic information we collect is anonymized and aggregated for reporting on the grants that fund the continued development of HNN. All questions are voluntary.

# Installation

See [Installation Guide][]. To install `hnn-core` with the minimum dependencies
on Mac or Linux, simply do:

    $ pip install hnn_core

Note that `hnn-core` currently only supports Python 3.8, 3.9, 3.10, 3.11, and 3.12, but *not* 3.13.

If you want to track the latest developments of `hnn-core`, you can
install the current version of the code (nightly) with:

    $ pip install --upgrade https://api.github.com/repos/jonescompneurolab/hnn-core/zipball/master

If you are interested in features like GUI, Optimization, or Parallel support, or are on Windows, then please see our [Installation Guide][].

# Documentation and examples

Once you have installed `hnn_core` and the dependencies for the features you
want, we recommend downloading and executing the [example
scripts](https://jonescompneurolab.github.io/hnn-core/stable/auto_examples/index.html)
provided on the [documentation
pages](https://jonescompneurolab.github.io/hnn-core/) (as well as in the
[GitHub repository](https://github.com/jonescompneurolab/hnn-core)).

Note that `python` plots are by default non-interactive (blocking): each
plot must thus be closed before the code execution continues. We
recommend using and 'interactive' python interpreter such as
`ipython`:

    $ ipython --matplotlib

and executing the scripts using the `%run`-magic:

    %run plot_simulate_evoked.py

When executed in this manner, the scripts will execute entirely, after
which all plots will be shown. For an even more interactive experience,
in which you execute code and interrogate plots in sequential blocks, we
recommend editors such as [VS Code](https://code.visualstudio.com) and
[Spyder](https://docs.spyder-ide.org/current/index.html).

# Bug reports

Use the [GitHub Issues
tracker](https://github.com/jonescompneurolab/hnn-core/issues) to report
bugs. For user questions and scientific discussions, please see our
[GitHub Discussions
page](https://github.com/jonescompneurolab/hnn-core/discussions).

# Interested in Contributing?

Please read our [Contributing Guide][] and make sure to abide by our [Code of Conduct](https://github.com/jonescompneurolab/hnn-core/blob/master/CODE_OF_CONDUCT.md).

# Governance Structure

Our [governance structure can be found here](https://jonescompneurolab.github.io/hnn-core/stable/governance.html).

# Citing

If you use HNN-core in your work, please cite our [publication in
JOSS](https://doi.org/10.21105/joss.05848):

> Jas et al., (2023). HNN-core: A Python software for cellular and
> circuit-level interpretation of human MEG/EEG. *Journal of Open Source
> Software*, 8(92), 5848, <https://doi.org/10.21105/joss.05848>

[Contributing Guide]: https://jonescompneurolab.github.io/hnn-core/stable/contributing.html
[Installation Guide]: https://jonescompneurolab.github.io/hnn-core/stable/install.html
