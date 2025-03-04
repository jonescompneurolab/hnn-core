(contributing)=
# Contributing Guide

Please read the contribution guide **until the end** before beginning
contributions.

Contributions are welcome in the form of pull requests.

Once the implementation of a piece of functionality is considered to be
bug free and properly documented (both API docs and an example script),
it can be incorporated into the master branch.

To help developing `hnn-core`, you will need a few adjustments to your
installation as shown below.

If your contributions will make use of parallel backends for using more
than one core, please see the additional installation steps for either Joblib or MPI in our {doc}`Installation Guide <install>`.

## Setting up your local development environment

### Configuring git

Instructions for how to configure git can be found on the git book
[configuration](https://git-scm.com/book/en/v2/Customizing-Git-Git-Configuration)
page.

## Making your first pull request

Changes to hnn-core are typically made by "forking" the hnn-core
repository, making changes to your fork (usually by "cloning" it to
your personal computer, making the changes locally, and then "pushing"
the local changes up to your fork on GitHub), and finally creating a
"pull request" to incorporate your changes back into the shared
"upstream" version of the codebase.

In general you'll be working with three different copies of the
hnn-core codebase: the official remote copy at
<https://github.com/jonescompneurolab/hnn-core> (usually called
`upstream`), your remote fork of the upstream repository (similar URL,
but with your username in place of `hnn-core`, and usually called
`origin`), and the local copy of the codebase on your computer. The
typical contribution process is to

1.  Make a fork of the
    [hnn-core](https://github.com/jonescompneurolab/hnn-core) repository
    to your own account on github. Look for the Fork button in the top
    right corner

2.  On the terminal of your local computer clone the fork:

        $ git clone https://github.com/<username>/hnn-core

3.  On the terminal of your local computer set up the remotes:

        $ cd hnn-core
        $ git remote add upstream https://github.com/jonescompneurolab/hnn-core

4.  Check that the remotes have been correctly added:

        $ git remote -v

    You should see:

        origin      https://github.com/<username>/hnn-core (fetch)
        origin      https://github.com/<username>/hnn-core (push)
        upstream    https://github.com/jonescompneurolab/hnn-core (fetch)
        upstream    https://github.com/jonescompneurolab/hnn-core (push)

5.  To start a new feature branch, we will copy the existing `master`
    branch from the `upstream` remote and give it a specific name:

        $ git fetch upstream master:cool_feature
        $ git checkout cool_feature

6.  Make your changes relevant to the pull request

7.  To make a commit, you first have to add the files you have changed
    to the staging area:

        $ git add -u

    ensure they have been added correctly:

        $ git status

    make a commit:

        $ git commit -m "your commit message"

    and finally check that the commit has been added:

        $ git log

    Note: see the [numpy contributing
    guide](https://numpy.org/doc/stable/dev/development_workflow.html#writing-the-commit-message)
    for tips on informative commit messages.

8.  Now push the feature branch to your fork:

        $ git push origin cool_feature

9.  Go to <https://github.com/jonescompneurolab/hnn-core/compare> to
    open the pull request by selecting your feature branch. You should
    also see this link on the terminal when you make the push.

10. After your pull request is reviewed, repeat steps 6-8 to update the
    pull request.

11. Once the pull request is ready to be merged, add the prefix [MRG]
    to the title.

See the [git book](https://git-scm.com/book/en/v2) for a more general
guide on using git.

## Installing editable hnn-core

For making changes to hnn-core, you will need to install an editable
version of hnn-core. For that you need to have the git cloned `hnn-core`
repository and use pip with the editable (`-e`) flag:

    $ git clone https://github.com/jonescompneurolab/hnn-core
    $ cd hnn-core
    $ pip install -e '.[dev]'
    $ python setup.py build_mod

The `pip install -e '.[dev]'` step will install all extra packages used
by developers to access all features and to perform testing and building
of documentation.

The last step builds `mod` files which specifies the dynamics of
specific cellular mechanisms. These are converted to C, and hence
require a compilation step. In the normal course of development, you
will not have to edit these files. However, if you do have to update
them, they will need to be rebuilt using the command:

    $ python setup.py build_mod

## Running tests

Once you have the editable hnn-core, you should install the requirements
for running the tests. Tests help ensure integrity of the package after
your change has been made. We recommend developers to run tests locally
on their computers after making changes.

We use the `pytest` testing framework.

To run the tests simply type into your terminal:

    $ make test

MPI tests are skipped if the `mpi4py` module is not installed. We highly
encourage contributors to follow the MPI portion of the
{doc}`Installation Guide <install>` so that
they can run the entire test suite locally on their computer.

As part of `make test`, your code is also "linted" (meaning checked for errors)
and spell-checked. If you would like to perform these checks individually, you
can do so by running the following command to lint with `ruff`:

    $ make lint

Or the following command to perform common spell-checking:

    $ make spell

## Updating documentation

When you update the documentation, it is recommended to build it locally
to check whether the documentation renders correctly in HTML.

Certain documentation files require explicit updates by the contributor
of a given change (i.e., you). These are:

-   `doc/api.rst` if you added a new function.
-   `doc/whats_new.rst` to document the fix or change so you can be
    credited on the next release.

Please update these documents once your pull request is ready to merge
to avoid rebase conflicts with other pull requests.

### Building the documentation

The documentation can be built using sphinx.

You can build the documentation locally using the command:

    $ cd doc/
    $ make html

If you want to build the documentation locally without running all the
examples, use the command:

    $ make html-noplot

Finally, to view the documentation, do:

    $ make view

### Writing the documentation

You are welcome to write documentation pages using
[reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#rst-primer) (RST). However,
you are also welcome to write using Sphinx's support for [MyST
Markdown](https://www.sphinx-doc.org/en/master/usage/markdown.html#markdown), which employs
[`myst-parser`](https://myst-parser.readthedocs.io/en/latest/). This gives you the power of RST with
the readability of Markdown.

If you want to take advantage of [Roles and Directives inside your Markdown files](https://myst-parser.readthedocs.io/en/latest/syntax/roles-and-directives.html#roles-directives), it is fairly straightforward to use them via ["MyST Markdown" syntax](https://myst-parser.readthedocs.io/en/latest/syntax/roles-and-directives.html#roles-directives). For example:

- If you want to refer to another local document like this: {doc}`Installation Guide <install>`, then:
    - In RST, write:
       ```
       :doc:`Installation Guide <install>`
       ```
    - In Markdown, write:
       ```
       {doc}`Installation Guide <install>`
       ```
- If you want to refer to a part of the HNN-Core API like this: {func}`~hnn_core.Network.add_electrode_array`, then:
    - In RST, write:
       ```
       :func:`~hnn_core.Network.add_electrode_array`
       ```
    - In Markdown, write:
       ```
       {func}`~hnn_core.Network.add_electrode_array`
       ```
- For convenience, to quickly insert a link to any specific GitHub issue at <https://github.com/jonescompneurolab/hnn-core>, like this: {gh}`705`, then:
    - In RST, write:
       ```
       :gh:`705`
       ```
    - In Markdown, write:
       ```
       {gh}`705`
       ```

## How to rebase

Commits in hnn-core follow a linear history, therefore we use a
"rebase" workflow instead of "merge" to resolve commits. See [this
article](https://www.atlassian.com/git/tutorials/merging-vs-rebasing)
for more details on the differences between these workflows.

To rebase, we do the following:

1.  Checkout the feature branch:

        $ git checkout cool_feature

2.  Delete the `master` branch and fetch a new copy:

        $ git branch -D master
        $ git fetch upstream master:master

3.  Start the rebase:

        $ git rebase master

4.  If there are conflicts, the easiest approach is to resolve them in
    an editor like VS code. See [this
    guide](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging)
    for more general information on resolve merge conflicts

5.  Once the conflicts have been resolved, add the resolved files to the
    staging area:

        $ git add -u
        $ git rebase --continue

In general it is best to rebase frequently if you are aware of pull
requests being merged into the `master` base.

If you face a lot of difficulting resolving merge conflicts, it may be
easier to
[squash](https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History)
before rebasing.

## Continuous Integration

The repository is tested via continuous integration with GitHub Actions
and Circle. The automated tests run on GitHub Actions while the
documentation is built on Circle.

To speed up the documentation-building process on CircleCI, we enabled
versioned [caching](https://circleci.com/docs/caching/).

Usually, you don't need to worry about it. But in case a complete
rebuild is necessary for a new version of the doc, you can modify the
content in `.circleci/build_cache`, as CircleCI uses the MD5 of that
file as the key for previously cached content. For consistency, we
recommend you to monotonically increase the version number in that file,
e.g., from "v2"->"v3".

## Notes on MPI for contributors

MPI parallelization with NEURON requires that the simulation be launched
with the `nrniv` binary from the command-line. The `mpiexec` command is
used to launch multiple `nrniv` processes which communicate via MPI.
This is done using `subprocess.Popen()` in `MPIBackend.simulate()` to
launch parallel child processes (`MPISimulation`) to carry out the
simulation. The communication sequence between `MPIBackend` and
`MPISimulation` is outlined below.

1.  In order to pass the network to simulate from `MPIBackend`, the
    child `MPISimulation` processes' `stdin` is used. The ready-to-use
    {class}`~hnn_core.Network` object is base64 encoded and pickled before
    being written to the child processes' `stdin` by way of a Queue in
    a non-blocking way. See how it is [used in MNE-Python][].
    The data is marked by start and end signals that are used to extract
    the pickled net object. After being unpickled, the parallel
    simulation begins.
2.  Output from the simulation (either to `stdout` or `stderr`) is
    communicated back to `MPIBackend`, where it will be printed to the
    console. Typical output at this point would be simulation progress
    messages as well as any MPI warnings/errors during the simulation.
3.  Once the simulation has completed, the rank 0 of the child process
    sends back the simulation data by base64 encoding and and pickling
    the data object. It also adds markings for the start and end of the
    encoded data, including the expected length of data (in bytes) in
    the end of data marking. Finally rank 0 writes the whole string with
    markings and encoded data to `stderr`.
4.  `MPIBackend` will look for these markings to know that data is being
    sent (and will not print this). It will verify the length of data it
    receives, printing a `UserWarning` if the data length received
    doesn't match the length part of the marking.
5.  To signal that the child process should terminate, `MPIBackend`
    sends a signal to the child proccesses' `stdin`. After sending the
    simulation data, rank 0 waits for this completion signal before
    continuing and letting all ranks of the MPI process exit
    successfully.
6.  At this point, `MPIBackend.simulate()` decodes and unpickles the
    data, populates the network's CellResponse object, and returns the
    simulation dipoles to the caller.

It is important that `flush()` is used whenever data is written to stdin
or stderr to ensure that the signal will immediately be available for
reading by the other side.

Tests for parallel backends utilize a special `@pytest.mark.incremental`
decorator (defined in `conftest.py`) that causes a test failure to skip
subsequent tests in the incremental block. For example, if a test
running a simple MPI simulation fails, subsequent tests that compare
simulation output between different backends will be skipped. These
types of failures will be marked as a failure in CI.

[used in MNE-Python]: https://github.com/mne-tools/mne-python/blob/148de1661d5e43cc88d62e27731ce44e78892951/mne/utils/misc.py#L124-L132

## Making changes to the default network

If you ever need to make scientific or technical changes to the default network, you must "re-generate" the smaller network we use for testing, in `hnn_core/tests/assets/jones2009_3x3_drives.json`. This is easily done via the following:

    $ make regenerate-test-network

Once you do this, make sure to re-run all the tests using `make test` to ensure that numerical tests dependent on the network itself have not broken.
