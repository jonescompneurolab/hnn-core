Contributions
-------------

Contributions are welcome in the form of pull requests.

Once the implementation of a piece of functionality is considered to be bug
free and properly documented (both API docs and an example script),
it can be incorporated into the master branch.

To help developing ``hnn-core``, you will need a few adjustments to your
installation as shown below.

If your contributions will make use of parallel backends for using more than
one core, please see the additional installation steps in our
:doc:`parallel backend guide <parallel>`.

Running tests
=============

To run the tests using ``pytest``, you need to have the git cloned ``hnn-core``
repository with an editable pip install::

    $ git clone https://github.com/jonescompneurolab/hnn-core --depth 1
    $ cd hnn-core
    $ pip install -e .
    $ python setup.py build_mod

Then, install the following python packages::

    $ pip install flake8 pytest pytest-cov

If you update a mod file, you will have to rebuild them using the command::

    $ python setup.py build_mod

MPI tests are skipped if the ``mpi4py`` module is not installed. This allows
testing features not related to parallelization without installing the extra
dependencies as described in our :doc:`parallel backend guide <parallel>`.

Updating documentation
======================

Update ``doc/api.rst`` and ``doc/whats_new.rst`` as appropriate.

Building the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

The documentation can be built using sphinx. For that, please additionally
install the following::

    $ pip install matplotlib sphinx numpydoc sphinx-gallery sphinx_bootstrap_theme pillow mpi4py joblib psutil nbsphinx

If you are using a newer version of pip, you may be prompted to use the flag
``--use-feature=2020-resolver``. If this happens, please add it as recommended::

    $ pip install --use-feature=2020-resolver matplotlib sphinx numpydoc sphinx-gallery sphinx_bootstrap_theme pillow mpi4py joblib psutil nbsphinx

You can build the documentation locally using the command::

    $ cd doc/
    $ make html

While MNE is not needed to install hnn-core, as a developer you will need to
install it to run all the examples successfully. Please find the installation
instructions on the `MNE website <https://mne.tools/stable/install/index.html>`_.

If you want to build the documentation locally without running all the examples,
use the command::

    $ make html-noplot

Continuous Integration
======================

The repository is tested via continuous integration with GitHub Actions and
Circle. The automated tests run on GitHub Actions while the documentation is
built on Circle.