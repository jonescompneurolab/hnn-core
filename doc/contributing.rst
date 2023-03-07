Contributions
-------------

Please read the contribution guide **until the end** before beginning
contributions.

Contributions are welcome in the form of pull requests.

Once the implementation of a piece of functionality is considered to be bug
free and properly documented (both API docs and an example script),
it can be incorporated into the master branch.

To help developing ``hnn-core``, you will need a few adjustments to your
installation as shown below.

If your contributions will make use of parallel backends for using more than
one core, please see the additional installation steps in our
:doc:`parallel backend guide <parallel>`.

Setting up your local development environment
=============================================

Configuring git
~~~~~~~~~~~~~~~

Instructions for how to configure git can be found on 
the git book `configuration <https://git-scm.com/book/en/v2/Customizing-Git-Git-Configuration>`_ page.

Making your first pull request
==============================
Changes to hnn-core are typically made by "forking" the hnn-core
repository, making changes to your fork (usually by "cloning" it to your
personal computer, making the changes locally, and then "pushing" the local
changes up to your fork on GitHub), and finally creating a "pull request" to incorporate
your changes back into the shared "upstream" version of the codebase.

In general you'll be working with three different copies of the hnn-core
codebase: the official remote copy at https://github.com/jonescompneurolab/hnn-core
(usually called ``upstream``), your remote fork of the upstream repository
(similar URL, but with your username in place of ``hnn-core``, and usually
called ``origin``), and the local copy of the codebase on your computer. The
typical contribution process is to

1. Make a fork of the `hnn-core <https://github.com/jonescompneurolab/hnn-core>`_
   repository to your own account on github. Look for the Fork button in the top right corner

2. On the terminal of your local computer clone the fork::

    $ git clone https://github.com/<username>/hnn-core

3. On the terminal of your local computer set up the remotes::

    $ cd hnn-core
    $ git remote add upstream https://github.com/jonescompneurolab/hnn-core

4. Check that the remotes have been correctly added::

    $ git remote -v

   You should see::

    origin	https://github.com/<username>/hnn-core (fetch)
    origin	https://github.com/<username>/hnn-core (push)
    upstream	https://github.com/jonescompneurolab/hnn-core (fetch)
    upstream	https://github.com/jonescompneurolab/hnn-core (push)

5. To start a new feature branch, we will copy the existing ``master`` branch from
   the ``upstream`` remote and give it a specific name::

    $ git fetch upstream master:cool_feature
    $ git checkout cool_feature

6. Make your changes relevant to the pull request

7. To make a commit, you first have to add the files you have changed to the staging area::

        $ git add -u

   ensure they have been added correctly::

        $ git status

   make a commit::

        $ git commit -m "your commit message"

   and finally check that the commit has been added::

        $ git log

   Note: see the `numpy contributing guide <https://numpy.org/doc/stable/dev/development_workflow.html#writing-the-commit-message>`_
   for tips on informative commit messages.

8. Now push the feature branch to your fork::

    $ git push origin cool_feature

9. Go to https://github.com/jonescompneurolab/hnn-core/compare to open the pull request by selecting your feature branch.
   You should also see this link on the terminal when you make the push. 

11. After your pull request is reviewed, repeat steps 6-8 to update the pull request.

12. Once the pull request is ready to be merged, add the prefix [MRG] to the title.

See the `git book <https://git-scm.com/book/en/v2>`_ for a more general guide on using git. 

Installing editable hnn-core
============================

For making changes to hnn-core, you will need to install an editable
version of hnn-core. For that you need to have the git cloned ``hnn-core``
repository and use pip with the editable (``-e``) flag::

    $ git clone https://github.com/jonescompneurolab/hnn-core
    $ cd hnn-core
    $ pip install -e '.[gui]'
    $ python setup.py build_mod

The last step builds ``mod`` files which specifies the dynamics of specific
cellular mechanisms. These are converted to C, and hence require a compilation
step. In the normal course of development, you will not have to edit these
files. However, if you do have to update them, they will need to be rebuilt
using the command::

    $ python setup.py build_mod

Running tests
=============

Once you have the editable hnn-core, you should install the requirements
for running the tests. Tests help ensure integrity of the package after
your change has been made. We recommend developers to run tests locally
on their computers after making changes.

We use the ``pytest`` testing framework.

Install the following python packages::

    $ pip install flake8 pytest pytest-cov

To run the tests simply type into your terminal::

    $ make test

MPI tests are skipped if the ``mpi4py`` module is not installed. We highly
encourage contributors to follow the :doc:`parallel backend guide <parallel>`
to install ``mpi4py`` so that they can run the entire test suite locally
on their computer.

Updating documentation
======================

When you update the documentation, it is recommended to build it locally to
check whether the documentation renders correctly in HTML.

Certain documentation files require explicit updates by the contributor of a given change (i.e., you). These are:

* ``doc/api.rst`` if you added a new function.
* ``doc/whats_new.rst`` to document the fix or change so you can be credited on the next release.

Please update these documents once your pull request is ready to merge to avoid rebase conflicts with other
pull requests.

Building the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

The documentation can be built using sphinx. For that, please additionally
install the following::

    $ pip install matplotlib sphinx numpydoc sphinx-gallery sphinx_bootstrap_theme sphinx-copybutton pillow joblib psutil nbsphinx

You can build the documentation locally using the command::

$ cd doc/
$ make html

While MNE is not needed to install hnn-core, as a developer you will need to
install it to run all the examples successfully. Please find the installation
instructions on the `MNE website <https://mne.tools/stable/install/index.html>`_.

If you want to build the documentation locally without running all the examples,
use the command::

    $ make html-noplot

Finally, to view the documentation, do::

    $ make view

How to rebase
=============

Commits in hnn-core follow a linear history, therefore we use a "rebase" workflow
instead of "merge" to resolve commits.
See `this article <https://www.atlassian.com/git/tutorials/merging-vs-rebasing>`_ for more details
on the differences between these workflows.

To rebase, we do the following:

1. Checkout the feature branch::

    $ git checkout cool_feature

2. Delete the ``master`` branch and fetch a new copy::

    $ git branch -D master
    $ git fetch upstream master:master

3. Start the rebase::

    $ git rebase master

4. If there are conflicts, the easiest approach is to resolve them in an editor
   like VS code.
   See `this guide <https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging>`_
   for more general information on resolve merge conflicts

5. Once the conflicts have been resolved, add the resolved files to the staging area::

    $ git add -u
    $ git rebase --continue

In general it is best to rebase frequently if you are aware of pull requests being merged
into the ``master`` base.

If you face a lot of difficulting resolving merge conflicts,
it may be easier to `squash <https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History>`_
before rebasing.

Continuous Integration
======================

The repository is tested via continuous integration with GitHub Actions and
Circle. The automated tests run on GitHub Actions while the documentation is
built on Circle.

To speed up the documentation-building process on CircleCI, we enabled versioned 
`caching <https://circleci.com/docs/caching/>`_.

Usually, you don't need to worry about it. But in case a complete rebuild is necessary 
for a new version of the doc, you can modify the content in ``.circleci/build_cache``, as 
CircleCI uses the MD5 of that file as the key for previously cached content.
For consistency, we recommend you to monotonically increase the version number
in that file, e.g., from "v2"-> "v3".
