---
orphan: true
---

(how_to_create_releases)=
# How to create releases

Last updated: 2025-06-26

## 0. Notes before starting

- We use at least 2 types of releases:
    1. "stable" releases (aka "full" version releases), which are updates to the *default* version that is publicly distributed. I.e. these are what users will install if they run `pip install hnn-core`, etc.
    2. "dev" releases, or development releases. These can be incremented as desired, but are mainly used to distinguish new work that is merged after the latest stable release. After any stable release, there should immediately be a follow-up PR (or commit) that begins creation of the next dev release. Dev releases *can* be pushed to Pypi as long as they are indicated as development versions. However, we typically only need to do all of the steps on this page for a stable release, not every dev release.
    3. (optional) "rc" releases, or "release candidates". If desired, a dev version can be changed to an rc version if we want to distribute specific rc versions for "beta-testing" for an soon and upcoming stable release.

- Version identifiers:
    - We try our best to follow a Python-compatible "Semantic versioning" style as [described briefly here](https://packaging.python.org/en/latest/discussions/versioning/) and in [much greater detail here](https://packaging.python.org/en/latest/specifications/version-specifiers/#version-specifiers).
    - Stable release version identifiers should therefore be of the `<major>.<minor>.<patch>` format. For example, `0.4.1`.
    - Dev release version identifiers should be of the `<major>.<minor>.<patch>.dev<dev-version>` format. For example, `0.4.2dev0`.
    - (optional) "RC" or release candidate version identifiers should be of the `<major>.<minor>.<patch>.rc<rc-version>` format. For example, `0.4.2rc3`.
    - For what defines major vs minor vs patch, the following is copied verbatim [from here](https://packaging.python.org/en/latest/discussions/versioning/):

>    *major* when they make incompatible API changes,
>
>    *minor* when they add functionality in a backwards-compatible manner, and
>
>    *patch*, when they make backwards-compatible bug fixes.

- However, in accordance with [semantic versioning](https://semver.org/#semantic-versioning-specification-semver), since HNN-Core has not yet reached 1.0, our `patch` and `minor` releases do NOT need to be backwards-compatible yet.

- *Pay careful attention* to the use of `v` in front of the version identifier. Some commands and config values need e.g. `v0.4`, while others should *not* include the `v`, as in `0.4`.

- Unless said otherwise, these instructions assume you are in the "root" (aka top-level) directory of the `hnn-core` code repository.

- Once you push a new version to TestPypi or Pypi, we CANNOT re-upload a different package to that same version. That version will always, *permanently*, be attached to that specific package file. This is a security limitation of Pypi and cannot be bypassed. If you make a mistake and upload a broken package however, there is still a solution: uploading a newer version with a higher version number. You can always increment the `patch` part of the version, then follow the process again to have a new, fixed version pushed to both TestPypi and Pypi.

- Due to the rule we use to detect when to auto-publish packages ([see here](https://github.com/jonescompneurolab/hnn-core/blob/843ed4aaed5a09d46c49ce34b5b620ec112d4e5a/.github/workflows/publish-packages-on-tag.yml#L10)) *any* tag pushed to `upstream` that begins with `v` will cause the workflow to run. This means that any new version tag pushed to upstream will cause a package to be built, and the version of the code inside that commit (in `hnn_core/__init__.py`, NOT the tag itself!) will be used when uploading the package to Pypi. If that is a valid version that has not been used before, Pypi will accept and publish that package according to the version in question. Because versions are immutable on Pypi, this means that that version will NOT be available again! So, double- and triple-check before you push a tag!

- The original "How to make a release" is [available here](https://github.com/jonescompneurolab/hnn-core/wiki/How-to-make-a-release), however it should be considered deprecated. It leaves out a few important steps (or suggests doing them out-of-order), and does not account for our new automatic Github-tag publishing workflow.

## 1. Before making a release

- Make sure all milestones are achieved or migrated to the next version.
- Make sure the CI passes on master.
- Make sure that for both TestPypi and Pypi, there are NOT already existing releases for the new version that you want to push. You can view [TestPypi releases here](https://test.pypi.org/project/hnn-core/#history) and [Pypi releases here](https://pypi.org/project/hnn-core/#history).

## 2. Prepare your local version of the release

1. Create and checkout a new local branch in your local `hnn-core` repository directory.
2. Edit `doc/whats_new.md` to have a new section for the new version, including all development work done since the last release.
     - If doing a stable release, we also add an authors section (this is not present for the "current" section).
     - Run `git shortlog v0.4.1..HEAD --summary --numbered` (replace the version tag to the most recent version) to find contributor names for this release.
     - Manually add authors who have done a lot of work (reviewing, advising, etc.) but don't show up in the shortlog because they did not submit PRs.
     - Then order the names alphabetically and add to the last section of the new version's changelog.
     - Feel free to change the sections of the release notes, but always make sure to include at minimum a brief summary, deprecations, and public API changes.
3. Increment `__version__` in `hnn_core/__init__.py`.
4. Increment `version` and `release` in `doc/conf.py`. It is safe to update both.
5. ONLY if you're doing a stable release: Update `doc/_static/versions.json` which handles versions of the documentation on the code-website:
    1. Add a new entry with the new version identifier to the list.
    2. For the entry with the name `Stable <current stable version>`, update the `name` (but NOT the `version`) so that Stable refers to the new version.

## 3. Test your local version of the release

1. Create a NEW virtual environment, however you prefer.

2. Perform the necessary steps to install MPI inside your environment, but BEFORE you install HNN-Core, by following [this part of the install guide](https://jonescompneurolab.github.io/hnn-core/stable/install.html#pip-mpi-macos-or-linux).

3. Begin by installing ONLY the most basic version of HNN-Core, with NO extra features, using:
```
pip install -e "."
```

4. Test that it can successfully run a simulation using the following:
```
python -c "from hnn_core import jones_2009_model, simulate_dipole ; simulate_dipole(jones_2009_model(), tstop=20)"
```
Simply testing the import with `python -c "import hnn_core"` is NOT enough! You must test an actual simulation run. If you encounter issues with your compiled MOD files, then run `make clean` to delete the existing ones, then try running a simulation again. If you continue to have issues, then halt the release and investigate!

5. Next, install the local version of HNN-Core again, but this time with all development features, using:
```
pip install -e ".[dev]"
```

6. Run all tests by using:
```
make test
```
Note that this will download and create some new files into the current directory. These new files should NOT be included in any future commits, especially those for a release. Obviously, if you run into errors, then halt the release and investigate.

7. Go into your `doc` directory, clean your local copy of the docs, then run a full build of the documentation. You can use the following to do this:
```
cd doc
make clean
make html
```
This can take a while depending on your computer speed. The build process should produce warnings but no errors. If it reports errors, investigate them.

8. Manually inspect the newly-built documentation pages to make sure there are no issues. You can do this by opening the file located at `doc/_build/html/index.html` in your web browser, and then navigating from there.

9. Make a copy of the newly built documentation; we will be using it later. Assuming you are still in the `doc` directory, you can do this easily using:
```
mkdir -p ~/new-docs
cp -r _build/html/* ~/new-docs
```

10. Change directory back up to the "root" directory of your repository.

11. Next, install the dependencies we need for building a local version of our package using:
```
pip install -U setuptools twine
```

12. Build our package locally using the following:
```
python setup.py sdist
```
This will create a "source tarball" package file of the new version in a new directory at `dist/hnn_core-<version>.tar.gz`. Note that we *do not build wheels*, we only build source tarballs.

13. Check some metadata of our package, using:
```
twine check dist/*
```
After this step, you can delete the package file inside `dist` or the `dist` directory itself, we will not be using that package file.

## 4. Create your release PR

1. Use `git add` to add ONLY the following files to your next commit:
    - `hnn_core/__init__.py`
    - `doc/conf.py`
    - `doc/whats_new.md`
    - `doc/_static/versions.json` (ONLY if you are pushing a new stable release)

You can do this easily with:
```
git add hnn_core/__init__.py doc/conf.py doc/whats_new.md doc/_static/versions.json
```
2. Make your commit, but do NOT tag it yet.
3. Push your commit to your branch.
4. On Github's website, make your PR.
5. Wait for all CI to pass successfully on your PR.
6. Finally, when ready to merge to `master`, use "SQUASH and merge" to merge the PR as a single commit. We will refer to the commit you just pushed to `master` as the "release commit".
7. Note: If you have "cyclical documentation dependencies", such as if you reference a new web-page on the `stable` version of the docs, but that page doesn't exist yet in the `stable` doc version, then that is fine. You can do the following:
    1. create this PR but don't merge it yet,
    2. follow the next section ("5. Update the documentation") to update the stable docs, then
    3. come back and re-run the `linkcheck` workflow, and finally
    4. merge the PR.
    5. Alternatively, you can merge your PR before you have gone to the next step while `linkcheck` is failing, but this is not recommended.

## 5. Update the documentation

1. This series of steps is more easily done as commits pushed directly onto `upstream`, instead of being merged through PRs.
2. Switch locally to the `gh-pages` branch, and make sure your branch is up to date.
3. Delete the existing files in the `stable` directory.
4. Take all the documentation files you previously saved (e.g. into `~/new-docs`, see previous steps about building the documentation), and copy them into the `stable` directory. You could do this with:
```
cp -r ~/new-docs/* stable
```
5. Create a new folder with the version identifier, and copy the *same* files into that directory *too*.
6. You should have two directories with identical files in them: `stable` and `v<version identifier>`. This step is necessary for the version to be indicated in the dropdown on the code-website.
7. Create a new git commit with ALL these changes and push it to `upstream/gh-pages`. You can instead do this through a PR, but you must make sure that the PR is for merging into the `gh-pages` branch of `jonescompneurolab/hnn-core`, NOT the `master` branch!

## 6. Push a git tag to build the release

1. Switch back to the `master` branch.
2. Update your `master` branch such that it is in sync with `upstream`, and that its current commit is the "release commit" you previously merged through your PR.
3. Create a tag for this release commit, where the tag is `v<new version>`, including the `v`. For example, if the new version that you just updated to is `0.4.4`, then you could do the following to create the tag locally:
```
git tag v0.4.4
```
4. Push the tag to `upstream`. Continuing the example from before, you could do this with:
```
git push upstream v0.4.4
```
5. Congrats! Github will now automatically begin building the package directly from the commit using a "Publish (etc.)" workflow (you can watch it in our [Actions here](https://github.com/jonescompneurolab/hnn-core/actions)). Once Github has built the package file, it will automatically publish that package to both Pypi and TestPypi.
    - The workflow code that Github uses to build and publish the packages is [located here](https://github.com/jonescompneurolab/hnn-core/blob/master/.github/workflows/publish-packages-on-tag.yml).
    - Note that if you ever change the filename of the workflow, you **must** go to TestPypi and Pypi and add a new "Publisher" to the `hnn-core` project. The new "Publisher" must use the new filename, in addition to other metadata. You can see the Pypi and TestPypi-specific [instructions here](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/#configuring-trusted-publishing). Note that to add a new "Publisher" to the `hnn-core` project in the first place, you must have the necessary permissions. Ask Austin or Dylan if you need to upgrade your permissions.
    - Note that the publishing workflow uses the version exactly from `hnn_core/__init__.py`, NOT from the git tag itself. This is why it is important to double check that your tag numbers are consistent. It will not detect if there is a tag version mismatch.
6. Assuming nothing went wrong with the Github Actions "Publish (etc.)" workflow, your new version should now be live on both [Pypi](https://pypi.org/project/hnn-core/#history) and [TestPypi](https://test.pypi.org/project/hnn-core/#history).

## 7. Post-publishing tests

1. Create yet another fresh Python environment.
2. Move somewhere outside of your `hnn-core` code repository, such as to your `~/Desktop`. We want to test that our new package installs and works correctly without relying on any code on your hard drive.
3. Wait 10 minutes for Pypi's stores to start distributing your new release (go grab a coffee). It may be ready instantly, or not. Maybe ponder how you can improve this package release process...
4. Install the basic version of `hnn-core` from Pypi with:
```
pip install hnn-core
```
Make sure that you check that it's the latest version that you just uploaded. If there are any issues during during installation, recoil in horror and start sweating.

5. Similar to before, test that a basic simulation runs, such as using:
```
python -c "from hnn_core import jones_2009_model, simulate_dipole ; simulate_dipole(jones_2009_model(), tstop=20)"
```
6. You can also test the TestPypi version in yet another environment if you want, but you should note that the command shown at the top of the [TestPypi page](https://test.pypi.org/project/hnn-core/#history) will NOT work, due to the `-i` argument. (The reason for this is that TestPypi does not actually provide the `setuptools` package that we need, [see here](https://stackoverflow.com/a/77948986)). If you want to install the TestPypi version in a way that actually works, you should do the following:
```
pip install --extra-index-url https://test.pypi.org/simple/ "hnn-core"
```
7. Finally, manually download a copy of your new package directly from <https://pypi.org/project/hnn-core/#files>, then save it somewhere like your Downloads.

## 8. More Github release steps

Only do these steps if you have made a stable release.

1. In your terminal, make a new "maintenance branch" by doing the following:
     1. Return to your `hnn-code` repository on the `master` branch in your terminal.
     2. Create a new branch matching `maint/<new version>` for the new version, beginning at the release commit.
     3. Push that new branch to `upstream`.
2. In your browser, make an official "Github Release":
     1. Go to <https://github.com/jonescompneurolab/hnn-core/releases> and click "Draft a new release".
     2. Copy and re-format the new version's release notes (which you added in `doc/whats_new.md` previously) into the release notes for this Github release. Doing this in a text editor with "find-replace" is recommended.
     3. Where it says "Attach binaries...", upload the package file you manually downloaded from <https://pypi.org/project/hnn-core/#files>.
     4. Make sure `Set as the latest release` is checked if you are indeed pushing the latest release that people should be using.
     5. Publish the release.

## 9. Increment version again, to a development version

1. Return to your `hnn-code` repository on the `master` branch in your terminal.
2. Start to realize that this is a lot of steps. Maybe too many.
3. Begin incrementing the version *again* to the *next development version*. You can do this via a personal branch then PR, or just do it as a single commit that you will later push to `upstream/master`.
4. View the [TestPypi page here](https://test.pypi.org/project/hnn-core/#history) and [Pypi here](https://pypi.org/project/hnn-core/#history), and check to make sure that the next development version you want to use has not already been taken by an existing package. It is possible for this to happen, due to the necessity to debug our package building pipeline, etc. Increment your next development version so it's higher. For example, if you just released version `0.4.4` but there is a pre-existing package for version `0.4.5dev1`, then you should prepare to increment the next development version to `0.4.5dev2`.
5. Update `hnn_core/__init__.py` and `doc/conf.py` to use the next development version. You do NOT need to make any changes to `doc/_static/versions.json`.
6. Add a new section to the top of `doc/whats_new.md` that says "Current", pushing the most recent version release notes downwards.
7. Git add these three files, such as using:
```
git add hnn_core/__init__.py doc/conf.py doc/whats_new.md
```
8. Make a commit, then push. You can push this to `upstream/master` directly. It is *not* recommended that you push a version tag for this commit, *unless* you need this development version to be available from `pip`. (If you for some reason need to be publishing a development version as a package available from pip, then you may be better off having that user install `hnn-core` from source instead).

## 10. Build and distribute Conda packages utilizing the new version

1. We're done....with Github and Pypi that is! Oh, you thought that was all? Nope. Now that our new version has been released on Pypi, we need to **use the new Pypi package to rebuild new Conda packages for the new version too**. Do not fret however! This part may seem intimidating, but you don't need to do that many steps.
2. Elsewhere on your hard drive, clone our repo here <https://github.com/jonescompneurolab/hnn-core-conda-packaging>. We will NOT be using your local copy of `hnn-core`'s source code for this, but instead be using the Pypi package directly.
3. Read through the that repo's ["How to use this repo to build and upload the packages" section](https://github.com/jonescompneurolab/hnn-core-conda-packaging?tab=readme-ov-file#how-to-use-this-repo-to-build-and-upload-the-packages). You do *not* need to read the entire README.
4. Follow the instructions until you are told to run `00-install-build-deps.sh` to install the conda building dependencies. Execute that script.
5. Let's start with the more complex conda package. `cd` into `hnn-core-all/recipe`.
6. Open the file `meta.yaml` (aka `hnn-core-all/recipe/meta.yaml`). You need to do the following:
     1. Increment the `version` variable at the top to the latest version on Pypi.
     2. Go to <https://pypi.org/project/hnn-core/#files>, and click on `view details`.
     3. Copy the SHA256 Hash digest.
     4. Paste it into the value for `sha256` inside `meta.yaml`.
7. If NEURON has released a new stable version (currently, newer than 8.2.7), then you also need to edit `hnn-core-all/recipe/build.sh`. Open that file.
     1. Inside that file, identify the places where `NRN_WHL_URL` values are set to URLs that provide the NEURON wheel files.
     2. Go to <https://pypi.org/project/NEURON/#files>.
     3. For the version of Python we support in our conda packages (currently 3.12 only), you must copy the *full URL link* of the relevant NEURON wheel file, and then replace its existing value in `build.sh`. The easiest way to do this is right-click on the relevant `.whl` file link and click `Copy Link`. You will know you're on the right track when the URL is long and includes a complete hash of the file, for example like: <https://files.pythonhosted.org/packages/3e/69/5a8d498fd3096726768ab875b0e9a633cfdb68f976a6d520b6158b07ed7c/neuron-8.2.7-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl>.
     4. Yes this is super annoying, but it is required for reasons described in our Conda package building repo's README.
8. Once you have updated `hnn-core-all/recipe/meta.yaml` and `hnn-core-all/recipe/build.sh`, do the same for the other conda package, `hnn-core`. You should be able to make the exact same changes in the same places to `hnn-core/recipe/meta.yaml` and `hnn-core/recipe/build.sh`.
9. Before trying to build the packages, save your work, and make a commit on your own branch of the repo or something.
10. From here, you can continue where you left off from ["How to use this repo to build and upload the packages" section](https://github.com/jonescompneurolab/hnn-core-conda-packaging?tab=readme-ov-file#how-to-use-this-repo-to-build-and-upload-the-packages), which should be around step 5. Details are in that repo.
11. To summarize the rest of the instructions from that repo, assuming you have no problems, the rest of the workflow will look like this:
      1. You `cd` into each Conda package's directory.
      2. Execute `01-build-pkg.sh`, which takes a few minutes to build the package.
      3. Assuming that succeeded, you test that you can install and run the package. As always, make sure you do this from somewhere that is NOT inside your main `hnn-core` code repository, otherwise Python will sometimes use your local code instead of the package you have installed in your environment.
      4. Satisfied, you upload the package.
      5. Repeat for the other package.
12. Do something to celebrate, you're done!
