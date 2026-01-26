(contributing)=
# Contributing Guide

Please read the contribution guide **until the end** before beginning contributions.

Contributions are welcome in the form of Pull Requests. You must abide by our [Code of
Conduct, found
here](https://github.com/jonescompneurolab/hnn-core/blob/master/CODE_OF_CONDUCT.md), which includes instructions on what kind of AI- and LLM-generated code contributions we accept. Our
{doc}`Governance Model can be found here <governance>`. Please tie your Pull Requests to specific Issues.

Once the implementation of a piece of functionality is considered to be bug free and
properly documented (both API docs and an example script), it can be incorporated into
the `master` branch, which is where our releases come from.

To contribute to `hnn-core` development, you need a special kind of installation, see the
["`pip` Source Installation" section of our Installation Guide][] on our [Textbook website][]. Note that this is **different** from the "`pip` Package Installation" type on that webpage!

We are experimenting with having once-monthly HNN Development meetings which are open to the public. These take place on the first Monday of every month at 1:00 PM, United States Eastern time (the [timezone of Providence, RI, viewable here](https://www.timeanddate.com/worldclock/usa/providence)). You can access the Zoom room here [https://brown.zoom.us/j/99212200748](https://brown.zoom.us/j/99212200748). Note that there is a waiting room and you *must raise your hand* before you will be granted audio privileges. We reserve the right to kick out anyone causing disruptions during the meeting.

## How to contribute code

We use the ["fork and pull model" on
Github](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/getting-started/about-collaborative-development-models)
to incorporate changes onto our `master` branch. We want commits in `hnn-core`
to follow a linear history, therefore we use a "rebase" workflow instead of "merge
commits" to incorporate work. See [this
article](https://www.atlassian.com/git/tutorials/merging-vs-rebasing) for more details
on the differences between these workflows. Importantly, we also usually "squash and
rebase" so that each Pull Request is tied to a single commit immediately before it is merged.

If these terms are unfamiliar, or you are new to open-source development using
[`git`](https://git-scm.com/) and [Github][], don't worry. A complete guide to using
both [the `git` program](https://git-scm.com/) and [the Github
website](https://github.com) is outside the scope of our documentation, but we strive to
provide you with the minimum commands necessary to get everything working.

Fortunately, there is a large amount of helpful guidance online for learning how to use
`git` and, through it, Github:

- <https://github.com/git-guides> (recommended for beginners)
- <https://docs.github.com/en/get-started> (recommended for beginners)
- <https://swcarpentry.github.io/git-novice/>
- <https://git-scm.com/doc/ext> (often more technical, but also more informative)

Finally, whenever you make a Pull Request, we recommend that you add mention of your
contribution to `doc/whats_new.md` so that you can can publicly receive credit.

### Making your first Pull Request, including *installing for development*

1. Go to <https://github.com> and create an account. For the sake of an example, let's
   pretend your new account is <https://github.com/asoplata> and your username is
   `asoplata`. (For all subsequent steps, you should replace where it says `asoplata`
   with your username.)

2. On the Github webpage for the [`hnn-core`
   repository](https://github.com/jonescompneurolab/hnn-core), create a new Fork. Look
   for the Fork button in the top right corner. Click that, and make a new Fork under
   your new Github account.

3. On your computer, [setup `git` using the instructions
   here](https://docs.github.com/en/get-started/git-basics/set-up-git). Note that this
   involves two steps:

    A. Install the `git` program [using these
    steps](https://docs.github.com/en/get-started/git-basics/set-up-git#setting-up-git).

    B. Configure `git` to know who you are and to **authenticate** with Github [using
    these
    steps](https://docs.github.com/en/get-started/git-basics/set-up-git#authenticating-with-github-from-git).

4. On your computer, open a "terminal" or "command line" program:
    - On Linux, you probably have a program named "Terminal Emulator" or similar.
    - On MacOS, you can use the program called Terminal.
    - On Windows (native), you can use the program called "Git Bash", which was
      installed when you installed `git` above.
    - On Windows (Windows Subsystem for Linux), you can use the WSL app that you
      installed, probably called "Ubuntu".

5. "Clone" your new fork. You can do this using the following command, but replacing
   `asoplata` with your username:

    ```
    git clone https://github.com/asoplata/hnn-core
    ```
6. You have just downloaded ("cloned") the code of your fork (the "repository" or
   "repo") into a new directory. Enter that directory using the following command:

    ```
    cd hnn-core
    ```

7. Next, let's look at our
   ["remotes"](https://git-scm.com/book/en/v2/Git-Basics-Working-with-Remotes). You can
   loosely think of remotes as "my backup copy of the versions of the code stored on
   Github's servers". To view your remotes, run the following command:

    ```
    git remote -v
    ```

    You should see:

    ```
    origin      https://github.com/asoplata/hnn-core (fetch)
    origin      https://github.com/asoplata/hnn-core (push)
    ```

    Your `origin` remote is essentially your local "cloud backup" of your fork sitting
    on Github's servers. This is the same remote that you used to "clone" your fork repo
    from. This is the remote that you will primarily "`push`" (a.k.a. upload)
    your code "`commits`" (a.k.a. changes or versions) to.

8. Add a new "remote" that points to the version of the code that you created your fork
   *from*. This remote (and the code version) is commonly called the "`upstream`". This is
   the remote you will primarily be using when you want to "`fetch`" (a.k.a. download) new
   commits that other people have made to the "`upstream`" (a.k.a. original) version of the
   code. To be explicit, this is so you can manage changes that other people make to the
   "upstream" code at <https://github.com/jonescompneurolab/hnn-core>, including
   eventually integrating those changes to both your local git repo and your fork at
   <https://github.com/asoplata/hnn-core>. Run the following command:

    ```
    git remote add upstream https://github.com/jonescompneurolab/hnn-core
    ```

9. Check that the remotes have been correctly added:

    ```
    git remote -v
    ```

   You should now see:

    ```
    origin      https://github.com/asoplata/hnn-core (fetch)
    origin      https://github.com/asoplata/hnn-core (push)
    upstream    https://github.com/jonescompneurolab/hnn-core (fetch)
    upstream    https://github.com/jonescompneurolab/hnn-core (push)
    ```

10. You should **never** write code directly to the `master` branch (this is important
    for the [next section on
    Rebasing](#keeping-your-code-up-to-date-by-rebasing)). Instead, all of your new work
    should be organized on new, separate "feature branches", which themselves are built
    off of the `master` version of the code. To start a new feature branch, we will copy
    the existing `master` branch from the `upstream` remote and give it a specific
    name. [See here for a guide on working with different
    branches](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging). You
    can use the following commands to do this, which will create and "`checkout`" (a.k.a.
    switch to) a feature branch called `cool-feature`:

    ```
    git fetch upstream master:cool-feature
    git checkout cool-feature
    ```

11. If you have not already done so, now is the time to install HNN. However, **you need
    to install using the ["`pip` Source Installation" section of our Installation
    Guide][]. You should read the instructions carefully**!

    - Note: You do **not** need to run the `git clone` command listed in the Install
    guide, since you already did this above in step 5.

    - The fact that you included the `--editable` flag means that your fork's code has
    been installed to your Python environment in a live, "editable" state. What this
    means that if you make a code change to a file, save that file, then start a new
    Python process, your code changes will *immediately* take effect. This is true even
    if you switch between `git` branches or commits. In other words, you do *not* need
    to reinstall using `pip` when you change versions of your code. The only exceptions
    to this are if you're changing the NEURON MOD files, our package dependencies, our
    package's "entry points", or if your installation had an error.

    - It is recommended, but not required, that you follow the additional instructions
    required for MPI installation, if you are on a supported platform (i.e. MacOS,
    Linux, and Windows using "Windows Subsystem for Linux").

12. Make sure you test that `hnn-core` has been installed correctly by following the
    section "Testing Your Installation" at the bottom of our [Installation Guide][]. If
    these commands fail, then stop and reach out to us for installation help at [our
    Github Discussions page][]. Installing HNN for development is the most complex
    install, but we should be able to help you solve your install issues.

13. Once your installation has been successful, it's time to start coding! Make your
    code changes and ["add"
    them](https://git-scm.com/book/en/v2/Git-Basics-Recording-Changes-to-the-Repository)
    (for example, using `git add -u`), then [create
    "commits"](https://git-scm.com/book/en/v2/Git-Basics-Recording-Changes-to-the-Repository)
    on your branch (for example, using `git commit -m "your commit message"`). Helpful
    tips:
     - If you are very new to using git, then before you run any git commands that you
       don't understand, it is recommended to make a backup copy of the directory
       containing your code somewhere else on your computer, just in case you mess
       something up. Unfortunately, it can be easy to get your git repository into a
       complex state (e.g. "merge conflicts") that makes it difficult to undo.
     - `git` is extremely powerful and has *many* tools to make your life easier. In
       particular, [`git stash`](https://git-scm.com/docs/git-stash), `git status` and
       `git log` are your friends!
     - Please try to follow the [Numpy contributing
       guide](https://numpy.org/doc/stable/dev/development_workflow.html#writing-the-commit-message)
       for commit messages, as this **greatly** helps us understand your changes. This
       includes things like beginning each commit with an acronym to indicate what kind
       of work it involves (see the link for a list). This also includes trying to keep
       the "title" (a.k.a. first line) of the commit message under 51 characters, and the
       "body" (subsequent lines) of the commit message under 72 characters.
     - In general, "more frequent commits with fewer code changes inside each commit" is preferable
       to "fewer commits with more code changes in each commit."
     - You can either use the command line to do these `git add` and `git commit`
       commands, or you can use plugins and features available in your code editor like
       [VS Code](https://code.visualstudio.com/download) that will do the same
       thing. Both ways are good. Code editors often have more visualization to show you
       changes, but it can also be helpful to use the command line to know **exactly**
       which commands do which things.

14. Test your code: go to the "top-level" directory of your repository (i.e. `cd` into
    the folder that has the file `setup.py` in it), then run the following command:

    ```
    make test
    ```

    This will perform all the checks and run all the tests described below in [Quality
    control](#quality-control). If there are any errors or failures, try to fix them
    before the next step. You will likely need to run `make format-overwrite` at some
    point; see [Formatting](#formatting) for more details. If you have tried your best
    but you don't know how to resolve your errors, that is fine, you can proceed to the
    next step.

15. Once your feature branch is ready for other developers to look at it, you need to
    "push" (a.k.a. upload) your feature branch to Github's copy of your fork (i.e. your
    `origin` remote):

    ```
    git push origin cool-feature
    ```

16. Now, if you go to the webpage for your fork
    (e.g. <https://github.com/asoplata/hnn-core> ) and click `master` on the left, you
    should see that your new branch is now viewable. Github is aware of your branch, but
    there is no Pull Request yet.

17. Let's finally create the Pull Request: go to
    <https://github.com/jonescompneurolab/hnn-core/compare>, then click "compare across
    forks". Don't change the `base repository` dropdown button (it should say
    `jonescompneurolab/hnn-core`) and don't change `base` (it should say
    `master`). Instead, click on the `head repository` dropdown button and select your
    fork. Then, click on the `compare` dropdown and select your new feature
    branch. Finally, click the green `Create Pull Request` to make it!
    - If your feature branch is "complete" and you are happy with it, add `[MRG]` to the
      beginning of your Pull Request's title to indicate that it is ready for merge into
      `master`.
    - If your feature branch is incomplete, add `[WIP]` to the beginning of your Pull
      Request's title to indicate that it is a Work In Progress.
    - There are also other, more convenient ways to create the Pull Request, but it must
      always be done on the website. For example, if you have pushed a feature branch
      recently, if you navigate to the webpage for your fork, there will often be a
      highlighted box at the top that says something similar to "Compare & Pull Request"
      which you can click. When you push, your terminal may also print a link that you
      can click on to quickly bring you to the Pull Request page.

18. After your Pull Request is reviewed, repeat steps 13, 14, and 15 based on changes
    that the developers request. The webpage for the Pull Request is automatically
    updated whenever you push new commits, so you only need to make new commits and then
    `push` them for everyone else to see the updates.

19. Once your Pull Request is accepted and merged, congratulations! However, your work
    is not yet over...keep reading below...

### Keeping your code up-to-date by rebasing

Imagine the following scenario: you just successfully got your Pull Request for
`cool-feature` merged into `master`, and you start work on a second feature called
`cool-feature-2`. However, while you're working on `cool-feature-2`, someone *else* got
their own Pull Request, called `other-persons-feature`, also merged into `master`. Now,
from your fork's perspective, both `master` and `cool-feature-2` are "out of date" with
respect to the `upstream` remote (a.k.a. the `master` version of the code at
<https://github.com/jonescompneurolab/hnn-core> ). What do you do? Use the magic of `git
rebase` to rewrite history! Great Scott!

Usually,
["merging"](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging)
means making *new* commits for the purpose of making two different branches compatible
with each other. In contrast,
["rebasing"](https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History) can be thought
of as *editing previous, existing commits* in order to make two different branches
compatible. We do **not** recommend using merging or "merge commits" to bring changes
from `master` into your feature branch, since using "merge commits" makes the git
history *significantly* more difficult to manage and rebase for us! (Note that "*merge
conflicts*" and "*merge commits*" are very different things.)

Rebasing can be a **destructive** operation, meaning that if you make a mistake, you can
**accidentally erase code that you didn't intend to!** It is therefore important to make
backups of your repository before you start doing it if you are unsure, and it is
important to understand what's going on. However, rebasing is extremely powerful and
makes for much cleaner commit histories.

Here are some useful guides that discuss "rebasing", including how it is different than
"merging" code:
- <https://www.atlassian.com/git/tutorials/merging-vs-rebasing>
- <https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History> (strongly recommended
  reading for understanding rebasing)
- <https://docs.github.com/en/get-started/using-git/about-git-rebase>
- <https://docs.github.com/en/get-started/using-git/resolving-merge-conflicts-after-a-git-rebase>

To rebase, we're going to download the latest commits from `upstream` remote's `master`
branch (also called `upstream/master`). Then we're going to rebase both our local and
`origin` remote's `master` branch (also called `origin/master`) onto
`upstream/master`. Then, finally, we're going to rebase our local `cool-feature-2`
branch onto our newly-updated `master` branch, bringing everything up to date.

1. First, make a **backup copy** of your code repository somewhere else on your
   computer. Until you are very confident with rebasing and altering `git` history, you
   should always do this! If you get your `git` repository into a very confusing state
   and want to start over, you can delete the repository and copy over from your backup
   copy (this only helps with your local files, however, and not the state of your
   `origin` remote). That said, we'll see later you **can** abort a rebase while it's in
   progress.

2. As a reminder, let's view our `git remotes`:

    ```
    git remote -v
    ```

   You should see:

    ```
    origin      https://github.com/asoplata/hnn-core (fetch)
    origin      https://github.com/asoplata/hnn-core (push)
    upstream    https://github.com/jonescompneurolab/hnn-core (fetch)
    upstream    https://github.com/jonescompneurolab/hnn-core (push)
    ```

    In the current situation, the `master` branch of our `upstream` remote has new work
    on it that we want to incorporate. This new work is not present on our local
    `master` branch, the `master` branch of our `origin` remote (i.e. Github's copy of
    our fork), nor on our local `cool-feature-2` branch.

3. We need to download those new commits from `upstream`. We can do this using the
   following command:

    ```
    git fetch upstream master
    ```

    This downloads the latest commits to `upstream`'s `master` branch (a.k.a.
    `upstream/master`), but **does not change** any other branches, including our copy
    of `origin`'s `master` branch (a.k.a. `origin/master`) or our local
    `cool-feature-2`. This only downloads the commits "in the background", so to speak.

4. Let's checkout our local `master` branch:

    ```
    git checkout master
    ```

5. Now that we're on our local `master` branch, we need to rebase this branch "onto"
   `upstream`'s `master` branch (also called `upstream/master`). We can do this with the
   following command:

    ```
    git rebase upstream/master
    ```

    This should work automatically, without any other changes needed. If there are
    problems, then see [Troubleshooting `git` problems](#troubleshooting-git-problems)
    below. You have probably added additional, local commits onto your local `master`
    branch, which you shouldn't do in the future.

6. Now that our local `master` is fully synchronized with the latest `upstream` commits,
   we also need to update `origin`'s `master` branch on Github's servers. This is easily
   doable with a typical push command:

    ```
    git push origin master
    ```

7. At this point, all of the `master` branches (local, `origin`, and `upstream`) are now
   successfully synchronized with `upstream/master`. That's the easy part, however. The
   next, second rebase often (but not always) requires some real work. Let's checkout
   our `cool-feature-2` branch:

    ```
    git checkout cool-feature-2
    ```

8. Next, we want to rebase our new commits in `cool-feature-2` "onto" the latest commits
   in `master`. At this point, our `master` branch is up-to-date and includes the work
   from both prior feature branches, including both `cool-feature` and
   `other-persons-feature`. When we rebase, it will be as if we're "replaying" our
   `cool-feature-2` commits, but pretending that we wrote them AFTER the latest commit
   in `master`. Let's begin the rebase:

    ```
    git rebase master
    ```

9. If you're lucky, the rebase will work automatically. If you're not, then there will
   be a "merge conflict", due to the fact that some code change in the new `master`
   commits is "conflicting" with one or more code changes in your `cool-feature-2`
   commits. In this case, the rebase will stop mid-way, and your entire `git` repository
   will be in a special kind of "merge conflict" state that prevents some normal `git`
   commands from working. This is because you need to rewrite / edit your
   `cool-feature-2` commits so that there is no conflict. How to deal with conflicts in
   a rebase it outside the scope of our help, but here is some guidance:
    - Read this:
      <https://docs.github.com/en/get-started/using-git/resolving-merge-conflicts-after-a-git-rebase>
    - As the above article says, since the rebase is stopped mid-way, you can "abort" the rebase
      using:

        ```
        git rebase --abort
        ```
      This will return the git repo to the state it was before you ran `git rebase master`
      in case you want to try a different solution.
    - For how to fix the conflicts in the files themselves, one helpful link from the
      above page is
      <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/addressing-merge-conflicts/resolving-a-merge-conflict-using-the-command-line>.
    - Another guide is here
      <https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging>.
    - Your code editor (such as [VS Code](https://code.visualstudio.com/download)) will
      often have great features that help you resolve the conflicts easily. These are
      instructions for how you handle a "merge conflict" where you need to go into
      individual files and edit the indicated lines to resolve the conflict. The
      terminology is a bit weird: these are called "merge conflicts", but they can arise
      when you are either "merging" two branches (which are are not doing) OR "rebasing"
      two branches (which we are doing).
    - For how to use rebase to make changes to `cool-feature-2`'s commits (including
      "dropping" commits, "squashing" two commits into one, etc.), see
      <https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History>. It is often helpful
      to "squash" multiple commits that have merge conflicts into a single commit
      *before* rebasing. Most of what you can do in the rebasing process can also be done
      using commands before starting a rebase.
    - As the above guides say, once you have resolved the "merge conflicts", you can add
      the files that are fixed to the index/stage and then continue the rebase from
      where it stopped, using the following commands. This allows the rebase to
      progress, and it will stop again if there are more merge conflicts with a later
      commit.

        ```
        git add -u
        git rebase --continue
        ```

    - To make something clear: yes, this means that you are responsible for making sure
      your proposed, un-merged code changes are compatible with `upstream/master` before
      we are willing to merge them, even if `upstream` is being updated as you are
      developing your own code. We are happy to help you with this process as well; for
      example, you are free to make a Pull Request that has merge conflicts with
      `upstream/master` or our tests, and ask for help in fixing the
      incompatibilities. But we will never rebase `upstream/master` itself -- *all
      rebasing and merge conflicts need to be handled on the feature-branches, not on
      `master`.*

10. Assuming the last rebase succeeded, that's it! Your `master` branches and your
    `cool-feature-2` are now up to date against the latest commits, and you can continue
    developing on `cool-feature-2`. Note that you only strictly need to do this process
    if your Github Pull Request explicitly says that your changes cannot automatically
    be merged. However, it is best practice to keep your feature branches up to
    date. Just because there are no "merge conflicts" doesn't mean that there aren't
    problems with the code itself (e.g. failing tests) due to your branch being out of
    date!

### Troubleshooting `git` problems

- Here's a very helpful website with how to fix or reverse your changes, if you get your
  `git` repository in a confusing or broken state: <https://dangitgit.com/en>

- If you have accidentally written *new* code commits to your local `master` or
  `origin`'s `master` branch (a.k.a. it is "out-of-sync" with `upstream/master`), below
  you will find one way that you can fix the problem. These steps will show you how to
  make a second branch that points to your current `master` branch's latest commit (a
  backup of your work), delete your `master` branch, then re-download the "upstream"
  version of `master`. At the end, your work will still be available on a separate
  branch, but your `master` branch should be identical to that of `upstream`'s `master`
  branch. Do the following:

    1. First, make a backup of your git repo somewhere else on your computer, just in
       case!

    2. Checkout `master`. This will bring you to the latest of your new commits.

    3. Create and checkout a new branch at that commit (let's call the new branch
       `cool-feature-3`), for example by using:

    ```
    git checkout -b cool-feature-3
    ```

    4. Set the upstream of this new branch to `origin` just in case:

    ```
    git branch --set-upstream-to=origin cool-feature-3
    ```

    5. **Delete** your local `master` branch, using the following:

    ```
    git branch -D master
    ```

    6. Download the latest version of `upstream`'s `master` branch, AND make a new
       `master` branch locally that is the same as `upstream`'s `master`, using the
       following:

    ```
    git fetch upstream master:master
    ```

    7. **Only if** you already pushed your out-of-sync commits to `origin`, you should
       force `origin`'s `master` to use your new, local `master` branch, using the
       following. **Note: You should NEVER `git push --force upstream` to your
       `upstream` remote.** The `--set-upstream` option is fine, but always make sure
       you only `git push --force origin`, and never `git push --force upstream`! You
       probably will not be allowed to, but you still need to be cautious.

    ```
    git checkout master
    git branch --set-upstream-to=origin master
    git push --force origin
    ```

    8. The situation should now be resolved: the code you had on `master` is now on
       `cool-feature-3`. Your local and `origin` `master` branch is now fully
       synchronized with `upstream`'s `master`. You can now continue your work
       developing on feature branches, but never the `master` branch.

## Documentation

We provide multiple sources of documentation (including websites) for HNN:

1. **Docstrings** for the code itself. These are inside the appropriate code blocks, but
   they are incorporated automatically into webpages on the HNN Developer Portal
   (described below). We use [NumPy conventions for our
   docstrings](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard).

2. The [HNN Frontpage website](https://hnn.brown.edu/). You probably do
   not have to worry about this. Changes to the Frontpage are handled by the source code
   at <https://github.com/GitHub-at-Brown/hnn-front-website>, not on
   <https://github.com/jonescompneurolab/hnn-core>.

3. The [HNN Textbook
   website](https://jonescompneurolab.github.io/textbook/content/preface.html). Changes
   to the Textbook are handled by the source code at
   <https://github.com/jonescompneurolab/textbook>, not on
   <https://github.com/jonescompneurolab/hnn-core>. The Textbook is the **main guide to
   HNN for users**, and includes both scientific explanations and **example Jupyter
   notebook tutorials**. If you are contributing a new public feature to HNN, then it
   needs to be explained in a tutorial Jupyter notebook on the Textbook.

4. The [HNN Developer Portal
   website](https://jonescompneurolab.github.io/hnn-core/stable/index.html), where you
   currently are. Changes to the Developer Portal are handled by the source code at the
   `doc` subdirectory of `hnn-core` located here:
   <https://github.com/jonescompneurolab/hnn-core/tree/master/doc>. This includes
   written content helpful for developers, but also the "Public API Documentation"
   available here: <https://jonescompneurolab.github.io/hnn-core/stable/api.html>. The
   API documentation is *automatically generated* from the [NumPy-style
   docstrings](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard)
   in our Python code by [`sphinx`](https://www.sphinx-doc.org/en/master/index.html),
   which creates nice-looking webpages to display the docstrings. This also
   automatically executes the scripts in our `examples` directory, subsequently and
   automatically creating webpages for the scripts (including executed output) and
   Jupyter notebooks for the scripts (not including executed output). We are currently
   in the process of moving this notebook execution to the Textbook repo. You can build
   a local version of the Developer Portal (including the Public API Documentation) for
   inspection by following [this section
   below](#building-developer-documentation-locally).

### Updating the Public API Documentation

If you ever add, remove, or rename a function or class from the Public API
(i.e. anything that is explicitly expected to be used by typical users), you must make
sure you update `doc/api.rst`. You should then [build the documentation locally (see
below)](#building-developer-documentation-locally) with your API changes and inspect the
HTML output.

### Updating the Developer Portal

You are welcome to edit or write new documentation pages for the HNN Developer Portal
website using
[reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#rst-primer)
(RST). However, you are also welcome to write using Sphinx's support for [MyST
Markdown](https://www.sphinx-doc.org/en/master/usage/markdown.html#markdown), which
employs [`myst-parser`](https://myst-parser.readthedocs.io/en/latest/). This gives you
the power of RST with the readability of Markdown.

If you want to take advantage of [Roles and Directives inside your Markdown
files](https://myst-parser.readthedocs.io/en/latest/syntax/roles-and-directives.html#roles-directives),
it is fairly straightforward to use them via ["MyST Markdown"
syntax](https://myst-parser.readthedocs.io/en/latest/syntax/roles-and-directives.html#roles-directives). For
example:

- If you want to refer to another local document like this: {doc}`Contributing Guide
  <contributing>` (corresponding to the content in the file `doc/contributing.md`), then:
    - In RST, write:
       ```
       :doc:`Contributing Guide <contributing>`
       ```
    - In Markdown, write:
       ```
       {doc}`Contributing Guide <contributing>`
       ```
- If you want to refer to a part of the HNN-Core API like this:
  {func}`~hnn_core.Network.add_electrode_array`, then:
    - In RST, write:
       ```
       :func:`~hnn_core.Network.add_electrode_array`
       ```
    - In Markdown, write:
       ```
       {func}`~hnn_core.Network.add_electrode_array`
       ```
- For convenience, to quickly insert a link to any specific GitHub issue at
  <https://github.com/jonescompneurolab/hnn-core>, like this: {gh}`705`, then:
    - In RST, write:
       ```
       :gh:`705`
       ```
    - In Markdown, write:
       ```
       {gh}`705`
       ```

Once you have made your changes, you should then [build the documentation locally (see
below)](#building-developer-documentation-locally) and inspect the HTML output.

### Building developer documentation locally

The Developer Portal website, including the Public API Documentation, can be built using
`sphinx` and some related extensions. These are already installed if you used the
`[doc]` or `[dev]` feature sets during your `pip` install (see the ["`pip` Source
Installation" section of our Installation Guide][]), which you already did if you used
the above installation instructions.

You can build a local version of the Developer Portal, **including execution of most
`examples` scripts**, using the following command:

```
cd doc/  # Unnecessary if you're already in the "hnn-core/doc" directory
make html
```

Alternatively, if you want to build the Developer Portal website locally **without
executing** any `examples` scripts, use the command:

```
cd doc/  # Unnecessary if you're already in the "hnn-core/doc" directory
make html-noplot
```

Finally, to view the website, do:

```
cd doc/  # Unnecessary if you're already in the "hnn-core/doc" directory
make view
```

If you've made documentation changes and you want to force a full local website rebuild
from scratch, you can delete all the local output files (such as HTML, script execution
output, etc.) by running the following:

```
cd doc/  # Unnecessary if you're already in the "hnn-core/doc" directory
make clean
```

By default, our automated website deployment checks many, but not all, external URL
links using a command similar to the following:

```
cd doc/  # Unnecessary if you're already in the "hnn-core/doc" directory
make linkcheck
```

However, we exclude certain kinds of link checks from this such as the very, very many
Github-Issue specific links in `doc/whats_new.md`. If you want to manually do an
external link check for **all valid URLs** (excluding a handful that never work with
`sphinx`'s `linkcheck`), then you can use the following command:

```
cd doc/  # Unnecessary if you're already in the "hnn-core/doc" directory
make linkcheck-all
```

Note that running too many local linkchecks will get your IP address temporarily
throttled, especially by Github, so be cautious.

## Quality control

All new code contributions must pass linting checks, spell checks, format checks, and
tests before they can be merged. If you used the above install instructions, then
everything you need to run these should already be installed. We *strongly recommend*
that Contributors first run all quality checks and tests *locally*, and *before* you
push new code to any Pull Requests. These same checks and tests are run automatically by
our Continuous Integration suite, and if any of them fail on your local machine,
then *they will fail* in the automated tests run on Github, and your contributions will
not be merge-able (until the errors are fixed). That said, if you have trouble getting
the tests to pass using your new code changes and you need help, feel free open a Pull
Request using the broken code and then ask us for assistance.

How to run these checks and tests locally is described below.

### Linting

"Linting" your code, which checks for code syntax errors and other errors, can be done
using the following command, which uses the `ruff` library:

```
make lint
```

If the above command prints any errors, then you will *need to fix* those errors before
your code will pass our Continuous Integration testing, which is required for it to be
merged. How to fix the error is up to you: the above command does not change your code,
but will often provide suggestions on potential fixes.

Note that linting is also done as part of the `make test` command (see "Testing" section
below), which means that you do not need to run `make lint` by itself unless you wish
to.

### Spell-checking

Spell-checking your code can be done by simply running the following command. This
command also only checks your code, but does not make changes to it:

```
make spell
```

Note that spell-checking is also done as part of the `make test` command (see "Testing"
section below), which means that you do not need to run `make spell` by itself unless
you wish to.

### Formatting

Formatting is handled differently, through two commands. The first command, below, is
used to check if your code is consistent with our enforced formatting style, which is
currently the [default `ruff` style](https://docs.astral.sh/ruff/formatter/). This
command does not change your code:

```
make format-check
```

Note that format-checking is also done as part of the `make test` command (see "Testing"
section below), which means that you do not need to run `make format-check` by itself
unless you wish to.

However, most of the code you write will probably need to be re-formatted to pass `make
format-check`. Fortunately, `ruff` provides a tool for safe, *automatic* formatting of
your code. If `make format-check` returns any errors and tells you that any number of
files "would be reformatted", and if you are ready to make a git commit, then you should
run the following command. This command **will almost always change your code
automatically**, since it re-formats your code:

```
make format-overwrite
```

Unlike linting, spell-checking, and testing, we do provide a way to automatically fix
formatting issues. That way is the above `make format-overwrite` command. Just to be
safe, you should *always* run tests (see "Testing" section below) after you run `make
format-overwrite`, just in case the auto-formatter broke something, which it hopefully
never will 🤞.

One nice thing about the autoformatter is that if you are defining something that has
multiple elements (e.g. a list `foo = [1, 2]`), and you change the code such that the
final element of that list ends in a comma (i.e., continuing the example, `foo = [1,
2,]`), then the autoformatter will automatically put each element of that list on its
own line. See <https://docs.astral.sh/ruff/settings/#format_skip-magic-trailing-comma>
for more details. Conversely, if you don't want this, then if you remove the "magic
trailing comma" in question (i.e., continuing the example, changing it back to `foo =
[1, 2]`), then the autoformatter will attempt to keep all the elements on the same line
(but only if it is less than the allowed line length).

### Testing

Tests are extremely important and help ensure integrity of the code functionality after
your changes have been made. We use the [`pytest` testing
framework](https://docs.pytest.org/en/stable/), and use the [`pytest-xdist`
extension](https://pytest-xdist.readthedocs.io/en/stable/) to run most tests in parallel
and as fast as possible. To run the tests, run the following command:

```
make test
```

Running tests will not change your code, but may download some additional files. MPI
tests are skipped if the `mpi4py` module is not installed. See our [Installation
Guide][] for how to install MPI and `mpi4py`.

Note that `make test` first runs the checks above for linting, spell-checking, and
formatting, and then runs the test suite only *after* your code successfully passes
those initial checks.

### Continuous Integration

The repository is tested via continuous integration with GitHub Actions and
CircleCI. All the above checks and tests run on GitHub Actions, while the Developer
Portal website is built on CircleCI.

To speed up the website-building process on CircleCI, we enabled versioned
[caching](https://circleci.com/docs/caching/).

Usually, you don't need to worry about it. But in case a complete rebuild is necessary
for a new version of the doc, you can modify the content in `.circleci/build_cache`, as
CircleCI uses the MD5 of that file as the key for previously cached content. For
consistency, we recommend you to monotonically increase the version number in that file,
e.g., from `v2` to `v3`.

## Notes on MPI for contributors

MPI parallelization with NEURON requires that the simulation be launched with the
`nrniv` binary from the command-line. The `mpiexec` command is used to launch multiple
`nrniv` processes which communicate via MPI.  This is done using `subprocess.Popen()` in
`MPIBackend.simulate()` to launch parallel child processes (`MPISimulation`) to carry
out the simulation. The communication sequence between `MPIBackend` and `MPISimulation`
is outlined below.

1.  In order to pass the network to simulate from `MPIBackend`, the child
    `MPISimulation` processes' `stdin` is used. The ready-to-use
    {class}`~hnn_core.Network` object is base64 encoded and pickled before being written
    to the child processes' `stdin` by way of a Queue in a non-blocking way. See how it
    is [used in MNE-Python][].  The data is marked by start and end signals that are
    used to extract the pickled net object. After being unpickled, the parallel
    simulation begins.
2.  Output from the simulation (either to `stdout` or `stderr`) is communicated back to
    `MPIBackend`, where it will be printed to the console. Typical output at this point
    would be simulation progress messages as well as any MPI warnings/errors during the
    simulation.
3.  Once the simulation has completed, the rank 0 of the child process sends back the
    simulation data by base64 encoding and pickling the data object. It also adds
    markings for the start and end of the encoded data, including the expected length of
    data (in bytes) in the end of data marking. Finally rank 0 writes the whole string
    with markings and encoded data to `stderr`.
4.  `MPIBackend` will look for these markings to know that data is being sent (and will
    not print this). It will verify the length of data it receives, printing a
    `UserWarning` if the data length received doesn't match the length part of the
    marking.
5.  To signal that the child process should terminate, `MPIBackend` sends a signal to
    the child proccesses' `stdin`. After sending the simulation data, rank 0 waits for
    this completion signal before continuing and letting all ranks of the MPI process
    exit successfully.
6.  At this point, `MPIBackend.simulate()` decodes and unpickles the data, populates the
    network's CellResponse object, and returns the simulation dipoles to the caller.

It is important that `flush()` is used whenever data is written to stdin or stderr to
ensure that the signal will immediately be available for reading by the other side.

Tests for parallel backends utilize a special `@pytest.mark.incremental` decorator
(defined in `conftest.py`) that causes a test failure to skip subsequent tests in the
incremental block. For example, if a test running a simple MPI simulation fails,
subsequent tests that compare simulation output between different backends will be
skipped. These types of failures will be marked as a failure in CI.

## Making changes to the default network

If you ever need to make scientific or technical changes to the default network
(i.e. the `jones_2009_model` network), you need to do three things:

1. Step 1: If needed, manually make changes to `hnn_core/param/default.json`. This is
   the base file used for the important `jones_2009_model()` function. Make sure that if
   you need to change certain parameters, then change them in this all-important file
   **manually**. Note that not all parameters are in this file. If your changes do not
   affect the parameters in this file, then you don't need to make any change to the
   file.

2. Step 2: Run the following command from the top-level of the repository:

    ```
    make regenerate-networks
    ```

    This command runs two scripts, each of which rebuild one of the "hierarchical JSON"
    network files used in HNN-Core. These two files are described below. Note that you do
    **not** need to make manual changes to these files; running the above command is
    sufficient. However, you **do** need to commit the new versions of these files. The
    two files:

    A. `hnn_core/param/jones2009_base.json`: This is the base file used for the
       GUI. This file has been built using the code in
       `hnn_core/params.py::convert_to_json` by way of
       `dev_scripts/regenerate_base_network.py`.

    B. `hnn_core/test/assets/jones2009_3x3_drives.json`: This is the base file used for
       many tests. This file has been built using the script in
       `hnn_core/tests/regenerate_test_network.py`.

3. Step 3: Once all the above versions of the network have been updated, make sure to
   re-run all the tests using `make test`! If the new network files break tests, then
   that breakage needs to be fixed before we can merge your updates.

## Publishing new releases

For an updated version of how we make releases, {doc}`see our guide here
<how_to_create_releases>`.

## Tips & Tricks

- You can run only specific tests (such a single test inside a single file) by invoking
  the `pytest` command with particular arguments, see
  <https://docs.pytest.org/en/6.2.x/usage.html#specifying-tests-selecting-tests>.
    - Furthermore, if you want to investigate an test failure, you can pass the `--pdb`
      argument to `pytest` ([see
      here](https://docs.pytest.org/en/6.2.x/usage.html#dropping-to-pdb-python-debugger-on-failures))
      which will open the Python Debugger `pdb` ([see
      here](https://docs.python.org/3/library/pdb.html#module-pdb)) at the point where
      the tests are first failing.

["`pip` Source Installation" section of our Installation Guide]: https://jonescompneurolab.github.io/textbook/content/01_getting_started/installation.html#local-installation
[Github]: https://github.com
[Installation Guide]: https://jonescompneurolab.github.io/textbook/content/01_getting_started/installation.html#local-installation
[our Github Discussions page]: https://github.com/jonescompneurolab/hnn-core/discussions
[Textbook website]: https://jonescompneurolab.github.io/textbook/content/preface.html
[used in MNE-Python]: https://github.com/mne-tools/mne-python/blob/148de1661d5e43cc88d62e27731ce44e78892951/mne/utils/misc.py#L124-L132
