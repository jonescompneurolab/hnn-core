#! /usr/bin/env python
# All package metadata and configuration lives in pyproject.toml. This file only
# exists to compile the NEURON mechanisms (.mod files) at build time.
import platform
from pathlib import Path
import subprocess
import shutil

from setuptools import setup, Command
from setuptools.command.build_py import build_py


# test install with:
# $ rm -rf hnn_core/mod/x86_64/
# $ rm -rf hnn_core/mod/arm64/
# $ python setup.py clean --all install
#
# to make sure there are no residual mod files
#
# also see following link to understand why build_py must be overridden:
# https://stackoverflow.com/questions/51243633/python-setuptools-setup-py-install-does-not-automatically-call-build
class BuildMod(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        print("=> Building mod files ...")

        if platform.system() == "Windows":
            shell = True
        else:
            shell = False

        mod_path = Path(__file__).parent / "hnn_core" / "mod"
        process = subprocess.Popen(
            ["nrnivmodl"],
            cwd=mod_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=shell,
        )
        outs, errs = process.communicate()
        print(outs)


class build_py_mod(build_py):
    def run(self):
        self.run_command("build_mod")

        build_dir = Path(self.build_lib)/ "hnn_core"/ "mod"
        mod_path = Path(__file__).parent / "hnn_core" / "mod"
        shutil.copytree(mod_path, build_dir)

        build_py.run(self)


if __name__ == "__main__":
    setup(cmdclass={"build_py": build_py_mod, "build_mod": BuildMod})
