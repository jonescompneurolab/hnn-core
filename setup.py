#! /usr/bin/env python
import platform
import os.path as op
import subprocess
import shutil

from setuptools import setup, Command
from setuptools.command.build_py import build_py

# test the build of wheel and sdist:
# First remove residual mod files
# $ rm -rf hnn_core/mod/x86_64/
# or for Apple silicon
# $ rm -rf hnn_core/mod/mod64/
# $ python -m build
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

        if platform.system() == 'Windows':
            shell = True
        else:
            shell = False

        mod_path = op.join(op.dirname(__file__), 'hnn_core', 'mod')
        process = subprocess.Popen(['nrnivmodl'], cwd=mod_path,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, shell=shell)
        outs, errs = process.communicate()
        print(outs)


class build_py_mod(build_py):
    def run(self):
        self.run_command("build_mod")

        build_dir = op.join(self.build_lib, 'hnn_core', 'mod')
        mod_path = op.join(op.dirname(__file__), 'hnn_core', 'mod')
        shutil.copytree(mod_path, build_dir)

        build_py.run(self)

setup(cmdclass={'build_py': build_py_mod, 'build_mod': BuildMod})

