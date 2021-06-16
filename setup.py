#! /usr/bin/env python
import platform
import os.path as op
import os
import subprocess
import shutil

from setuptools import setup, find_packages

from distutils.command.build_py import build_py
from distutils.cmd import Command

descr = """Code for biophysical simulation of a cortical column using Neuron"""

DISTNAME = 'hnn-core'
DESCRIPTION = descr
MAINTAINER = 'Mainak Jas'
MAINTAINER_EMAIL = 'mainakjas@gmail.com'
URL = ''
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'http://github.com/jonescompneurolab/hnn-core'

# get the version
version = None
with open(os.path.join('hnn_core', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')


# test install with:
# $ rm -rf hnn_core/mod/x86_64/
# $ python setup.py clean --all install
#
# to make sure there are no residual mod files
#
# also see following link to understand why build_py must be overriden:
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
            mod_path = op.join(op.dirname(__file__), 'mod')
            shell = True
        else:
            mod_path = op.join(op.dirname(__file__), 'hnn_core', 'mod')
            shell = False

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


if __name__ == "__main__":
    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=version,
          download_url=DOWNLOAD_URL,
          long_description=open('README.rst').read(),
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS',
          ],
          platforms='any',
          install_requires=[
              'numpy >=1.14',
              'NEURON >=7.7',
              'matplotlib',
              'scipy'
          ],
          packages=find_packages(),
          package_data={'hnn_core': [
              'param/*.json',
              'mod/*',
              'mod/x86_64/*',
              'mod/x86_64/.lib/*']},
          cmdclass={'build_py': build_py_mod, 'build_mod': BuildMod}
          )
