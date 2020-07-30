#! /usr/bin/env python
import os.path as op
import os
import subprocess

from setuptools import setup, find_packages

from distutils.command.build import build
from distutils.cmd import Command

descr = """Experimental code for simulating evoked using Neuron"""

DISTNAME = 'hnn-core'
DESCRIPTION = descr
MAINTAINER = 'Mainak Jas'
MAINTAINER_EMAIL = 'mainakjas@gmail.com'
URL = ''
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'http://github.com/jonescompneurolab/hnn-core'
VERSION = '0.1.dev0'


class BuildMod(Command):
    user_options = []

    def initialize_options(self):
        """Abstract method that is required to be overwritten"""
        pass

    def finalize_options(self):
        """Abstract method that is required to be overwritten"""
        pass

    def run(self):
        print("=> Building mod files ...")
        mod_path = op.join(op.dirname(__file__), 'hnn_core', 'mod')
        process = subprocess.Popen(['nrnivmodl'], cwd=mod_path,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        outs, errs = process.communicate()
        print(outs)


class my_build(build):
    def run(self):
        self.run_command("build_mod")
        build.run(self)


if __name__ == "__main__":
    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
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
          packages=find_packages(),
          include_package_data=True,
          cmdclass={'build': my_build, 'build_mod': BuildMod}
          )
