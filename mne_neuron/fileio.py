# fileio.py - general file input/output functions
#
# v 1.10.0-py35
# rev 2016-05-01 (SL: return_data_dir() instead of hardcoded everywhere, etc.)
# last rev: (SL: toward python3)

import datetime
import fnmatch
import os
import sys

# creates data dirs and a dictionary of useful types
# self.dfig is a dictionary of experiments, which is each a
# dictionary of data type
# keys and the specific directories that contain them.


class SimulationPaths(object):

    def __init__(self):
        # hard coded data types
        # fig extensions are not currently being used as well as they could be
        # add new directories here to be automatically created for every
        # simulation
        self.__datatypes = {'rawspk': 'spk.txt',
                            'rawdpl': 'rawdpl.txt',
                            # same output name - do not need both raw and
                            # normalized dipole - unless debugging
                            'normdpl': 'dpl.txt',
                            'rawcurrent': 'i.txt',
                            'rawspec': 'spec.npz',
                            'rawspeccurrent': 'speci.npz',
                            'avgdpl': 'dplavg.txt',
                            'avgspec': 'specavg.npz',
                            'figavgdpl': 'dplavg.png',
                            'figavgspec': 'specavg.png',
                            'figdpl': 'dpl.png',
                            'figspec': 'spec.png',
                            'figspk': 'spk.png',
                            'param': 'param.txt',
                            }
        # empty until a sim is created or read
        self.fparam = None
        self.sim_prefix = None
        self.trial_prefix_str = None
        self.expmt_groups = []
        self.dproj = None
        self.ddate = None
        self.dsim = None
        self.dexpmt_dict = {}
        self.dfig = {}

    # reads sim information based on sim directory and param files
    def read_sim(self, dproj, dsim):
        # nested import to avoid circular dependency
        from .paramrw import read_expmt_groups, read_sim_prefix

        self.dproj = dproj
        self.dsim = dsim
        # match the param from this sim
        self.fparam = file_match(dsim, '.param')[0]
        self.expmt_groups = read_expmt_groups(self.fparam)
        self.sim_prefix = read_sim_prefix(self.fparam)
        # this should somehow be supplied by the ExpParams() class, but doing
        # it here
        self.trial_prefix_str = self.sim_prefix + "-%03d-T%02d"
        self.dexpmt_dict = self.__create_dexpmt(self.expmt_groups)
        # create dfig
        self.dfig = self.__read_dirs()
        return self.dsim

    # only run for the creation of a new simulation
    def create_new_sim(self, dproj, expmt_groups, sim_prefix='test'):
        self.dproj = dproj
        self.expmt_groups = expmt_groups
        # prefix for these simulations in both filenames and directory in ddate
        self.sim_prefix = sim_prefix
        # create date and sim directories if necessary
        self.ddate = self.__datedir()
        self.dsim = self.__simdir()
        self.dexpmt_dict = self.__create_dexpmt(self.expmt_groups)
        # dfig is just a record of all the fig directories, per experiment
        # will only be written to at time of creation, by create_dirs
        # dfig is a terrible variable name, sorry!
        self.dfig = self.__ddata_dict_template()

    # this is a hack
    # checks root expmt_group directory for any files i've thrown there
    def find_aggregate_file(self, expmt_group, datatype):
        # file name is in format: '%s-%s-%s' % (sim_prefix, expmt_group,
        # datatype-ish)
        fname = '%s-%s-%s.txt' % (self.sim_prefix, expmt_group, datatype)
        # get a list of txt files in the expmt_group
        # local=1 forces the search to be local to this directory and not
        # recursive
        local = 1
        flist = file_match(self.dexpmt_dict[expmt_group], fname, local)
        return flist

    # returns a filename for an example type of data
    def return_filename_example(self, datatype, expmt_group,
                                sim_no=None, tr=None, ext='png'):
        fname_short = "%s-%s" % (self.sim_prefix, expmt_group)
        if sim_no is not None:
            fname_short += "-%03i" % (sim_no)
        if tr is not None:
            fname_short += "-T%03i" % (tr)
        # add the extension
        fname_short += ".%s" % (ext)
        fname = os.path.join(self.dfig[expmt_group][datatype], fname_short)
        return fname

    # creates a dict of dicts for each experiment and all the datatype directories
    # this is the empty template that gets filled in later.
    def __ddata_dict_template(self):
        dfig = dict.fromkeys(self.expmt_groups)
        for key in dfig:
            dfig[key] = dict.fromkeys(self.__datatypes)
        return dfig

    # read directories for an already existing sim
    def __read_dirs(self):
        dfig = self.__ddata_dict_template()
        for expmt_group, dexpmt in self.dexpmt_dict.items():
            for key in self.__datatypes.keys():
                ddatatype = os.path.join(dexpmt, key)
                dfig[expmt_group][key] = ddatatype
        return dfig

    # create the data directory for the sim
    def create_datadir(self):
        dout = self.__simdir()
        if not safemkdir(dout):
            print("ERR: could not create output dir", dout)

    # Returns date directory
    # this is NOT safe for midnight
    def __datedir(self):
        self.str_date = datetime.datetime.now().strftime("%Y-%m-%d")
        ddate = os.path.join(self.dproj, self.str_date)
        return ddate

    # returns the directory for the sim
    def __simdir(self):
        return os.path.join(os.path.dirname(__file__), '..', 'data',
                            self.sim_prefix)

    # creates all the experimental directories based on dproj
    def __create_dexpmt(self, expmt_groups):
        d = dict.fromkeys(expmt_groups)
        for expmt_group in d:
            d[expmt_group] = os.path.join(self.dsim, expmt_group)
        return d

    # Get the data files matching file_ext in this directory
    # functionally the same as the previous function but with a local scope
    def file_match(self, expmt_group, key):
        # grab the relevant fext
        fext = self.__datatypes[key]
        file_list = []
        ddata = self.__simdir()
        # search the sim directory for all relevant files
        if os.path.exists(ddata):
            for root, dirnames, filenames in os.walk(ddata):
                for fname in fnmatch.filter(filenames, '*' + fext):
                    file_list.append(os.path.join(root, fname))
        # sort file list? untested
        file_list.sort()
        return file_list

# Cleans input files


def clean_lines(file):
    with open(file) as f_in:
        lines = (line.rstrip() for line in f_in)
        lines = [line for line in lines if line]
    return lines

# create file name for temporary spike file
# that every processor is aware of


def file_spike_tmp(dproj):
    filename_spikes = 'spikes_tmp.spk'
    file_spikes = os.path.join(dproj, filename_spikes)
    return file_spikes

# Get the data files matching file_ext in this directory
# this function traverses ALL directories
# local=1 makes the search local and not recursive


def file_match(dsearch, file_ext, local=0):
    file_list = []
    if not local:
        if os.path.exists(dsearch):
            for root, dirnames, filenames in os.walk(dsearch):
                for fname in fnmatch.filter(filenames, '*' + file_ext):
                    file_list.append(os.path.join(root, fname))
    else:
        file_list = [os.path.join(dsearch, file) for file in os.listdir(
            dsearch) if file.endswith(file_ext)]
    # sort file list? untested
    file_list.sort()
    return file_list

# check any directory


def dir_check(d):
    if not os.path.isdir(d):
        return 0
    else:
        return os.path.isdir(d)

# only create if check comes back 0

# list spike raster eps files and then rasterize them to HQ png files,
# lossless compress,
# reencapsulate as eps, and remove backups when successful


def safemkdir(dn):
    """Make dir, catch exceptions."""

    try:
        os.mkdir(dn)
        return True
    except OSError:
        if not os.path.exists(dn):
            print('ERR: could not create', dn)
            return False
        else:
            return True

# returns the data dir


def return_data_dir():
    dfinal = os.path.join('.', 'data')
    if not safemkdir(dfinal):
        sys.exit(1)
    return dfinal


if __name__ == '__main__':
    return_data_dir()
