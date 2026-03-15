#!/usr/bin/env python3

# This file is part of a demonstrator for Map/Reduce Monte-Carlo
# methods.

# This is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.

# Copyright (c) 2020 Patrick B Warren <patrickbwarren@gmail.com>
# apart from where otherwise stated.

# You should have received a copy of the GNU General Public License
# along with this file.  If not, see <http://www.gnu.org/licenses/>.

"""Reduce outputs from jobs run on a condor cluster

Eg: ./reducer.py --header=mytest --njobs=8
"""

import os
import argparse
import numpy as np

# The following code snippet comes from
# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

def add_bool_arg(parser, name, default=False, help=None):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true',
                       help=help + (' (default)' if default else ''))
    group.add_argument('--no-' + name, dest=name, action='store_false',
                       help="don't " +help + (' (default)' if not default else ''))
    parser.set_defaults(**{name:default})

# The arguments here are used to construct the list of data files

parser = argparse.ArgumentParser(__doc__)
parser.add_argument('header', help='the name of the output and job files')
parser.add_argument('--njobs', default=None, type=int, help='the number of condor jobs')
parser.add_argument('--wipe', default='out,err', help='file extensions for cleaning, default out,err')
parser.add_argument('--data-types', default=None, help='over-ride list of data types')
add_bool_arg(parser, 'overwrite', default=False, help='overwrite data files')
add_bool_arg(parser, 'clean', default=False, help='clean up intermediate files')
add_bool_arg(parser, 'prepend', default=True, help='prepend mapper call to log file')
parser.add_argument('-v', '--verbose', action='count', default=0, help='increasing verbosity')
args = parser.parse_args()

# Extract the list of data types from the log file (or override).
# In the log file, look for a line like
# ... data collected for ... : type1, type2, type3,...
# split on the ':', take the second half, and split again on ',' then
# strip off any white space to recover a python list of strings (data_types)

if args.data_types:
    data_types = args.data_types.split(',')
else:
    with open(args.header + '.log') as f:
        for line in f:
            if 'data collected for' in line:
                data_types = [s.strip() for s in line.split(':')[1].split(',')]

# Now reduce each data type, using numpy to do the statistics

def process(data_file):
    """process the data in the given file"""
    with open(data_file) as f:
        for line in f:
            val, tag = line.rstrip('\n').split('\t')[:2]
            if tag in data:
                data[tag].append(val)
            else:
                data[tag] = [val]

for data_type in data_types:
    data = {}
    if args.njobs:
        for k in range(args.njobs):
            process(f'{args.header}__{k}_{data_type}.dat')
    else:
        process(f'{args.header}_{data_type}.dat')
    data_file = f'{args.header}_{data_type}.dat'
    if not args.overwrite and os.path.exists(data_file):
        print(f'{data_file} exists, use --overwrite option to overwrite')
    else:
        with open(data_file, 'w') as f:
            for tag in data:
                arr = np.array([float(v) for v in data[tag]])
                mean, var, npt = np.mean(arr), np.var(arr, ddof=1), np.size(arr)
                sem = np.sqrt(var / npt) # standard error in the mean
                f.write('%g\t%g\t%s\t%d\n' % (mean, sem, tag, npt))
        if args.verbose:
            print(f'{data_type} > {data_file}')

# Prepend the mapper command line extracted from the first line of condor job description, based on
# https://stackoverflow.com/questions/4454298/prepend-a-line-to-an-existing-file-in-python

if args.prepend:
    condor_job = args.header + '__condor.job'
    try:
        with open(condor_job) as f:
            mapper_command = f.readline() # readline keeps the newline character '\n'
    except IOError:
        print(f'failed to find {condor_job}, skipping prepend')
        mapper_command = None
    if mapper_command:
        log_file = args.header + '.log'
        try:
            with open(log_file, 'r+') as f:
                contents = f.read() # slurp the existing contents
                f.seek(0) # rewind to the beginning
                f.write(mapper_command) # write the mapper command
                f.write(contents) # then the rest of the contents
        except IOError:
            print(f'failed to find {log_file}, skipping prepend')

# Clean up output and error files

if args.clean:
    for k in range(args.njobs):
        if args.wipe:
            for extension in args.wipe.split(','):
                os.remove(f'{args.header}__{k}.{extension}')
        for data_type in data_types:
            os.remove(f'{args.header}__{k}_{data_type}.dat')

# End of script
