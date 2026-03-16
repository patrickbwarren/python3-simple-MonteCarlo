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
# apart from where otherise stated.

# You should have received a copy of the GNU General Public License
# along with this file.  If not, see <http://www.gnu.org/licenses/>.

"""Map jobs onto a condor cluster

Eg: ./mapper.py throw_darts.py --header=mytest --seed=12345 --ntrial=10 \
 --nthrow=10^6 --njobs=8 --module=ThrowDarts --run
"""

import os
import sys
import argparse
import subprocess

# keep this to include in the job description

command_line = ' '.join(sys.argv)

# The following code snippet comes from
# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

def add_bool_arg(parser, name, default=False, help=None):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true',
                       help=help + (' (default)' if default else ''))
    group.add_argument('--no-' + name, dest=name, action='store_false',
                       help="don't " +help + (' (default)' if not default else ''))
    parser.set_defaults(**{name:default})
    
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("script", help="script to be run")
parser.add_argument('--header', required=True, help='set the name of the output and job files')
parser.add_argument('--njobs', required=True, type=int, help='the number of condor jobs')
parser.add_argument('--run', action='store_true', help='run the condor or DAGMan job')
parser.add_argument('--fast', action='store_true', help='run with Mips > min mips')
parser.add_argument('--min-mips', type=int, default=20000, help='min mips for fast option, default 20000')
parser.add_argument('--modules', default=None, help='supporting module(s), default None')
parser.add_argument('--extensions', default='so,py,pm', help='file extensions for module(s), default so,py,pm')
parser.add_argument('--executable', default=sys.executable, help=f'executable to run script, if not {sys.executable}')
parser.add_argument('--transfers', default=None, help='additional files to transfer, default None')
parser.add_argument('--wipe', default='out,err', help='file extensions for cleaning, default out,err')
add_bool_arg(parser, 'reduce', default=True, help='use DAGMan to reduce the output')
add_bool_arg(parser, 'clean', default=True, help='clean up intermediate files')
add_bool_arg(parser, 'prepend', default=True, help='prepend mapper call to log file')
parser.add_argument('-v', '--verbose', action='count', default=0, help='increasing verbosity')
args, rest = parser.parse_known_args()

header, njobs = args.header, args.njobs

# Find the files to transfer; include files in the current
# directory where the file name matches any of the modules in
# args.modules (comma-separated list) and which have an extension in
# args.extensions (comma-separated list).  We first convert the
# comma-separated lists to python lists, then filter a list of the
# files in the current directory.

modules, extensions, transfers = [ [] if s is None else s.split(',') for s in
                                   [args.modules, args.extensions, args.transfers] ]

file_list = [f.name for f in os.scandir() if f.is_file()] # all files in current directory

if modules and extensions:
    transfers.extend(filter(lambda f: any(f.endswith(f'.{e}') for e in extensions)
                            and any(m in f for m in modules), file_list))

transfers.append(args.script) # add the script itself to the list

# Create the condor job file

condor_job = header + '__condor.job'

# Add a requirements line if requested (newlines are required to
# insert as lines in constructing the script below).

extra = f'\nrequirements = Mips > {args.min_mips}' if args.fast else ''

# Reconstruct the verbosity and stick on the end of the unmatched arguments

if args.verbose:
    rest.append('-' + 'v' * args.verbose)
    
opts = ' '.join(rest) # now contains all the unmatched arguments

# The actual job description using an f-string

lines = [f'# {command_line}',
         'should_transfer_files = YES',
         'when_to_transfer_output = ON_EXIT',
         'notification = never',
         'universe = vanilla',
         f'opts = {opts}{extra}',
         'transfer_input_files = ' + ','.join(transfers),
         f'executable = {args.script}',
         f'arguments = --header={header} $(opts) --process=$(Process) --njobs={njobs}',
         f'output = {header}__$(Process).out',
         f'error = {header}__$(Process).err',
         f'queue {njobs}']

with open(condor_job, 'w') as f:
    f.write('\n'.join(lines) + '\n')

if args.verbose:
    print('Created:', condor_job)

if not args.reduce: # we just need to run the condor job

    run_command = 'condor_submit ' + condor_job

else: # create a DAGMan master job

    dag_job = header + '__dag.job'

    opts = ['--clean' if args.clean else '--no-clean', 
            '--prepend' if args.prepend else '--no-prepend',
            f'--wipe={args.wipe}', f'--njobs={njobs}']
    
    script = f"{args.executable} reducer.py {header} {' '.join(opts)}"

    lines = [f'JOB A {condor_job}',
             f'SCRIPT POST A {script}']

    with open(dag_job, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    if args.verbose:
        print('Created:', dag_job)

    run_command = 'condor_submit_dag -notification Never ' + dag_job

# We run if required, otherwise print out the run command for the user

if args.run: 
    subprocess.call(run_command, shell=True)
else:
    print(run_command)

# End of script
