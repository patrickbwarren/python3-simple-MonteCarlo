#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Run a DPD Monte-Carlo simulation
# This file is part of the simple Monte-Carlo collection.

# This is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.

# Copyright (c) 2026 Patrick B Warren <patrick.warren@stfc.ac.uk>

# You should have received a copy of the GNU General Public License
# along with this file.  If not, see <http://www.gnu.org/licenses/>.

# Can be run with map/reduce as for example,
# ./mapper.py dpd.py --header=puretest--seed=12345 --ntrial=10 --nthrow=10^5 --njobs=20 --run

import sys
import argparse
import numpy as np
from itertools import product

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--header', required=True, help='set the name of the output files')
parser.add_argument('--seed', default=12345, type=int, help='the RNG seed, default 12345')
parser.add_argument('--process', default=0, type=int, help='process number, default 0')
parser.add_argument('--njobs', default=1, type=int, help='the number of condor jobs, deault 1')
parser.add_argument('-e', '--es', default=10.0, type=float, help='box size, default 10.0')
parser.add_argument('-r', '--rho', default=3.0, type=float, help='density, default 3.0')
parser.add_argument('--npart', default='rho*vol', help='number of particles, default computed')
parser.add_argument('--nmove', default='npart', help='number of MC moves per sweep, default npart')
parser.add_argument('--nequil', default=10, type=int, help='number of equilibration sweeps, default 10')
parser.add_argument('--nsamp', default=0, type=int, help='number of additional samples, default 0')
parser.add_argument('--delta', default=0.2, type=float, help='trial displacement, default 0.2')
parser.add_argument('-A', '--A', default=25.0, type=float, help='repulsion amplitude, default 25.0')
parser.add_argument('-v', '--verbose', action='count', default=0, help='increasing verbosity')
args = parser.parse_args()

pid, njobs = args.process, args.njobs
rng = np.random.default_rng(seed=args.seed).spawn(njobs)[pid] # select a local RNG stream

ep_file = '%s__%d_ep.dat' % (args.header, pid)
gr_file = '%s__%d_gr.dat' % (args.header, pid)
log_file = '%s.log' % args.header

A = args.A
rho = args.rho
delta = args.delta

def kM_replace(s):
    return s.replace('k', '*1e3').replace('M', '*1e6')

es, esby2, vol = args.es, args.es/2, args.es**3
npart = eval('int(%s)' % kM_replace(args.npart))
rho = args.rho if args.npart is None else npart/vol
nmove = eval('int(%s)' % kM_replace(args.nmove))
nequil, nsamp = args.nequil, args.nsamp

pos = rng.uniform(0.0, es, size=(npart, 3)) # initialise particle positions

ncell = int(es) # number of cells along one axis
cell_size = es / ncell # cell size
cell_coord = list(range(ncell)) # list of coords along one axis
all_cell_coords = product(cell_coord, cell_coord, cell_coord) # coords of all cells, as tuples
contents = dict([(cell, set()) for cell in all_cell_coords]) # empty cell sets, indexed by cell coords

for i in range(npart): # go through all particles
    cell = (pos[i]/cell_size).astype(int) # calculate cell coordinates
    contents[tuple(cell)].add(i) # add particle to set of particles in cell

neighbours = [np.array(x, dtype=int) for x in product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])]

pairs = [(i, j) for i, j in product(range(npart), range(npart)) if i < j]

def brute_force():
    energy, virial = 0, 0
    for i, j in pairs:
        Δr = pos[j] - pos[i]
        Δr = Δr - np.where(Δr > esby2, es, 0) + np.where(Δr < -esby2, es, 0)
        rsq = np.sum(Δr**2)
        if rsq < 1:
            r = np.sqrt(rsq)
            energy = energy + (A/2)*(1-r)**2
            virial = virial + A*r*(1-r)
    return energy/vol, rho + virial/(3*vol)

def energy_pressure():
    energy, virial = 0, 0
    for i in range(npart):
        pos_i = pos[i]
        cell = (pos_i/cell_size).astype(int)
        for neighbour in neighbours:
            neighbour_cell = (cell + neighbour) % ncell
            for j in contents[tuple(neighbour_cell)]:
                if i < j:
                    Δr = pos[j] - pos_i
                    Δr = Δr - np.where(Δr > esby2, es, 0) + np.where(Δr < -esby2, es, 0)
                    rsq = np.sum(Δr**2)
                    if rsq < 1:
                        r = np.sqrt(rsq)
                        energy = energy + (A/2)*(1-r)**2
                        virial = virial + A*r*(1-r)
    return energy/vol, rho + virial/(3*vol)

def part_energy(i, cell, pos_i):
    energy = 0
    for neighbour in neighbours:
        neighbour_cell = (cell + neighbour) % ncell
        for j in contents[tuple(neighbour_cell)]:
            if i != j:
                Δr = pos[j] - pos_i
                Δr = Δr - np.where(Δr > esby2, es, 0) + np.where(Δr < -esby2, es, 0)
                rsq = np.sum(Δr**2)
                if rsq < 1:
                    r = np.sqrt(rsq)
                    energy = energy + (A/2)*(1-r)**2
    return energy

def test_energy():
    e, p = energy_pressure()
    print(f'cell list methods  =\t{e}\t{p}')
    e, p = brute_force()
    print(f'brute force method =\t{e}\t{p}')
    energy = 0
    for i in range(npart):
        cell = (pos[i]/cell_size).astype(int) # calculate cell coordinates
        energy = energy + 0.5*part_energy(i, cell, pos[i])
    print(f'part_energy        =\t{energy/vol}')

if args.verbose > 1:
    test_energy()

def mc_moves(nmove):
    naccept = 0
    parts = rng.integers(0, npart, size=nmove)
    disps = rng.normal(0.0, delta, size=(nmove, 3))
    probs = rng.random(size=nmove)
    for i, d, p in zip(parts, disps, probs):
        old_pos = pos[i]
        old_cell = (old_pos/cell_size).astype(int)
        old_energy = part_energy(i, old_cell, old_pos)
        new_pos = (old_pos + d) % es
        new_cell = (new_pos/cell_size).astype(int)
        new_energy = part_energy(i, new_cell, new_pos)
        if (p < np.exp(-(new_energy-old_energy))):
            naccept = naccept + 1
            pos[i] = new_pos
            contents[tuple(old_cell)].remove(i)
            contents[tuple(new_cell)].add(i)
    return naccept/nmove

for sweep in range(nequil):
    acc_rat = mc_moves(nmove)
    e, p = energy_pressure()
    if args.verbose:
        print('equilibration:\t%i\t%g\t%g\t%g' % (sweep, acc_rat, e, p))

results = [(e, p, acc_rat)]

for samp in range(nsamp):
    acc_rat = mc_moves(nmove)
    e, p = energy_pressure()
    results.append((e, p, acc_rat))
    if args.verbose:
        print('production:\t%i\t%g\t%g\t%g' % (samp, acc_rat, e, p))

#    mode = 'w' if samp == 0 else 'a'
#    with open(gr_file, mode) as f:
#        for ig in range(nbins):
#            r = (ig + 0.5) / nbins
#            area_annulus = np.pi * ((ig+1)**2 - ig**2) / nbins**2
#            g = 4.0 * gr_bins[ig] / (nthrows * area_annulus)
#            f.write('%g\tgr__%g\n' % (g, r))

with open(ep_file, 'w') as f:
    for e, p, acc_rat in results:
        f.write('%g\te\n' % e)
        f.write('%g\tp\n' % p)
        f.write('%g\tacc_rat\n' % acc_rat)

if args.verbose > 1:
    test_energy()

run_opts = [sys.argv[0], f'--header={args.header}', f'--seed={args.seed}',
            f'--nequil={nequil}', f'--nsamp={nsamp}', f'--nmove={nmove}',
            f'--A={A}', f'--rho={rho}', f'--es={es}']

reports = ['command_line: ' + ' '.join(sys.argv),
           'full_command: ' + ' '.join(run_opts),
           'data collected for: ep']

if args.verbose:
    for line in reports:
        print(line)

if args.process == 0:
    with open(log_file, 'w') as f:
        for line in reports:
            f.write('# ' + line + '\n')
