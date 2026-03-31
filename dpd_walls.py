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
# ./mapper.py dpd.py --header=test --seed=12345 --es=6 --njobs=20 --run

import argparse
import numpy as np
from numpy import pi as π
from itertools import product

def eval_kM_replace(s):
    return eval('int({})'.format(s.replace('k', '*1e3').replace('M', '*1e6')))
    
# Extend the ArgumentParser class to be able to add boolean options, adapted from
# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

class ExtendedArgumentParser(argparse.ArgumentParser):

    def add_bool_arg(self, long_opt, short_opt=None, default=False, help=None):
        '''Add a mutually exclusive --opt, --no-opt group with optional short opt'''
        opt = long_opt.removeprefix('--')
        group = self.add_mutually_exclusive_group(required=False)
        help_string = None if not help else help if not default else f'{help}, default'
        if short_opt:    
            group.add_argument(short_opt, f'--{opt}', dest=opt, action='store_true', help=help_string)
        else:
            group.add_argument(f'--{opt}', dest=opt, action='store_true', help=help_string)
        help_string = None if not help else f"don't {help}" if default else f"don't {help}, default"        
        group.add_argument(f'--no-{opt}', dest=opt, action='store_false', help=help_string)
        self.set_defaults(**{opt:default})

parser = ExtendedArgumentParser(description=__doc__)
parser.add_argument('--header', default=None, help='set the name of the output files')
parser.add_argument('--seed', default=12345, type=int, help='the RNG seed, default 12345')
parser.add_argument('--process', default=0, type=int, help='process number, default 0')
parser.add_argument('--njobs', default=1, type=int, help='the number of condor jobs, deault 1')
parser.add_argument('-e', '--es', default=10.0, type=float, help='box size, default 10.0')
parser.add_argument('-r', '--rho', default=3.0, type=float, help='density, default 3.0')
parser.add_argument('-A', '--A', default=25.0, type=float, help='repulsion amplitude, default 25.0')
parser.add_argument('--Awall', default='0.0', help='wall repulsion amplitude, default 0.0')
parser.add_argument('--npart', default='rho*vol', help='number of particles, default computed')
parser.add_argument('--nmove', default='npart', help='number of MC moves per sweep, default npart')
parser.add_argument('--nequil', default='10', help='number of equilibration sweeps, default 10')
parser.add_argument('--delta', default=0.2, type=float, help='trial displacement, default 0.2')
parser.add_argument('--nbins', default=80, type=int, help='number of bins in density profile, default 40')
parser.add_bool_arg('--walls', short_opt='-w', default=True, help='include walls')
parser.add_bool_arg('--uniform', short_opt='-u', default=False, help='uniform wall model')
parser.add_argument('-v', '--verbose', action='count', default=0, help='increasing verbosity')
args = parser.parse_args()

pid, njobs = args.process, args.njobs
rng = np.random.default_rng(seed=args.seed).spawn(njobs)[pid] # select a local RNG stream

A, rho, ΔR = args.A, args.rho, args.delta

Awall = eval(args.Awall) # catch things like A*rho, used in uniform wall model

es, esby2 = args.es, args.es/2

zlo, zhi = (0.5, es-0.5) if args.walls else (0, es)

vol = es**2*(zhi-zlo) # volume between walls

npart = eval_kM_replace(args.npart)
nmove = eval_kM_replace(args.nmove)
nequil = eval_kM_replace(args.nequil)

pos = rng.uniform(zlo, zhi, size=(npart, 3)) # initialise particle positions between walls

ncell = int(es) # number of cells along one axis
cell_size = es / ncell # cell size
cell_coord = list(range(ncell)) # list of coords along one axis
cell_coord_triple = [cell_coord] * 3 # three such lists
all_cells = product(*cell_coord_triple) # iterator for coords of all cells, as tuples (triples)
contents = dict([(cell, set()) for cell in all_cells]) # empty cell sets, indexed by cell coords

box = list(range(npart)) # list of particles

for i in box: # go through all particles
    cell = (pos[i]/cell_size).astype(int) # calculate cell coordinates
    contents[tuple(cell)].add(i) # add particle to set of particles in cell

neighbour_coord = [-1, 0, 1] # neighbour offsets along one axis
neighbour_coord_triple = [neighbour_coord] * 3 # three such offsets
all_neighbours = product(*neighbour_coord_triple) # iterator for neighbour offset triples
neighbours = [np.array(x, dtype=int) for x in all_neighbours] # convert to a list of numpy integer arrays

def brute_force():
    energy, virial = 0, 0
    for i in box:
        for j in box:
            if i < j:
                Δr = pos[j] - pos[i]
                Δr = Δr - np.where(Δr > esby2, es, 0) + np.where(Δr < -esby2, es, 0)
                rsq = np.sum(Δr**2)
                if rsq < 1:
                    r = np.sqrt(rsq)
                    energy += (A/2)*(1-r)**2
                    virial += A*r*(1-r)
    return 3*npart/(2*vol) + energy/vol, npart/vol + virial/(3*vol)

def energy_pressure():
    energy = 0
    virial = np.zeros(3)
    for i in box:
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
                        energy += (A/2)*(1-r)**2
                        virial += Δr**2 / r * A*(1-r) # resolves into components
    return 3*npart/(2*vol) + energy/vol, npart/vol + virial/vol

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
                    energy += (A/2)*(1-r)**2
    return energy

def tot_part_energy(): # test of above
    energy = 3*npart/2
    for i in box:
        cell = (pos[i]/cell_size).astype(int) # calculate cell coordinates
        energy += 0.5*part_energy(i, cell, pos[i])
    return energy/vol

def test_energy():
    e, p = energy_pressure()
    print('cell list methods  =\t{}\t{}'.format(e, np.mean(p)))
    print('brute force method =\t{}\t{}'.format(*brute_force()))
    print('tot part energy    =\t{}'.format(tot_part_energy()))

if args.verbose > 1:
    test_energy()

def vanilla_wall_energy(z):
    energy = 0
    if z < zlo + 1:
        zz = z - zlo
        energy = (Awall/2)*(1-zz)**2
    elif z > zhi - 1:
        zz = zhi - z
        energy = (Awall/2)*(1-zz)**2
    return energy

def uniform_wall_energy(z):
    energy = 0
    if z < zlo + 1:
        zz = z - zlo
        energy = (π*Awall/60)*(1-zz)**4*(2+3*zz)
    elif z > zhi - 1:
        zz = zhi - z
        energy = (π*Awall/60)*(1-zz)**4*(2+3*zz)
    return energy

wall_energy = uniform_wall_energy if args.uniform else vanilla_wall_energy

# This is the actual Monte-Carlo algorithm

for sweep in range(nequil): # do a number of sweeps of nmove trial moves
    naccept = 0
    parts = rng.integers(0, npart, size=nmove)
    disps = rng.normal(0.0, ΔR, size=(nmove, 3))
    probs = rng.random(size=nmove)
    for i, disp, prob in zip(parts, disps, probs):
        old_pos = pos[i]
        old_cell = (old_pos/cell_size).astype(int)
        old_energy = part_energy(i, old_cell, old_pos) + wall_energy(old_pos[2])
        new_pos = (old_pos + disp) % es
        if new_pos[2] < zlo or new_pos[2] > zhi: # reject if falls outside walls
            continue
        new_cell = (new_pos/cell_size).astype(int)
        new_energy = part_energy(i, new_cell, new_pos) + wall_energy(new_pos[2])
        if (prob < np.exp(-(new_energy-old_energy))): # acceptance criterion
            naccept += 1
            pos[i] = new_pos
            contents[tuple(old_cell)].remove(i)
            contents[tuple(new_cell)].add(i)
    (e, p), a = energy_pressure(), naccept/nmove
    if args.verbose:
        print('equilibration: {:3d} {:0.5f} {:0.5f}'.format(sweep, e, a))

pxx, pyy, pzz = p # diagonal components of pressure tensor

gamma = 0.5*(zhi-zlo)*(pzz - 0.5*(pxx+pyy)) # surface tension

stats = dict(energy=e, pxx=pxx, pyy=pyy, pzz=pzz, gamma=gamma, accrat=a)

if args.verbose > 1:
    test_energy()

# density profile in z direction

nbins = args.nbins
counts, edges = np.histogram(pos[:, 2], nbins, range=(0, es), density=False)
midpoint = 0.5*(edges[1:]+edges[:-1])
density = counts * nbins / (es**3) 

run_opts = [f'--header={args.header}', f'--seed={args.seed}',
            f'--nequil={nequil}', f'--nmove={nmove}',
            f'--A={A}', f'--npart={npart}', f'--es={es}']

if args.verbose > 1:
    print('opts:', ' '.join(run_opts))

if args.header is not None:

    dd, ff, ss = '{:d}', '{:0.8f}', '{:s}'

    stats_file = f'{args.header}__{pid:d}_stats.dat'
    fmt_string = '\t'.join([ff, ss]) + '\n'
    with open(stats_file, 'w') as f:
        for k in stats:
            f.write(fmt_string.format(stats[k], k))

    zprof_file = f'{args.header}__{pid:d}_zprof.dat'
    fmt_string = '\t'.join([ff, ff]) + '\n'
    with open(zprof_file, 'w') as f:
        for i in range(nbins):
            f.write(fmt_string.format(density[i], midpoint[i]))

    files = [stats_file, zprof_file]

    if args.process == 0:
        log_file = f'{args.header}.log'
        with open(log_file, 'w') as f:
            f.write('# opts: ' + ' '.join(run_opts) + '\n')
            f.write('# reduce data for: stats, zprof\n')
            f.write(f'# derived parameters: npart = {npart}, vol = {vol}, rho = {npart/vol}\n')

    if args.verbose:
        print('data >', ', '.join(files))

# end of code
