#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# dpd_walls_main.py  –  Driver for the Cython-accelerated DPD walls simulation.
#
# Drop-in replacement for dpd_walls.py.  All argument parsing, initialisation,
# output and file I/O remain in pure Python; every hot inner loop is delegated
# to dpd_walls_cy (the compiled Cython extension).
#
# Usage is identical to dpd_walls.py, e.g.:
#   python dpd_walls_main.py --seed=12345 --es=6 --nequil=20 -v -w
#
# Before first use, compile the Cython module:
#   python setup_dpd_walls.py build_ext --inplace
#
# Copyright (c) 2026 Patrick B Warren <patrick.warren@stfc.ac.uk>
# Licence: GPLv3+

import argparse
import numpy as np
from numpy import pi as π
from itertools import product

# ---- Import Cython module -------------------------------------------------
try:
    import dpd_walls_cy as cy
except ImportError:
    raise ImportError(
        "Cython extension not found.\n"
        "Please compile it first with:\n"
        "    python setup_dpd_walls.py build_ext --inplace"
    )

# ---------------------------------------------------------------------------
# Argument parsing  (identical to original)
# ---------------------------------------------------------------------------

def eval_kM_replace(s):
    return eval('int({})'.format(s.replace('k', '*1e3').replace('M', '*1e6')))

class ExtendedArgumentParser(argparse.ArgumentParser):
    def add_bool_arg(self, long_opt, short_opt=None, default=False, help=None):
        opt = long_opt.removeprefix('--')
        group = self.add_mutually_exclusive_group(required=False)
        help_string = None if not help else help if not default else f'{help}, default'
        if short_opt:
            group.add_argument(short_opt, f'--{opt}', dest=opt, action='store_true', help=help_string)
        else:
            group.add_argument(f'--{opt}', dest=opt, action='store_true', help=help_string)
        help_string = None if not help else f"don't {help}" if default else f"don't {help}, default"
        group.add_argument(f'--no-{opt}', dest=opt, action='store_false', help=help_string)
        self.set_defaults(**{opt: default})

parser = ExtendedArgumentParser(description=__doc__)
parser.add_argument('--header', default=None)
parser.add_argument('--seed', default=12345, type=int)
parser.add_argument('--process', default=0, type=int)
parser.add_argument('--njobs', default=1, type=int)
parser.add_argument('-e', '--es', default=10.0, type=float)
parser.add_argument('-r', '--rho', default=3.0, type=float)
parser.add_argument('-A', '--A', default=25.0, type=float)
parser.add_argument('--Awall', default='0.0')
parser.add_argument('--npart', default='rho*vol')
parser.add_argument('--nmove', default='npart')
parser.add_argument('--nequil', default='10')
parser.add_argument('--delta', default=0.2, type=float)
parser.add_argument('--nbins', default=80, type=int)
parser.add_bool_arg('--walls', short_opt='-w', default=True, help='include walls')
parser.add_bool_arg('--uniform', short_opt='-u', default=False, help='uniform wall model')
parser.add_argument('-v', '--verbose', action='count', default=0)
args = parser.parse_args()

pid, njobs = args.process, args.njobs
rng = np.random.default_rng(seed=args.seed).spawn(njobs)[pid]

A, rho, ΔR = args.A, args.rho, args.delta
Awall = eval(args.Awall)

es, esby2 = args.es, args.es / 2
zlo, zhi = (0.5, es - 0.5) if args.walls else (0.0, es)
vol = es**2 * (zhi - zlo)

npart = eval_kM_replace(args.npart)
nmove = eval_kM_replace(args.nmove)
nequil = eval_kM_replace(args.nequil)

use_uniform = bool(args.uniform)  # passed as bint to Cython

# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

pos = np.ascontiguousarray(
    rng.uniform(zlo, zhi, size=(npart, 3)), dtype=np.float64
)
# x and y can span the full box; only z is constrained to [zlo, zhi]
pos[:, 0] = rng.uniform(0, es, size=npart)
pos[:, 1] = rng.uniform(0, es, size=npart)

ncell     = int(es)
cell_size = es / ncell

cell_coord = list(range(ncell))
all_cells  = product(cell_coord, cell_coord, cell_coord)
contents   = {cell: set() for cell in all_cells}

box = list(range(npart))

for i in box:
    cell = tuple((pos[i] / cell_size).astype(int))
    contents[cell].add(i)

neighbour_offsets = list(product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]))
neighbours        = [np.array(x, dtype=np.int32) for x in neighbour_offsets]

# ---------------------------------------------------------------------------
# Pure-Python reference functions (brute_force, test_energy) — unchanged
# ---------------------------------------------------------------------------

def brute_force():
    energy, virial = 0, 0
    for i in box:
        for j in box:
            if i < j:
                Δr  = pos[j] - pos[i]
                Δr -= np.where(Δr >  esby2, es, 0)
                Δr += np.where(Δr < -esby2, es, 0)
                rsq = np.sum(Δr**2)
                if rsq < 1:
                    r       = np.sqrt(rsq)
                    energy += (A/2)*(1-r)**2
                    virial += A*r*(1-r)
    return 3*npart/(2*vol) + energy/vol, npart/vol + virial/(3*vol)

# Python-level wall energy (used only in test_energy / tot_part_energy)
def vanilla_wall_energy(z):
    if z < zlo + 1:
        zz = z - zlo
        return (Awall/2)*(1-zz)**2
    elif z > zhi - 1:
        zz = zhi - z
        return (Awall/2)*(1-zz)**2
    return 0.0

def uniform_wall_energy(z):
    if z < zlo + 1:
        zz = z - zlo
        return (π*Awall/60)*(1-zz)**4*(2+3*zz)
    elif z > zhi - 1:
        zz = zhi - z
        return (π*Awall/60)*(1-zz)**4*(2+3*zz)
    return 0.0

wall_energy = uniform_wall_energy if args.uniform else vanilla_wall_energy

def part_energy_py(i, cell, pos_i):
    """Pure-Python part_energy — used only in tot_part_energy for testing."""
    energy = 0
    for neighbour in neighbours:
        neighbour_cell = tuple(((np.array(cell) + neighbour + ncell) % ncell))
        for j in contents[neighbour_cell]:
            if i != j:
                Δr  = pos[j] - pos_i
                Δr -= np.where(Δr >  esby2, es, 0)
                Δr += np.where(Δr < -esby2, es, 0)
                rsq = np.sum(Δr**2)
                if rsq < 1:
                    r = np.sqrt(rsq)
                    energy += (A/2)*(1-r)**2
    return energy

def tot_part_energy():
    energy = 3*npart/2
    for i in box:
        cell = (pos[i]/cell_size).astype(int)
        energy += 0.5*(part_energy_py(i, cell, pos[i]) + wall_energy(pos[i][2]))
    return energy/vol

def test_energy():
    e, p = cy.energy_pressure(
        pos, contents, neighbours, box,
        ncell, cell_size, es, esby2,
        A, Awall, zlo, zhi, vol, npart, use_uniform
    )
    print('cell list methods  =\t{}\t{}'.format(e, np.mean(p)))
    print('brute force method =\t{}\t{}'.format(*brute_force()))
    print('tot part energy    =\t{}'.format(tot_part_energy()))

if args.verbose > 1:
    test_energy()

# ---------------------------------------------------------------------------
# Monte-Carlo equilibration  (hot loop in Cython)
# ---------------------------------------------------------------------------

for sweep in range(nequil):
    parts = np.ascontiguousarray(
        rng.integers(0, npart, size=nmove), dtype=np.int32
    )
    disps = np.ascontiguousarray(
        rng.normal(0.0, ΔR, size=(nmove, 3)), dtype=np.float64
    )
    probs = np.ascontiguousarray(
        rng.random(size=nmove), dtype=np.float64
    )

    naccept = cy.mc_sweep(
        pos, contents, neighbours,
        ncell, cell_size, es, esby2,
        A, Awall, zlo, zhi,
        parts, disps, probs, nmove,
        use_uniform
    )

    e, p = cy.energy_pressure(
        pos, contents, neighbours, box,
        ncell, cell_size, es, esby2,
        A, Awall, zlo, zhi, vol, npart, use_uniform
    )
    a = naccept / nmove

    if args.verbose:
        print('equilibration: {:3d} {:0.5f} {:0.5f} {:0.5f}'.format(
            sweep, e, float(np.mean(p)), a))

pxx, pyy, pzz = p
gamma = 0.5*(zhi - zlo)*(pzz - 0.5*(pxx + pyy))
stats = dict(energy=e, pxx=pxx, pyy=pyy, pzz=pzz, gamma=gamma, accrat=a)

if args.verbose > 1:
    test_energy()

# ---------------------------------------------------------------------------
# Density profile in z direction  (unchanged)
# ---------------------------------------------------------------------------

nbins = args.nbins
counts, edges = np.histogram(pos[:, 2], nbins, range=(0, es), density=False)
midpoint = 0.5*(edges[1:] + edges[:-1])
density  = counts * nbins / (es**3)

# ---------------------------------------------------------------------------
# Output  (identical to original)
# ---------------------------------------------------------------------------

run_opts = [f'--header={args.header}', f'--seed={args.seed}',
            f'--nequil={nequil}', f'--nmove={nmove}',
            f'--A={A}', f'--npart={npart}', f'--es={es}']

if args.verbose > 1:
    print('opts:', ' '.join(run_opts))

if args.header is not None:

    ff, ss = '{:0.8f}', '{:s}'

    stats_file = f'{args.header}__{pid:d}_stats.dat'
    with open(stats_file, 'w') as f:
        fmt = '\t'.join([ff, ss]) + '\n'
        for k in stats:
            f.write(fmt.format(stats[k], k))

    zprof_file = f'{args.header}__{pid:d}_zprof.dat'
    with open(zprof_file, 'w') as f:
        fmt = '\t'.join([ff, ff]) + '\n'
        for i in range(nbins):
            f.write(fmt.format(density[i], midpoint[i]))

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
