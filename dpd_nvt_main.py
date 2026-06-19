#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# dpd_nvt_main.py  –  Driver for the Cython-accelerated DPD NVT simulation.
#
# This file is a drop-in replacement for dpd_nvt.py.  It keeps all argument
# parsing, initialisation, output formatting and file I/O in pure Python
# (these are not performance-critical), while delegating every hot inner loop
# to dpd_nvt_cy (the compiled Cython extension).
#
# Usage is identical to dpd_nvt.py, e.g.:
#
#   python dpd_nvt_main.py --seed=12345 --es=6 --nequil=20 -v
#
# Before first use, compile the Cython module:
#
#   python setup_dpd.py build_ext --inplace
#
# Copyright (c) 2026 Patrick B Warren <patrick.warren@stfc.ac.uk>
# Licence: GPLv3+

import argparse
import numpy as np
from itertools import product

# ---- Import Cython module -------------------------------------------------
try:
    import dpd_nvt_cy as cy
except ImportError:
    raise ImportError(
        "Cython extension not found.\n"
        "Please compile it first with:\n"
        "    python setup_dpd.py build_ext --inplace"
    )

# ---------------------------------------------------------------------------
# Argument parsing  (identical to original)
# ---------------------------------------------------------------------------

def eval_kM_replace(s):
    return eval('int({})'.format(s.replace('k', '*1e3').replace('M', '*1e6')))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--header', default=None)
parser.add_argument('--seed', default=12345, type=int)
parser.add_argument('--process', default=0, type=int)
parser.add_argument('--njobs', default=1, type=int)
parser.add_argument('-e', '--es', default=10.0, type=float)
parser.add_argument('-r', '--rho', default=3.0, type=float)
parser.add_argument('-A', '--A', default=25.0, type=float)
parser.add_argument('--npart', default='rho*vol')
parser.add_argument('--nmove', default='npart')
parser.add_argument('--nequil', default='10')
parser.add_argument('--delta', default=0.2, type=float)
parser.add_argument('--rmax', default=4.0, type=float)
parser.add_argument('--nbins', default=80, type=int)
parser.add_argument('-v', '--verbose', action='count', default=0)
args = parser.parse_args()

pid, njobs = args.process, args.njobs
rng = np.random.default_rng(seed=args.seed).spawn(njobs)[pid]

A, rho, ΔR = args.A, args.rho, args.delta
es, esby2, vol = args.es, args.es / 2, args.es ** 3

npart = eval_kM_replace(args.npart)
nmove = eval_kM_replace(args.nmove)
nequil = eval_kM_replace(args.nequil)

# ---------------------------------------------------------------------------
# Initialisation  (identical to original)
# ---------------------------------------------------------------------------

pos = np.ascontiguousarray(
    rng.uniform(0, es, size=(npart, 3)), dtype=np.float64
)
wld = np.zeros(npart, dtype=np.float64)

ncell     = int(es)
cell_size = es / ncell

# Cell-list dict  (Python sets, same structure as the original)
cell_coord        = list(range(ncell))
all_cells         = product(cell_coord, cell_coord, cell_coord)
contents          = {cell: set() for cell in all_cells}

box = list(range(npart))

for i in box:
    cell = tuple((pos[i] / cell_size).astype(int))
    contents[cell].add(i)

# Neighbour offset list  (27 entries, stored as int32 arrays for Cython)
neighbour_offsets = list(product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]))
neighbours        = [np.array(x, dtype=np.int32) for x in neighbour_offsets]

# ---------------------------------------------------------------------------
# Helper: build cell array for a position  (used in Python-level calls)
# ---------------------------------------------------------------------------

def cell_of(pos_i):
    return (pos_i / cell_size).astype(np.int32)

# ---------------------------------------------------------------------------
# Wrapped calls to Cython kernels (mirroring original function signatures)
# ---------------------------------------------------------------------------

def energy_pressure_mean_wld():
    return cy.energy_pressure_mean_wld(
        pos, wld, contents, neighbours, box,
        ncell, cell_size, es, esby2, A, npart, vol
    )

def part_energy(i, cell, pos_i):
    return cy.part_energy(
        i,
        np.asarray(cell, dtype=np.int32),
        np.asarray(pos_i, dtype=np.float64),
        pos, contents, neighbours,
        ncell, cell_size, es, esby2, A
    )

def tot_part_energy():
    energy = 3 * npart / 2
    for i in box:
        cell = cell_of(pos[i])
        energy += 0.5 * part_energy(i, cell, pos[i])
    return energy / vol

# ---------------------------------------------------------------------------
# Brute-force reference  (unchanged from original – not performance-critical)
# ---------------------------------------------------------------------------

def brute_force():
    energy, virial, wgt = 0, 0, 0
    for i in box:
        for j in box:
            if i < j:
                Δr  = pos[j] - pos[i]
                Δr -= np.where(Δr >  esby2, es, 0)
                Δr += np.where(Δr < -esby2, es, 0)
                rsq = np.sum(Δr ** 2)
                if rsq < 1:
                    r      = np.sqrt(rsq)
                    energy += (A / 2) * (1 - r) ** 2
                    virial += A * r * (1 - r)
                    wgt    += 0.25 * (1 - r) ** 2
    return (3*npart/(2*vol) + energy/vol,
            npart/vol + virial/(3*vol),
            2*wgt/npart)

def test_energy():
    print('cell list methods  =\t{}\t{}\t{}'.format(*energy_pressure_mean_wld()))
    print('brute force method =\t{}\t{}\t{}'.format(*brute_force()))
    print('tot part energy    =\t{}'.format(tot_part_energy()))

if args.verbose > 1:
    test_energy()

# ---------------------------------------------------------------------------
# Monte-Carlo equilibration  (hot loop delegated to Cython)
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
        pos, wld, contents, neighbours,
        ncell, cell_size, es, esby2, A,
        parts, disps, probs, nmove
    )

    (e, p, w), a = energy_pressure_mean_wld(), naccept / nmove
    if args.verbose:
        print('equilibration: {:3d} {:0.5f} {:0.5f}'.format(sweep, e, a))

stats = dict(energy=e, pressure=p, wmean=w, accrat=a)

if args.verbose > 1:
    test_energy()

# ---------------------------------------------------------------------------
# Widom insertion for chemical potential  (unchanged)
# ---------------------------------------------------------------------------

def even_parity(cell):
    return (cell[0] + cell[1] + cell[2]) % 2 == 0

origin_pos  = rng.uniform(0, 1, size=3)
insert_eng  = np.array([
    part_energy(npart, np.array(cell, dtype=np.int32), origin_pos + np.array(cell))
    for cell in contents if even_parity(cell)
])
mu = -np.log(vol * np.mean(np.exp(-insert_eng)) / (npart + 1))

stats['mu'] = mu

# ---------------------------------------------------------------------------
# Pair distribution function  (hot loop delegated to Cython)
# ---------------------------------------------------------------------------

nbins, Δg = args.nbins, args.rmax / args.nbins

count = np.zeros(nbins, dtype=np.int32)
sumnr = np.zeros(nbins, dtype=np.float64)

i_arr  = np.arange(nbins)
rmid   = (i_arr + 0.5) * Δg
vshell = 4 * np.pi / 3 * (3*i_arr**2 + 3*i_arr + 1) * Δg**3

cy.rdf_accumulate(pos, wld, count, sumnr, box, es, esby2, Δg, nbins)

npairs = npart * (npart - 1) // 2
gr     = count * vol / (npairs * vshell)

# ---------------------------------------------------------------------------
# Output  (identical to original)
# ---------------------------------------------------------------------------

run_opts = [f'--header={args.header}', f'--seed={args.seed}',
            f'--nequil={nequil}', f'--nmove={nmove}',
            f'--A={A}', f'--npart={npart}', f'--es={es}']

if args.verbose > 1:
    print('opts:', ' '.join(run_opts))

if args.header is not None:

    dd, ff, ss = '{:d}', '{:0.8f}', '{:s}'

    stats_file = f'{args.header}__{pid:d}_stats.dat'
    with open(stats_file, 'w') as f:
        fmt = '\t'.join([ff, ss]) + '\n'
        for k in stats:
            f.write(fmt.format(stats[k], k))

    rdfs_file = f'{args.header}__{pid:d}_rdfs.dat'
    with open(rdfs_file, 'w') as f:
        fmt = '\t'.join([dd, dd, ff, dd, ff, ff]) + '\n'
        for i in range(nbins):
            f.write(fmt.format(pid, i, rmid[i], count[i], sumnr[i], gr[i]))

    files = [stats_file, rdfs_file]
    concats = ['rdfs']

    if args.process == 0:
        log_file = f'{args.header}.log'
        with open(log_file, 'w') as f:
            f.write('# opts: ' + ' '.join(run_opts) + '\n')
            f.write('# reduce data for: stats\n')
            f.write('# concatenate data for: ' + ', '.join(concats) + '\n')
            f.write(f'# derived parameters: npart = {npart}, vol = {vol}, rho = {npart/vol}\n')

    if args.verbose:
        print('data >', ', '.join(files))

# end of code
