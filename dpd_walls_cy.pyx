# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False
#
# dpd_walls_cy.pyx  –  Cython-accelerated core for dpd_walls.py
#
# Compile with:
#   python setup_dpd_walls.py build_ext --inplace
# Then use dpd_walls_main.py as the driver.
#
# The simulation box is fully periodic in all three dimensions.
# Walls act only as an energy penalty / confinement check on the z-coordinate.
#
# Copyright (c) 2026 Patrick B Warren <patrick.warren@stfc.ac.uk>
# Licence: GPLv3+

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, exp, floor, pi

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
DTYPE_F = np.float64
DTYPE_I = np.int32
ctypedef np.float64_t F64
ctypedef np.int32_t   I32

# ---------------------------------------------------------------------------
# Minimum-image displacement (scalar, inline, no cdivision needed)
# ---------------------------------------------------------------------------
cdef inline double mic(double dx, double es, double esby2) nogil:
    """Wrap dx into (-esby2, +esby2]."""
    if dx >  esby2: return dx - es
    if dx < -esby2: return dx + es
    return dx

# ---------------------------------------------------------------------------
# Floor-based periodic wrap for a coordinate (always non-negative)
# ---------------------------------------------------------------------------
cdef inline double wrap(double x, double es) nogil:
    """Wrap x into [0, es) using floor division — safe for negative x."""
    return x - floor(x / es) * es

# ---------------------------------------------------------------------------
# Wall energy functions (both variants, inlined as C functions)
# ---------------------------------------------------------------------------
cdef inline double vanilla_wall_energy(double z,
                                        double zlo, double zhi,
                                        double Awall) nogil:
    cdef double zz
    if z < zlo + 1.0:
        zz = z - zlo
        return (Awall * 0.5) * (1.0 - zz) * (1.0 - zz)
    elif z > zhi - 1.0:
        zz = zhi - z
        return (Awall * 0.5) * (1.0 - zz) * (1.0 - zz)
    return 0.0

cdef inline double uniform_wall_energy(double z,
                                        double zlo, double zhi,
                                        double Awall) nogil:
    cdef double zz
    if z < zlo + 1.0:
        zz = z - zlo
        return (pi * Awall / 60.0) * (1.0 - zz)**4 * (2.0 + 3.0*zz)
    elif z > zhi - 1.0:
        zz = zhi - z
        return (pi * Awall / 60.0) * (1.0 - zz)**4 * (2.0 + 3.0*zz)
    return 0.0

# ---------------------------------------------------------------------------
# part_energy  — particle i's DPD interaction energy + wall energy
# ---------------------------------------------------------------------------
def part_energy(int i,
                np.ndarray[I32, ndim=1] cell not None,
                np.ndarray[F64, ndim=1] pos_i not None,
                np.ndarray[F64, ndim=2] pos not None,
                contents,
                list neighbours,
                int ncell,
                double cell_size,
                double es,
                double esby2,
                double A,
                double Awall,
                double zlo,
                double zhi,
                bint use_uniform):
    cdef:
        double energy = 0.0
        double dx, dy, dz, rsq, r
        double pix = pos_i[0], piy = pos_i[1], piz = pos_i[2]
        int cx = cell[0], cy = cell[1], cz = cell[2]
        int nx, ny, nz, j
        np.ndarray[I32, ndim=1] nb
        F64[:] pos_j

    # DPD pair interactions via cell list
    for nb in neighbours:
        nx = (cx + nb[0] + ncell) % ncell
        ny = (cy + nb[1] + ncell) % ncell
        nz = (cz + nb[2] + ncell) % ncell
        cell_set = contents[(nx, ny, nz)]
        for j in cell_set:
            if j == i:
                continue
            pos_j = pos[j]
            dx = mic(pos_j[0] - pix, es, esby2)
            dy = mic(pos_j[1] - piy, es, esby2)
            dz = mic(pos_j[2] - piz, es, esby2)
            rsq = dx*dx + dy*dy + dz*dz
            if rsq < 1.0:
                r = sqrt(rsq)
                energy += (A * 0.5) * (1.0 - r) * (1.0 - r)

    # Wall energy on z-coordinate only
    if use_uniform:
        energy += uniform_wall_energy(piz, zlo, zhi, Awall)
    else:
        energy += vanilla_wall_energy(piz, zlo, zhi, Awall)

    return energy


# ---------------------------------------------------------------------------
# energy_pressure  — full system energy and vector virial (3 components)
# ---------------------------------------------------------------------------
def energy_pressure(
        np.ndarray[F64, ndim=2] pos not None,
        contents,
        list neighbours,
        list box,
        int ncell,
        double cell_size,
        double es,
        double esby2,
        double A,
        double Awall,
        double zlo,
        double zhi,
        double vol,
        int npart,
        bint use_uniform):
    """
    Returns (e, (pxx, pyy, pzz)) matching the original energy_pressure().
    The virial is resolved into x, y, z components for the pressure tensor.
    """
    cdef:
        double energy = 0.0
        double vxx = 0.0, vyy = 0.0, vzz = 0.0
        double dx, dy, dz, rsq, r, fac
        double pix, piy, piz
        int cx, cy, cz, nx, ny, nz, i, j
        np.ndarray[I32, ndim=1] nb
        F64[:] pos_i, pos_j

    for i in box:
        pos_i = pos[i]
        pix = pos_i[0]; piy = pos_i[1]; piz = pos_i[2]
        cx = int(pix / cell_size) % ncell
        cy = int(piy / cell_size) % ncell
        cz = int(piz / cell_size) % ncell

        for nb in neighbours:
            nx = (cx + nb[0] + ncell) % ncell
            ny = (cy + nb[1] + ncell) % ncell
            nz = (cz + nb[2] + ncell) % ncell
            cell_set = contents[(nx, ny, nz)]
            for j in cell_set:
                if j <= i:
                    continue
                pos_j = pos[j]
                dx = mic(pos_j[0] - pix, es, esby2)
                dy = mic(pos_j[1] - piy, es, esby2)
                dz = mic(pos_j[2] - piz, es, esby2)
                rsq = dx*dx + dy*dy + dz*dz
                if rsq < 1.0:
                    r = sqrt(rsq)
                    energy += (A * 0.5) * (1.0 - r) * (1.0 - r)
                    # fac = A*(1-r)/r  so that virial_alpha = fac * d_alpha^2
                    fac = A * (1.0 - r) / r
                    vxx += fac * dx * dx
                    vyy += fac * dy * dy
                    vzz += fac * dz * dz

    cdef double e_out   = 3.0*npart/(2.0*vol) + energy/vol
    cdef double pxx_out = npart/vol + vxx/vol
    cdef double pyy_out = npart/vol + vyy/vol
    cdef double pzz_out = npart/vol + vzz/vol
    return e_out, (pxx_out, pyy_out, pzz_out)


# ---------------------------------------------------------------------------
# mc_sweep  — full Monte-Carlo sweep, walls-aware
# ---------------------------------------------------------------------------
def mc_sweep(
        np.ndarray[F64, ndim=2] pos not None,
        contents,
        list neighbours,
        int ncell,
        double cell_size,
        double es,
        double esby2,
        double A,
        double Awall,
        double zlo,
        double zhi,
        np.ndarray[I32, ndim=1] parts not None,
        np.ndarray[F64, ndim=2] disps not None,
        np.ndarray[F64, ndim=1] probs not None,
        int nmove,
        bint use_uniform):
    """
    Execute one full MC sweep of nmove trial moves.
    Returns the number of accepted moves.

    Moves that would place the particle outside [zlo, zhi] are rejected
    before energy evaluation, exactly as in the original Python code.
    """
    cdef:
        int naccept = 0
        int i, j, m
        int old_cx, old_cy, old_cz
        int new_cx, new_cy, new_cz
        int nx, ny, nz
        double old_px, old_py, old_pz
        double new_px, new_py, new_pz
        double dx, dy, dz, rsq, r
        double old_energy, new_energy, delta_e
        double wall_old, wall_new
        np.ndarray[I32, ndim=1] nb
        F64[:] pos_j

    for m in range(nmove):
        i = parts[m]

        # ---- current state ------------------------------------------------
        old_px = pos[i, 0];  old_py = pos[i, 1];  old_pz = pos[i, 2]
        old_cx = int(old_px / cell_size) % ncell
        old_cy = int(old_py / cell_size) % ncell
        old_cz = int(old_pz / cell_size) % ncell

        # ---- trial position -----------------------------------------------
        # x, y: fully periodic wrap using floor (safe for negative values)
        # z: periodic wrap then reject if outside wall bounds
        new_px = wrap(old_px + disps[m, 0], es)
        new_py = wrap(old_py + disps[m, 1], es)
        new_pz = wrap(old_pz + disps[m, 2], es)

        if new_pz < zlo or new_pz > zhi:
            continue   # reject — outside wall confinement

        new_cx = int(new_px / cell_size) % ncell
        new_cy = int(new_py / cell_size) % ncell
        new_cz = int(new_pz / cell_size) % ncell

        # ---- old DPD pair energy ------------------------------------------
        old_energy = 0.0
        for nb in neighbours:
            nx = (old_cx + nb[0] + ncell) % ncell
            ny = (old_cy + nb[1] + ncell) % ncell
            nz = (old_cz + nb[2] + ncell) % ncell
            cell_set = contents[(nx, ny, nz)]
            for j in cell_set:
                if j == i:
                    continue
                pos_j = pos[j]
                dx = mic(pos_j[0] - old_px, es, esby2)
                dy = mic(pos_j[1] - old_py, es, esby2)
                dz = mic(pos_j[2] - old_pz, es, esby2)
                rsq = dx*dx + dy*dy + dz*dz
                if rsq < 1.0:
                    r = sqrt(rsq)
                    old_energy += (A * 0.5) * (1.0 - r) * (1.0 - r)

        # ---- old wall energy ----------------------------------------------
        if use_uniform:
            old_energy += uniform_wall_energy(old_pz, zlo, zhi, Awall)
        else:
            old_energy += vanilla_wall_energy(old_pz, zlo, zhi, Awall)

        # ---- new DPD pair energy ------------------------------------------
        new_energy = 0.0
        for nb in neighbours:
            nx = (new_cx + nb[0] + ncell) % ncell
            ny = (new_cy + nb[1] + ncell) % ncell
            nz = (new_cz + nb[2] + ncell) % ncell
            cell_set = contents[(nx, ny, nz)]
            for j in cell_set:
                if j == i:
                    continue
                pos_j = pos[j]
                dx = mic(pos_j[0] - new_px, es, esby2)
                dy = mic(pos_j[1] - new_py, es, esby2)
                dz = mic(pos_j[2] - new_pz, es, esby2)
                rsq = dx*dx + dy*dy + dz*dz
                if rsq < 1.0:
                    r = sqrt(rsq)
                    new_energy += (A * 0.5) * (1.0 - r) * (1.0 - r)

        # ---- new wall energy ----------------------------------------------
        if use_uniform:
            new_energy += uniform_wall_energy(new_pz, zlo, zhi, Awall)
        else:
            new_energy += vanilla_wall_energy(new_pz, zlo, zhi, Awall)

        # ---- Metropolis acceptance ----------------------------------------
        delta_e = new_energy - old_energy
        if probs[m] < exp(-delta_e):
            naccept += 1
            pos[i, 0] = new_px
            pos[i, 1] = new_py
            pos[i, 2] = new_pz
            contents[(old_cx, old_cy, old_cz)].discard(i)
            contents[(new_cx, new_cy, new_cz)].add(i)

    return naccept
