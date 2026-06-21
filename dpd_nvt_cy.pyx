# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=False
# cython: nonecheck=False
# cython: initializedcheck=False
#
# dpd_nvt_cy.pyx  –  Cython-accelerated core for dpd_nvt.py
#
# Compile with:
#   python setup_dpd.py build_ext --inplace
# Then use dpd_nvt_main.py as the driver instead of dpd_nvt.py.
#
# Copyright (c) 2026 Patrick B Warren <patrick.warren@stfc.ac.uk>
# Licence: GPLv3+

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, exp, fabs, floor
from libc.stdlib cimport malloc, free

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
DTYPE_F = np.float64
DTYPE_I = np.int32
ctypedef np.float64_t F64
ctypedef np.int32_t  I32

# ---------------------------------------------------------------------------
# Minimum-image displacement (scalar, inline)
# ---------------------------------------------------------------------------
cdef inline double mic(double dx, double es, double esby2) nogil:
    """Return dx wrapped into (-esby2, +esby2]."""
    if dx >  esby2: return dx - es
    if dx < -esby2: return dx + es
    return dx

# ---------------------------------------------------------------------------
# Floor-based periodic wrap for a coordinate (always non-negative)
# ---------------------------------------------------------------------------
cdef inline double wrap(double x, double es) nogil:
    """Wrap x into [0, es) using floor division — safe for negative x.
    Used in preference to the % operator: with cdivision=False Cython's %
    already matches Python semantics (non-negative for positive divisor),
    but this explicit form is kept for clarity and consistency with the
    dpd_walls_cy module, and is robust even if cdivision is ever toggled."""
    return x - floor(x / es) * es

# ---------------------------------------------------------------------------
# Cell-list helpers
# ---------------------------------------------------------------------------
# We store cell contents as a flat dict[tuple[int,int,int], set] exactly as
# the original Python code does, so the calling driver can share the same
# data structure.  The hot inner loop reads the dict in Cython.

# ---------------------------------------------------------------------------
# part_energy  (the single hottest function – called 2 × nmove times)
# ---------------------------------------------------------------------------
def part_energy(int i,
                np.ndarray[I32, ndim=1] cell not None,
                np.ndarray[F64, ndim=1] pos_i not None,
                np.ndarray[F64, ndim=2] pos not None,
                contents,          # dict[(cx,cy,cz)] -> set of particle ids
                list neighbours,   # list of 27 numpy int arrays, shape (3,)
                int ncell,
                double cell_size,
                double es,
                double esby2,
                double A):
    """
    Compute the DPD potential energy of particle i at position pos_i.

    Parameters match the original Python signature but all array arguments
    are typed so Cython generates pure-C inner loops with no Python overhead
    on the arithmetic.
    """
    cdef:
        double energy = 0.0
        double dx, dy, dz, rsq, r
        double pix = pos_i[0], piy = pos_i[1], piz = pos_i[2]
        int cx = cell[0], cy = cell[1], cz = cell[2]
        int nx, ny, nz, j
        np.ndarray[I32, ndim=1] nb
        F64[:]  pos_j

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

    return energy


# ---------------------------------------------------------------------------
# energy_pressure_mean_wld  (second most expensive – full pair loop)
# ---------------------------------------------------------------------------
def energy_pressure_mean_wld(
        np.ndarray[F64, ndim=2] pos not None,
        np.ndarray[F64, ndim=1] wld not None,
        contents,
        list neighbours,
        list box,
        int ncell,
        double cell_size,
        double es,
        double esby2,
        double A,
        int npart,
        double vol):
    """
    Compute total energy, pressure and mean weighted local density using the
    cell list.  Populates *wld* in-place as a side effect.
    """
    cdef:
        double energy = 0.0, virial = 0.0
        double dx, dy, dz, rsq, r, wgt
        double pix, piy, piz
        int cx, cy, cz, nx, ny, nz, i, j
        np.ndarray[I32, ndim=1] nb
        np.ndarray[I32, ndim=1] cell_arr = np.empty(3, dtype=DTYPE_I)
        F64[:] pos_i, pos_j

    wld[:] = 0.0

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
                if j <= i:          # count each pair once (i < j)
                    continue
                pos_j = pos[j]
                dx = mic(pos_j[0] - pix, es, esby2)
                dy = mic(pos_j[1] - piy, es, esby2)
                dz = mic(pos_j[2] - piz, es, esby2)
                rsq = dx*dx + dy*dy + dz*dz
                if rsq < 1.0:
                    r = sqrt(rsq)
                    energy += (A * 0.5) * (1.0 - r) * (1.0 - r)
                    virial += A * r * (1.0 - r)
                    wgt = 0.25 * (1.0 - r) * (1.0 - r)
                    wld[i] += wgt
                    wld[j] += wgt

    cdef double e_out = 3.0*npart/(2.0*vol) + energy/vol
    cdef double p_out = npart/vol + virial/(3.0*vol)
    cdef double w_out = np.mean(wld)
    return e_out, p_out, w_out


# ---------------------------------------------------------------------------
# mc_sweep  (the main Monte-Carlo loop – entire sweep in one C-level call)
# ---------------------------------------------------------------------------
def mc_sweep(
        np.ndarray[F64, ndim=2] pos not None,
        np.ndarray[F64, ndim=1] wld not None,
        contents,
        list neighbours,
        int ncell,
        double cell_size,
        double es,
        double esby2,
        double A,
        np.ndarray[I32, ndim=1] parts not None,
        np.ndarray[F64, ndim=2] disps not None,
        np.ndarray[F64, ndim=1] probs not None,
        int nmove):
    """
    Execute one full MC sweep of *nmove* trial moves.

    Returns the number of accepted moves.

    All RNG arrays (parts, disps, probs) are pre-drawn by the caller using
    numpy, exactly as in the original code.
    """
    cdef:
        int naccept = 0
        int i, k, m
        int old_cx, old_cy, old_cz
        int new_cx, new_cy, new_cz
        int nx, ny, nz
        double old_px, old_py, old_pz
        double new_px, new_py, new_pz
        double dx, dy, dz, rsq, r
        double old_energy, new_energy, delta_e
        np.ndarray[I32, ndim=1] nb
        F64[:] pos_j

    for m in range(nmove):
        i = parts[m]

        # ---- current state ------------------------------------------------
        old_px = pos[i, 0];  old_py = pos[i, 1];  old_pz = pos[i, 2]
        old_cx = int(old_px / cell_size) % ncell
        old_cy = int(old_py / cell_size) % ncell
        old_cz = int(old_pz / cell_size) % ncell

        # ---- energy of particle i at old position -------------------------
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

        # ---- trial position (periodic wrap) -------------------------------
        new_px = wrap(old_px + disps[m, 0], es)
        new_py = wrap(old_py + disps[m, 1], es)
        new_pz = wrap(old_pz + disps[m, 2], es)
        new_cx = int(new_px / cell_size) % ncell
        new_cy = int(new_py / cell_size) % ncell
        new_cz = int(new_pz / cell_size) % ncell

        # ---- energy of particle i at new position -------------------------
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

        # ---- Metropolis acceptance criterion ------------------------------
        delta_e = new_energy - old_energy
        if probs[m] < exp(-delta_e):
            naccept += 1
            pos[i, 0] = new_px
            pos[i, 1] = new_py
            pos[i, 2] = new_pz
            contents[(old_cx, old_cy, old_cz)].discard(i)
            contents[(new_cx, new_cy, new_cz)].add(i)

    return naccept


# ---------------------------------------------------------------------------
# rdf_accumulate  (pair-distribution function accumulation loop)
# ---------------------------------------------------------------------------
def rdf_accumulate(
        np.ndarray[F64, ndim=2] pos not None,
        np.ndarray[F64, ndim=1] wld not None,
        np.ndarray[I32, ndim=1] count not None,
        np.ndarray[F64, ndim=1] sumnr not None,
        list box,
        double es,
        double esby2,
        double delta_g,
        int nbins):
    """
    Accumulate pair counts and wld-weighted sums into *count* and *sumnr*.

    Uses an O(N²) loop (same as the original) since rdf accumulation is
    called only once after equilibration.
    """
    cdef:
        int i, j, k, n = len(box)
        double dx, dy, dz, r
        F64[:] pi, pj

    for i in range(n):
        pi = pos[i]
        for j in range(i+1, n):
            pj = pos[j]
            dx = mic(pj[0] - pi[0], es, esby2)
            dy = mic(pj[1] - pi[1], es, esby2)
            dz = mic(pj[2] - pi[2], es, esby2)
            r = sqrt(dx*dx + dy*dy + dz*dz)
            k = int(r / delta_g)
            if k < nbins:
                count[k] += 1
                sumnr[k] += 0.5 * (wld[i] + wld[j])
