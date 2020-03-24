# -*- coding: utf-8 -*-
""" optical lindblad master equation solver.

    :author: keksnicoh
"""
import numpy as np
from .util import vectorize, unvectorize, ketbra, thermal_dist, sqmat, eigh, is_H
from . import settings

DTYPE_JUMP = np.dtype([
    ('I', settings.DTYPE_INT, 2),
    ('d', settings.DTYPE_COMPLEX),
    ('w', settings.DTYPE_FLOAT),
    ('n', settings.DTYPE_FLOAT),
])

class ReducedSystem():
    """ classification of a reduced system in an optical environment.

        this contains the most basic information to compile a kernel
        which then performs integration and manages runtime state.
    """

    @classmethod
    def from_dipole_eb(cls, h0, dipole_eb):
        """ create HeatBathCoupling from a **dipole_operator**
            in energy eigenbase. Therefore, the (ij) component
            of the operator represents the jump energy-state j to i.

            Arguments:
            ----------

            :h0: MxM matrix representing the stationary reduced
                systems
                hamiltonian

            :dipole_eb: MxM matrix representing the dipole transitions
                in energy eigenbase.

            """
        pass


    def __init__(self, h0, dipole=None):
        # normalize
        h0 = sqmat(h0).astype(settings.DTYPE_COMPLEX)
        dipole = None if dipole is None else sqmat(dipole)

        # validate
        if not is_H(h0):
            raise ValueError('arg h0 must be hermitian.')

        if dipole is not None:
            if not is_H(dipole):
                raise ValueError('dipole transition moment must be hermitian')

            if not np.all(dipole.shape == h0.shape):
                raise ValueError('dipole transition moment must match h0')

        self.h0 = h0
        self.dipole = dipole
        self.ev, self.s = eigh(h0)
        self.sarr = np.array(self.s)
        self.dimH = self.h0.shape[0]
        self.tw = None


    def create_rs_dipole_ladder(self):
        """
        XXX Test this
        """
        dip_eb = self.dipole_eb()
        r = np.arange(0, self.dimH-1)
        ldip_eb = np.zeros_like(dip_eb)
        ldip_eb[r, r + 1] = dip_eb[r, r + 1]
        ldip_eb[r + 1, r] = dip_eb[r + 1, r]
        return ReducedSystem(self.h0, self.eb2op(ldip_eb))


    def get_jumps(self):
        """ returns jumping configuration for the system
            """
        jumps = [None for _ in range(self.h0.shape[0])]

        for w in self.get_possible_tw():
            tr = self.get_transitions(w)
            i = len(tr) - 1
            jumps[i] = tr if jumps[i] is None else np.concatenate((jumps[i], tr))

        return jumps


    def get_possible_tw(self):
        """ returns all possible transition frequencies
            bewteen the energy eigenstates of the system.
        """
        if self.tw is not None:
            return self.tw
        ev = self.ev
        f = np.array([np.abs(a - b) for a in ev for b in ev if not np.isclose(a, b)])
        return sorted(f[~(np.triu(np.abs(f[:, None] - f) <= settings.EQ_COMPARE_TOL, 1)).any(0)])


    def get_transitions(self, w):
        """ returns an array of `DTYPE_JUMP` describing the
            possible transitions at frequency `w`.
            """
        return np.array([((i, j), self.dij(j, i), np.abs(e1 - e2), 0)
                         for j, e1 in enumerate(self.ev)
                         for i, e2 in enumerate(self.ev)
                         if np.isclose(e1 - e2, w)], dtype=DTYPE_JUMP)


    def dij(self, i, j):
        """ dipole transition moment component (i, j) in eigenbase.
            if no dipole defined, then all components with i != j are 1.
            """
        if self.dipole is None:
            return 1 if i != j else 0

        # <s_i|D|s_j>
        return (self.s[j:j+1] @ self.dipole @ self.s[i:i+1].T.conj())[0, 0]


    def dipole_eb(self):
        """ returns dipole operator in energy eigenbase.
            """
        d = np.zeros_like(self.h0)

        for i in range(self.h0.shape[0]):
            for j in range(self.h0.shape[0]):
                d[i, j] = self.dij(i, j)

        return d


    def jump_operators_eb(self, list_empty=False):
        """ returns all jump operators in given base
            """
        aw = []
        for k, jumps in enumerate(self.get_jumps()):
            if jumps is None:
                continue

            if k == 0:
                for t in jumps:
                    aw.append((t['w'], 1, t['d'] * ketbra(self.s, *t['I'])))
            else:
                for jump in jumps.reshape((int(len(jumps) / (k + 1)), k + 1)):
                    op = sum(t['d'] * ketbra(self.s, *t['I']) for t in jump)
                    aw.append((jump[0]['w'], k + 1, op))

        return [a for a in aw if list_empty or not np.allclose(a[2], 0)]


    def thermal_state(self, T):
        """ thermal state at temperature ``T``.

            Arguments:
            ----------

            :T: floating point temperature or an array of temperatures.

            Example:
            --------

            ```python

            rho_at_t44 = rs.thermal_state(4.4)
            rho = rs.thermal_state([0, 1, 2, 3])

            ```
            """
        #can = lambda t: enumerate(thermal_dist(self.ev, t))
        #rho = lambda t: sum(p * ketbra(self.s, i) for (i, p) in can(t))
        #vpack(T, rho(t) for t in vunpack(T))

        return unvectorize([
            sum(
                p * ketbra(self.s, i)
                for (i, p) in enumerate(thermal_dist(self.ev, t))
            ) for t in vectorize(T)
        ])


    def pure_energy_state(self, i):
        """ creates a state |i><i| where |i> is
            the ``i``-th energy eigenstate. If i defines
            many states then a list of pure
            energy states is returned.
        """

        #vpack(i, ketbra(self.s, j, j) for j in vunpack(i))

        return unvectorize(
            ketbra(self.s, i, i)
            for i in vectorize(i)
        )


    def op2eb(self, op):
        """ transforms `op` into eigenbase.
            `op` must be ndarray of shape `(M,M)` or `(N,M,M)`
            """
        return self.sarr @ op @ self.sarr.conj().T


    def eb2op(self, op):
        """ transforms `op` into eigenbase.
            `op` must be ndarray of shape `(M,M)` or `(N,M,M)`
            """
        return self.sarr.conj().T @ op @ self.sarr

