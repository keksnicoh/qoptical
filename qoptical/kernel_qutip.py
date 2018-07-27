# -*- coding: utf-8 -*-
""" contains qutip kernels for solving optical
    master equation.
    :author: keksnicoh
"""
from qutip import Qobj, mesolve, expect
import numpy as np
from .settings import *
from .util import *
from .result import OpMeResult

class QutipKernel():
    """ Performs sequential integration of multiple states
        using QuTip: http://qutip.org/docs/4.1/modules/qutip/mesolve.html
    """
    def __init__(self, system):
        """ create QutipKernel for given ``system`` """
        # api state
        self.system    = system
        self.state     = None
        self.t_bath    = None
        self.y_0       = None
        self.y_w       = None
        self.htl       = None
        self.e_ops     = None
        # Qobj
        self.q_state   = None
        self.q_h0      = None
        self.q_htl     = []
        self.q_L       = None
        self.q_e_ops   = None
        # prepared state
        self.n_dst     = None
        # flags
        self.synced    = False
        self.compiled  = False


    def compile(self):
        """ create Qobj for system hamiltonian as well as for the
            basic jumping operators. """
        assert not self.compiled

        self.q_h0 = Qobj(self.system.h0)
        self.q_L = []
        for k, jumps in enumerate(self.system.get_jumps()):
            if jumps is None:
                continue

            if k == 0:
                for t in jumps:
                    self.q_L.append((t['w'], Qobj(t['d'] * self.tij(*t['I']))))
            else:
                for jump in jumps.reshape((int(len(jumps) / (k + 1)), k + 1)):
                    self.q_L.append((
                        jump[0]['w'],
                        Qobj(sum(t['d'] * self.tij(*t['I']) for t in jump))
                    ))

        self.compiled = True


    def tij(self, i, j):
        """ projection from i to j """
        return ketbra(self.system.s, j, i)


    def sync(self, state=None, t_bath=None, y_0=None, htl=None, e_ops=None):
        """ sync data into kernel environment. """
        assert self.compiled

        self.synced = False

        if state is None and self.state is None:
            raise RuntimeError('state must be defined at least once.')
        if t_bath is None and self.t_bath is None:
            raise RuntimeError('t_bath must be defined at least once.')
        if y_0 is None and self.y_0 is None:
            raise RuntimeError('y_0 must be defined at least once.')
        if self.system.n_htl > 0 and htl is None and self.htl is None:
            raise RuntimeError('htl must be defined at least once.')
        if self.system.n_e_ops > 0 and e_ops is None and self.e_ops is None:
            raise RuntimeError('e_ops must be defined at least once.')
        if self.system.n_e_ops == 0 and e_ops is not None:
            raise RuntimeError('e_ops is defined by reduced system n_e_ops is zero.')

        if state is not None:
            self.state = npmat_manylike(self.system.h0, state)
            self.q_state = [Qobj(s) for s in self.state]

        if t_bath is not None:
            self.t_bath = nparr_manylike(self.state, t_bath, DTYPE_FLOAT)
            assert np.all(self.t_bath >= 0)
            self.n_dst = [boson_stat(t) for t in self.t_bath]

        if y_0 is not None:
            self.y_0 = nparr_manylike(self.state, y_0, DTYPE_FLOAT)
            assert np.all(self.y_0 >= 0)

        if self.system.n_htl > 0 and htl is not None:
            self.htl = list(htl)
            assert len(self.htl) == self.system.n_htl
            self.q_htl = [[Qobj(sqmat(ht[0])), ht[1]] for ht in self.htl]

        if self.system.n_e_ops > 0 and e_ops is not None:
            self.e_ops = npmat_manylike(self.system.h0, e_ops)
            assert len(self.e_ops) == self.system.n_e_ops
            self.q_e_ops = [Qobj(e) for e in self.e_ops]


        self.synced = True


    def run(self, tlist, sync_state=False):
        """ runs qutip.mesolve for given system and
            synced state. Note the this runner does not
            run multiple systems in parallel, indeed it
            will perform each system sequentially.

            if `sync_state` is `True` then the final state
            will be synced to this kernel after integration.
            **Note** This is not possible if **e_ops** is
            `None`.
        """
        assert self.compiled and self.synced
        if sync_state is True and self.q_e_ops is not None:
            raise ValueError('sync_state must be False if system.n_e_ops > 0.')

        # result data preparation
        tstate = None
        if self.q_e_ops is None:
            tstate = np.empty(
                (self.state.shape[0], len(tlist), *self.state.shape[1:]),
                dtype=settings.DTYPE_COMPLEX)
        texpect = None
        if self.q_e_ops is not None:
            texpect = np.empty(
                (self.state.shape[0], len(self.q_e_ops), len(tlist)),
                dtype=settings.DTYPE_COMPLEX)

        # integrate states
        q_state_new = []
        zipped      = zip(self.q_state, self.t_bath, self.y_0, self.n_dst)
        for (i, (q_s, t_bath, y_0, n_dst)) in enumerate(zipped):

            # prepare lindblad operators
            C = lambda w: w**3 * (1 + n_dst(w)) if w >= 0 \
                    else -w**3 * n_dst(-w)
            L = []
            for w, l in self.q_L:
                L.append(np.sqrt(y_0 * C(w)) * l)
                L.append(np.sqrt(y_0 * C(-w)) * l.dag())

            # mesolve
            Lf  = [l for l in L if not np.allclose(l.full(), 0)]
            H   = [self.q_h0] + self.q_htl
            res = mesolve(H     = H[0] if len(H) == 1 else H,
                          rho0  = q_s,
                          tlist = tlist,
                          c_ops = Lf)

            # work with result
            if texpect is not None:
                texpect[i] = [expect(e, res.states) for e in self.q_e_ops]
            if tstate is not None:
                tstate[i] = [s.full() for s in res.states]
                q_state_new.append(res.states[-1])

        fstate = None
        if tstate is not None:
            fstate = npmat_manylike(self.system.h0, [q_s.full() for q_s in q_state_new])

        # assign inner state
        if sync_state:
            self.q_state = q_state_new
            self.state   = fstate

        # copy date to avoid references
        return OpMeResult(tlist=tlist[:],
                          state=fstate[:]    if fstate  is not None else None,
                          tstate=tstate[:]   if tstate  is not None else None,
                          texpect=texpect[:] if texpect is not None else None)


