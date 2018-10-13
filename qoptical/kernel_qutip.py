# -*- coding: utf-8 -*-
""" contains qutip kernels for solving optical
    master equation.
    :author: keksnicoh
"""
from qutip import Qobj, mesolve, expect
import numpy as np
from .settings import QOP, print_debug
from .util import *
from .result import OpMeResult
from time import time

class QutipKernel():
    """ Performs sequential integration of multiple states
        using QuTip: http://qutip.org/docs/4.1/modules/qutip/mesolve.html
    """
    def __init__(self, system, n_htl=0, n_e_ops=0, debug=None):
        """ create QutipKernel for given ``system`` """
        # api state
        self.system    = system
        self.state     = None
        self.t_bath    = None
        self.y_0       = None
        self.htl       = None
        self.n_htl     = n_htl
        self.n_e_ops   = n_e_ops
        self.hu        = npmat_manylike(self.system.h0, [self.system.h0])
        self.args      = None

        self.e_ops     = None
        # Qobj
        self.q_state   = None
        self.q_hu      = None
        self.q_htl     = []
        self.q_L       = None
        self.q_e_ops   = None
        # prepared state
        self.n_dst     = None
        self.r_y_0     = None
        # flags
        self.synced    = False
        self.compiled  = False

        self.debug = debug or QOP.DEBUG

        self.init()

    def init(self):
        if self.debug:
            print_debug("")
            print_debug("                           //")
            print_debug("   boaaa!!1              _oo\ ")
            print_debug("                        (__/ \  _  _")
            print_debug("   feed me and I           \  \/ \/ \ ")
            print_debug("   will calculate          (         )\ ")
            print_debug("   numbers for you.         \_______/  \ ")
            print_debug("                             [[] [[]")
            print_debug("   qutip kernel v0.0         [[] [[]")
            print_debug("")


    def compile(self):
        """ create Qobj for system hamiltonian as well as for the
            basic jumping operators. """
        assert not self.compiled

        tij = lambda i, j: ketbra(self.system.s, i, j)
        self.q_L = []
        for k, jumps in enumerate(self.system.get_jumps()):
            if jumps is None:
                continue

            if k == 0:
                for t in jumps:
                    self.q_L.append((t['w'], Qobj(t['d'] * tij(*t['I']))))
            else:
                for jump in jumps.reshape((int(len(jumps) / (k + 1)), k + 1)):
                    self.q_L.append((
                        jump[0]['w'],
                        Qobj(sum(t['d'] * tij(*t['I']) for t in jump))
                    ))

        self.compiled = True


    def sync(self, state=None, t_bath=None, y_0=None, hu=None, htl=None, e_ops=None, args=None):
        """ sync data into kernel environment. """
        assert self.compiled

        self.synced = False

        self._validate_sync_args(state, t_bath, y_0, hu, htl, e_ops)

        if state is not None:
            self.state = npmat_manylike(self.system.h0, state)
        if hu is not None:
            self.hu = npmat_manylike(self.system.h0, hu)
        if t_bath is not None:
            self.t_bath = vectorize(t_bath, dtype=QOP.T_FLOAT)
            assert np.all(self.t_bath >= 0)
        if y_0 is not None:
            self.y_0 = vectorize(y_0, dtype=QOP.T_FLOAT)
            assert np.all(self.y_0 >= 0)
        if args is not None:
            self.args = args
            #assert np.all(self.args >= 0)

        # XXX todo make observables listable
        if self.n_e_ops > 0 and e_ops is not None:
            self.e_ops = npmat_manylike(self.system.h0, e_ops)
            assert len(self.e_ops) == self.n_e_ops
            self.q_e_ops = [Qobj(e) for e in self.e_ops]

        # XXX todo make time dependent operators listable
        if self.n_htl > 0 and htl is not None:
            self.htl = list(htl)
            assert len(self.htl) == self.n_htl
            self.q_htl = [[Qobj(sqmat(ht[0])), ht[1]] for ht in self.htl]


        # normalize all buffers such that length are the same.
        tb = self.t_bath if self.t_bath is not None else [0.0]
        self.q_state, \
        self.q_hu, \
        self.n_dst, \
        self.r_y_0 = self._sync_fill_up(
            q_state = [Qobj(s) for s in self.state],
            q_hu    = [Qobj(hu) for hu in self.hu],
            n_dst   = [boson_stat(t) for t in tb],
            r_y_0   = self.y_0 if self.y_0 is not None else np.array([0.0], dtype=QOP.T_FLOAT))

        self.synced = True


    def _sync_fill_up(self, q_state, q_hu, n_dst, r_y_0):
        """ some buffers might be specified by only one value while other buffers
            have `l` values. In this case we fill up the singleton buffers by
            copying the values `l` times to normalize all buffers lengths.
            if this is not possible, a `RuntimeError` is raised.

            the normalized buffers are returned in the same order as the
            list of arguments of this function.
            """
        lengths = list(set([len(s) for s in (q_state, q_hu, n_dst, r_y_0)]))
        if len(lengths) != 1:
            # we can only normalize if there are two lengths and one length is 1.
            # this means that we have n buffers of length l and m buffers of length 1.
            # the m buffers of length 1 are then filled up by copy the data n times.
            if len(lengths) == 2:
                l = lengths[0] if lengths[1] == 1 else lengths[1]
                if len(q_state) == 1:  q_state  = l * [q_state[0]]
                if len(q_hu) == 1:     q_hu     = l * [q_hu[0]]
                if len(n_dst) == 1:    n_dst    = l * [n_dst[0]]
                if len(r_y_0) == 1:    r_y_0    = np.array([r_y_0[0]] * l, dtype=r_y_0.dtype)
            else:
                raise RuntimeError("data length mismatch.")

        return q_state, q_hu, n_dst, r_y_0


    def _validate_sync_args(self, state, t_bath, y_0, hu, htl, e_ops):
        """ ensure that we have data in all buffers """

        if state is None and self.state is None:
            raise RuntimeError('state must be defined at least once.')

        if hu is None and self.hu is None:
            raise RuntimeError('hu must be defined at least once.')

        if len(self.q_L) > 0:
            if t_bath is None and self.t_bath is None:
                raise RuntimeError('t_bath must be defined at least once.')

            if y_0 is None and self.y_0 is None:
                raise RuntimeError('y_0 must be defined at least once.')

        if self.n_htl > 0 and htl is None and self.htl is None:
            raise RuntimeError('htl must be defined at least once.')

        if self.n_e_ops > 0 and e_ops is None and self.e_ops is None:
            raise RuntimeError('e_ops must be defined at least once.')

        if self.n_e_ops == 0 and e_ops is not None:
            raise RuntimeError('e_ops is defined by reduced system n_e_ops is zero.')


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
            raise ValueError('sync_state must be False if n_e_ops > 0.')

        # result data preparation
        tstate = None
        if self.q_e_ops is None:
            shape = (self.state.shape[0], len(tlist), *self.state.shape[1:])
            tstate = np.empty(shape, dtype=settings.QOP.T_COMPLEX)
        texpect = None
        if self.q_e_ops is not None:
            shape = (self.state.shape[0], len(self.q_e_ops), len(tlist))
            texpect = np.empty(shape, dtype=settings.QOP.T_COMPLEX)

        # integrate states
        t0          = time()
        q_state_new = []
        zipped      = zip(self.q_hu, self.q_state, self.r_y_0, self.n_dst)
        for (i, (q_hu, q_s, y_0, n_dst)) in enumerate(zipped):
            if QOP.DEBUG:
                print_debug('{}/{}'.format(i, len(self.q_hu)))
            # prepare lindblad operators
            C = lambda w: w**3 * (1 + n_dst(w)) if w >= 0 \
                    else -w**3 * n_dst(-w)
            L = []
            for w, l in self.q_L:
                L.append(np.sqrt(y_0 * C(w)) * l)
                L.append(np.sqrt(y_0 * C(-w)) * l.dag())

            # args
            args = None
            if self.args is not None:
                args = self.args[i] # XXX TEST ME

            # mesolve
            Lf  = [l for l in L if not np.allclose(l.full(), 0)]
            H   = [q_hu] + self.q_htl
            res = mesolve(H     = H[0] if len(H) == 1 else H,
                          rho0  = q_s,
                          tlist = tlist,
                          c_ops = Lf,
                          args  = args)

            # work with result
            if texpect is not None:
                texpect[i] = [expect(e, res.states) for e in self.q_e_ops]
            if tstate is not None:
                tstate[i] = [s.full() for s in res.states]
                q_state_new.append(res.states[-1])

        tf = time()
        if tstate is not None:
            tstate = np.swapaxes(tstate, axis1=0, axis2=1)
            self.debug and print_debug("1/1 calculated {} steps, took {:.4f}s".format(tstate.shape[0:2], tf - t0))

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


