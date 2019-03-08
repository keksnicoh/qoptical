# -*- coding: utf-8 -*-
"""
Metropolis Algorithm util

:author: keksnicoh
"""

from .settings import QOP
import numpy as np
from enum import Enum

P_DTYPE = np.dtype([
    ('p',   QOP.T_FLOAT),
    ('dp',  QOP.T_FLOAT),
    ('p0',  QOP.T_FLOAT),
    ('p1',  QOP.T_FLOAT),
    ('eps', QOP.T_FLOAT),
])


class Optimize(Enum):
    MAXIMIZE = +1
    MINIMIZE = -1


def mc_param(p0, p_forb=None, keymap={}, rand=np.random.rand):
    """
    modifies a random component of a np.ndarray of parameters
    by given `dp` and random sign.

    Arguments:
    ----------

    :p0: (dimP, ) shaped arr of parameters

    :p_forb: (dimP, ) shaped arr of forbidden values

        if None then p0['p'] is used instead

    :keymap: optional, in case of RuntimeError this dict
        might be used to label field by index

    :rand: random generator

    Returns:
    --------

    a new (dimP, 0) shaped arr of parameters

    """

    if p_forb is None:
        p_forb = p0['p']

    # get random field and signs
    MCidx  = np.int32(rand(p0.shape[0]) * p0.shape[1])
    MCsign = np.sign(rand(p0.shape[0]) * 2 - 1)

    # apply to parameters
    p1 = np.array(p0, copy=True)
    for i, j in enumerate(MCidx):
        if MCsign[i] > 0:
            ismax = np.isclose(p0[i, j]['p'], p0[i, j]['p1'])
            sign = -2 * int(ismax) + 1
        elif MCsign[i] < 0:
            ismin = np.isclose(p0[i, j]['p'], p0[i, j]['p0'])
            sign = 2 * int(ismin) - 1
        else:
            sign = MCsign[i]

        # find changed value within boundaries
        param = p0[i, j]
        p_unbound = param['p'] + sign * param['dp']
        p = np.max((param['p0'], np.min((param['p1'], p_unbound))))

        # check whether the new value is forbidden, if so then
        # flip the sign
        if np.isclose(p, p_forb[i, j]):
            p_unbound = param['p'] - sign * param['dp']
            p = np.max((param['p0'], np.min((param['p1'], p_unbound))))

        # if still forbidden then we are in some bad state
        if np.isclose(p, p_forb[i, j]):
            err = 'could not modify p[{}][{}] to a legal value.'
            raise RuntimeError(err.format(i, keymap.get(j, j)))

        p1[i, j]['p'] = p

    return p1


def arg_discard(dx, beta=0, rand=np.random.rand):
    """
    modifies a random component of a np.ndarray of parameters
    by given `dp` and random sign.

    Arguments:
    ----------

    :dx: difference of values (positive <=> enhancement)

    :beta: fluctuating factor

    :rand: random generator

    Returns:
    --------

    a new (dimP, xxx) shaped arr of parameters
    """
    if np.isinf(beta):
        return np.argwhere(dx >= 0)
    return np.argwhere(np.exp(-(beta * dx)) <= rand(*dx.shape))


def zero_param(n, fields):
    km = dict(zip(fields, range(len(fields))))
    return km, np.zeros((n, len(fields)), dtype=P_DTYPE)


def step(
    func,
    p0,
    x0=None,
    beta=np.inf, 
    rand=np.random.rand,
    optimize=Optimize.MAXIMIZE,
    n_mc_param=1,
):
    """
    metropolis step higher order function

    Arguments:
    ----------

    :func: function which received the current parameters `p0`

    :p0: parameters

    :x0: value at `p0`

    :beta: fluctuating parameter

    :rand: random gen

    :optimize: optimize or minimize

    :n_mc_param: how many times the `mc_param` should be
        applied to `p0`

    Returns:
    --------

    tuple `(p1, x1)`

    """

    p1 = mc_param(p0, rand=rand)
    for i in range(n_mc_param - 1):
        p1 = mc_param(p1, rand=rand)

    x1 = func(p1)

    if x0 is not None:
        didx = arg_discard(
            dx=optimize.value * (x0 - x1),
            beta=beta,
            rand=rand
        )

        x1[didx], p1[didx] = x0[didx], p0[didx]

    return p1, x1


def arg_eps(p, eps, tol=0.000001):
    """
    returns the indices of a parameter record list until
    a convergence criteria is fullfilled.

    Arguments:
    ----------

    :p: (N_rec, dimP, nP)

    :eps: (N_rec, dimP, nP)

    :tol:

    Returns:
    --------

    the highest indeces to fullfill the criteria
    """
    if p.shape != eps.shape:
        err = "p.shape {} does not equals eps.shape {}"
        raise ValueError(err.format(p.shape, eps.shape))

    arg = np.zeros(p.shape[1:], np.int32)
    aX  = np.where(p[0] == p[0])
    for i in range(1, p.shape[0] + 0):

        mean = np.mean(p[0:i + 1], axis=0)

        for j in range(i):
            kidx = np.where(np.abs(mean - p[j]) - eps[j] <= tol)

            # intersection of the previous active indices
            # and the active indices in the current level j
            interidx = set(zip(*aX)) & set(zip(*kidx))

            # no intersection => deepest progression for
            # all parameters reached. Nothing to do here
            # anymore...
            if len(interidx) == 0:
                return arg

            # rebuild the active indices arr from set of tuples
            aX = tuple(l for l in zip(*interidx))

        # active indeces has gone one level deeper
        arg[aX] += 1

    return arg


class MetropolisMC():

    def __init__(self, km, beta=0, func=None, rand=None, optimize=None, tol=0.000001):
        """

        """
        self.beta       = beta
        self.func       = func
        self.optimize   = optimize or Optimize.MAXIMIZE
        self.rand       = rand or np.random.rand
        self.km         = km
        self.tol        = tol
        self.n_mc_param = 2


    def zero_param(self, n, dp=None, p0=None, p1=None):
        """

        initializes parameters

        Arguments:
        ----------

        :n: number of parameters to be spawned

        :fields: arr of field, implies dimension of parameterspace

        :dp: initial dp

        :p0: initial p0

        :p1: initial p1

        Returns:
        --------

        tuple of keymap and ndarray

        """
        p = np.zeros((n, len(self.km)), dtype=P_DTYPE)

        if dp is not None:
            p['dp'] = dp

        if p0 is not None:
            p['p0'] = p0

        if p1 is not None:
            p['p1'] = p1

        # if boundaries are set then set p to random value
        # in between those.
        if p0 is not None and p1 is not None:
            return self.rand_param(p)

        return p


    def rand_param(self, p):
        """
        takes parameters and randomizes them

        Arguments:
        ----------

        :p: parameters to be randomized

        Returns:
        --------

        randomize parameters

        """
        pr = np.array(p, copy=True)
        pr['p'] = p['p0'] + self.rand(*p.shape) * (p['p1'] - p['p0'])
        return pr


    def mc_param(self, p0, p_forb=None, n_mc_param=None):
        """

        modifies a random component of a np.ndarray of parameters
        by given `dp` and random sign.

        Arguments:
        ----------

        :p0: (dimP, ) shaped arr of parameters

        :p_forb: (dimP, ) shaped arr of forbidden values

        Returns:
        --------

        a new (dimP, 0) shaped arr of parameters

        """

        for i in range(n_mc_param or self.n_mc_param):
            p0 = mc_param(
                p0=p0,
                p_forb=p_forb,
                keymap=self.km,
                rand=self.rand
            )

        return p0


    def step(self, p0, x0=None, n_mc_param=None):

        return step(
            func=self.func,
            p0=p0,
            x0=x0,
            beta=self.beta,
            rand=self.rand,
            optimize=self.optimize,
            n_mc_param=n_mc_param or self.n_mc_param,
        )


    def arg_discard(self, x0, x1):

        return arg_discard(
            dx=self.optimize.value * (x0 - x1),
            beta=self.beta,
            rand=self.rand,
        )


    def n_eps(self, P):
        return oc.arg_eps(
            p=P['p'][::-1],
            eps=P['eps'][::-1],
            tol=self.tol
        )
