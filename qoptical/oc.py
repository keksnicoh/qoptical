# -*- coding: utf-8 -*-
"""
quantum optimal control

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

        this argument allows to avoid going back in the
        parameterpace.

                      |
          |     1/3   |   1/3
        --.-----------.------.....
          |           |
          |           |
          | 1/3       | 1
          |           |
        --.-----------X.....
          |   1/3     |


        likelyhood of going back in a dimP=2 parameterspace
        after starting from X in any direction:

            1 * (1/3)^3 * 2 = 2 / 27

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

    discard_idx = np.argwhere(dx > 0)[..., 0]
    th_weight   = np.exp(-(beta * dx)[discard_idx])
    ddidx       = np.argwhere(th_weight >= rand(*th_weight.shape))
    return discard_idx[ddidx]


def zero_parameters(n, fields):
    km = dict(zip(fields, range(len(fields))))
    return km, np.zeros((n, len(fields)), dtype=P_DTYPE)


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


class OptimalControl():

    def __init__(self, fields, beta=0, rand=None, optimize=None, tol=0.000001):
        """

        """
        self.beta     = beta
        self.optimize = optimize or Optimize.MAXIMIZE
        self.rand     = rand or np.random.rand
        self.fields   = fields
        self.keymap   = dict(zip(fields, range(len(fields))))
        self.tol      = tol


    def zero_parameters(self, n):
        """
        initializes parameters

        Arguments:
        ----------

        :n: number of parameters to be spawnd

        :fields: arr of field, implies dimension of parameterspace

        Returns:
        --------

        tuple of keymap and ndarray

        """
        return np.zeros((n, len(self.fields)), dtype=P_DTYPE)


    def mc_param(self, p0, p_forb=None):
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

        return mc_param(
            p0=p0,
            p_forb=p_forb,
            keymap=self.keymap,
            rand=self.rand
        )

    def run(self, p):
        pass


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
