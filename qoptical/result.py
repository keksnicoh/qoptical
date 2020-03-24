# -*- coding: utf-8 -*-
"""
    :author: keksnicoh
"""

import numpy as np

class OpMeResult2():

    def __init__(self, state, tlist, tstate, texpect):
        self.state   = state
        self.tlist   = tlist
        self.tstate  = tstate
        self.texpect = texpect


    def plot_expect(self,
                    si        = 0,
                    labels    = None,
                    plot_real = True,
                    plot_imag = True,
                    plot_abs  = False,
                    mavg      = None,
                    ref       = None,
    ):
        """ plot expectation values for given system `si`.
            a legend can be defined using `labels` which must be a list
            with the same length as the number of `e_ops`.

            the flags `plot_real`, `plot_imag` and `plot_abs` allow to
            define how to plot complex numbers """
        import matplotlib.pyplot as plt

        if labels is not None:
            assert len(labels) == len(self.texpect[si])

        for i in range(self.texpect.shape[1]):
            if plot_real:
                label = 'Real({})'.format(labels[i]) if labels is not None else None
                plt.plot(self.tlist, self.texpect[si, i].real, label=label)
                mavg is not None and plt.plot(*self.mavg(mavg, self.texpect[si, i].real))
                ref  is not None and plt.plot(self.tlist, [self.texpect[si, i].real[ref]] * len(self.tlist))
            if plot_imag:
                label = 'Imag({})'.format(labels[i]) if labels is not None else None
                plt.plot(self.tlist, self.texpect[si, i].imag, label=label)
                mavg is not None and plt.plot(*self.mavg(mavg, self.texpect[si, i].imag))
                ref  is not None and plt.plot(self.tlist, [self.texpect[si, i].imag[ref]] * len(self.tlist))
            if plot_abs:
                label = 'Abs({})'.format(labels[i]) if labels is not None else None
                plt.plot(self.tlist, np.abs(self.texpect[si, i]), label=label)
                mavg is not None and plt.plot(*self.mavg(mavg, np.abs(self.texpect[si, i])))
                ref  is not None and plt.plot(self.tlist, [np.abs(self.texpect[si, i])[ref]] * len(self.tlist))

        if labels is not None:
            plt.legend()

        plt.xlabel('time t')
        return plt


    def plot_state(self, si=0, ti=-1, matrixf=np.abs):
        import matplotlib.pyplot as plt

        plt.imshow(1.0-matrixf(self.tstate[si][ti]), cmap="Blues")
        return plt


    def save(self, filename):
        raise NotImplementedError('sryyyyyyy...')

    def mavg(self, w, data):
        # add constant lists of window length `w` to start and
        # end of the list to avoid moving avg starting/enging at zero.
        ad = np.empty(len(data) + 2 * w).astype(data.dtype)
        ad[0:w], ad[w:-w], ad[-w:] = data[0], data, data[-1]
        mavg_vec = np.convolve(ad, np.ones(int(w))/float(w), 'same')
        return self.tlist, mavg_vec[w:-w]