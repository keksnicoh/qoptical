# -*- coding: utf-8 -*-
""" contains math functions and operations which are useful.
    """

import numpy as np

# XXX Test this function
def dft_single_freq_window(x, freqs, dt, tperiod):
    """ transforms N signals **x** of shape `(N, n)` where `n` is the
        number of samples into frequency space. **freqs** is a list
        of `N` frequencies to calculate `FFT(freq[i], signals[i]).
        The signal must be equally spaced by `dt`. The spectrum
        is calculated such that the integration starts at `tperiod[0]`
        and end such that

            T[i] * M + R = tperiod[1]

        with 0 < R < T[i] and T[i] the period of the i-th frequency freqs[i].

        Example:
        --------

        tr = (0, 100, 0.0001) # full time space
        wp = [1, 2, 3, 4]
        x  = [.., .., .., ..] # shape (4, len(np.arange(*tr)))
        dft_single_freq_window(x, wp, dt=tr[2], tperiod=(10, 90))

        """
    t0, tf = tperiod
    tp = 2 * np.pi / freqs
    ntp = np.floor((tf - t0) / tp)
    wsig = np.zeros(len(freqs), dtype=np.complex64)

    assert np.min(ntp) >= 1.0, 'time window (tperiod) too small.'

    for i, freq in enumerate(freqs):
        ltr = np.arange(t0, t0 + ntp[i] * tp[i], dt)
        exp = np.exp(1.0j * freq * ltr)
        i0 = int(t0 / dt)
        wsig[i] = np.sum(x[i0:i0 + len(ltr), i] * exp) / (ltr[-1] - ltr[0])

    return dt * wsig
