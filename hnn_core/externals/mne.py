import operator
import functools

from warnings import warn
from pathlib import Path

import numpy as np


####################################################
# mne/parallel.py
def parallel_func(func, n_jobs):
    if n_jobs != 1:
        try:
            from joblib import Parallel, delayed
        except ImportError:
            warn('joblib not installed. Cannot run in parallel.')
            n_jobs = 1
    if n_jobs == 1:
        my_func = func
        parallel = list
    else:
        parallel = Parallel(n_jobs)
        my_func = delayed(func)

    return parallel, my_func


####################################################
# mne/filter.py
def next_fast_len(target):
    """Find the next fast size of input data to `fft`, for zero-padding, etc.
    SciPy's FFTPACK has efficient functions for radix {2, 3, 4, 5}, so this
    returns the next composite of the prime factors 2, 3, and 5 which is
    greater than or equal to `target`. (These are also known as 5-smooth
    numbers, regular numbers, or Hamming numbers.)
    Parameters
    ----------
    target : int
        Length to start searching from.  Must be a positive integer.
    Returns
    -------
    out : int
        The first 5-smooth number greater than or equal to `target`.
    Notes
    -----
    Copied from SciPy with minor modifications.
    """
    from bisect import bisect_left
    hams = (8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48,
            50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100, 108, 120, 125, 128,
            135, 144, 150, 160, 162, 180, 192, 200, 216, 225, 240, 243, 250,
            256, 270, 288, 300, 320, 324, 360, 375, 384, 400, 405, 432, 450,
            480, 486, 500, 512, 540, 576, 600, 625, 640, 648, 675, 720, 729,
            750, 768, 800, 810, 864, 900, 960, 972, 1000, 1024, 1080, 1125,
            1152, 1200, 1215, 1250, 1280, 1296, 1350, 1440, 1458, 1500, 1536,
            1600, 1620, 1728, 1800, 1875, 1920, 1944, 2000, 2025, 2048, 2160,
            2187, 2250, 2304, 2400, 2430, 2500, 2560, 2592, 2700, 2880, 2916,
            3000, 3072, 3125, 3200, 3240, 3375, 3456, 3600, 3645, 3750, 3840,
            3888, 4000, 4050, 4096, 4320, 4374, 4500, 4608, 4800, 4860, 5000,
            5120, 5184, 5400, 5625, 5760, 5832, 6000, 6075, 6144, 6250, 6400,
            6480, 6561, 6750, 6912, 7200, 7290, 7500, 7680, 7776, 8000, 8100,
            8192, 8640, 8748, 9000, 9216, 9375, 9600, 9720, 10000)

    if target <= 6:
        return target

    # Quickly check if it's already a power of 2
    if not (target & (target - 1)):
        return target

    # Get result quickly for small sizes, since FFT itself is similarly fast.
    if target <= hams[-1]:
        return hams[bisect_left(hams, target)]

    match = float('inf')  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)

            p2 = 2 ** int(quotient - 1).bit_length()

            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match


####################################################
# mne/fixes.py
@functools.lru_cache(None)
def _import_fft(name):
    single = False
    if not isinstance(name, tuple):
        name = (name,)
        single = True
    try:
        from scipy.fft import rfft  # noqa analysis:ignore
    except ImportError:
        from numpy import fft  # noqa
    else:
        from scipy import fft  # noqa
    out = [getattr(fft, n) for n in name]
    if single:
        out = out[0]
    return out


####################################################
# mne/utils/check.py
def _ensure_int(x, name='unknown', must_be='an int'):
    """Ensure a variable is an integer."""
    # This is preferred over numbers.Integral, see:
    # https://github.com/scipy/scipy/pull/7351#issuecomment-299713159
    try:
        # someone passing True/False is much more likely to be an error than
        # intentional usage
        if isinstance(x, bool):
            raise TypeError()
        x = int(operator.index(x))
    except TypeError:
        raise TypeError('%s must be %s, got %s' % (name, must_be, type(x)))
    return x


class _IntLike(object):
    @classmethod
    def __instancecheck__(cls, other):
        try:
            _ensure_int(other)
        except TypeError:
            return False
        else:
            return True


int_like = _IntLike()
path_like = (str, Path)


class _Callable(object):
    @classmethod
    def __instancecheck__(cls, other):
        return callable(other)


_multi = {
    'str': (str,),
    'numeric': (np.floating, float, int_like),
    'path-like': path_like,
    'int-like': (int_like,),
    'callable': (_Callable(),),
}


def _validate_type(item, types=None, item_name=None, type_name=None):
    """Validate that `item` is an instance of `types`.
    Parameters
    ----------
    item : object
        The thing to be checked.
    types : type | str | tuple of types | tuple of str
         The types to be checked against.
         If str, must be one of {'int', 'str', 'numeric', 'path-like'}.
    """
    if types == "int":
        _ensure_int(item, name=item_name)
        return  # terminate prematurely

    if not isinstance(types, (list, tuple)):
        types = [types]

    check_types = sum(((type(None),) if type_ is None else (type_,)
                       if not isinstance(type_, str) else _multi[type_]
                       for type_ in types), ())
    if not isinstance(item, check_types):
        if type_name is None:
            type_name = ['None' if cls_ is None else cls_.__name__
                         if not isinstance(cls_, str) else cls_
                         for cls_ in types]
            if len(type_name) == 1:
                type_name = type_name[0]
            elif len(type_name) == 2:
                type_name = ' or '.join(type_name)
            else:
                type_name[-1] = 'or ' + type_name[-1]
                type_name = ', '.join(type_name)
        raise TypeError('%s must be an instance of %s, got %s instead'
                        % (item_name, type_name, type(item),))


def _check_option(parameter, value, allowed_values, extra=''):
    """Check the value of a parameter against a list of valid options.
    Return the value if it is valid, otherwise raise a ValueError with a
    readable error message.
    Parameters
    ----------
    parameter : str
        The name of the parameter to check. This is used in the error message.
    value : any type
        The value of the parameter to check.
    allowed_values : list
        The list of allowed values for the parameter.
    extra : str
        Extra string to append to the invalid value sentence, e.g.
        "when using ico mode".
    Raises
    ------
    ValueError
        When the value of the parameter is not one of the valid options.
    Returns
    -------
    value : any type
        The value if it is valid.
    """
    if value in allowed_values:
        return value

    # Prepare a nice error message for the user
    extra = ' ' + extra if extra else extra
    msg = ("Invalid value for the '{parameter}' parameter{extra}. "
           '{options}, but got {value!r} instead.')
    allowed_values = list(allowed_values)  # e.g., if a dict was given
    if len(allowed_values) == 1:
        options = f'The only allowed value is {repr(allowed_values[0])}'
    else:
        options = 'Allowed values are '
        options += ', '.join([f'{repr(v)}' for v in allowed_values[:-1]])
        options += f', and {repr(allowed_values[-1])}'
    raise ValueError(msg.format(parameter=parameter, options=options,
                                value=value, extra=extra))


####################################################
# mne/time_frequency/tfr.py
def _check_decim(decim):
    """Aux function checking the decim parameter."""
    _validate_type(decim, ('int-like', slice), 'decim')
    if not isinstance(decim, slice):
        decim = slice(None, None, int(decim))
    # ensure that we can actually use `decim.step`
    if decim.step is None:
        decim = slice(decim.start, decim.stop, 1)
    return decim


def _centered(arr, newsize):
    """Aux Function to center data."""
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


def morlet(sfreq, freqs, n_cycles=7.0, sigma=None, zero_mean=False):
    """Compute Morlet wavelets for the given frequency range.
    Parameters
    ----------
    sfreq : float
        The sampling Frequency.
    freqs : array
        Frequency range of interest (1 x Frequencies).
    n_cycles : float | array of float, default 7.0
        Number of cycles. Fixed number or one per frequency.
    sigma : float, default None
        It controls the width of the wavelet ie its temporal
        resolution. If sigma is None the temporal resolution
        is adapted with the frequency like for all wavelet transform.
        The higher the frequency the shorter is the wavelet.
        If sigma is fixed the temporal resolution is fixed
        like for the short time Fourier transform and the number
        of oscillations increases with the frequency.
    zero_mean : bool, default False
        Make sure the wavelet has a mean of zero.
    Returns
    -------
    Ws : list of array
        The wavelets time series.
    """
    Ws = list()
    n_cycles = np.atleast_1d(n_cycles)

    freqs = np.array(freqs)
    if np.any(freqs <= 0):
        raise ValueError("all frequencies in 'freqs' must be "
                         "greater than 0.")

    if (n_cycles.size != 1) and (n_cycles.size != len(freqs)):
        raise ValueError("n_cycles should be fixed or defined for "
                         "each frequency.")
    for k, f in enumerate(freqs):
        if len(n_cycles) != 1:
            this_n_cycles = n_cycles[k]
        else:
            this_n_cycles = n_cycles[0]
        # fixed or scale-dependent window
        if sigma is None:
            sigma_t = this_n_cycles / (2.0 * np.pi * f)
        else:
            sigma_t = this_n_cycles / (2.0 * np.pi * sigma)
        # this scaling factor is proportional to (Tallon-Baudry 98):
        # (sigma_t*sqrt(pi))^(-1/2);
        t = np.arange(0., 5. * sigma_t, 1.0 / sfreq)
        t = np.r_[-t[::-1], t[1:]]
        oscillation = np.exp(2.0 * 1j * np.pi * f * t)
        gaussian_enveloppe = np.exp(-t ** 2 / (2.0 * sigma_t ** 2))
        if zero_mean:  # to make it zero mean
            real_offset = np.exp(- 2 * (np.pi * f * sigma_t) ** 2)
            oscillation -= real_offset
        W = oscillation * gaussian_enveloppe
        W /= np.sqrt(0.5) * np.linalg.norm(W.ravel())
        Ws.append(W)
    return Ws


def _cwt_gen(X, Ws, *, fsize=0, mode="same", decim=1, use_fft=True):
    """Compute cwt with fft based convolutions or temporal convolutions.
    Parameters
    ----------
    X : array of shape (n_signals, n_times)
        The data.
    Ws : list of array
        Wavelets time series.
    fsize : int
        FFT length.
    mode : {'full', 'valid', 'same'}
        See numpy.convolve.
    decim : int | slice, default 1
        To reduce memory usage, decimation factor after time-frequency
        decomposition.
        If `int`, returns tfr[..., ::decim].
        If `slice`, returns tfr[..., decim].
        .. note:: Decimation may create aliasing artifacts.
    use_fft : bool, default True
        Use the FFT for convolutions or not.
    Returns
    -------
    out : array, shape (n_signals, n_freqs, n_time_decim)
        The time-frequency transform of the signals.
    """
    fft, ifft = _import_fft(('fft', 'ifft'))
    _check_option('mode', mode, ['same', 'valid', 'full'])
    decim = _check_decim(decim)
    X = np.asarray(X)

    # Precompute wavelets for given frequency range to save time
    _, n_times = X.shape
    n_times_out = X[:, decim].shape[1]
    n_freqs = len(Ws)

    # precompute FFTs of Ws
    if use_fft:
        fft_Ws = np.empty((n_freqs, fsize), dtype=np.complex128)
        for i, W in enumerate(Ws):
            fft_Ws[i] = fft(W, fsize)

    # Make generator looping across signals
    tfr = np.zeros((n_freqs, n_times_out), dtype=np.complex128)
    for x in X:
        if use_fft:
            fft_x = fft(x, fsize)

        # Loop across wavelets
        for ii, W in enumerate(Ws):
            if use_fft:
                ret = ifft(fft_x * fft_Ws[ii])[:n_times + W.size - 1]
            else:
                ret = np.convolve(x, W, mode=mode)

            # Center and decimate decomposition
            if mode == 'valid':
                sz = int(abs(W.size - n_times)) + 1
                offset = (n_times - sz) // 2
                this_slice = slice(offset // decim.step,
                                   (offset + sz) // decim.step)
                if use_fft:
                    ret = _centered(ret, sz)
                tfr[ii, this_slice] = ret[decim]
            elif mode == 'full' and not use_fft:
                start = (W.size - 1) // 2
                end = len(ret) - (W.size // 2)
                ret = ret[start:end]
                tfr[ii, :] = ret[decim]
            else:
                if use_fft:
                    ret = _centered(ret, n_times)
                tfr[ii, :] = ret[decim]
        yield tfr


def _time_frequency_loop(X, Ws, output, use_fft, mode, decim):
    """Aux. function to _compute_tfr.
    Loops time-frequency transform across wavelets and epochs.
    Parameters
    ----------
    X : array, shape (n_epochs, n_times)
        The epochs data of a single channel.
    Ws : list, shape (n_tapers, n_wavelets, n_times)
        The wavelets.
    output : str
        * 'complex' : single trial complex.
        * 'power' : single trial power.
        * 'phase' : single trial phase.
        * 'avg_power' : average of single trial power.
        * 'itc' : inter-trial coherence.
        * 'avg_power_itc' : average of single trial power and inter-trial
          coherence across trials.
    use_fft : bool
        Use the FFT for convolutions or not.
    mode : {'full', 'valid', 'same'}
        See numpy.convolve.
    decim : slice
        The decimation slice: e.g. power[:, decim]
    """
    # Set output type
    dtype = np.float64
    if output in ['complex', 'avg_power_itc']:
        dtype = np.complex128

    # Init outputs
    decim = _check_decim(decim)
    n_epochs, n_times = X[:, decim].shape
    n_freqs = len(Ws[0])
    if ('avg_' in output) or ('itc' in output):
        tfrs = np.zeros((n_freqs, n_times), dtype=dtype)
    else:
        tfrs = np.zeros((n_epochs, n_freqs, n_times), dtype=dtype)

    # Loops across tapers.
    for W in Ws:
        # No need to check here, it's done earlier (outside parallel part)
        nfft = _get_nfft(W, X, use_fft, check=False)
        coefs = _cwt_gen(
            X, W, fsize=nfft, mode=mode, decim=decim, use_fft=use_fft)

        # Inter-trial phase locking is apparently computed per taper...
        if 'itc' in output:
            plf = np.zeros((n_freqs, n_times), dtype=np.complex128)

        # Loop across epochs
        for epoch_idx, tfr in enumerate(coefs):
            # Transform complex values
            if output in ['power', 'avg_power']:
                tfr = (tfr * tfr.conj()).real  # power
            elif output == 'phase':
                tfr = np.angle(tfr)
            elif output == 'avg_power_itc':
                tfr_abs = np.abs(tfr)
                plf += tfr / tfr_abs  # phase
                tfr = tfr_abs ** 2  # power
            elif output == 'itc':
                plf += tfr / np.abs(tfr)  # phase
                continue  # not need to stack anything else than plf

            # Stack or add
            if ('avg_' in output) or ('itc' in output):
                tfrs += tfr
            else:
                tfrs[epoch_idx] += tfr

        # Compute inter trial coherence
        if output == 'avg_power_itc':
            tfrs += 1j * np.abs(plf)
        elif output == 'itc':
            tfrs += np.abs(plf)

    # Normalization of average metrics
    if ('avg_' in output) or ('itc' in output):
        tfrs /= n_epochs

    # Normalization by number of taper
    tfrs /= len(Ws)
    return tfrs


def _compute_tfr(epoch_data, freqs, sfreq=1.0, method='morlet',
                 n_cycles=7.0, zero_mean=None, time_bandwidth=None,
                 use_fft=True, decim=1, output='complex', n_jobs=1,
                 verbose=None):
    """Compute time-frequency transforms.
    Parameters
    ----------
    epoch_data : array of shape (n_epochs, n_channels, n_times)
        The epochs.
    freqs : array-like of floats, shape (n_freqs)
        The frequencies.
    sfreq : float | int, default 1.0
        Sampling frequency of the data.
    method : 'morlet'
        The time-frequency method. 'morlet' convolves a Morlet wavelet.
    n_cycles : float | array of float, default 7.0
        Number of cycles in the wavelet. Fixed number
        or one per frequency.
    zero_mean : bool | None, default None
        None means True for method='multitaper' and False for method='morlet'.
        If True, make sure the wavelets have a mean of zero.
    time_bandwidth : float, default None
        If None and method=multitaper, will be set to 4.0 (3 tapers).
        Time x (Full) Bandwidth product. Only applies if
        method == 'multitaper'. The number of good tapers (low-bias) is
        chosen automatically based on this to equal floor(time_bandwidth - 1).
    use_fft : bool, default True
        Use the FFT for convolutions or not.
    decim : int | slice, default 1
        To reduce memory usage, decimation factor after time-frequency
        decomposition.
        If `int`, returns tfr[..., ::decim].
        If `slice`, returns tfr[..., decim].
        .. note::
            Decimation may create aliasing artifacts, yet decimation
            is done after the convolutions.
    output : str, default 'complex'
        * 'complex' : single trial complex.
        * 'power' : single trial power.
        * 'phase' : single trial phase.
        * 'avg_power' : average of single trial power.
        * 'itc' : inter-trial coherence.
        * 'avg_power_itc' : average of single trial power and inter-trial
          coherence across trials.
    %(n_jobs)s
        The number of epochs to process at the same time. The parallelization
        is implemented across channels.
    %(verbose)s
    Returns
    -------
    out : array
        Time frequency transform of epoch_data. If output is in ['complex',
        'phase', 'power'], then shape of out is (n_epochs, n_chans, n_freqs,
        n_times), else it is (n_chans, n_freqs, n_times). If output is
        'avg_power_itc', the real values code for 'avg_power' and the
        imaginary values code for the 'itc': out = avg_power + i * itc
    """
    # Check data
    epoch_data = np.asarray(epoch_data)
    if epoch_data.ndim != 3:
        raise ValueError('epoch_data must be of shape (n_epochs, n_chans, '
                         'n_times), got %s' % (epoch_data.shape,))

    # Check params
    freqs, sfreq, zero_mean, n_cycles, time_bandwidth, decim = \
        _check_tfr_param(freqs, sfreq, method, zero_mean, n_cycles,
                         time_bandwidth, use_fft, decim, output)

    decim = _check_decim(decim)
    if (freqs > sfreq / 2.).any():
        raise ValueError('Cannot compute freq above Nyquist freq of the data '
                         '(%0.1f Hz), got %0.1f Hz'
                         % (sfreq / 2., freqs.max()))

    # We decimate *after* decomposition, so we need to create our kernels
    # for the original sfreq
    if method == 'morlet':
        W = morlet(sfreq, freqs, n_cycles=n_cycles, zero_mean=zero_mean)
        Ws = [W]  # to have same dimensionality as the 'multitaper' case

    # Check wavelets
    if len(Ws[0][0]) > epoch_data.shape[2]:
        raise ValueError('At least one of the wavelets is longer than the '
                         'signal. Use a longer signal or shorter wavelets.')

    # Initialize output
    n_freqs = len(freqs)
    n_epochs, n_chans, n_times = epoch_data[:, :, decim].shape
    if output in ('power', 'phase', 'avg_power', 'itc'):
        dtype = np.float64
    elif output in ('complex', 'avg_power_itc'):
        # avg_power_itc is stored as power + 1i * itc to keep a
        # simple dimensionality
        dtype = np.complex128

    if ('avg_' in output) or ('itc' in output):
        out = np.empty((n_chans, n_freqs, n_times), dtype)
    else:
        out = np.empty((n_chans, n_epochs, n_freqs, n_times), dtype)

    # Parallel computation
    all_Ws = sum([list(W) for W in Ws], list())
    _get_nfft(all_Ws, epoch_data, use_fft)
    parallel, my_cwt = parallel_func(_time_frequency_loop, n_jobs)

    # Parallelization is applied across channels.
    tfrs = parallel(
        my_cwt(channel, Ws, output, use_fft, 'same', decim)
        for channel in epoch_data.transpose(1, 0, 2))

    # FIXME: to avoid overheads we should use np.array_split()
    for channel_idx, tfr in enumerate(tfrs):
        out[channel_idx] = tfr

    if ('avg_' not in output) and ('itc' not in output):
        # This is to enforce that the first dimension is for epochs
        out = out.transpose(1, 0, 2, 3)
    return out


def tfr_array_morlet(epoch_data, sfreq, freqs, n_cycles=7.0,
                     zero_mean=False, use_fft=True, decim=1, output='complex',
                     n_jobs=1, verbose=None):
    """Compute Time-Frequency Representation (TFR) using Morlet wavelets.
    Same computation as `~mne.time_frequency.tfr_morlet`, but operates on
    :class:`NumPy arrays <numpy.ndarray>` instead of `~mne.Epochs` objects.
    Parameters
    ----------
    epoch_data : array of shape (n_epochs, n_channels, n_times)
        The epochs.
    sfreq : float | int
        Sampling frequency of the data.
    freqs : array-like of float, shape (n_freqs,)
        The frequencies.
    n_cycles : float | array of float, default 7.0
        Number of cycles in the Morlet wavelet. Fixed number or one per
        frequency.
    zero_mean : bool | False
        If True, make sure the wavelets have a mean of zero. default False.
    use_fft : bool
        Use the FFT for convolutions or not. default True.
    decim : int | slice
        To reduce memory usage, decimation factor after time-frequency
        decomposition. default 1
        If `int`, returns tfr[..., ::decim].
        If `slice`, returns tfr[..., decim].
        .. note::
            Decimation may create aliasing artifacts, yet decimation
            is done after the convolutions.
    output : str, default 'complex'
        * 'complex' : single trial complex.
        * 'power' : single trial power.
        * 'phase' : single trial phase.
        * 'avg_power' : average of single trial power.
        * 'itc' : inter-trial coherence.
        * 'avg_power_itc' : average of single trial power and inter-trial
          coherence across trials.
    %(n_jobs)s
        The number of epochs to process at the same time. The parallelization
        is implemented across channels. Default 1.
    %(verbose)s
    Returns
    -------
    out : array
        Time frequency transform of epoch_data. If output is in ['complex',
        'phase', 'power'], then shape of out is (n_epochs, n_chans, n_freqs,
        n_times), else it is (n_chans, n_freqs, n_times). If output is
        'avg_power_itc', the real values code for 'avg_power' and the
        imaginary values code for the 'itc': out = avg_power + i * itc.
    See Also
    --------
    mne.time_frequency.tfr_morlet
    mne.time_frequency.tfr_multitaper
    mne.time_frequency.tfr_array_multitaper
    mne.time_frequency.tfr_stockwell
    mne.time_frequency.tfr_array_stockwell
    Notes
    -----
    .. versionadded:: 0.14.0
    """
    return _compute_tfr(epoch_data=epoch_data, freqs=freqs,
                        sfreq=sfreq, method='morlet', n_cycles=n_cycles,
                        zero_mean=zero_mean, time_bandwidth=None,
                        use_fft=use_fft, decim=decim, output=output,
                        n_jobs=n_jobs, verbose=verbose)


# Low level convolution

def _get_nfft(wavelets, X, use_fft=True, check=True):
    n_times = X.shape[-1]
    max_size = max(w.size for w in wavelets)
    if max_size > n_times:
        msg = (f'At least one of the wavelets ({max_size}) is longer than the '
               f'signal ({n_times}). Consider using a longer signal or '
               'shorter wavelets.')
        if check:
            if use_fft:
                warn(msg)  # warn(msg, UserWarning)
            else:
                raise ValueError(msg)
    nfft = n_times + max_size - 1
    nfft = next_fast_len(nfft)  # 2 ** int(np.ceil(np.log2(nfft)))
    return nfft


def _check_tfr_param(freqs, sfreq, method, zero_mean, n_cycles,
                     time_bandwidth, use_fft, decim, output):
    """Aux. function to _compute_tfr to check the params validity."""
    # Check freqs
    if not isinstance(freqs, (list, np.ndarray)):
        raise ValueError('freqs must be an array-like, got %s '
                         'instead.' % type(freqs))
    freqs = np.asarray(freqs, dtype=float)
    if freqs.ndim != 1:
        raise ValueError('freqs must be of shape (n_freqs,), got %s '
                         'instead.' % np.array(freqs.shape))

    # Check sfreq
    if not isinstance(sfreq, (float, int)):
        raise ValueError('sfreq must be a float or an int, got %s '
                         'instead.' % type(sfreq))
    sfreq = float(sfreq)

    # Default zero_mean = True if multitaper else False
    zero_mean = method == 'multitaper' if zero_mean is None else zero_mean
    if not isinstance(zero_mean, bool):
        raise ValueError('zero_mean should be of type bool, got %s. instead'
                         % type(zero_mean))
    freqs = np.asarray(freqs)

    if (method == 'multitaper') and (output == 'phase'):
        raise NotImplementedError(
            'This function is not optimized to compute the phase using the '
            'multitaper method. Use np.angle of the complex output instead.')

    # Check n_cycles
    if isinstance(n_cycles, (int, float)):
        n_cycles = float(n_cycles)
    elif isinstance(n_cycles, (list, np.ndarray)):
        n_cycles = np.array(n_cycles)
        if len(n_cycles) != len(freqs):
            raise ValueError('n_cycles must be a float or an array of length '
                             '%i frequencies, got %i cycles instead.' %
                             (len(freqs), len(n_cycles)))
    else:
        raise ValueError('n_cycles must be a float or an array, got %s '
                         'instead.' % type(n_cycles))

    # Check time_bandwidth
    if (method == 'morlet') and (time_bandwidth is not None):
        raise ValueError('time_bandwidth only applies to "multitaper" method.')
    elif method == 'multitaper':
        time_bandwidth = (4.0 if time_bandwidth is None
                          else float(time_bandwidth))

    # Check use_fft
    if not isinstance(use_fft, bool):
        raise ValueError('use_fft must be a boolean, got %s '
                         'instead.' % type(use_fft))
    # Check decim
    if isinstance(decim, int):
        decim = slice(None, None, decim)
    if not isinstance(decim, slice):
        raise ValueError('decim must be an integer or a slice, '
                         'got %s instead.' % type(decim))

    # Check output
    _check_option('output', output, ['complex', 'power', 'phase',
                                     'avg_power_itc', 'avg_power', 'itc'])
    _check_option('method', method, ['multitaper', 'morlet'])

    return freqs, sfreq, zero_mean, n_cycles, time_bandwidth, decim
