"""External drives to network."""

# Authors: Christopher Bailey <bailey.cj@gmail.com>


def _check_drive_parameter_values(drive_type, **kwargs):
    if 'sigma' in kwargs:
        if kwargs['sigma'] < 0.:
            raise ValueError('XStandard deviation cannot be negative')
    if 'numspikes' in kwargs:
        if not kwargs['numspikes'] > 0:
            raise ValueError('Number of spikes must be greater than zero')
    if 't0' in kwargs:
        if kwargs['t0'] < 0:
            raise ValueError(f'Start time of {drive_type} drive cannot be '
                             'negative')
    if 'T' in kwargs:
        if kwargs['T'] < 0.:
            raise ValueError(f'End time of {drive_type} drive cannot be '
                             'negative')
    if 'T' in kwargs and 'tstop' in kwargs:
        if kwargs['T'] > kwargs['tstop']:
            raise ValueError(f"End time of {drive_type} drive cannot exceed "
                             f"simulation end time {kwargs['tstop']}. Got "
                             f"{kwargs['T']}.")
    if 't0' in kwargs and 'T' in kwargs:
        if kwargs['T'] - kwargs['t0'] < 0.:
            raise ValueError(f'Duration of {drive_type} drive cannot be '
                             'negative')

    if 'numspikes' in kwargs and 'spike_isi' in kwargs and 'burst_f' in kwargs:
        nspikes = kwargs['numspikes']
        isi = kwargs['spike_isi']
        burst_period = 1000. / kwargs['burst_f']
        if (nspikes - 1) * isi > burst_period:
            raise ValueError('Burst duration cannot be greater than period')
