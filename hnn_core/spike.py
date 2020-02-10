"""Class to handle the spiking info."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>
#          Blake Caldwell <blake_caldwell@brown.edu>

import numpy as np


class Spike(object):
    """Spike class.

    Parameters
    ----------
    spiketimes : array
    spikegids : array
    cells: dict
    tstop: float

    Attributes
    ----------
    spiketimes : array
    spikegids : array
    cells: dict
    tstop: float
    """

    def __init__(self, spiketimes, spikegids, cells, tstop):
        self.spiketimes = spiketimes
        self.spikegids = spikegids
        self.cells = cells
        self.tstop = tstop

    def plot_input_hist(self, ax=None, show=True):
        """Plot the histogram of inputs.
        Parameters
        ----------
        ax : instance of matplotlib axis | None
            An axis object from matplotlib. If None,
            a new figure is created.
        show : bool
            If True, show the figure.
        Returns
        -------
        fig : instance of matplotlib Figure
            The matplotlib figure handle.
        """
        import matplotlib.pyplot as plt
        spikes = np.array(self.spiketimes)
        valid_gids = np.r_[[gid for gid in self.cells.keys()
                            if self.cells[gid].startswith('evokedProximal')]]
        mask_evprox = np.in1d(self.spikegids, valid_gids)
        valid_gids = np.r_[[gid for gid in self.cells.keys()
                            if self.cells[gid].startswith('evokedDistal')]]
        mask_evdist = np.in1d(self.spikegids, valid_gids)
        bins = np.linspace(0, self.tstop, 50)

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.hist(spikes[mask_evprox], bins, color='r', label='Proximal')
        ax.hist(spikes[mask_evdist], bins, color='g', label='Distal')
        plt.legend()
        if show:
            plt.show()
        return ax.get_figure()

    def plot(self, ax=None, show=True):
        """Plot the spiking activity for each cell type.
        Parameters
        ----------
        ax : instance of matplotlib axis | None
            An axis object from matplotlib. If None,
            a new figure is created.
        show : bool
            If True, show the figure.
        Returns
        -------
        fig : instance of matplotlib Figure
            The matplotlib figure object
        """
        import matplotlib.pyplot as plt
        spikes = np.array(self.spiketimes)
        gids = np.array(self.spikegids)
        spike_times = np.zeros((4, spikes.shape[0]))
        cell_types = ['L5Pyr', 'L5Basket', 'L2Pyr', 'L2Basket']
        for idx, key in enumerate(cell_types):
            valid_gids = np.r_[[gid for gid in self.cells.keys()
                                if self.cells[gid].startswith(key)]]
            mask = np.in1d(gids, valid_gids)
            spike_times[idx, mask] = spikes[mask]

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        ax.eventplot(spike_times, colors=['r', 'b', 'g', 'w'])
        ax.legend(cell_types, ncol=2)
        ax.set_facecolor('k')
        ax.set_xlabel('Time (ms)')
        ax.get_yaxis().set_visible(False)
        ax.set_ylim((-1, 4.5))

        if show:
            plt.show()
        return ax.get_figure()
