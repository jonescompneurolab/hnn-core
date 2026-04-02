"""Dipole contribution analysis and visualization."""

# Authors: HNN-core developers

import numpy as np
import matplotlib.pyplot as plt
from .externals.mne import _validate_type


def _identify_up_down_components(net, dpls):
    """Identify components that contribute to UP vs DOWN dipole deflections.
    
    Parameters
    ----------
    net : Network
        The network object containing drive and connectivity information.
    dpls : list of Dipole
        List of dipole objects from simulation.
        
    Returns
    -------
    up_components : dict
        Dictionary of components that contribute to UP (positive) deflections.
        Keys are component names, values are contribution time series.
    down_components : dict  
        Dictionary of components that contribute to DOWN (negative) deflections.
        Keys are component names, values are contribution time series.
    """
    up_components = {}
    down_components = {}
    
    # Get the first dipole for timing reference
    dpl = dpls[0] if isinstance(dpls, list) else dpls
    times = dpl.times
    
    # L2 pyramidal cells generally contribute UP (positive deflections)
    # L5 pyramidal cells generally contribute DOWN (negative deflections)
    if 'L2' in dpl.data:
        up_components['L2_pyramidal'] = dpl.data['L2']
    if 'L5' in dpl.data:
        down_components['L5_pyramidal'] = dpl.data['L5']
    
    # Analyze drives based on their location and target
    for drive_name, drive_info in net.external_drives.items():
        location = drive_info.get('location', 'unknown')
        drive_type = drive_info.get('type', 'unknown')
        
        # Proximal drives (to basal dendrites) generally cause UP deflections
        # Distal drives (to apical dendrites) generally cause DOWN deflections
        if location == 'proximal':
            # Proximal drives contribute to UP deflections
            # Create a synthetic contribution based on drive timing
            contribution = _estimate_drive_contribution(
                drive_info, times, polarity='up'
            )
            up_components[f'{drive_name}_({drive_type})'] = contribution
            
        elif location == 'distal':
            # Distal drives contribute to DOWN deflections  
            contribution = _estimate_drive_contribution(
                drive_info, times, polarity='down'
            )
            down_components[f'{drive_name}_({drive_type})'] = contribution
    
    # Analyze tonic biases
    for bias_name, bias_info in net.external_biases.items():
        # Tonic biases can contribute to both UP and DOWN depending on amplitude
        for cell_type, bias_params in bias_info.items():
            amplitude = bias_params.get('amplitude', 0)
            t0 = bias_params.get('t0', 0)
            tstop = bias_params.get('tstop', times[-1])
            
            # Create tonic contribution
            contribution = np.zeros_like(times)
            mask = (times >= t0) & (times <= tstop)
            contribution[mask] = amplitude * 0.1  # Scale factor for visualization
            
            if amplitude > 0:
                up_components[f'{bias_name}_{cell_type}_tonic'] = contribution
            else:
                down_components[f'{bias_name}_{cell_type}_tonic'] = contribution
    
    return up_components, down_components


def _estimate_drive_contribution(drive_info, times, polarity='up'):
    """Estimate the contribution of a drive to the dipole signal.
    
    Parameters
    ----------
    drive_info : dict
        Drive information from network.
    times : array
        Time points for the simulation.
    polarity : str
        'up' for positive contributions, 'down' for negative.
        
    Returns
    -------
    contribution : array
        Estimated contribution time series.
    """
    contribution = np.zeros_like(times)
    dynamics = drive_info.get('dynamics', {})
    drive_type = drive_info.get('type', 'unknown')
    
    if drive_type == 'evoked':
        # Evoked drives create brief deflections
        mu = dynamics.get('mu', 0)
        sigma = dynamics.get('sigma', 1)
        numspikes = dynamics.get('numspikes', 1)
        
        # Create Gaussian-shaped contribution
        amplitude = numspikes * 10  # Scaling factor
        if polarity == 'down':
            amplitude = -amplitude
            
        contribution = amplitude * np.exp(-0.5 * ((times - mu) / sigma) ** 2)
        
    elif drive_type == 'poisson':
        # Poisson drives create sustained activity
        tstart = dynamics.get('tstart', 0)
        tstop = dynamics.get('tstop', times[-1])
        rate = dynamics.get('rate_constant', 10)
        
        # Create sustained contribution during drive period
        amplitude = rate * 0.5  # Scaling factor
        if polarity == 'down':
            amplitude = -amplitude
            
        mask = (times >= tstart) & (times <= tstop)
        contribution[mask] = amplitude
        
    elif drive_type == 'bursty':
        # Bursty drives create rhythmic contributions
        tstart = dynamics.get('tstart', 0)
        tstop = dynamics.get('tstop', times[-1])
        burst_rate = dynamics.get('burst_rate', 10)
        
        # Create rhythmic contribution
        amplitude = 20  # Scaling factor
        if polarity == 'down':
            amplitude = -amplitude
            
        mask = (times >= tstart) & (times <= tstop)
        # Add rhythmic modulation
        freq = burst_rate / 1000.0  # Convert to Hz
        rhythmic = np.sin(2 * np.pi * freq * times[mask])
        contribution[mask] = amplitude * (1 + 0.5 * rhythmic)
    
    return contribution


def plot_dipole_contributions(
    net, 
    dpls, 
    tmin=None, 
    tmax=None, 
    ax=None, 
    show_components=True,
    show_total=True,
    show=True
):
    """Plot dipole contributions showing UP vs DOWN components.
    
    This visualization shows how different network components (drives, layers,
    biases) contribute to the overall dipole signal, with components that
    typically cause UP (positive) deflections shown above zero and components
    that cause DOWN (negative) deflections shown below zero.
    
    Parameters
    ----------
    net : Network
        The network object containing drive and connectivity information.
    dpls : Dipole | list of Dipole
        Dipole object(s) from simulation. If list, will use the first one.
    tmin : float | None
        Start time for plotting (ms). If None, use full simulation.
    tmax : float | None  
        End time for plotting (ms). If None, use full simulation.
    ax : matplotlib.axes.Axes | None
        Axes to plot into. If None, creates new figure.
    show_components : bool
        Whether to show individual component contributions.
    show_total : bool
        Whether to show the total dipole signal.
    show : bool
        Whether to display the figure.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """
    import matplotlib.pyplot as plt
    
    # Validate inputs
    _validate_type(net, 'Network', 'net')
    if isinstance(dpls, list):
        dpl = dpls[0]
    else:
        dpl = dpls
    
    # Get time range
    times = dpl.times
    if tmin is not None:
        start_idx = np.argmin(np.abs(times - tmin))
    else:
        start_idx = 0
        tmin = times[0]
        
    if tmax is not None:
        end_idx = np.argmin(np.abs(times - tmax))
    else:
        end_idx = len(times)
        tmax = times[-1]
    
    times_plot = times[start_idx:end_idx]
    
    # Identify UP and DOWN components
    up_components, down_components = _identify_up_down_components(net, dpl)
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure
    
    # Color schemes
    up_colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(up_components)))
    down_colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(down_components)))
    
    # Plot UP components (positive contributions)
    if show_components:
        for i, (name, contribution) in enumerate(up_components.items()):
            contrib_plot = contribution[start_idx:end_idx]
            ax.fill_between(
                times_plot, 0, contrib_plot, 
                alpha=0.6, color=up_colors[i], 
                label=f'UP: {name}'
            )
    
    # Plot DOWN components (negative contributions)  
    if show_components:
        for i, (name, contribution) in enumerate(down_components.items()):
            contrib_plot = contribution[start_idx:end_idx]
            ax.fill_between(
                times_plot, 0, contrib_plot,
                alpha=0.6, color=down_colors[i],
                label=f'DOWN: {name}'
            )
    
    # Plot total dipole signal
    if show_total:
        total_signal = dpl.data['agg'][start_idx:end_idx]
        ax.plot(times_plot, total_signal, 'k-', linewidth=2, 
                label='Total Dipole', zorder=10)
    
    # Add zero line
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Formatting
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Dipole Moment (nAm)')
    ax.set_title('Dipole Contributions: UP vs DOWN Components')
    ax.grid(True, alpha=0.3)
    
    # Legend
    if show_components or show_total:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add annotations
    ax.text(0.02, 0.98, 'UP Components\n(Positive Deflections)', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.1))
    
    ax.text(0.02, 0.02, 'DOWN Components\n(Negative Deflections)',
            transform=ax.transAxes, verticalalignment='bottom', 
            bbox=dict(boxstyle='round', facecolor='blue', alpha=0.1))
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig


def analyze_dipole_contributions(net, dpls, verbose=True):
    """Analyze and summarize dipole contributions.
    
    Parameters
    ----------
    net : Network
        The network object.
    dpls : Dipole | list of Dipole
        Dipole object(s) from simulation.
    verbose : bool
        Whether to print analysis summary.
        
    Returns
    -------
    analysis : dict
        Dictionary containing analysis results.
    """
    if isinstance(dpls, list):
        dpl = dpls[0]
    else:
        dpl = dpls
    
    up_components, down_components = _identify_up_down_components(net, dpl)
    
    analysis = {
        'up_components': up_components,
        'down_components': down_components,
        'total_up_contribution': sum(up_components.values()) if up_components else np.zeros_like(dpl.times),
        'total_down_contribution': sum(down_components.values()) if down_components else np.zeros_like(dpl.times),
        'net_contribution': None
    }
    
    # Calculate net contribution
    total_up = analysis['total_up_contribution']
    total_down = analysis['total_down_contribution']
    analysis['net_contribution'] = total_up + total_down
    
    if verbose:
        print("Dipole Contribution Analysis")
        print("=" * 40)
        print(f"UP Components ({len(up_components)}):")
        for name in up_components.keys():
            print(f"  - {name}")
        
        print(f"\nDOWN Components ({len(down_components)}):")
        for name in down_components.keys():
            print(f"  - {name}")
        
        # Calculate RMS contributions
        if up_components:
            up_rms = np.sqrt(np.mean(analysis['total_up_contribution'] ** 2))
            print(f"\nTotal UP RMS contribution: {up_rms:.2f} nAm")
        
        if down_components:
            down_rms = np.sqrt(np.mean(analysis['total_down_contribution'] ** 2))
            print(f"Total DOWN RMS contribution: {down_rms:.2f} nAm")
    
    return analysis