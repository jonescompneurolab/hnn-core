"""
==================================================
Visualize dipole contributions: UP vs DOWN components
==================================================

This example demonstrates how to visualize the different components that
contribute to UP (positive) vs DOWN (negative) dipole deflections throughout
a simulation. This helps understand which drives, layers, and biases are
responsible for different aspects of the recorded signal.

The visualization shows:
- UP components (typically proximal drives, L2 pyramidal activity) in red
- DOWN components (typically distal drives, L5 pyramidal activity) in blue  
- The total dipole signal overlaid in black
"""

# Authors: HNN-core developers

import matplotlib.pyplot as plt
import numpy as np

import hnn_core
from hnn_core import jones_2009_model, simulate_dipole
from hnn_core.dipole_contributions import plot_dipole_contributions, analyze_dipole_contributions

###############################################################################
# Create a network with multiple drives to demonstrate the visualization

# Load the base Jones 2009 model
net = jones_2009_model()

# Add additional drives to show different contributions
# Proximal drive (contributes to UP deflections)
net.add_evoked_drive(
    name='proximal_early',
    mu=50.0,
    sigma=5.0,
    numspikes=1,
    location='proximal',
    weights_ampa={'L2_pyramidal': 0.01, 'L5_pyramidal': 0.01},
    weights_nmda={'L2_pyramidal': 0.01, 'L5_pyramidal': 0.01},
)

# Distal drive (contributes to DOWN deflections)  
net.add_evoked_drive(
    name='distal_late',
    mu=120.0,
    sigma=8.0,
    numspikes=1,
    location='distal',
    weights_ampa={'L2_pyramidal': 0.02, 'L5_pyramidal': 0.02},
    weights_nmda={'L2_pyramidal': 0.02, 'L5_pyramidal': 0.02},
)

# Add a Poisson drive for sustained activity
net.add_poisson_drive(
    name='background_poisson',
    tstart=25.0,
    tstop=150.0,
    rate_constant=5.0,
    location='proximal',
    weights_ampa={'L2_pyramidal': 0.005, 'L5_pyramidal': 0.005},
    weights_nmda={'L2_pyramidal': 0.005, 'L5_pyramidal': 0.005},
)

# Add tonic bias
net.add_tonic_bias(
    amplitude={'L2_pyramidal': 0.5, 'L5_pyramidal': -0.3},
    t0=0.0,
    tstop=200.0
)

###############################################################################
# Simulate the dipole

print("Simulating dipole...")
dpls = simulate_dipole(net, tstop=200.0, n_trials=1)
dpl = dpls[0]

###############################################################################
# Analyze dipole contributions

print("\nAnalyzing dipole contributions...")
analysis = analyze_dipole_contributions(net, dpl, verbose=True)

###############################################################################
# Create the contribution visualization

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Full contribution breakdown
plot_dipole_contributions(
    net, dpl, 
    ax=axes[0],
    show_components=True,
    show_total=True,
    show=False
)
axes[0].set_title('Dipole Contributions: UP vs DOWN Components (Full View)')

# Plot 2: Focus on a specific time window to see details
plot_dipole_contributions(
    net, dpl,
    tmin=40.0, tmax=140.0,
    ax=axes[1], 
    show_components=True,
    show_total=True,
    show=False
)
axes[1].set_title('Dipole Contributions: Detailed View (40-140 ms)')

plt.tight_layout()
plt.show()

###############################################################################
# Create a summary plot showing the relationship between components

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Plot total UP contributions
if analysis['up_components']:
    total_up = analysis['total_up_contribution']
    axes[0].fill_between(dpl.times, 0, total_up, alpha=0.6, color='red', 
                        label='Total UP Contribution')
    axes[0].set_ylabel('UP Components\n(nAm)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

# Plot total DOWN contributions  
if analysis['down_components']:
    total_down = analysis['total_down_contribution']
    axes[1].fill_between(dpl.times, 0, total_down, alpha=0.6, color='blue',
                        label='Total DOWN Contribution')
    axes[1].set_ylabel('DOWN Components\n(nAm)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

# Plot actual dipole signal
axes[2].plot(dpl.times, dpl.data['agg'], 'k-', linewidth=2, label='Actual Dipole')
axes[2].set_ylabel('Total Dipole\n(nAm)')
axes[2].set_xlabel('Time (ms)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle('Dipole Component Analysis Summary', fontsize=14)
plt.tight_layout()
plt.show()

###############################################################################
# Demonstrate the method on the Dipole object directly

print("\nUsing Dipole object methods...")

# Use the plot_contributions method
fig = dpl.plot_contributions(net, tmin=0, tmax=200, show=False)
plt.title('Using dpl.plot_contributions() method')
plt.show()

# Use the analyze_contributions method
analysis_direct = dpl.analyze_contributions(net, verbose=False)
print(f"Found {len(analysis_direct['up_components'])} UP components")
print(f"Found {len(analysis_direct['down_components'])} DOWN components")

###############################################################################
# Educational summary

print("\n" + "="*60)
print("UNDERSTANDING UP vs DOWN DIPOLE CONTRIBUTIONS")
print("="*60)
print("""
This visualization helps understand the mechanistic origins of dipole signals:

UP COMPONENTS (Positive Deflections):
- L2/3 pyramidal cell activity
- Proximal drives (targeting basal dendrites)
- Positive tonic biases
- Generally represent excitatory input to superficial layers

DOWN COMPONENTS (Negative Deflections):  
- L5 pyramidal cell activity
- Distal drives (targeting apical dendrites)
- Negative tonic biases
- Generally represent excitatory input to deep layer apical dendrites

KEY INSIGHTS:
1. The timing of UP vs DOWN components reveals the sequence of activation
2. The relative magnitude shows which inputs dominate the signal
3. This analysis helps interpret experimental MEG/EEG data mechanistically
4. Different drive locations create characteristic dipole signatures

This type of analysis is particularly useful for:
- Understanding evoked response components (e.g., M50, M100 in MEG)
- Interpreting the effects of different stimulation protocols
- Designing experiments to target specific circuit elements
""")

print("Visualization complete!")