
import torch
import numpy as np
import matplotlib.pyplot as plt

def vis_freq(snap):
    # --- Perform Fourier Transform ---
    snap_fft = torch.fft.rfft(snap, dim=0)

    # --- Visualization of Individual Frequency Components ---
    freq_indices_to_plot = range(23) # Plot ω = 0 to 22 as requested

    # Determine the grid size for the subplots
    n_freqs = len(freq_indices_to_plot)
    n_cols = 6  # Let's arrange them in a grid with 6 columns
    n_rows = (n_freqs + n_cols - 1) // n_cols # Calculate necessary rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12))
    fig.suptitle(f'Amplitude of Frequency Components ω=0 to {n_freqs-1} for Sample ', fontsize=20)

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    for i, k in enumerate(freq_indices_to_plot):
        ax = axes[i]
        
        # Get the 2D complex data for the k-th frequency component
        freq_component = snap_fft[k, :, :]
        
        # Calculate the amplitude (magnitude) of the complex numbers
        amplitude_map = torch.abs(freq_component).numpy()
        
        # Plot the amplitude map
        # The DC component (k=0) can have a very different scale, so we use LogNorm for better visualization
        if k == 0:
            im = ax.imshow(amplitude_map, cmap='viridis')
        else:
            # For AC components, LogNorm helps to see details when amplitudes vary a lot
            from matplotlib.colors import LogNorm
            im = ax.imshow(amplitude_map, cmap='viridis', norm=LogNorm(vmin=np.percentile(amplitude_map, 5), vmax=np.percentile(amplitude_map, 99.5)))

        ax.set_title(f'ω = {k}')
        ax.axis('off') # Hide axes ticks for clarity

    # Turn off any unused subplots
    for i in range(n_freqs, len(axes)):
        axes[i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- Optional: Visualize Phase as well ---
    fig_phase, axes_phase = plt.subplots(n_rows, n_cols, figsize=(18, 12))
    fig_phase.suptitle(f'Phase of Frequency Components ω=0 to {n_freqs-1} for Sample', fontsize=20)
    axes_phase = axes_phase.flatten()

    for i, k in enumerate(freq_indices_to_plot):
        ax = axes_phase[i]
        
        # Get the 2D complex data for the k-th frequency component
        freq_component = snap_fft[k, :, :]
        
        # Calculate the phase (angle) of the complex numbers
        phase_map = torch.angle(freq_component).numpy()
        
        # Plot the phase map. 'twilight_shifted' is a good cyclic colormap for phase.
        im = ax.imshow(phase_map, cmap='twilight_shifted')

        ax.set_title(f'ω = {k}')
        ax.axis('off')

    for i in range(n_freqs, len(axes_phase)):
        axes_phase[i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("FreqVis.png")