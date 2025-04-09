import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd

def plot_mean_with_color_gradient(df: pd.DataFrame, img=None, fig=None, ax=None, show_bar: bool = True, vmin=0, vmax=None):
    """
    Plot the mean (x, y) for each node with a color gradient representing uncertainty.

    Parameters:
      df (pd.DataFrame): DataFrame with columns ['Node', 'Sample', 'X', 'Y'].
      img: Optional background image.
      fig, ax: Matplotlib figure and axes objects.
      show_bar (bool): Whether to display a colorbar.
      vmin, vmax: Color mapping limits.

    Returns:
      fig, ax, scatter: The figure, axes, and scatter plot object.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    node_groups = df.groupby('Node')
    means = node_groups[['X', 'Y']].mean()
    stds = node_groups[['X', 'Y']].std()
    sigma_avg = (stds['X'] + stds['Y']) / 2

    if img is not None:
        ax.imshow(img, cmap='gray')

    scatter = ax.scatter(means['X'], means['Y'], c=sigma_avg, cmap='hot', s=50, vmin=vmin, vmax=vmax)
    if show_bar:
        fig.colorbar(scatter, ax=ax, label='Average Std (Uncertainty)')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(0, 1024)
    ax.set_ylim(0, 1024)
    ax.invert_yaxis()

    return fig, ax, scatter

def plot_mean_with_uncertainty(df: pd.DataFrame, img=None, use_error_bars: bool = False, scale_factor: float = 1.0, fig=None, ax=None):
    """
    Plot mean node locations with uncertainty visualization using either error bars or ellipses.

    Parameters:
      df (pd.DataFrame): DataFrame with columns 'Node', 'X', and 'Y'.
      img: Optional background image.
      use_error_bars (bool): If True, use error bars; otherwise, use ellipses.
      scale_factor (float): Factor to scale the uncertainty visualization.
      fig, ax: Matplotlib figure and axes objects.

    Returns:
      fig, ax: The figure and axes objects.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    node_groups = df.groupby('Node')
    means = node_groups[['X', 'Y']].mean()
    stds = node_groups[['X', 'Y']].std()

    if img is not None:
        ax.imshow(img, cmap='gray')

    ax.plot(means['X'], means['Y'], 'o', color='blue', label='Mean')

    if use_error_bars:
        ax.errorbar(means['X'], means['Y'],
                    xerr=1.96 * stds['X'] * scale_factor,
                    yerr=1.96 * stds['Y'] * scale_factor,
                    fmt='none', ecolor='red', alpha=0.5, elinewidth=0.5, capsize=3)
    else:
        for node, group in node_groups:
            cov = np.cov(group['X'], group['Y'])
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            axis_length_1 = np.sqrt(eigenvalues[0] * 5.991) * scale_factor
            axis_length_2 = np.sqrt(eigenvalues[1] * 5.991) * scale_factor
            angle = np.rad2deg(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            ell = Ellipse(xy=means.loc[node].values,
                          width=2 * axis_length_1,
                          height=2 * axis_length_2,
                          angle=angle,
                          edgecolor='red', facecolor='none', alpha=0.5)
            ax.add_patch(ell)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(0, 1024)
    ax.set_ylim(0, 1024)
    ax.invert_yaxis()

    return fig, ax, None

def plot_comparison(df_orig, img_orig, df_corr, img_corr, plot_fn, show_global_bar=False, **kwargs):
    """
    Create a side-by-side comparison plot using the provided plotting function.

    Parameters:
      df_orig (pd.DataFrame): DataFrame for the original image.
      img_orig (np.ndarray): Original image.
      df_corr (pd.DataFrame): DataFrame for the corrupted image.
      img_corr (np.ndarray): Corrupted image.
      plot_fn (callable): Plotting function to use (e.g., plot_mean_with_color_gradient or plot_mean_with_uncertainty).
    
    Returns:
      tuple: (fig, (ax1, ax2))
    """
    # Create figure and two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot for original image
    fig, ax1, scatter1 = plot_fn(df_orig, img=img_orig, fig=fig, ax=ax1, **kwargs)
    ax1.set_title('Original', fontsize=16)

    # Plot for corrupted image
    fig, ax2, scatter2 = plot_fn(df_corr, img=img_corr, fig=fig, ax=ax2, **kwargs)
    ax2.set_title('Corrupted', fontsize=16)

    # If 'show_bar' is specified and True, add a shared colorbar
    if show_global_bar:
        cbar = fig.colorbar(scatter1, ax=[ax1, ax2], orientation='vertical', fraction=0.05)
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)
        cbar.ax.set_position([0.87, 0.15, 0.03, 0.7])

    return fig, (ax1, ax2)

def plot_sigmas_mean(sigma_dict, yscale='linear', grid=True):
    """Plots sigma values for a given sigma dictionary and returns fig, ax."""
    n_images = len(sigma_dict)
    n_cols = 3
    n_rows = (n_images + n_cols - 1) // n_cols  

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), sharex=True)
    axes = axes.flatten()

    for i, (img_name, vals) in enumerate(sigma_dict.items()):
        ax = axes[i]
        ax.scatter(vals['corr_levels'], np.array(vals['sigmas']).mean(axis=1))
        ax.set_title(f"{img_name[:20]}")
        ax.set_yscale(yscale)
        if grid: 
            ax.grid(True, color='gray', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    return fig, axes  # Return fig, axes instead of showing it

def plot_global_uncertainty(sigma_dict, title=''):
    """Plots global uncertainty using a boxplot and returns fig, ax."""
    data = []
    labels = []

    for key, value in sigma_dict.items():
        data.append(value)
        labels.append(key)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=data, ax=ax)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_xlabel('Corruption')
    ax.set_ylabel('Avg std')
    ax.set_title(title)

    plt.tight_layout()
    return fig, ax