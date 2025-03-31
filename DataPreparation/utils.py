import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from collections import defaultdict
from natsort import natsorted


def load_image_and_samples(img_dir, output_dir, img_name):
    """
    Load a grayscale image and extract landmark coordinates from all sample files.

    Parameters:
    - img_dir (str): Directory containing the input image.
    - output_dir (str): Directory containing the sample output files.
    - img_name (str): Name of the image file (e.g., 'image.png').

    Returns:
    - df (pd.DataFrame): DataFrame with columns 'Node', 'Sample', 'X', 'Y'.
    - img (np.ndarray): Loaded grayscale image as a NumPy array.
    """
    # Construct the full image path
    img_path = os.path.join(img_dir, img_name)
    
    # Load the grayscale image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {img_path}")

    base_name = img_name.split('.png')[0]

    # Find all sample files in the output directory matching the pattern
    sample_files = [f for f in os.listdir(output_dir) if f.startswith(base_name) and f.endswith('.txt')]
    n_samples = len(sample_files)
    if n_samples == 0:
        raise FileNotFoundError(f"No sample files found for {base_name} in {output_dir}")

    # Sort files to ensure consistent sample numbering (e.g., _1.txt, _2.txt, etc.)
    sample_files.sort(key=lambda x: int(x.split('_')[-1].split('.txt')[0]))

    data = []

    # Loop through each sample file
    for sample_idx, sample_file in enumerate(sample_files, 1):
        output_path = os.path.join(output_dir, sample_file)
        x_coords = []
        y_coords = []

        # Read coordinates from the file
        with open(output_path, 'r') as file:
            for line in file:
                x, y = map(float, line.strip().split())
                x_coords.append(x)
                y_coords.append(y)

        # Add coordinates to the data list
        for i in range(len(x_coords)):
            data.append({
                'Node': i + 1,  # Landmark index (1-based)
                'Sample': sample_idx,  # Sample number (1-based)
                'X': x_coords[i],
                'Y': y_coords[i]
            })

    # Create DataFrame
    df = pd.DataFrame(data)

    return df, img

def extract_landmarks(file_name, split=True):
    landmarks_dir = 'Annotations'
    landmarks_dict = {}
    
    if split:
        file_name = '_'.join(file_name.split('_')[:-1])

    for key in ['RL', 'LL', 'H']:
        file_path = os.path.join(landmarks_dir, key, file_name + '.npy')
        if os.path.exists(file_path):
            landmarks_dict[key] = np.load(file_path)
        else:
            landmarks_dict[key] = None  
    
    return landmarks_dict


def get_error(img_dir, output_dir, img_name):
    """Calculate the error and std for each landmark in the image."""
    df, _ = load_image_and_samples(img_dir, output_dir, img_name)
    node_groups = df.groupby('Node')
    
    means = node_groups[['X', 'Y']].mean().values
    stds = node_groups[['X', 'Y']].std()
    sigmas = (stds['X'] + stds['Y']) / 2
    
    # Take mean (X,Y) location as the prediction
    RL_pred = means[:44]
    LL_pred = means[44:94]
    H_pred = means[94:]
    
    # Extract ground-truth landmarks
    landmarks_gt = extract_landmarks(img_name)
    RL_gt = landmarks_gt['RL']
    LL_gt = landmarks_gt['LL']
    H_gt = landmarks_gt['H']

    if H_gt is not None:
        pred = np.concatenate([RL_pred, LL_pred, H_pred])
        gt = np.concatenate([RL_gt, LL_gt, H_gt])
    else:
        pred = np.concatenate([RL_pred, LL_pred])
        gt = np.concatenate([RL_gt, LL_gt])
        sigmas = sigmas[:94]
    
    error = np.linalg.norm(pred - gt, axis=1)
    return error, sigmas.values


def find_avg_std(img_dir, output_dir, img_file):
    corr_level = int(img_file.split('_')[-1].strip('.png'))

    df, _ = load_image_and_samples(img_dir, output_dir, img_file)
    node_groups = df.groupby('Node')
    stds = node_groups[['X', 'Y']].std()
    sigma = (stds['X'] + stds['Y']) / 2

    return sigma, corr_level


def process_corr_images(img_dir_corr, output_dir_corr, selected_images):
    """Process all corrupted images and calculate sigma averages for each selected image."""
    sigma_dict = defaultdict(lambda: {"sigma_avgs": [], "corr_levels": []})

    for img_name in selected_images:
        img_prefix = img_name.replace('.png', '')
        
        corr_images = natsorted(
            [f for f in os.listdir(img_dir_corr) if f.startswith(img_prefix)]
        )

        for img_file in corr_images:
            sigma, corr_level = find_avg_std(img_dir_corr, output_dir_corr, img_file)
            sigma_dict[img_name]["sigma_avgs"].append(sigma.mean())
            sigma_dict[img_name]["corr_levels"].append(corr_level)
    
    return sigma_dict


def read_sigma_file(file_path):
    sigma_dict = {}

    with open(file_path, 'r') as f:
        for line in f:
            # Split by space to get image name and sigma avg
            img_name, sigma_avg = line.strip().split()
            sigma_avg = float(sigma_avg)  # Convert sigma avg to float

            # Extract corruption level (last part of the filename)
            corr_level = int(img_name.split('_')[-1].split('.')[0])

            # Get the base image name without corruption level
            base_img_name = '_'.join(img_name.split('_')[:-1]) + '.png'

            # If base_img_name is not in the dictionary, add it
            if base_img_name not in sigma_dict:
                sigma_dict[base_img_name] = {'corr_levels': [], 'sigma_avgs': []}
            
            # Append the corruption level and sigma average
            sigma_dict[base_img_name]['corr_levels'].append(corr_level)
            sigma_dict[base_img_name]['sigma_avgs'].append(sigma_avg)

    return sigma_dict


def apply_occlusion(image, landmarks, size):
    """
    Apply occlusion around the specified landmarks.

    Parameters:
    - image: Grayscale image as a numpy array of shape (height, width).
    - landmarks: List of tuples (x, y), where x is the column and y is the row.
    - size: Integer, the size of the square region around each landmark to apply the occlusion.

    Returns:
    - occluded_image: The image with occlusions applied around the landmarks, as uint8.
    """
    if not isinstance(size, int) or size < 0:
        raise ValueError("size must be a non-negative integer")

    occluded_image = image.copy()

    for x, y in landmarks:
        x_start = max(0, x - size)
        x_end = min(image.shape[1], x + size + 1)
        y_start = max(0, y - size)
        y_end = min(image.shape[0], y + size + 1)

        occluded_image[y_start:y_end, x_start:x_end] = 0  # Set region to black

    return occluded_image.astype(np.uint8)


def apply_blur(image, landmarks, size, severity):
    """
    Apply Gaussian blur around the specified landmarks.

    Parameters:
    - image: Grayscale image as a numpy array of shape (height, width).
    - landmarks: List of tuples (x, y), where x is the column and y is the row.
    - size: Integer, the size of the square region around each landmark to apply the blur.
    - severity: Float between 0 and 1, indicating the severity of the blur.

    Returns:
    - blurred_image: The image with blurring applied around the landmarks, as uint8.
    """
    if severity < 0:
        raise ValueError("severity must be non-negative")
    if not isinstance(size, int) or size < 0:
        raise ValueError("size must be a non-negative integer")

    blurred_image = image.astype(np.float32).copy()

    for x, y in landmarks:
        x_start = max(0, x - size)
        x_end = min(image.shape[1], x + size + 1)
        y_start = max(0, y - size)
        y_end = min(image.shape[0], y + size + 1)
        region = blurred_image[y_start:y_end, x_start:x_end]

        ksize = 2*severity + 1 # Kernel size must be odd
        if ksize > 1:
            region = cv2.GaussianBlur(region, (ksize, ksize), 0)

        region = np.clip(region, 0, 255)
        blurred_image[y_start:y_end, x_start:x_end] = region

    return blurred_image.astype(np.uint8)



def plot_mean_with_color_gradient(df, img=None, fig=None, ax=None, show_bar=True, vmin=0, vmax=None):
    """
    Plot the mean (x,y) for each node and use a color gradient to show uncertainty.
    Uncertainty is computed as the average of the std deviations in the x and y directions.

    Args:
      df (pd.DataFrame): DataFrame with columns ['Node', 'Sample', 'X', 'Y']
      img (np.array, optional): Underlying image to overlay the landmarks.
      fig (matplotlib.figure.Figure, optional): Existing figure to plot on.
      ax (matplotlib.axes.Axes, optional): Existing axes to plot on.
      show_bar (bool, optional): Whether to show the colorbar (default: True).

    Returns:
      fig (matplotlib.figure.Figure): The figure object.
      ax (matplotlib.axes.Axes): The axes object.
      scatter (matplotlib.collections.PathCollection): The scatter object (if created).
    """
    # Create figure and axes if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    # Calculate mean and standard deviation for each node
    node_groups = df.groupby('Node')
    means = node_groups[['X', 'Y']].mean()
    stds = node_groups[['X', 'Y']].std()
    sigma_avg = (stds['X'] + stds['Y']) / 2  # Average std as uncertainty measure
    
    # Plot background image if provided
    if img is not None:
        ax.imshow(img, cmap='gray')
    
    # Plot points with color gradient for uncertainty
    scatter = ax.scatter(means['X'], means['Y'], c=sigma_avg, cmap='hot', s=50, vmin=vmin, vmax=vmax)
    
    # Add colorbar only if show_bar is True
    if show_bar:
        fig.colorbar(scatter, ax=ax, label='Average Std (Uncertainty)')
    
    # Finalize the plot settings
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(0, 1024)
    ax.set_ylim(0, 1024)
    ax.invert_yaxis()

    return fig, ax, scatter


def plot_mean_with_uncertainty(df, img=None, use_error_bars=False, scale_factor=1.0, fig=None, ax=None):
    """
    Plot mean node locations with uncertainty, either as ellipses or error bars.

    Parameters:
    - df: DataFrame with 'Node', 'X', and 'Y' coordinates for each node.
    - img: Optional background image to display (default: None).
    - use_error_bars: Boolean to choose error bars (True) or ellipses (False) (default: False).
    - scale_factor: Multiplicative factor to scale the uncertainty visualizations (default: 1.0).
    - fig: Existing matplotlib figure object (optional).
    - ax: Existing matplotlib axes object (optional).

    Returns:
      fig (matplotlib.figure.Figure): The figure object.
      ax (matplotlib.axes.Axes): The axes object.
    """
    # Create figure and axes if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Calculate mean and standard deviation for each node
    node_groups = df.groupby('Node')
    means = node_groups[['X', 'Y']].mean()
    stds = node_groups[['X', 'Y']].std()
    
    # Plot background image if provided
    if img is not None:
        ax.imshow(img, cmap='gray')
    
    # Plot the mean points for all nodes
    ax.plot(means['X'], means['Y'], 'o', color='blue', label='Mean')
    
    if use_error_bars:
        # Add error bars scaled by 1.96 and the scale_factor
        ax.errorbar(means['X'], means['Y'], 
                    xerr=1.96 * stds['X'] * scale_factor, 
                    yerr=1.96 * stds['Y'] * scale_factor,
                    fmt='none', ecolor='red', alpha=0.5, elinewidth=0.5, capsize=3)
    else:
        # Add uncertainty ellipses for each node
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

    # Finalize the plot settings
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(0, 1024)
    ax.set_ylim(0, 1024)
    ax.invert_yaxis()

    return fig, ax