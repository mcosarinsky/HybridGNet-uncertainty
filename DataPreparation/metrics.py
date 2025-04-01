import os
import numpy as np
import pandas as pd
from collections import defaultdict
from natsort import natsorted
from .io import load_image_and_samples, extract_landmarks

def get_error(img_dir: str, output_dir: str, img_name: str):
    """
    Calculate the error and standard deviation for each landmark in the image.

    Returns:
      error (np.ndarray): Norm error for each landmark.
      sigmas (np.ndarray): Standard deviation values.
    """
    df, _ = load_image_and_samples(img_dir, output_dir, img_name)
    node_groups = df.groupby('Node')
    means = node_groups[['X', 'Y']].mean().values
    stds = node_groups[['X', 'Y']].std()
    sigmas = (stds['X'] + stds['Y']) / 2
    
    RL_pred = means[:44]
    LL_pred = means[44:94]
    H_pred = means[94:]
    
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

def find_avg_std(img_dir: str, output_dir: str, img_file: str):
    """
    Calculate the average standard deviation for a given image file.

    Returns:
      sigma (np.ndarray): Standard deviation values.
      corr_level (int): Corruption level extracted from filename.
    """
    corr_level = int(img_file.split('_')[-1].strip('.png'))
    df, _ = load_image_and_samples(img_dir, output_dir, img_file)
    node_groups = df.groupby('Node')
    stds = node_groups[['X', 'Y']].std()
    sigma = (stds['X'] + stds['Y']) / 2
    return sigma, corr_level

def process_corr_images(img_dir_corr: str, output_dir_corr: str, selected_images: list) -> dict:
    """
    Process corrupted images and calculate sigma averages for each selected image.

    Returns:
      dict: Dictionary mapping image names to sigma averages and corruption levels.
    """
    sigma_dict = defaultdict(lambda: {"sigmas": [], "corr_levels": []})
    for img_name in selected_images:
        img_prefix = img_name.replace('.png', '')
        corr_images = natsorted([f for f in os.listdir(img_dir_corr) if f.startswith(img_prefix)])
        for img_file in corr_images:
            sigma, corr_level = find_avg_std(img_dir_corr, output_dir_corr, img_file)
            sigma_dict[img_name]["sigmas"].append(sigma.values)
            sigma_dict[img_name]["corr_levels"].append(corr_level)
    return sigma_dict

def read_sigma_file(file_path: str) -> dict:
    """
    Read a sigma file and parse sigma averages for each image.

    Returns:
      dict: Mapping of base image names to their corruption levels and sigma averages.
    """
    sigma_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            img_name = parts[0]
            sigmas = np.array(parts[1:], dtype=np.float32)
            corr_level = int(img_name.split('_')[-1].split('.')[0])
            base_img_name = '_'.join(img_name.split('_')[:-1]) + '.png'

            if base_img_name not in sigma_dict:
                sigma_dict[base_img_name] = {'corr_levels': [], 'sigmas': []}
            sigma_dict[base_img_name]['corr_levels'].append(corr_level)
            sigma_dict[base_img_name]['sigmas'].append(sigmas)
    return sigma_dict

def compute_global_vmax(df_original, df_corrupted):
    """Compute the maximum sigma_avg across both original and corrupted datasets."""
    def calc_sigma_avg(df):
        stds = df.groupby('Node')[['X', 'Y']].std()
        return ((stds['X'] + stds['Y']) / 2).max()
    
    max_original = calc_sigma_avg(df_original)
    max_corrupted = calc_sigma_avg(df_corrupted)
    
    return max(max_original, max_corrupted)