import os
import random
import pathlib
import cv2
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from natsort import natsorted

def load_image_and_samples(img_name, img_dir, skips=True):
    base_name = os.path.splitext(img_name)[0]
    img = cv2.imread(os.path.join(img_dir, img_name), cv2.IMREAD_GRAYSCALE)

    output_dir = img_dir.replace('Images', f"Predictions/{'Skip' if skips else 'NoSkip'}")
    samples_path = os.path.join(output_dir, base_name + '.csv')

    if not os.path.exists(samples_path):
        raise FileNotFoundError(f"Samples file not found: {samples_path}")

    samples = pd.read_csv(samples_path, index_col=0)
    return samples, img

def extract_landmarks(file_name: str) -> dict:
    file_name = file_name.split('.')[0]  # Remove file extension
    landmarks_dir = 'Annotations'
    landmarks_dict = {}
    
    for key in ['RL', 'LL', 'H']:
        file_path = os.path.join(landmarks_dir, key, file_name + '.npy')
        if os.path.exists(file_path):
            landmarks_dict[key] = np.load(file_path)
        else:
            landmarks_dict[key] = None  
    return landmarks_dict

def sample_landmarks(landmark_dict, organ, n_samples=1):
    if landmark_dict.get(organ) is None:
        return np.empty((0,2))
    else:
        landmarks = landmark_dict[organ]
        n_samples = min(n_samples, len(landmarks))
        
        indices = np.random.choice(len(landmarks), n_samples, replace=False)

        return landmarks[indices]

def get_error(output_dir, file_name):
    base_name = os.path.splitext(file_name)[0]
    samples_path = os.path.join(output_dir, file_name)
    df = pd.read_csv(samples_path, index_col=0)

    landmarks_dict = extract_landmarks(file_name)
    gt = np.concatenate([landmarks_dict[k] for k in ['RL', 'LL', 'H'] if landmarks_dict[k] is not None])
    n_nodes = gt.shape[0]

    pred = np.array(df[['Mean x', 'Mean y']])[:n_nodes]
    sigmas = np.array((df['Std x'] + df['Std y']) / 2)[:n_nodes]

    error = np.linalg.norm(pred - gt, axis=1)

    return error, sigmas

def compute_global_vmax(df_original, df_corrupted):
    """Compute the maximum sigma_avg across both original and corrupted DataFrames."""
    def calc_sigma_avg(df):
        return ((df['Std x'] + df['Std y']) / 2).max()

    max_original = calc_sigma_avg(df_original)
    max_corrupted = calc_sigma_avg(df_corrupted)

    return max(max_original, max_corrupted)

def find_avg_std(file: str):
    df = pd.read_csv(file, index_col=0)
    sigma = (df['Std x'] + df['Std y']) / 2
    return np.array(sigma)

def read_sigma_files(file_path):
    sigma_dict = defaultdict(lambda: {"sigmas": [], "corr_levels": []})

    for folder in natsorted(os.listdir(file_path)):
        subdir = os.path.join(file_path, folder)
        sigma_file = [f for f in os.listdir(subdir) if f.endswith('.txt')][0]

        with open(os.path.join(subdir, sigma_file), 'r') as f:
            for line in f:
                parts = line.strip().split()
                img_name = parts[0].split('.')[0]

                # Latents are stored as variance, take sqrt to get std
                sigmas = np.sqrt(np.array(parts[1:], dtype=np.float32))
                sigma_dict[img_name]['corr_levels'].append(folder)
                sigma_dict[img_name]['sigmas'].append(sigmas)

    return sigma_dict

def process_and_store_sigma(corr_dir):
    sigma_dict = defaultdict(lambda: {"sigmas": [], "corr_levels": []})

    for folder in natsorted(os.listdir(corr_dir)):
        for file in os.listdir(os.path.join(corr_dir, folder)):
            if file.endswith('.csv'):
                sigma = find_avg_std(os.path.join(corr_dir, folder, file))
                file_name = os.path.splitext(file)[0]
                sigma_dict[file_name]["sigmas"].append(sigma)
                sigma_dict[file_name]["corr_levels"].append(folder)
    return sigma_dict

def read_latents(dataset_path):
    all_sigmas = []

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                print(file_path)
                with open(file_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) > 1:
                            sigmas = np.sqrt(np.array(parts[1:], dtype=np.float32))  # Convert variances to std dev
                            all_sigmas.append(sigmas)

def find_uncertainty(dataset_path, reduce=False):
    all_sigmas = []

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.csv'):
                sigmas = find_avg_std(os.path.join(root, file))
                if reduce:
                    all_sigmas.append(sigmas.mean())
                else:
                    all_sigmas.append(sigmas)
    return np.array(all_sigmas).flatten()