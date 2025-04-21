import argparse
from models.HybridGNet2IGSC import Hybrid, HybridNoSkip
from tqdm import tqdm

import os 
import numpy as np
from torchvision import transforms
import torch

from utils.utils import scipy_to_torch_sparse, genMatrixesLungsHeart
import scipy.sparse as sp

import cv2
import pathlib
import re

def natural_key(string_):
    """Sort helper for natural string sorting."""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def getDenseMask(RL, LL, H):
    img = np.zeros([1024, 1024], dtype='uint8')
    
    RL = RL.reshape(-1, 1, 2).astype('int')
    LL = LL.reshape(-1, 1, 2).astype('int')
    H = H.reshape(-1, 1, 2).astype('int')

    img = cv2.drawContours(img, [RL], -1, 1, -1)
    img = cv2.drawContours(img, [LL], -1, 1, -1)
    img = cv2.drawContours(img, [H], -1, 2, -1)
    
    return img


def main(img_name, n_samples, output_file, base_folder, folder_name=None, skip_connections=True):
    """
    Process images to generate multiple samples per image efficiently by
    encoding once and decoding multiple times with different z values.
    
    Args:
        img_name: Name pattern of the image(s) to process (or None for all images)
        n_samples: Number of samples to generate per image
        folder_name: Optional subfolder name
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load and prepare matrices
    A, AD, D, U = genMatrixesLungsHeart()
    N1 = A.shape[0]
    N2 = AD.shape[0]

    A = sp.csc_matrix(A).tocoo()
    AD = sp.csc_matrix(AD).tocoo()
    D = sp.csc_matrix(D).tocoo()
    U = sp.csc_matrix(U).tocoo()

    D_ = [D.copy()]
    U_ = [U.copy()]

    config = {
        'n_nodes': [N1, N1, N1, N2, N2, N2],
        'latents': 64,
        'inputsize': 1024,
        'K': 6,
        'filters': [2, 32, 32, 32, 16, 16, 16],
        'skip_features': 32,
        'eval_sampling': True  # Enable sampling during evaluation
    }

    A_ = [A.copy(), A.copy(), A.copy(), AD.copy(), AD.copy(), AD.copy()]
    A_t, D_t, U_t = ([scipy_to_torch_sparse(x).to(device) for x in X] for X in (A_, D_, U_))

    base_folder = os.path.join('../Datasets/Chestxray', base_folder)

    # Load model and setup folders
    if skip_connections:
        print('Using skip connections')
        hybrid = Hybrid(config, D_t, U_t, A_t).to(device)
        hybrid.load_state_dict(torch.load("../Weights/Hybrid_LH_FULL/bestMSE.pt", map_location=device))
        base_output_folder = base_folder + '/Output_Skip'
    else:
        print('Using no skip connections')
        hybrid = HybridNoSkip(config, D_t, U_t, A_t).to(device)
        hybrid.load_state_dict(torch.load("../Weights/NoSkip/best.pt", map_location=device))
        base_output_folder = base_folder + '/Output_NoSkip'
    
    hybrid.eval()
    print('Model loaded')

    base_input_folder = base_folder + '/images'
    input_folder = os.path.join(base_input_folder, folder_name) if folder_name else base_input_folder
    output_folder = os.path.join(base_output_folder, folder_name) if folder_name else base_output_folder
    os.makedirs(output_folder, exist_ok=True)

    print('Writing to', output_folder)

    # Find all images to process
    data_root = pathlib.Path(input_folder)
    valid_extensions = ['.png', '.jpg']
    all_files = [str(path) for path in data_root.rglob("*") if path.suffix.lower() in valid_extensions]
    all_files.sort(key=natural_key)  
    print(f'Processing {len(all_files)} images in {input_folder}')

    logs = []
    with torch.no_grad():
        for image in tqdm(all_files):
            relative_path = os.path.relpath(image, input_folder)
            
            # Define corresponding output path
            output_subfolder = os.path.join(output_folder, os.path.dirname(relative_path))  
            os.makedirs(output_subfolder, exist_ok=True)
            
            image_name = os.path.basename(image)
            image_path = os.path.join(output_subfolder, os.path.splitext(image_name)[0] + ".txt")

            # Load and preprocess the image
            img = cv2.imread(image, 0) / 255.0
            data = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device).float()
            
            # Encode only once - this is the key efficiency improvement
            mu, log_var, conv6, conv5 = hybrid.encode(data)
            log_var_np = log_var.cpu().numpy()
            logs.append((image_name, np.exp(log_var_np))) # store variance of latents
            
            # Generate n_samples using the same encoding but different random z values
            for sample_id in range(n_samples):
                # Sample a new z from the latent distribution
                z = hybrid.sampling(mu, log_var)
                
                # Decode with the sampled z
                output, _, _ = hybrid.decode(z, conv6, conv5)
                
                # Process and save the output
                output_np = output.cpu().numpy().reshape(-1, 2) * 1024
                output_np = output_np.round().astype('int')
                
                output_sample_path = image_path.replace('.txt', f'_{sample_id + 1}.txt')
                np.savetxt(output_sample_path, output_np, fmt='%i', delimiter=' ')
            
    with open(f'{base_folder}/{output_file}', 'w') as f:
        for image_name, var in logs:
            var_str = " ".join(map(str, var.flatten()))
            f.write(f"{image_name} {var_str}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference for images.')
    parser.add_argument('--img_name', type=str, default='', help='Prefix of the image filenames to process. If not provided, all images will be processed.')
    parser.add_argument('--n_samples', type=int, default=50, help='Number of samples to generate for each image.')
    parser.add_argument('--folder_name', type=str, default=None, help='Folder name for input/output/mask subdirectories (optional).')
    parser.add_argument('--base_folder', type=str, default='', help='Base folder for input/output/mask subdirectories.')
    parser.add_argument('--output_file', type=str, default='output_sigma.txt', help='Output file for sigma values.')
    parser.add_argument('--skip_connections', action='store_true', default=False, help='Use skip connections in the model.')    
    
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    main(args.img_name, args.n_samples, args.output_file, args.base_folder, args.folder_name, args.skip_connections)
