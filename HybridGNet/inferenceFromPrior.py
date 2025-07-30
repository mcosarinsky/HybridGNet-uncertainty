import os
import numpy as np
import torch

from models.HybridGNet2IGSC import HybridNoSkip
from utils.utils import scipy_to_torch_sparse, genMatrixesLungsHeart
import scipy.sparse as sp

def main(n_samples=50):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_folder = "../Datasets/PriorSamples/"

    # Load graph matrices
    A, AD, D, U = genMatrixesLungsHeart()
    A = [sp.csc_matrix(A).tocoo()] * 3 + [sp.csc_matrix(AD).tocoo()] * 3
    D = [sp.csc_matrix(D).tocoo()]
    U = [sp.csc_matrix(U).tocoo()]
    
    A_t, D_t, U_t = ([scipy_to_torch_sparse(x).to(device) for x in X] for X in (A, D, U))

    config = {
        'n_nodes': [A[0].shape[0]] * 3 + [A[3].shape[0]] * 3,
        'latents': 64,
        'inputsize': 1024,
        'K': 6,
        'filters': [2, 32, 32, 32, 16, 16, 16],
        'skip_features': 32,
        'eval_sampling': True
    }

    latents = config['latents']
    hybrid = HybridNoSkip(config, D_t, U_t, A_t).to(device)
    hybrid.load_state_dict(torch.load("../Weights/NoSkip/best.pt", map_location=device))
    hybrid.eval()

    os.makedirs(output_folder, exist_ok=True)
    print(f"Generating {n_samples} samples in '{output_folder}'")

    with torch.no_grad():
        for i in range(n_samples):
            mu = torch.zeros((1,latents), device=device)
            log_var = torch.zeros((1,latents), device=device)

            # Sample z ~ N(0, I) and decode
            z = hybrid.sampling(mu, log_var)
            output, _, _ = hybrid.decode(z, None, None)

            # Process and save
            output_np = output.cpu().numpy().reshape(-1, 2) * 1024
            output_np = output_np.round().astype(int)

            out_path = os.path.join(output_folder, f'sample_{i+1}.txt')
            np.savetxt(out_path, output_np, fmt='%i', delimiter=' ')

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    main(n_samples=50)
