import os
import re
import engineer
from engineer.schedulers.cosine import CosineAnnealingLR

import torch
import torch.nn as nn
import numpy as np

from models.o3_cgmlp import O3CGMLP
from algebra.cliffordalgebra import CliffordAlgebra

from models.modules.fcgp import FullyConnectedSteerableGeometricProductLayer
from models.modules.mvsilu import MVSiLU

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def exclude_layer(module):
    if isinstance(module, nn.Sequential) or isinstance(module, O3CGMLP) or isinstance(module, CliffordAlgebra):
        return True
    return False

def include_layer(module):
    if isinstance(module, FullyConnectedSteerableGeometricProductLayer) or isinstance(module, MVSiLU):
        return True
    return False

def get_embeddings(model, input, n_layer):
    algebra = CliffordAlgebra([1., 1., 1.]).to('cuda')
    layers = [module for module in model.modules() if include_layer(module)][:n_layer]
    input = algebra.embed_grade(input, 1).to('cuda')
    print(input.shape)
    for i, layer in enumerate(layers):
        input = layer(input)
    return input

def main(config):
    dataset_config = config["dataset"]
    dataset = engineer.load_module(dataset_config.pop("module"))(**dataset_config)

    train_loader = dataset.train_loader()
    val_loader = dataset.val_loader()
    test_loader = dataset.test_loader()

    model_config = config["model"]
    model_module = engineer.load_module(model_config.pop("module"))
    model = model_module(**model_config)

    model = model.cuda()
    
    model.load_state_dict(torch.load(os.path.join(".", "models", "saved_models", config["save_name"])))
    model.eval()  # Remove dropouts etc
    

    batches = [batch for batch in train_loader]
    input_data = batches[0][0].to('cuda')

    embeddings = []
    for i in range(6):
        embeddings.append(get_embeddings(model, input_data, i))
    
    # Embedding shape: [BATCH, OUTPUT, CLIFFORD_DIM]
    #import pdb; pdb.set_trace()
"""
    trajwise_pca = PCA(n_components=10)
    trajwise = trajwise_pca.fit_transform(embeddings[0])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    THRESHOLD = np.mean(np.max(np.abs(trajwise_pca.components_), axis=0))  # 0.05
    relevant_components = trajwise_pca.components_[:, np.max(np.abs(trajwise_pca.components_), axis=0) > THRESHOLD]
    ax.imshow(np.abs(relevant_components), cmap='inferno')

    fig.suptitle(f"Trajectory-wise PCA components heatmap")
    fig.tight_layout()
    fig.savefig(f"plots\test_heatmap.png", dpi=300)
"""
    
    
    




if __name__ == "__main__":
    engineer.fire(main)
    """print([name+'\n' for (name, layer) in model.named_modules()])

    param_list = []
    pattern = re.compile('0')

    for name, param in model.named_parameters():
        #print(f"{name}: {param.shape}")
        if '.0.weight' in name:
            param_list.append((name, param))
            #print("Appended to parameter list!")
    
    for name, param in param_list:
        print(f"{name}: {param.shape}")"""

