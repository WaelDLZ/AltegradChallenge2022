"""
Author: Ambroise Odonnat
Purpose: Contains embeddings functions
"""
# Install bio_embeddings using the command: pip install bio-embeddings[all]

import numpy as np

import torch

from bio_embeddings.embed import ProtTransBertBFDEmbedder, SeqVecEmbedder

def create_SeqVec_embeddings(seq):
    embedder = SeqVecEmbedder()
    embedding = embedder.embed(seq)
    protein_embd = torch.tensor(embedding).sum(dim=0)  # Vector with shape [L x 1024]
    np_arr = protein_embd.cpu().detach().numpy()

    return np_arr


def create_ProBert_embeddings(seq):
    embedder = ProtTransBertBFDEmbedder()
    protein_embd = torch.tensor(embedder.embed(seq))  # Vector with shape [L x 1024]
    np_arr = protein_embd.cpu().detach().numpy()

    return np_arr

# Embed all sequences
with open('sequences.txt', 'r') as f:
    SeqVec_embeddings = {}
    ProBert_embeddings = {}
    for i, line in enumerate(f):
        if i%100 == 0:
            print('line number: ',i+1)
        seq = line[:-1]
        SeqVec_embeddings[i] = create_SeqVec_embeddings(seq)
        ProBert_embeddings[i] = create_ProBert_embeddings(seq)

np.save('embeddings/SeqVec_embeddings.npy', SeqVec_embeddings)
np.save('embeddings/ProBert_embeddings.npy', ProBert_embeddings)
print('Complete')
