"""
Author: Ambroise Odonnat
Purpose: Contains functions to train and evaluate models
"""

import time
import numpy as np
import scipy.sparse as sp

import torch

from utils import sparse_mx_to_torch_sparse_tensor


def training(model, optimizer, criterion, epochs, batch_size,
             adj_train, features_train, y_train, device, LOG=5):

    # Initialize losses
    N_train = len(adj_train)
    losses = []

    # Iterate over epochs
    for epoch in range(epochs):
        t = time.time()
        model.train()
        train_loss = 0
        correct = 0
        count = 0

        # Iterate over the batches
        for i in range(0, N_train, batch_size):
            adj_batch = list()
            features_batch = list()
            idx_batch = list()
            y_batch = list()

            # Create tensors
            for j in range(i, min(N_train, i + batch_size)):
                n = adj_train[j].shape[0]
                adj_batch.append(adj_train[j] + sp.identity(n))
                features_batch.append(features_train[j])
                idx_batch.extend([j - i] * n)
                y_batch.append(y_train[j])

            adj_batch = sp.block_diag(adj_batch)
            features_batch = np.vstack(features_batch)

            adj_batch = sparse_mx_to_torch_sparse_tensor(adj_batch).to(device)
            features_batch = torch.FloatTensor(features_batch).to(device)
            idx_batch = torch.LongTensor(idx_batch).to(device)
            y_batch = torch.LongTensor(y_batch).to(device)

            optimizer.zero_grad()
            output = model(features_batch, adj_batch, idx_batch)
            loss = criterion(output, y_batch)
            train_loss += loss.item() * output.size(0)
            count += output.size(0)
            preds = output.max(1)[1].type_as(y_batch)
            correct += torch.sum(preds.eq(y_batch).double())
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 1)
            optimizer.step()

        # Update losses
        losses.append(train_loss / count)
        if epoch % LOG == 0:
            print('Epoch: {:03d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(train_loss / count),
                  'acc_train: {:.4f}'.format(correct / count),
                  'time: {:.4f}s'.format(time.time() - t))

    return losses

def evaluation(model, batch_size, adj_test, features_test, device):

    N_test = len(adj_test)
    model.eval()
    y_pred_proba = list()

    # Iterate over the batches
    for i in range(0, N_test, batch_size):
        adj_batch = list()
        idx_batch = list()
        features_batch = list()
        y_batch = list()

        # Create tensors
        for j in range(i, min(N_test, i + batch_size)):
            n = adj_test[j].shape[0]
            adj_batch.append(adj_test[j] + sp.identity(n))
            features_batch.append(features_test[j])
            idx_batch.extend([j - i] * n)

        adj_batch = sp.block_diag(adj_batch)
        features_batch = np.vstack(features_batch)

        adj_batch = sparse_mx_to_torch_sparse_tensor(adj_batch).to(device)
        features_batch = torch.FloatTensor(features_batch).to(device)
        idx_batch = torch.LongTensor(idx_batch).to(device)

        output = model(features_batch, adj_batch, idx_batch)
        y_pred_proba.append(output)

    y_pred_proba = torch.cat(y_pred_proba, dim=0)
    y_pred_proba = torch.exp(y_pred_proba)

    return y_pred_proba