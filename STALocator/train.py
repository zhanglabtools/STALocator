import os
import numpy as np
import scanpy as sc
import pandas as pd
import anndata

import ot

import torch
import torch.nn as nn
import torch.nn.functional as F

loss1 = nn.L1Loss()
loss2 = nn.MSELoss()

def get_max_index(
    vector
):
    """This function Returns the index of the vector's maximum value.

        Args:
            vector: A vector that store values.

        Return:
            max_index: The index of the vector's maximum value.
    """
    max_index=np.where(vector==np.max(vector))[0][0]
    return max_index

def trans_plan_b(
    latent_A,
    latent_B,
    metric='correlation',
    reg=0.1,
    numItermax=10,
    device='cpu'
):
    """This function Returns the optimal transport (OT) plan.

        Args:
            latent_A, latent_B: Two set of data points.
            metric: Metric of OT. Default is 'correlation'.
            reg: The weight of entropy regularized term. Default is 0.1.
            numItermax: Iterations of OT. Default is 10.

        Return:
            plan: The index of the vector's maximum value.
    """
    cost = ot.dist(latent_A.detach().cpu().numpy(), latent_B.detach().cpu().numpy(), metric=metric)
    cost = torch.from_numpy(cost).float().to(device)

    length_A = latent_A.shape[0]
    length_B = latent_B.shape[0]

    P = torch.exp(-cost/reg)

    p_s = torch.ones(length_A, 1) / length_A
    p_t = torch.ones(length_B, 1) / length_B
    p_s = p_s.to(device)
    p_t = p_t.to(device)

    u_s = torch.ones(length_A, 1) / length_A
    u_t = torch.ones(length_B, 1) / length_B
    u_s = u_s.to(device)
    u_t = u_t.to(device)

    for i in range(numItermax):
        p_t = u_t / torch.mm(torch.transpose(P, 0, 1), p_s)
        p_s = u_s / torch.mm(P, p_t)

    plan = torch.transpose(p_t, 0, 1) * P * p_s

    return plan

def rand_projections(
    embedding_dim,
    num_samples=50,
    device='cpu'
):
    """This function generates `num_samples` random samples from the latent space's unit sphere.

        Args:
            embedding_dim (int): embedding dimensionality
            num_samples (int): number of random projection samples

        Return:
            torch.Tensor: tensor of size (num_samples, embedding_dim)
    """

    projections = [w / np.sqrt((w**2).sum())  # L2 normalization
                   for w in np.random.normal(size=(num_samples, embedding_dim))]
    projections = np.asarray(projections)
    return torch.from_numpy(projections).type(torch.FloatTensor).to(device)


def _sliced_wasserstein_distance(
    encoded_samples,
    distribution_samples,
    num_projections=50,
    p=2,
    device='cpu'
):
    """ Sliced Wasserstein Distance between encoded samples and drawn distribution samples.

        Args:
            encoded_samples (toch.Tensor): tensor of encoded training samples
            distribution_samples (torch.Tensor): tensor of drawn distribution training samples
            num_projections (int): number of projections to approximate sliced wasserstein distance
            p (int): power of distance metric
            device (torch.device): torch device (default 'cpu')

        Return:
            torch.Tensor: tensor of wasserstrain distances of size (num_projections, 1)
    """

    # derive latent space dimension size from random samples drawn from latent prior distribution
    embedding_dim = distribution_samples.size(1)
    # generate random projections in latent space
    projections = rand_projections(embedding_dim, num_projections).to(device)
    # calculate projections through the encoded samples
    encoded_projections = encoded_samples.matmul(projections.transpose(0, 1).to(device))
    # calculate projections through the prior distribution random samples
    distribution_projections = (distribution_samples.matmul(projections.transpose(0, 1)))
    # calculate the sliced wasserstein distance by
    # sorting the samples per random projection and
    # calculating the difference between the
    # encoded samples and drawn random samples
    # per random projection
    wasserstein_distance = (torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
                            torch.sort(distribution_projections.transpose(0, 1), dim=1)[0])
    # distance between latent space prior and encoded distributions
    # power of 2 by default for Wasserstein-2
    wasserstein_distance = torch.pow(wasserstein_distance, p)
    # approximate mean wasserstein_distance for each projection
    return wasserstein_distance.mean()


def sliced_wasserstein_distance(
    encoded_samples,
    transformed_samples,
    num_projections=50,
    p=2,
    device='cpu'
):
    """ Sliced Wasserstein Distance between encoded samples and drawn distribution samples.

        Args:
            encoded_samples (toch.Tensor): tensor of encoded training samples
            distribution_samples (torch.Tensor): tensor of drawn distribution training samples
            num_projections (int): number of projections to approximate sliced wasserstein distance
            p (int): power of distance metric
            device (torch.device): torch device (default 'cpu')

        Return:
            torch.Tensor: tensor of wasserstrain distances of size (num_projections, 1)
    """
    # derive batch size from encoded samples
    # draw random samples from latent space prior distribution

    # approximate mean wasserstein_distance between encoded and prior distributions
    # for each random projection
    swd = _sliced_wasserstein_distance(encoded_samples, transformed_samples, num_projections, p, device)
    return swd