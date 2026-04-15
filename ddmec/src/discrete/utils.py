import torch
import numpy as np
from collections import Counter

import torch


def compute_entropy(marginal):

    marginal = marginal[marginal > 0]  # Mask zero probabilities
    entropy = -torch.sum(marginal * torch.log(marginal))
    return entropy.item()


def compute_entropy_lower_bound(marginal_x, marginal_y):

    entropy_x = compute_entropy(marginal_x)
    entropy_y = compute_entropy(marginal_y)
    return entropy_x + entropy_y

from sklearn.metrics import mutual_info_score

def mutual_info(x,y,num_categories1,num_categories2):
    return mutual_info_score(x.flatten(),y.flatten())


def compute_marginals(data, num_categories):
    data = torch.tensor(data)

    one_hot_data = torch.nn.functional.one_hot(data, num_classes=num_categories)

    # Sum over samples to get category counts
    category_counts = one_hot_data.sum(dim=0)

    # Normalize to get marginal probabilities
    total_samples = data.size(0)
    marginals = category_counts.float() / total_samples

    return marginals

def kl_divergence_marginals(P, Q):

    P = P / P.sum()
    Q = Q / Q.sum()
    
    epsilon = 1e-10
    P = torch.clamp(P, min=epsilon)
    Q = torch.clamp(Q, min=epsilon)
    
    kl_div = torch.sum(P * torch.log(P / Q))
    return kl_div.item()

def estimate_density(X, Y):
    X = np.asarray(X)
    Y = np.asarray(Y)
    total_samples = len(X)
    joint_counts = Counter(zip(X, Y))
    joint_probs = {k: v / total_samples for k, v in joint_counts.items()}
    X_counts = Counter(X)
    Y_counts = Counter(Y)
    P_X = {k: v / total_samples for k, v in X_counts.items()}
    P_Y = {k: v / total_samples for k, v in Y_counts.items()}
    return joint_probs, P_X, P_Y


def compute_mi_from_density(X,Y):
    joint_probs, P_X, P_Y = estimate_density(X,Y)
    mi = 0.0
    for (x, y), p_xy in joint_probs.items():
        p_x = P_X[x]
        p_y = P_Y[y]
        mi += p_xy * np.log(p_xy / (p_x * p_y))
    return mi
