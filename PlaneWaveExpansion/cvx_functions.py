import cvxpy as cp
import numpy as np


def loss_fn(H, Y, H):
    return cp.norm2(H @ H - Y)**2

def regularizer(H):
    return cp.norm1(H)

def objective_fn(H, Y, beta, lambd):
    return loss_fn(H, Y, beta) + lambd * regularizer(beta)
