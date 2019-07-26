import os
import argparse
import logging
import time
import numpy as np
import numpy.random as npr
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import pickle
from scipy.stats import norm
from class_model import ODEF, ODEAdjoint, NeuralODE, LinearODEF, RandomLinearODEF, TestODEF, NNODEF, ODEVAE, \
    SpiralFunctionExample
from IPython.display import clear_output
import seaborn as sns
sns.color_palette("bright")
import tslearn.datasets


def to_np(x):
    return x.detach().cpu().numpy()


def extrapolation(PATH=None
):

    device = torch.device("cpu")

    with open(PATH+"_variable.txt", "rb") as fp:
            [input, nhidden, latent, n_epoch] = pickle.load(fp)

    X_train, Y_train, X_test, Y_test = tslearn.datasets.UCR_UEA_datasets().load_dataset('ECG5000')
    X_ts = np.linspace(1, 140, num=140)
    print("first")
    print(X_train.shape)
    print(X_train[1:10, 1:10])
    plt.plot(X_train[2])
    plt.savefig("truth.png")
    plt.clf()
    vae = ODEVAE(input, nhidden, latent)
    checkpoint = torch.load("/share/home/fpainblanc/virtualenvironment/virtenv/bin/" + PATH + "_50.pth")
    vae.load_state_dict(checkpoint["ode_trained"])

    """fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 9))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.scatter(to_np(seed_trajs[:, i, 0]), to_np(seed_trajs[:, i, 1]), c=to_np(ts[frm:to_seed, i, 0]), cmap=cm.plasma)
        ax.plot(to_np(orig_trajs[frm:to, i, 0]), to_np(orig_trajs[frm:to, i, 1]))
        ax.plot(samp_trajs_p[:, i, 0], samp_trajs_p[:, i, 1])"""

    X_train = np.reshape(X_train, (140, 500, 1))
    print("second")
    print(X_train.shape)
    print(X_train[1:10, 1])
    print(X_train[1, 1:10])
    X_train = torch.from_numpy(X_train)
    X_ts = np.tile(X_ts, (500, 1, 1))
    X_ts = np.reshape(X_ts, (140, 500, 1))
    X_ts = torch.from_numpy(X_ts)
    frm, to, to_seed = 0, 140, 50
    print("bof last")
    print(X_train.shape)
    print(X_train[1:10, 1])

    seed_trajs = X_train[frm:to_seed]
    ts = X_ts[frm:to]
    #print(seed_trajs)
    samp_trajs_p = to_np(vae.generate_with_seed(seed_trajs, ts))
    #samp_trajs_p = vae.generate_with_seed(seed_trajs, ts)
    #print(samp_trajs_p)
    #print(samp_trajs_p.shape)
    #print(type(samp_trajs_p))
    #samp_trajs_p = samp_trajs_p.detach().numpy()
    X_train = X_train.detach().numpy()
    print("last")
    print(X_train.shape)
    print(X_train[1:10, 1])
    plt.plot(X_train[1:10])
    plt.savefig("reshape.png")
    plt.clf()
    plt.subplot(2, 1, 1)
    #plt.plot(np.linspace(1, 140, 140), samp_trajs_p[:, 2], color="orange", label="Reconstructed")
    plt.plot(X_train[2])
    plt.subplot(2, 1, 2)
    plt.plot(X_train[:, 1], color="blue", label="Ground truth")
    plt.legend()

    plt.savefig(PATH+"sequence_ecg.png")

    clear_output(wait=True)

    return plt

extrapolation("ECG1")
