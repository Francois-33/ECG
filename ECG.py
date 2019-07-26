import os
import math
import arff
import numpy as np
import numpy.random as npr
from IPython.display import clear_output
from tqdm import tqdm_notebook as tqdm
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
#matplotlib inline
#import seaborn as sns
#sns.color_palette("bright")
import matplotlib as mpl
import matplotlib.cm as cm
import psutil
import torch
from scipy import stats
import scipy.io.arff
from torch import Tensor
import tslearn.datasets
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from class_model import ODE, ODEF, ODEAdjoint, NeuralODE, LinearODEF, RandomLinearODEF, TestODEF, NNODEF, ODEVAE, \
    SpiralFunctionExample

use_cuda = torch.cuda.is_available()


def gen_batch(batch_size, n_sample=140, samp_trajs=None, samp_ts=None):
    n_batches = samp_trajs.shape[1] // batch_size
    time_len = samp_trajs.shape[0]
    n_sample = min(n_sample, time_len)
    for i in range(n_batches):
        if n_sample > 0:
            t0_idx = npr.multinomial(1, [1. / (time_len - n_sample)] * (time_len - n_sample))
            t0_idx = np.argmax(t0_idx)
            tM_idx = t0_idx + n_sample
        else:
            t0_idx = 0
            tM_idx = time_len

        frm, to = batch_size*i, batch_size*(i+1)
        yield samp_trajs[t0_idx:tM_idx, frm:to], samp_ts[t0_idx:tM_idx, frm:to]


input = 1
nhidden = 64
latent = 6
n_epochs = 500
variable = [input, nhidden, latent, n_epochs]

vae = ODEVAE(input, nhidden, latent)
vae = vae.cuda()

optim = torch.optim.Adam(vae.parameters(), betas=(0.9, 0.999), lr=0.001)

with open("ECG1_variable.txt", "wb") as fp:
    pickle.dump(variable, fp)

X_train, Y_train, X_test, Y_test = tslearn.datasets.UCR_UEA_datasets().load_dataset('ECG5000')
X_ts = np.linspace(1, 140, num=140)

memory = []
losses = []

X_train = np.reshape(X_train, (140, 500, 1))
X_train = torch.from_numpy(X_train)
X_ts = np.tile(X_ts, (500, 1, 1))
X_ts = np.reshape(X_ts, (140, 500, 1))
X_ts = torch.from_numpy(X_ts)

for epoch_idx in range(n_epochs):
    process = psutil.Process(os.getpid())
    #batch_idx = sorted(npr.choice(range(499), size=100, replace=False))
    #batch = X_train[batch_idx, :, :]

    batch = gen_batch(batch_size=100, samp_trajs=X_train, samp_ts=X_ts, n_sample=0)
    for x, t in batch:
        optim.zero_grad()
        #x = batch[i, :, :]
        #t = batch_idx
        #t = torch.from_numpy(np.asarray(t))
        #if use_cuda:
        x, t = x.cuda(), t.cuda()
        max_len = np.random.choice([30, 50, 100])
        permutation = np.random.permutation(t.shape[0])
        np.random.shuffle(permutation)
        permutation = np.sort(permutation[:max_len])
        x, t = x[permutation], t[permutation]

        x_p, z, z_mean, z_log_var = vae(x, t)
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean**2 - torch.exp(z_log_var), -1)
        x_p = x_p.double()
        kl_loss = kl_loss.double()
        x = x.double()
        loss = 0.5 * ((x-x_p)**2).sum(-1).sum(0) + kl_loss
        loss = torch.mean(loss)
        loss /= max_len
        loss.backward()
        optim.step()
        losses.append(loss.item())

    print("Epoch "+str(epoch_idx))
    memory.append(process.memory_info().rss)
    print("Memory: "+str(memory[-1]))
    if epoch_idx % 50 == 0:
        torch.save({"ode_trained": vae.state_dict()},
           "ECG1_"+str(epoch_idx)+".pth")

torch.save({"ode_trained": vae.state_dict()},
           "ECG1.pth")

with open("ECG1_losses.txt", "wb") as fp:
    pickle.dump(losses, fp)

with open("ECG1_memory.txt", "wb") as fp:
    pickle.dump(memory, fp)


