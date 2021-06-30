import numpy as np
from numpy.random import standard_normal
from detecta import detect_peaks
import torch
import deepfreq
import os
import errno
np.warnings.filterwarnings('ignore')

def awgn(s, snr, L=1):
    gamma = 10**(snr/10)
    if s.ndim == 1:
        P = L*np.sum(np.abs(s)**2)/len(s)
    else:
        P = L*np.sum(np.sum(np.abs(s)**2))/len(s)
    N0 = P/gamma
    if np.isrealobj(s):
        n = np.sqrt(N0/2)*standard_normal(s.shape)
    else:
        n = np.sqrt(N0/2)*(standard_normal(s.shape)+1j*standard_normal(s.shape))
    r = s + n
    return r

def ls(y,f):
    N = len(y)
    K = len(f)
    f = f.reshape(1,K)
    A = np.exp(1j * np.arange(0, N, 1).reshape(N,1) @ f)
    Ap = A.conj().T@A
    if np.abs(np.linalg.det(Ap)) > 1e-6:
        a = np.abs(np.linalg.solve(Ap,A.conj().T@y))
    else:
        a = np.abs(np.linalg.lstsq(Ap,A.conj().T@y))[0]
    return a
    
def freqsort(f,a):
    a = a.reshape(len(a),1)
    f = f.reshape(len(f),1)
    mat = np.concatenate((a, f), axis=1)
    mat = mat[np.argsort(mat[:, 1])]
    amp = mat[:, 0]
    omega = mat[:, 1]
    return omega, amp

def detectpeaksort(f,Y,K):
    indexes = detect_peaks(Y)
    difference = K-len(indexes)
    if difference == K:
        peaks = np.zeros((difference,1))
        locs = np.ones((difference,1))*np.pi
        omegaa = locs
        ampa = peaks
    elif difference > 0:
        peaks = Y[indexes].reshape(len(indexes), 1)
        locs = f[indexes]
        peaks = np.concatenate((peaks,np.zeros((difference,1))))
        locs = np.concatenate((locs,np.ones((difference,1))*np.pi))
        mat = np.concatenate((peaks, locs), axis=1)
        mat = mat[np.argsort(mat[:, 1])]
        ampa = mat[:, 0]
        omegaa = mat[:, 1]
    else:
        peaks = Y[indexes].reshape(len(indexes), 1)
        locs = f[indexes]
        mat = np.concatenate((peaks, locs), axis=1)
        mat = mat[np.argsort(mat[:, 0])]
        mat = mat[len(indexes)-K:, :]
        mat = mat[np.argsort(mat[:, 1])]
        ampa = mat[:, 0]
        omegaa = mat[:, 1]
    return omegaa, ampa

def detectpeak(f,Y,K):
    indexes = detect_peaks(Y)
    difference = K-len(indexes)
    if difference == K:
        peaks = np.zeros(difference,1)
        locs = np.ones(difference,1)*np.pi
    elif difference > 0:
        peaks = Y[indexes].reshape(len(indexes), 1)
        locs = f[indexes]
        peaks = np.concatenate((peaks,np.zeros(difference,1)))
        locs = np.concatenate((locs,np.ones(difference,1)*np.pi))
    elif difference == 0:
        peaks = Y[indexes].reshape(len(indexes), 1)
        locs = f[indexes]
    else:
        raise ValueError('Too many peaks detected!')
    
    return locs, peaks

def load(fn, module_type, device = torch.device('cuda')):
    checkpoint = torch.load(fn, map_location=device)
    args = checkpoint['args']
    if device == torch.device('cpu'):
        args.use_cuda = False
    if module_type == 'fr':
        model = deepfreq.set_fr_module(args)
    elif module_type == 'fc':
        model = deepfreq.set_fc_module(args)
    else:
        raise NotImplementedError('Module type not recognized')
    model.load_state_dict(checkpoint['model'])
    optimizer, scheduler = set_optim(args, model, module_type)
    if checkpoint["scheduler"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer, scheduler, args, checkpoint['epoch']

def set_optim(args, module, module_type):
    if module_type == 'fr':
        optimizer = torch.optim.Adam(module.parameters(), lr=args.lr_fr) # Compare with v3 which uses Adam
    elif module_type == 'fc':
        optimizer = torch.optim.Adam(module.parameters(), lr=args.lr_fc) # Compare with v3 which uses Adam
    else:
        raise(ValueError('Expected module_type to be fr or fc but got {}'.format(module_type)))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.5, verbose=True)
    return optimizer, scheduler

def model_parameters(model):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    return num_params

def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e