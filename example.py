#%% Clear console and variables
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

#%% Code
from pytictoc import TicToc
import numpy as np
import matplotlib.pyplot as plt
from utilities import awgn, ls, detectpeaksort, freqsort, load

from uni_esprit import uni_esprit
from relax import relax
from cfh import cfh
from nomp.nomp import nomp

# Define signal parameters
N = 2**7
K = 5
snr = 40
distance_min = 2/N
pi = np.pi
d = 2/N

# DeepFreq auxiliary
import torch
from pathlib import Path
weights = str(N)
weights_file = weights + '.pth'
fr_path = Path('weights') / weights_file
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fr_module, _, _, _, _ = load(fr_path, 'fr', device)
fr_module = fr_module.double()
fr_module.cpu()
fr_module.eval()
xgrid = np.linspace(0, 2*np.pi, fr_module.fr_size, endpoint=False).reshape(fr_module.fr_size, 1)

# Generate signal with frequencies
normalized_omega_wrap = np.array([0.1, 0.14, 0.3, 0.4, 0.6]).reshape(K,1)
normalized_omega = np.copy(normalized_omega_wrap)
normalized_omega[normalized_omega >
                  0.5] = normalized_omega[normalized_omega > 0.5] - 1
normalized_omega = np.sort(normalized_omega)

omega_wrap = 2*pi*normalized_omega_wrap

# Generate complex amplitudes with magnitudes normalized to 1
beta = np.ones((K,2)) @ np.array([1, 1j]).reshape(2, 1)
beta = beta/np.abs(beta)
amp = np.abs(beta)

# Create noiseless signal
x = np.exp(1j * np.arange(0, N, 1).reshape(N,1) @ omega_wrap.reshape(1,K)) @ beta
x = x.squeeze()
noiseVar = np.mean(np.power(np.abs(x),2)) / (10**(snr/10))

# Create noisy signal
y = awgn(x,snr)

# Timer class
t = TicToc()

# Dicrete Fourier Transform (FFT)
t.tic()

Y = (np.abs(np.fft.fft(y)) / N).squeeze()
f = np.linspace(0, 2*pi, N, endpoint=False).reshape(N, 1)
omega_0, amp_0 = detectpeaksort(f,Y,K)

t_0 = t.tocvalue()

# plt.plot(f, Y)
# plt.plot(omega_0,amp_0,'o')
# plt.grid()

# Unitary ESPRIT
t.tic()

omega_1 = uni_esprit(y, K)
amp_1 = ls(y,omega_1)
omega_1, amp_1 = freqsort(omega_1, amp_1)

t_1 = t.tocvalue()

# RELAX
t.tic()

omega_2, amp_2 = relax(y, K)
omega_2, amp_2 = freqsort(omega_2, amp_2)

t_2 = t.tocvalue()

# CFH
t.tic()

omega_3, amp_3 = cfh(y, K)
omega_3, amp_3 = freqsort(omega_3, amp_3)

t_3 = t.tocvalue()

# NOMP
t.tic()

omega_4, amp_4 = nomp(y, K)
omega_4, amp_4 = freqsort(omega_4, amp_4)

t_4 = t.tocvalue()

# DeepFreq
t.tic()

y2 = np.concatenate((np.real(y).reshape(1,N),np.imag(y).reshape(1,N)),axis=1)
with torch.no_grad():
    freqrep = fr_module(torch.tensor(y2))
freqrep = freqrep.numpy()
freqrep = freqrep.squeeze() 
frL, frR = np.split(freqrep,2)
freqrep = np.concatenate((frR,frL))
omega_5, amp_5 = detectpeaksort(xgrid,freqrep,K)

t_5 = t.tocvalue()

# plt.plot(xgrid, freqrep.squeeze())
# plt.plot(omega_5,amp_5,'o')
# plt.grid()