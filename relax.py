"""
%=============================================================================
% DESCRIPTION:
% This is the Python implementation of RELAX, for line spectral 
% estimation (LSE) problems. This was written for the submitted article:
%
% "Real-time sinusoidal parameter estimation for ultrasonic fatigue tests."
%
%=============================================================================
% Version 2.2, Authored by:
% Shawn L. KISER (Msc) @ https://www.linkedin.com/in/shawn-kiser/
%   Laboratoire PIMM, Arts et Metiers Institute of Technology, CNRS, Cnam,
%   HESAM Universite, 151 boulevard de l’Hopital, 75013 Paris (France)
%
% Based on:
% [1] Jian Li and P. Stoica, “Efficient mixed-spectrum estimation with 
%     applications to target feature extraction,” IEEE Trans. Signal Process., 
%     vol. 44, no. 2, pp. 281–295, Feb. 1996, doi: 10.1109/78.485924.
%
%=============================================================================
% The MIT License (MIT)
% 
% Copyright (c) 2021 Shawn L. KISER
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in
% all copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
% THE SOFTWARE.

%=============================================================================
"""

import numpy as np
import math

def relax(y, P):
    N = len(y)
    if P > 1:
        Theta = np.zeros((2,2),dtype=complex)
    else:
        Theta = np.zeros((2,1),dtype=complex)
    iterMax = 25
    dynRange = 60 #dB
    zpFactor = 4
    zeroPad = zpFactor*N
    tol = 1/(2*zeroPad)
    ypad = np.concatenate((y.reshape(N,1),np.zeros(((zpFactor-1)*len(y),1))),axis=0).squeeze()
    cost = np.fft.fftshift(np.power(np.abs(np.fft.fft(ypad)),2))
    idx = np.argmax(cost)
    Theta[1,0] = (1/zeroPad * idx) - 0.5;
    Theta[0,0] = 1/N * newcost(Theta[:,0],y,N)
    ThetaOld = np.copy(Theta)
    for i in range(2,P+1):
        iterc = 0
        while True:
            for k in range(i,0,-1):
                x_i = serialsep(y,Theta,k)
                xpad = np.concatenate((x_i,np.zeros(((zpFactor-1)*len(x_i),1))),axis=0).squeeze()
                cost = np.fft.fftshift(np.power(np.abs(np.fft.fft(xpad)),2))
                idx = np.argmax(cost)
                if (np.size(Theta,1)>=k):
                    Theta[1,k-1] = ((1/zeroPad) * idx) - 0.5;
                    Theta[0,k-1] = 1/N * newcost(Theta[:,k-1],x_i,N)
                else:
                    Theta = np.concatenate((Theta,np.zeros((2,1))),axis=1)
                    Theta[1,k-1] = ((1/zeroPad) * idx) - 0.5;
                    Theta[0,k-1] = 1/N * newcost(Theta[:,k-1],x_i,N)
            if ((abs(Theta[1,:] - ThetaOld[1,:]) < tol).all()):
                ThetaOld = np.concatenate((ThetaOld,np.zeros((2,1),dtype=complex)),axis=1)
                break
            else:
                ThetaOld = np.copy(Theta)
                iterc += 1
                if iterc > iterMax:
                    ThetaOld = np.concatenate((ThetaOld,np.zeros((2,1),dtype=complex)),axis=1)
                    break
        if (abs(Theta[0,i-1])<10**(-dynRange/20)*abs(Theta[0,0])):
            break
    a = np.copy(Theta[0,:])         
    f = np.real(Theta[1,:])
    f[f<0] = f[f<0] + 1
    f = f*2*np.pi
    a = np.abs(a)
    return f, a

def newcost(theta,x,N):
    f = (theta[1]).real
    if (N % 2) == 0:
        n = np.arange(0, N, 1).reshape(N,1)
    else:
        n = np.arange(-math.floor(N/2),math.floor(N/2)+1,1).reshape(N,1)
    C = np.exp(1j * 2 * np.pi * f * n).conjugate() * x.reshape(N,1)
    c = np.sum(C)
    return c

def serialsep(y,Theta,k):
    N = len(y)
    comp = np.arange(1, Theta.shape[1]+1, 1)
    comp = np.delete(comp, np.where(comp == k))
    x_i = np.copy(y).reshape(N,1)
    for i in comp:
        x_i = x_i - sigmodel(Theta[:,i-1],N)
    return x_i

def sigmodel(Theta,N):
    beta = Theta[0]
    f = (Theta[1]).real
    if (N % 2) == 0:
        n = np.arange(0, N, 1).reshape(N,1)
    else:
        n = np.arange(-math.floor(N/2),math.floor(N/2)+1,1).reshape(N,1)
    
    y = beta * np.exp(1j * 2 * np.pi * f * n)
    return y