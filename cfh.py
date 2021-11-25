"""
%=============================================================================
% DESCRIPTION:
% This is the Python implementation of Coarse to fine HAQSE (CFH), for line 
% spectral estimation (LSE) problems. This was written for the submitted 
% manuscript:
%
% "Real-time sinusoidal parameter estimation for damage growth monitoring 
%  during ultrasonic very high cycle fatigue tests."
%
%=============================================================================
% Version 1.1.0, Authored by:
% Shawn L. KISER (Msc) @ https://www.linkedin.com/in/shawn-kiser/
%   Laboratoire PIMM, Arts et Metiers Institute of Technology, CNRS, Cnam,
%   HESAM Universite, 151 boulevard de l’Hopital, 75013 Paris (France)
%
% Based on:
% [1] A. Serbes and K. Qaraqe, “A Fast Method for Estimating Frequencies of 
%     Multiple Sinusoidals,” IEEE Signal Process. Lett., vol. 27, pp. 386–390, 
%     2020, doi: 10.1109/lsp.2020.2970837.
%
% Modifications include transcription from MATLAB sourced material into Python
% code, appropriate optimizations, and change of notations.
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

def cfh(y, P):
    N = len(y)
    n = np.arange(0, N, 1).reshape(N,1)
    q = min(((1/N)**(1/3),0.32))
    cq = (1 - np.pi * q * 1/math.tan(np.pi * q)) / (q * math.cos(np.pi * q)**2)
    Q1opt = math.ceil(math.log(math.log2(N / math.log(N))) / math.log(3))
    Q2opt = math.ceil(math.log(N / math.log(N)) / math.log(2))
    Qopt = max((2, Q1opt, Q2opt))
    p = np.zeros((P,1))
    delta = np.zeros((P,1))
    A = np.zeros((1,P),dtype=complex)
    sTilde = y.reshape(N,1)
    for k in range(1,P+1):
        p[k-1] =  np.argmax(np.power(np.abs(np.fft.fft(sTilde.squeeze())),2))
        Sq  = np.sum(sTilde * np.exp(-1j * 2 * np.pi / N * (p[k-1] + q) * n))
        Snq = np.sum(sTilde * np.exp(-1j * 2 * np.pi / N * (p[k-1] - q) * n))
        delta[k-1] = ( (1/cq) * (Sq - Snq) / (Sq + Snq) ).real
        A[0,k-1] = np.exp(-1j * 2 * np.pi * (p[k-1] + delta[k-1]) * n.T / N ) @ sTilde / N
        sTilde = sTilde - A[0,k-1] * np.exp(1j * 2 * np.pi * (p[k-1] + delta[k-1]) * n / N)
    f = p + delta
    for i in range(1,Qopt+1):
        for k in range(1,P+1):
            at = np.delete(A, [k-1])
            ft = np.delete(f, [k-1])
            shat = y.reshape(N,1)
            for m in range(1,P):
                shat = shat - at[m-1] * np.exp(1j * 2 * np.pi / N * (ft[m-1]) * n)
            Sq  = np.sum(shat * np.exp(-1j * 2 * np.pi / N * (f[k-1] + q) * n))
            Snq = np.sum(shat * np.exp(-1j * 2 * np.pi / N * (f[k-1] - q) * n))
            delta[k-1] = ( (1/cq) * (Sq - Snq) / (Sq + Snq) ).real
            f[k-1] = f[k-1] + delta[k-1]
            A[0,k-1] = np.exp(-1j * 2 * np.pi * (f[k-1]) * n.T / N ) @ shat / N
    f = (f/N*2*np.pi).squeeze()
    a = np.abs(A).squeeze()
    return f,a