"""
%=============================================================================
% DESCRIPTION:
% This is the Python implementation of Unitary ESPRIT, for line spectral 
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
% [1] M. Haardt and J. A. Nossek, “Unitary ESPRIT: how to obtain increased 
%     estimation accuracy with a reduced computational burden,” 
%     IEEE Transactions on Signal Processing, vol. 43, no. 5, pp. 1232–1242, 
%     May 1995, doi: 10.1109/78.382406.
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
from scipy.linalg import hankel, svd, lstsq, eig
import math

def uni_esprit(y, P):
    N = len(y)
    nSeg = math.floor(N/2)
    J2 = np.concatenate((np.zeros((nSeg-1, 1)), np.eye(nSeg-1)), axis=1)
    tmp = unitary_rot(nSeg-1).T
    tmp[math.floor(N/4):(nSeg-1), :] = -1 * tmp[math.floor(N/4):(nSeg-1), :]
    UniSelect = tmp @ J2 @ unitary_rot(nSeg)
    J1prime = np.real(UniSelect)
    J2prime = np.imag(UniSelect)
    Y = hankel(y[0:nSeg], y[nSeg-1:N])
    tmp = unitary_rot(nSeg).T
    tmp[math.floor(N/4):(nSeg), :] = -1 * tmp[math.floor(N/4):(nSeg), :]
    Y = tmp @ Y
    U = svd(np.concatenate((np.real(Y), np.imag(Y)), axis=1))[0]
    ULeft = U[:,0:P]
    Upsilon = lstsq(J1prime @ ULeft, J2prime @ ULeft)[0]
    f = np.arctan(np.real(eig(Upsilon)[0]))/np.pi
    f[f < 0] = f[f < 0] + 1
    return f*2*np.pi

def unitary_rot(m):
    imag = complex(0,1)
    sqrt2 = np.sqrt(2)
    if(np.remainder(m,2) != 0):
        d = int((m-1)/2)
        eye = np.eye(d)
        antiEye = np.fliplr(eye)
        Q = 1/sqrt2 * np.concatenate((np.concatenate((eye, np.zeros((d, 1)), imag*eye), axis=1),
                                      np.concatenate(
                                          (np.zeros((1, d)), np.asarray(sqrt2).reshape(1, 1), np.zeros((1, d))), axis=1),
                                      np.concatenate((antiEye, np.zeros((d, 1)), -imag*antiEye), axis=1)), axis=0)
    else:
        d = int(m/2)
        eye = np.eye(d)
        antiEye = np.fliplr(eye)
        Q = 1/sqrt2 * np.concatenate((np.concatenate((eye, imag*eye), axis=1),
                                      np.concatenate((antiEye, -imag*antiEye), axis=1)), axis=0)
    return Q