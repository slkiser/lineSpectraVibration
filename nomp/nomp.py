"""
%=============================================================================
% DESCRIPTION:
% This is the Python implementation of Newtonianized Orthogonal Matching 
% Pursuit (NOMP), for line spectral estimation (LSE) problems. This was 
% written for the submitted manuscript:
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
% [1] B. Mamandipoor, D. Ramasamy, and U. Madhow, “Newtonized Orthogonal 
%     Matching Pursuit: Frequency Estimation Over the Continuum,” IEEE Trans. 
%     Signal Process., vol. 64, no. 19, pp. 5066–5081, Oct. 2016, 
%     doi: 10.1109/tsp.2016.2580523.
% 
% Modifications include transcription from MATLAB sourced material into Python
% code, appropriate optimizations, and change of notations. Specifically, this 
% version of NOMP is written only for complete data, and does not zero-pad the
% FFT operation(s). It also relies on a LS estimate defined by eq. (22) in  
% "Real-time sinusoidal parameter estimation for ultrasonic fatigue tests."
%
%=============================================================================
% Copyright 2015 Dinesh Ramasamy
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%     http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.
%
%=============================================================================
% The MIT License (MIT)
% 
% Modifications Copyright (c) 2021 Shawn L. KISER
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
%
%=============================================================================
"""
import numpy as np
import math
import random

def nomp(y, P):
    Rc = 3
    OSR = 2
    N = len(y)
    R = N*OSR
    n = np.arange(0, R, 1).reshape(R,1)
    ncoarse = n*2*np.pi/R
    nshift = np.arange(0, N, 1).reshape(N,1) - (N-1)/2
    y_i = np.copy(y).reshape(N,1)
    omegalist = np.zeros((1,1),dtype=float)
    gainlist = np.zeros((1,1),dtype=complex)
    iterc = 0
    while True:
        gains = (np.fft.fft(y_i.squeeze(),R)/math.sqrt(N)).reshape(R,1)
        if (nshift[0] != 0):
            gains = gains * np.exp(-1j*ncoarse*nshift[0])
        cost = np.power(np.abs(gains),2)
        idx = np.argmax(cost)
        newomega = ncoarse[idx]
        newgain = gains[idx]
        x = np.exp(1j*nshift*newomega)/math.sqrt(N)
        y_i = y.reshape(N,1) - newgain*x;
        iterc += 1
        if iterc > P:
            break
        y_i,newomega,newgain = refineone(y_i,newomega,newgain,nshift,N,False)
        if iterc == 1:
            omegalist[0] = newomega
            gainlist[0] = newgain
        else: 
            omegalist = np.concatenate((omegalist,newomega),axis=0)
            gainlist = np.concatenate((gainlist,newgain),axis=0)
        y_i,omegalist,gainlist = refineall(y_i,omegalist,gainlist,nshift,N,Rc)
        A = np.exp(1j * nshift @ omegalist.T)/math.sqrt(N)
        Ap = A.conj().T@A
        if np.abs(np.linalg.det(Ap)) > 1e-6:
            gainList = (np.linalg.solve(Ap,A.conj().T@y)).reshape(iterc,1)
        else:
            gainList = (np.linalg.lstsq(Ap,A.conj().T@y))[0].reshape(iterc,1)  
        y_i = y.reshape(N,1) - A@gainList 
    f = omegalist.reshape(1,P)    
    A = np.exp(1j * np.arange(0, N, 1).reshape(N,1) @ f)
    a = np.abs(np.linalg.solve(A.conj().T@A,A.conj().T@y))
    return f.squeeze(),a               
            
def refineone(y_i,newomega,newgain,nshift,N,orth):   
    x = np.exp(1j*nshift*newomega)/math.sqrt(N)
    dx = 1j * nshift * x
    d2x = -np.power(nshift,2) * x
    y = y_i + newgain * x
    if orth==True:
        gain_newton = x.conjugate()*y
        y_i_newton = y - gain_newton*x
    dcost = -2*np.real(newgain*y_i.conjugate().T@dx)
    d2cost = -2*np.real(newgain*y_i.conjugate().T@d2x) + 2*np.power(abs(newgain),2)*(dx.conjugate().T@dx)
    if d2cost > 0:
        omega_newton = newomega - dcost/d2cost   
    else:
        omega_newton = newomega - np.sign(dcost)*np.pi/(2*N)*random.random()
    x = np.exp(1j*nshift*omega_newton)/math.sqrt(N)
    gain_newton = x.conjugate().T@y
    y_i_newton = y - gain_newton*x
    if ((y_i_newton.conjugate().T@y_i_newton) <= (y_i.conjugate().T@y_i)):
        newomega = np.real(omega_newton)
        newgain = gain_newton
        y_i = y_i_newton
    else: 
        newomega = newomega.reshape(1,1)
        newgain = newgain.reshape(1,1)
    return y_i,newomega,newgain
    
def refineall(y_i,omegalist,gainlist,nshift,N,Rc):
    l = len(omegalist)
    for i in range(1,Rc+1):
        for j in range(1,l+1):
            omega = omegalist[j-1]
            gain = gainlist[j-1]
            y_i,omega,gain = refineone(y_i,omega,gain,nshift,N,False)
            omegalist[j-1] = omega
            gainlist[j-1] = gain
    return y_i,omegalist,gainlist
