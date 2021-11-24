<div id="top"></div>

# Line spectral estimators for ultrasonic vibration
This implements the line spectral estimators written in Python used in the submitted manuscript "Real-time sinusoidal parameter estimation for damage growth monitoring during ultrasonic very high cycle fatigue tests". 

Abstract:
> Ultrasonic fatigue tests (UFT) are used to study the fatigue life behavior of metallic components undergoing a very high number of cycles (typically 10^7-10^9 cycles) under relatively low mechanical loads. By soliciting fatigue specimens at 20 kHz, ultrasonic fatigue machines are indispensable for monitoring damage growth and fatigue failures in a reasonable amount of time. As fatigue damage accumulates in the specimen, the specimen's free-end exhibits a nonlinear dynamic response. The resulting quasi-stationary, harmonic signals have sinusoidal parameters (frequency and amplitude) which are slowly time-varying with respect to the excitation frequency. The discrete Fourier transform (DFT) is typically used to extract these evolving sinusoidal parameters from a window of finite data of the vibration signal. Alternative spectral estimation methods, specifically line spectra estimators (LSEs), exploit a priori information of the signal via their modeling basis and overcome limitations seen by the DFT. Many LSEs are known to have state-of-the-art results when benchmarked on purely stationary signals with unit amplitudes. However, their performances are unknown in the context of slowly time-varying signals typical of UFT, leading to a widespread use of the DFT. Thus, this paper benchmarks classical and modern LSEs against specific synthetic signals which arise in UFTs. Adequate algorithms are then recommended and made publicly available to process experimental data coming from ultrasonic fatigue tests depending on performance metrics and experimental restraints.
  

## Setup and usage
We recommend the usage of the [Anaconda distribution](https://www.anaconda.com/products/individual) and the included [Spyder IDE](https://www.spyder-ide.org/). Spyder is a unique Python IDE that offers a [MATLAB](https://www.mathworks.com/products/matlab.html)-like experience with an integrated variable viewer.



### Download
Two zip files are provided due to size constraints:
Filename (download) | Size | Description
--- | --- | ---
[lineSpectraVibration-Full-v1.0.0.zip](https://github.com/slkiser/lineSpectraVibration/releases/download/v1.0.0/lineSpectraVibration-Full-v1.0.0.zip) | 679.50 MB | Includes the source code and all DeepFreq weight files for all signal lengths tested in the submitted manuscript.
[lineSpectraVibration-Lite-v1.0.0.zip](https://github.com/slkiser/lineSpectraVibration/releases/download/v1.0.0/lineSpectraVibration-Lite-v1.0.0.zip) | 23.80 MB | Includes the source code and only the DeepFreq weight file corresponding to a signal length (N = 128).



### Package installation
For required packages, refer to [`requirements.txt`](requirements.txt). The packages can be installed quickly using pip and/or in a virtual environment:
```
pip install -r requirements.txt
```
Or if using Anaconda:
```
conda install pip
pip install -r requirements.txt
```


### How to use

*Make sure to make the main folder the console's working directory.*

`example.py` is a script that generates a random number of **stationary** cissoids `K = 5` with a normalized frequency distance of `d = 2/N`  as a discrete signal length `N = 128` with an SNR of `snr = 30`. These parameters are indicated on lines 14-17:

```
N = 2**7
K = 5
snr = 30
d = 2/N
```

The script utilizes a timer class `TicToc()` which behaves very similarly to MATLAB's `tic` `toc`. For all 6 algorithms, results of time taken, and frequency and their respective amplitude estimates are given. 

At the end, a plot is created comparing the discrete Fourier Transform (via FFT), Unitary ESPRIT, and DeepFreq frequency and amplitude estimates. 

The vertical black dotted lines correspond to the true normalized wrapped frequencies and the horizontal line corresponds to the correct amplitude.

![Image of plot](https://github.com/slkiser/lineSpectraVibration/blob/main/plot.png)

The `example.py` script should be understood as a reference for syntax (especially for PyTorch), function calls, and the object-oriented organization.

___

## Acknowledgments

This research was part of a PhD thesis funded by [Arts et Métiers](https://artsetmetiers.fr/) (École nationale supérieure d'arts et métiers). Laboratory equipment was provided by H2020 FastMat (fast determination of fatigue properties of materials beyond one billion cycles) project under the European Research Council (ERC) (grant agreement No 725142) at [PIMM laboratory](https://pimm.artsetmetiers.fr/).

![Image of logos](https://github.com/slkiser/lineSpectraVibration/blob/main/logos.png)

## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>
