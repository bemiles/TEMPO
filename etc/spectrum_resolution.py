"""
Created on Sun Jul 20 16:49:11 2025

@author: ellabutler
"""

"""
Created on Sun Jun 22 17:16:27 2025

@author: ellabutler

This script reads in Sonora spectral models from .tar.gz archives and computes 
the spectral resolution as a function of wavelength. Spectra are read directly 
from compressed .tar.gz files without extracting them to the disk.

Spectra must be ASCII format with a header of 2 lines and two columns:
wavelength (microns) and flux (erg/cm²/s/Hz).
"""

import os
import io
import tarfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def get_filepath(filename, base_dir="/Users/ellabutler/MilesGroup/5063476"):
    """
    Load spectrum data from within a .tar.gz archive.

    Parameters
    ----------
    filename : str
        Base filename (without .tar.gz extension)

    Returns
    -------
    wavelength : ndarray
        Wavelength array in microns
    flux : ndarray
        Flux array in erg/cm²/s/Hz
    """
    filepath = os.path.join(base_dir, filename + ".tar.gz")
    return filepath


def load_bobcat_spectrum_from_tar(filename):
    """
    Load Sonora Bobcat spectrum data from within a .tar.gz file.

    Parameters
    ----------
    filename : str
        Base filename (without .tar.gz extension)

    Returns
    -------
    wavelength : ndarray
        Wavelength array in microns
    flux : ndarray
        Flux array in erg/cm²/s/Hz
    """
    filepath = get_filepath(filename)
    with tarfile.open(filepath, mode='r:gz') as tar:
        # Get the first file in the archive
        member = tar.getmembers()[0]
        f = tar.extractfile(member)
        if f is None:
            raise IOError(f"Could not extract file from {filepath}")
        content = f.read().decode("utf-8")
        data = np.genfromtxt(io.StringIO(content), skip_header=2)
    
    wavelength = data[:, 0]  # wavelength in microns
    flux = data[:, 1]        # flux in erg/cm^2/s/Hz
    return wavelength, flux

def load_diamondback_spectrum_from_tar(filename):
    """
    Load Sonora Diamondback spectrum data from within a .tar.gz file.

    Parameters
    ----------
    filename : str
        Base filename (without .tar.gz extension)

    Returns
    -------
    wavelength : ndarray
        Wavelength array in microns
    flux : ndarray
        Flux array in erg/cm²/s/Hz
    """
    filepath = get_filepath(filename)
    with tarfile.open(filepath, mode='r:gz') as tar:
        # Get the first file in the archive
        member = tar.getmembers()[0]
        f = tar.extractfile(member)
        if f is None:
            raise IOError(f"Could not extract file from {filepath}")
        content = f.read().decode("utf-8")
        data = np.genfromtxt(io.StringIO(content), skip_header=2)
    
    wavelength = data[:, 0]  # wavelength in microns
    flux = data[:, 1]        # flux in erg/cm^2/s/Hz
    return wavelength, flux
    
    
def load_elfowl_spectrum_from_tar(filename):
    """
    Load Sonora Elf Owl spectrum data from within a .tar.gz file.

    Parameters
    ----------
    filename : str
        Base filename (without .tar.gz extension)

    Returns
    -------
    wavelength : ndarray
        Wavelength array in microns
    flux : ndarray
        Flux array in erg/cm²/s/Hz
    """
    filepath = get_filepath(filename)
    with tarfile.open(filepath, mode='r:gz') as tar:
        # Get the first file in the archive
        member = tar.getmembers()[0]
        f = tar.extractfile(member)
        if f is None:
            raise IOError(f"Could not extract file from {filepath}")
        content = f.read().decode("utf-8")
        data = np.genfromtxt(io.StringIO(content), skip_header=2)
    
    wavelength = data[:, 0]  # wavelength in microns
    flux = data[:, 1]        # flux in erg/cm^2/s/Hz
    return wavelength, flux
    

def compute_resolution(wavelength):
    """
    Compute the spectral resolution- R = λ / Δλ.

    Parameters
    ----------
    wavelength : ndarray
        Wavelength array in microns

    Returns
    -------
    resolution : ndarray
        Spectral resolution array (one element shorter than input)  
    """
    delta_lambda = np.diff(wavelength)
    resolution = wavelength[1:] / delta_lambda
    return resolution


def get_model_filename(temperature, gravity, metallicity):
    """
    Generate the model filename based on input parameters.

    Parameters
    ----------
    temperature : int
        Stellar effective temperature (e.g., 200 for 2000 K)
    gravity : float
        log10(surface gravity in cgs, i.e. log(g/cm/s²))
    metallicity : float
        Metallicity [M/H] (e.g., -0.5, +0.5)

    Returns
    -------
    filename : str
        Model filename string
    """
    g_str = f"g{int(gravity * 10)}"
    m_sign = "+" if metallicity > 0 else "-" if metallicity < 0 else "0.0"
    m_str = f"m{m_sign}{metallicity}"
    model_filename = f"sp_t{temperature}{g_str}nc_{m_str}"
    return model_filename


def plot_resolution(wavelength, resolution, label=None):
    """
    Plot spectral resolution as a function of wavelength.

    Parameters
    ----------
    wavelength : ndarray
        Original wavelength array (includes the first point not used in resolution).

    resolution : ndarray
        Computed resolution values (length = len(wavelength) - 1).

    label : str, optional
        Legend label for the plot.
    
    Returns
    -------
    None
    """
    plt.plot(wavelength[1:], resolution, label=label)
    plt.xlabel('Wavelength (microns)')
    plt.ylabel('Resolution (λ / Δλ)')
    plt.title("Spectral Resolution vs. Wavelength")
    plt.xlim(1.0, 14.0)
    plt.ylim(0, np.nanmax(resolution) * 1.1)
    if label:
        plt.legend()
    plt.grid(True)


def rebin_spectrum_to_resolution(wavelength, flux, target_resolution):
    """
    Rebin a spectrum to a constant desired resolution R = λ / Δλ using
    linear spacing.

    Parameters
    ----------
    wavelength : ndarray
        Original wavelength array in microns 
        (must be sorted in ascending order).
    flux : ndarray
        Original flux array (same length as wavelength).
    target_resolution : float
        Desired constant spectral resolution.

    Returns
    -------
    rebinned_wavelength : ndarray
        Array of rebinned wavelength centers.
    rebinned_flux : ndarray
        Array of rebinned fluxes interpolated to those centers.
    """
    sort_idx = np.argsort(wavelength)
    wavelength = wavelength[sort_idx]
    flux = flux[sort_idx]

    start = wavelength[0]
    end = wavelength[-1]

    rebinned_wavelength = []
    rebinned_flux = []

    current_lambda = start
    flux_interp = interp1d(wavelength, flux, kind='linear', bounds_error=False, 
                                                      fill_value="extrapolate")

    while current_lambda < end:
        delta_lambda = current_lambda / target_resolution
        next_lambda = current_lambda + delta_lambda
        bin_center = (current_lambda + next_lambda) / 2
        bin_flux = flux_interp(bin_center)

        rebinned_wavelength.append(bin_center)
        rebinned_flux.append(bin_flux)

        current_lambda = next_lambda

    return np.array(rebinned_wavelength), np.array(rebinned_flux)



# Add a way to allow users to use Bobcat, Elf Owl, and Diamondback models. Elf 
# Owl and Diamondback will be called in by a separate function 


    
    # 1.1 - 14 microns 
    # first: 7.5-14 microns
    # 1.1-5.2 microns to match paper 
    
        
    # find spacing Dr. Miles needs - use values from Google sheets

# make it so that the user can plug in a temperature, surface gravity, and 
# metallicity, to get back the inputs requested 

# add a user input for resolution and mode (wavelength min and max is fixed)

# later down the line: fix the hardcoded file path so it can also take a Madden
# file

# document the different integers for the available models

# add documentation for each line

# try expecto (PHOENIX stellar spectra) 

# try pulling in TEMPO observing modes txt file (not the imaging section)
