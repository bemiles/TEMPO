"""
Created on Sun Jun 22 17:16:27 2025

@author: ellabutler
"""

"""
This script reads in two PHOENIX stellar spectral models and computes the 
spectral resolution as a function of wavelength. It is designed for use with 
models stored locally, specifically those generated by the PHOENIX/BT-Settl 
grid. Given filenames (or optionally, parameters like temperature, 
surface gravity, and metallicity), the code loads the spectra, extracts the 
wavelength and flux data, computes the spectral resolution (R = λ / Δλ), 
and returns it for further analysis or plotting. Users can visualize 
the resolution or rebin/interpolate it to compare with observational 
requirements.

Spectra are expected to be in ASCII format with a header of 2 lines and 
two columns: wavelength (microns) and flux (erg/cm²/s/Hz).
"""
import os
#pip install expecto
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
# import gaussian_filter1d from scipy.ndimage


    
def get_filepath(filename, base_dir="/Users/ellabutler/MilesGroup/5063476"):
    """
    Construct full file path for a given filename.
        
    Parameters
    ----------
    filename : str
    Name of the spectral file (e.g., "sp_t200g10nc_m-0.5")
        
    Returns
    -------
    str
    Full file path
    """
    return os.path.join(base_dir, filename)


def load_sonora_bobcat_spectrum(filename):
    """
    Load spectrum data from a file.

    Parameters
    ----------
    filename : str
        Name of the spectral file

    Returns
    -------
    wavelength : ndarray
        Wavelength array in microns
    flux : ndarray
        Flux array in erg/cm²/s/Hz
    """
    filepath = get_filepath(filename)
    data = np.genfromtxt(filepath, skip_header=2)
    wavelength = data[:, 0]   # wavelength in microns
    flux = data[:, 1]   # flux in erg/cm^2/s/Hz
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


def model_filename(temperature, gravity, metallicity):
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
    g_str = f"g{int(gravity * 10)}"  # log(g) = 1.0 -> g10
    m_sign = "+" if metallicity >= 0 else ""
    m_str = f"m{m_sign}{metallicity}"
    return f"sp_t{temperature}{g_str}nc_{m_str}"
    

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
    
# bin in linear space! 
def rebin_spectrum_to_resolution(wavelength, flux, target_resolution):
    """
    Rebin a spectrum to a constant desired resolution R = λ / Δλ.

    Parameters
    ----------
    wavelength : ndarray
        Original wavelength array in microns.
    flux : ndarray
        Original flux array.
    target_resolution : float
        Desired constant spectral resolution.

    Returns
    -------
    rebinned_wavelength : ndarray
        Rebin center wavelengths.
    rebinned_flux : ndarray
        Flux interpolated at rebinned wavelengths.
    """
    # Define log-spacing for equal resolution: Δλ/λ = 1/R
    loglam_start = np.log(wavelength[0])
    loglam_end = np.log(wavelength[-1])
    n_bins = int(np.log(wavelength[-1] - wavelength[0]) * target_resolution)
    loglam_bins = np.linspace(loglam_start, loglam_end, n_bins)
    rebinned_wavelength = np.exp(loglam_bins)

    # Interpolate flux to the rebinned wavelength grid
    flux_interp = interp1d(wavelength, flux, kind='linear', bounds_error=False,
                           fill_value="extrapolate")
    rebinned_flux = flux_interp(rebinned_wavelength)

    return rebinned_wavelength, rebinned_flux
    
    
    # 1.1 - 14 microns 
    # first: 7.5-14 microns
    # 1.1-5.2 microns to match paper 
    
    
    # resolution: central wavelength /  wavelength range 
    
    # find spacing Dr. Miles needs - use values from Google sheets

# make it so that the user can plug in a temperature, surface gravity, and 
# metallicity, to get back the inputs requested 

# add a user input for resolution and mode (wavelength min and max is fixed)

# later down the line: fix the hardcoded file path so it can also take a Madden file

# document the different integers for the available models

# add documentation for each line

# try expecto (PHOENIX stellar spectra) 

# try pulling in TEMPO observing modes txt file (not the imaging section)



