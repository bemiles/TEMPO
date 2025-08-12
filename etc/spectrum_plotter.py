import io
import glob
import gzip
import zipfile
import tarfile
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def build_filename(temperature_effective, gravity, metallicity, co_ratio="", cloud_seeding_parameter="", eddy_coefficient=""):
    """
    Constructs a spectral filename string based on input atmospheric parameters 
    and model type, then attempts to validate and load the corresponding file.

    The file naming convention varies depending on whether the file is from:
    - Diamondback models (specified by `cloud_seeding_parameter`)
    - Elf Owl models (specified by `eddy_coefficient`)
    - Bobcat models (no `cloud_seeding_parameter` or `eddy_coefficient`)

    Only one of `cloud_seeding_parameter` or `eddy_coefficient` should be provided.
    

    Parameters
    ----------
    temperature_effective : float or int
        Effective temperature (Teff) of the model atmosphere.
    gravity : float or int
        Surface gravity of the model atmosphere.
    metallicity : float
        Metallicity of the model atmosphere (e.g., 0.0 for solar, -0.5 for sub-solar).
    co_ratio : float or str, optional
        Carbon-to-oxygen ratio (C/O). Default is "" (not used).
    cloud_seeding_parameter : float or int, optional
        Parameter used for Diamondback model files. Leave as "" if not using this model.
    eddy_coefficient : float or int, optional
        Parameter used for Elf Owl model files. Leave as "" if not using this model.

    Raises
    ------
    ValueError
        If both `cloud_seeding_parameter` and `eddy_coefficient` are provided.
    
    Returns
    -------
    ndarray or None
        Array of spectral data if the constructed filename is valid and the file exists.
        Returns None if the file is invalid or not found.

    """
    
    # Ensure only one model-specific parameter is provided
    if cloud_seeding_parameter != "" and eddy_coefficient != "":
        raise ValueError("Can only accept one input, not two. \n" 
        "Please choose either a Diamondback file which uses the cloud seeding parameter,"
        "or an Elf Owl file which uses the eddy coefficient parameter. \n"
        "Leave the other parameter blank by inserting an empty string.")
    
    # --- Diamondback file naming convention ---
    elif cloud_seeding_parameter != "":
        filename = f"t{temperature_effective}g{gravity}"
        
        # 'nc' denotes 'no clouds' if parameter is zero, otherwise include seeding factor
        if cloud_seeding_parameter == 0:
            filename += "nc_"
        else:
            filename += f"f{cloud_seeding_parameter}_"
            
        # Append metallicity; positive metallicities get a '+' sign
        if metallicity > 0.0:
            filename += f"m+{metallicity}_co1.0.spec"
        else:
            filename += f"m{metallicity}_co1.0.spec"
            
    # --- Elf Owl file naming convention ---
    elif eddy_coefficient != "":
        filename = f"spectra_logzz_{float(eddy_coefficient)}_teff_{float(temperature_effective)}_grav_{float(gravity)}_mh_{float(metallicity)}_co_{float(co_ratio)}.nc"
    
    # --- Bobcat file naming convention ---
    else:
        filename = f"sp_t{temperature_effective}g{gravity}nc_m"
        if not co_ratio == "" and co_ratio != 0.0:
            filename += f"+0.0_co{co_ratio}"
        if metallicity > 0.0:
            filename += f"+{metallicity}"
        else:
            filename += f"{metallicity}"
            
    # Validate constructed filename and attempt to load the file
    if file_validate(filename):
        return file_to_array(filename)
    else:
        return


def file_to_array(file):
    """
    Loads a spectrum file into a NumPy array, extracting it from the appropriate 
    compressed archive based on filename patterns.

    The function supports:
    - **Bobcat** model files (`sp_t` in filename) stored in `.tar.gz` archives
      containing plain text or gzipped spectra.
    - **Diamondback** model files (`.spec` in filename) stored in a `.zip` archive.
    - **Elf Owl** model files (neither of the above) stored in `.tar.gz` archives
      containing NetCDF files.

    File type and archive location are determined from keywords in the filename.
    
    Parameters
    ----------
    file : str
        Name of the spectral file (without full path) to extract and load.

    Returns
    -------
    array : ndarray
        2D NumPy array containing the spectrum:
        - For plain text or gzipped Bobcat/Diamondback files: wavelength and flux columns.
        - For NetCDF Elf Owl files: wavelength and flux columns extracted from dataset variables.

    """
    
    # --- Bobcat files ---
    # Identified by 'sp_t' in the filename
    if "sp_t" in file:
        # Choose archive path based on CO ratio or metallicity keywords in the filename
        if "co_0.5" in file:
            path = "spectra_m+0.0_co0.5_g1000nc.tar.gz"
        elif "co1.5" in file:
            path = "spectra_m+0.0_co1.5_g1000nc.tar.gz"
        elif "+0.5" in file:
            path = "spectra_m+0.5.tar.gz"
        elif "-0.5" in file:
            path = "spectra_m-0.5.tar.gz"
        else:
            path = "spectra_m+0.0.tar.gz"
        # Open the tar.gz archive and locate the file
        with tarfile.open(path, "r|gz") as tar:
            for member in tar:
                # Case: file is plain text
                if member.name.endswith(file):
                    array = np.loadtxt(io.StringIO(tar.extractfile(member).read().decode('utf-8')), skiprows=2)
                    break
                # Case: file is gzipped within the archive
                elif member.name.endswith(f"{file}.gz"):
                    with gzip.open(io.BytesIO(tar.extractfile(member).read()), "rt") as data:
                        array = np.loadtxt(data, skiprows=2)
                        break
                    
    # --- Diamondback files ---
    # Identified by the `.spec` extension
    elif ".spec" in file:
        with zipfile.ZipFile("spectra.zip", "r") as zf:
            with zf.open(f"spectra/{file}") as f:                         
                array = np.loadtxt(io.TextIOWrapper(f, encoding='utf-8'), skiprows=3)
                
    # --- Elf Owl files ---
    # Neither 'sp_t' nor '.spec' in filename, typically NetCDF data            
    else:
        # Choose archive path based on temperature ranges found in filename
        if any(n in file for n in ["1300.0", "1400.0", "1500.0"]):
            path = "output_1300.0_1400.tar.gz"
        elif any(n in file for n in ["1600.0", "1700.0", "1800.0"]):
            path = "output_1600.0_1800.tar.gz"
        elif any(n in file for n in ["1900.0", "2000.0", "2100.0"]):
            path = "output_1900.0_2100.tar.gz"
        else:
            path = "output_2200.0_2400.tar.gz"
        
        # Extract and read NetCDF file from tar.gz archive
        with tarfile.open(path, "r:gz") as tar:
            for member in tar:
                if member.name.endswith(file):
                    ds = xr.open_dataset(tar.extractfile(member).read(), engine="scipy")
                    # Stack wavelength and flux into a single 2D array
                    array = np.column_stack((ds['wavelength'].values, ds['flux'].values))
                    break
    return array


def file_validate(file_name):
    """
    Validates whether a given spectral filename matches allowed parameter 
    combinations derived from available local data archives.

    The function:
    1. Scans all local spectral archive files (`*.tar.gz` and `*.zip`).
    2. Builds sets of valid parameter values (temperature, gravity, metallicity, 
       C/O ratio, cloud seeding parameter, eddy coefficient) from the archive contents.
    3. Parses the provided `file_name` according to its model type:
       - **Diamondback**: `.spec` files
       - **Elf Owl**: `.nc` NetCDF files
       - **Bobcat**: other formats
    4. Checks whether all parameter values in `file_name` are present in the 
       corresponding valid sets.

    Parameters
    ----------
    file_name : str
        The filename (not path) of the spectrum to validate.

    Returns
    -------
    bool
        True if the filename's parameters are valid for the available datasets.
        False if any parameter is not found, with an error message printed.
    """
    
    # Parameter sets to be populated from local archive files
    temps = set()
    gravs = set()
    metals = set()
    ratios = set()
    clouds = set()
    eddys = set()
    
    # Collect available spectral archive files
    files = set()
    files.update(glob.glob("spectra*.tar.gz"))
    files.update(glob.glob("*.zip"))
    files.update(glob.glob("output*.tar.gz"))
    
    # --- Build valid parameter sets from archive filenames ---
    for f in files:
        if f.endswith(".zip"):
            # Diamondback archive (.spec format)
            temps.update([900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 
                          1800, 1900, 2000, 2100, 2200, 2300, 2400])
            gravs.update([31, 100, 316, 1000, 3160])
            clouds.update([0, 1, 2, 3, 4, 8])
            metals.update([-0.5, 0.0, 0.5])
            ratios.update([1.0])
        elif f.startswith("spectra") and f.endswith(".tar.gz"):
            # Bobcat archive (.txt or .gz inside .tar.gz)
            temps.update([200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 
                          450, 475, 500, 525, 550, 575, 600, 650, 700, 750, 
                          800, 850, 900, 950, 1000, 1100, 1200, 1300, 1400, 
                          1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 
                          2300, 2400])
            gravs.update([1000])
            
            # Add metallicity values from filename
            if "m+0.0" in f:
                metals.update([0.0])
            elif "m+0.5" in f:
                metals.update([0.5])
            elif "m-0.5" in f:
                metals.update([-0.5])
            
            # Add C/O ratio from filename
            if "co0.5" in f:
                ratios.update([0.5])
            elif "co1.5" in f:
                ratios.update([1.5])
            else:
                ratios.update([0.0])
            
            # Add extra gravities for multi-gravity archives
            if "0.5.tar.gz" in f or "0.0.tar.gz" in f:
                gravs.update([10, 17, 31, 100, 316, 1000, 3160])
                if "+0.5" in f:
                    gravs.update([56, 178, 562])
                elif "-0.5" in f:
                    gravs.update([1780])
                else:
                    gravs.update([56, 178, 562, 1780])
        else:
            # Elf Owl archive (.nc NetCDF inside .tar.gz)
            ratios.update([0.5, 1.0, 1.5, 2.5]) 
            eddys.update([2.0, 4.0, 7.0, 8.0, 9.0])
            gravs.update([17.0, 31.0, 56.0, 100.0, 178.0, 
                          316.0, 562.0, 1000.0, 1780.0, 3160.0])
            metals.update([-1.0, -0.5, 0.0, 0.5, 0.7, 1.0])
            
            # Add temperatures and extra parameters based on filename range
            if "1300.0_1400" in f:
                temps.update([1300.0, 1400.0, 1500.0])
                gravs.update([10.0])
            elif "1600.0_1800" in f:
                temps.update([1600.0, 1700.0, 1800.0])
                ratios.update([2.0])
                gravs.update([10.0])
            elif "1900.0_2100" in f:
                temps.update([1900.0, 2000.0, 2100.0])
                gravs.update([10.0])
            else:
                temps.update([2200.0, 2300.0, 2400.0])  
    
    # --- Validate filename parameters based on model type ---
    if file_name.endswith(".spec"):
        # Diamondback format: t{temp}g{grav}f{cloud}m{metal}_co{ratio}.spec
        name = file_name[1:-5].replace("g","_").replace("f", "_").replace("nc","_0").replace("+","").replace("m","").replace("co","").split("_")
        if int(name[0]) not in temps:
            print("ERROR - temperature value not found")
            print(f"Valid Temperatures: {sorted(temps)}")
            print("No spectral archive files found in current directory.")
            return False
        if int(name[1]) not in gravs:
            print("ERROR - gravity value not found")
            print(f"Valid Gravities: {sorted(gravs)}")
            return False
        if int(name[2]) not in clouds:
            print("ERROR - cloud seeding value not found")
            print(f"Valid Cloud Seeding: {sorted(clouds)}")
            return False
        if float(name[3]) not in metals:
            print("ERROR - metal value not found")
            print(f"Valid Metals: {sorted(metals)}")
            return False
        if float(name[4]) not in ratios:
            print("ERROR - ratio value not found")
            print(f"Valid Ratios: {sorted(ratios)}")
            return False
    elif file_name.endswith(".nc"):
        # Elf Owl format: spectra_logzz_{eddy}_teff_{temp}_grav_{grav}_mh_{metal}_co_{ratio}.nc
        name = file_name[14:-3].replace("_teff_","_").replace("_grav_", "_").replace("_mh_","_").replace("+","").replace("_co_","_").split("_")
        if float(name[0]) not in eddys:
            print("ERROR - eddy coefficient value not found")
            print(f"Valid Eddy Coefficients: {sorted(eddys)}")
            return False
        if float(name[1]) not in temps:
            print("ERROR - temperature value not found")
            print(f"Valid Temperatures: {sorted(temps)}")
            return False
        if float(name[2]) not in gravs:
            print("ERROR - gravity value not found")
            print(f"Valid Gravities: {sorted(gravs)}")
            return False
        if float(name[3]) not in metals:
            print("ERROR - metal value not found")
            print(f"Valid Metals: {sorted(metals)}")
            return False
        if float(name[4]) not in ratios:
            print("ERROR - ratio value not found")
            print(f"Valid Ratios: {sorted(ratios)}")
            return False
    else:
        # Bobcat format: sp_t{temp}g{grav}nc_m{metal}[optional_co{ratio}]
        name = file_name[4:].replace("g", "_").replace("nc_m","_").replace("+","").replace("sp_","").replace("co", "").split("_")
        if len(name) == 3:
        # If C/O ratio not provided, append default 0.0
                name.append(0.0)
        if int(name[0]) not in temps:
            print("ERROR - temperature value not found")
            print(f"Valid Temperatures: {sorted(temps)}")
            return False
        if int(name[1]) not in gravs:
            print("ERROR - gravity value not found")
            print(f"Valid Gravities: {sorted(gravs)}")
            return False
        if float(name[2]) not in metals:
            print("ERROR - metal value not found")
            print(f"Valid Metals: {sorted(metals)}")
            return False
        if float(name[3]) not in ratios:
            print("ERROR - ratio value not found")
            print(f"Valid Ratios: {sorted(ratios)}")
            return False      
    return True


def plot_resolution(wavelength, resolution, label=None):
    """
    Plot spectral resolution as a function of wavelength.

    Parameters
    ----------
    wavelength : array-like
        Array of wavelength values in microns.
    resolution : array-like
        Corresponding resolution values (λ / Δλ) for each wavelength.
    label : str, optional
        Label for the plotted line, used in the legend if provided.

    Returns
    -------
    None
        The function generates and displays a plot of resolution vs. wavelength.

    Notes
    -----
    - The x-axis is fixed to the range 1.0–14.0 microns.
    - The y-axis range is set from 0 to 10% above the maximum resolution value.
    - If `label` is given, a legend will be displayed.
    - A grid is added for readability.
    """
    
    plt.plot(wavelength, resolution, label=label)
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
    Rebin a spectrum to a constant spectral resolution.

    This function resamples a spectrum such that the resolution
    R = λ / Δλ remains constant across the wavelength range,
    using linearly spaced bins in wavelength according to the
    target resolution.

    Parameters
    ----------
    wavelength : array-like
        Original wavelength array in microns. Must be sorted in ascending
        order; if not, the function will internally sort it.
    flux : array-like
        Original flux array corresponding to the input wavelengths. Must have
        the same length as `wavelength`.
    target_resolution : float
        Desired constant spectral resolution (R = λ / Δλ).

    Returns
    -------
    rebinned_wavelength : ndarray
        1D array of rebinned wavelength bin centers in microns.
    rebinned_flux : ndarray
        1D array of flux values interpolated to the rebinned wavelength centers.

    Notes
    -----
    - Uses linear interpolation (`scipy.interpolate.interp1d`) to estimate
      flux at rebinned bin centers.
    - The bin width Δλ is calculated for each bin as λ / R, where λ is the
      start of the current bin.
    - Values outside the original wavelength range will be extrapolated.
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


# for this code to work, make sure it exists on the same level as all the 
# tar.gz and .zip files. 

array = build_filename(225, 316, 0.0, 0.0, "", "")  # Bobcat Special Test
plot_resolution(array[:,0], array[:,1], "Bobcat Special - TEST")
plt.show()

print("BOBCAT SPECIAL COMPLETED")
print(build_filename(200, 3160, 0.5, 0.0,"","")) # Bobcat Normal Test
print("BOBCAT NORMAL COMPLETED")
#print(build_filename(2000, 3160, 0.5, 0.0,3,"")) # Diamondback Test
print("DIAMONDBACK COMPLETED")
#print(build_filename(1700.0, 1000.0, 0.5, 1.0,"",4.0))  # Elf Owl Test
print("ELF OWL COMPLETED")

# stress test 
# make some plots of different temperatures, cloud seeding parameters, and eddy diffusion coefficient
# try to recreate Marley (2021)
# add documentation 
# leave skeleton code for Madden files
# submit a ReadMe file 

# do this all by Wednesday 

