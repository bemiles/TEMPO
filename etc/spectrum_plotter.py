import glob
import xarray as xr
import numpy as np
import zipfile
import io
import tarfile
import gzip
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def build_filename(temperature_effective, gravity, metallicity, co_ratio="", cloud_seeding_parameter="", eddy_coefficient=""):
    if cloud_seeding_parameter != "" and eddy_coefficient != "":
        raise ValueError("Can only accept one input, not two. \n Please choose either a Diamondback file which uses the cloud seeding parameter, or an Elf Owl file which uses the eddy coefficient parameter. \n Leave the other parameter blank by inserting an empty string.")
    elif cloud_seeding_parameter != "":
        filename = f"t{temperature_effective}g{gravity}"
        if cloud_seeding_parameter == 0:
            filename += "nc_"
        else:
            filename += f"f{cloud_seeding_parameter}_"
        if metallicity > 0.0:
            filename += f"m+{metallicity}_co1.0.spec"
        else:
            filename += f"m{metallicity}_co1.0.spec"
    elif eddy_coefficient != "":
        filename = f"spectra_logzz_{float(eddy_coefficient)}_teff_{float(temperature_effective)}_grav_{float(gravity)}_mh_{float(metallicity)}_co_{float(co_ratio)}.nc"
    else:
        filename = f"sp_t{temperature_effective}g{gravity}nc_m"
        if not co_ratio == "" and co_ratio != 0.0:
            filename += f"+0.0_co{co_ratio}"
        if metallicity > 0.0:
            filename += f"+{metallicity}"
        else:
            filename += f"{metallicity}"
    if file_validate(filename):
        return file_to_array(filename)
    else:
        return None

def file_to_array(file):
    if "sp_t" in file:
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
        with tarfile.open(path, "r|gz") as tar:
            for member in tar:
                if member.name.endswith(file):
                    data_array = np.loadtxt(io.StringIO(tar.extractfile(member).read().decode('utf-8')), skiprows=2)
                    break
                elif member.name.endswith(f"{file}.gz"):
                    with gzip.open(io.BytesIO(tar.extractfile(member).read()), "rt") as data:
                        data_array = np.loadtxt(data, skiprows=2)
                        break
    elif ".spec" in file:
        with zipfile.ZipFile("spectra.zip", "r") as zf:
            with zf.open(f"spectra/{file}") as f:                         
                data_array = np.loadtxt(io.TextIOWrapper(f, encoding='utf-8'), skiprows=3)
                data_array[:,1] = data_array[:,1] * (10**3) * ((data_array[:,0]/1e+6)**2) / 2.998e+8
    else:
        if any(n in file for n in ["1300.0", "1400.0", "1500.0"]):
            path = "teff_1300.0_1400.tar.gz"
        elif any(n in file for n in ["1600.0", "1700.0", "1800.0"]):
            path = "teff_1600.0_1800.tar.gz"
        elif any(n in file for n in ["1900.0", "2000.0", "2100.0"]):
            path = "teff_1900.0_2100.tar.gz"
        elif any(n in file for n in ["275.0", "300.0", "325.0"]):
            path = "teff_275_325.tar.gz"
        elif any(n in file for n in ["350.0", "375.0", "400.0"]):
            path = "teff_350_400.tar.gz"
        elif any(n in file for n in ["425.0", "450.0", "475.0"]):
            path = "teff_425_475.tar.gz"
        elif any(n in file for n in ["500.0", "525.0", "550.0"]):
            path = "teff_500_550.tar.gz"
        elif any(n in file for n in ["575.0", "600.0", "650.0"]):
            path = "teff_575.0_650.tar.gz"
        elif any(n in file for n in ["700.0", "750.0", "800.0"]):
            path = "teff_700.0_800.tar.gz"
        elif any(n in file for n in ["850.0", "900.0", "950.0"]):
            path = "teff_850.0_950.tar.gz"
        elif any(n in file for n in ["1000.0", "1100.0", "1200.0"]):
            path = "teff_1000.0_1200.tar.gz"
        else:
            path = "teff_2200.0_2400.tar.gz"
        with tarfile.open(path, "r|gz") as tar:
            for member in tar:
                if member.name.endswith(file):
                    ds = xr.open_dataset(io.BytesIO(tar.extractfile(member).read()), engine="scipy")
                    flux = ds['flux'].values * ((ds['wavelength'].values / 10000)**2)/2.998e+10
                    data_array = np.column_stack((ds['wavelength'].values, flux))
                    break
    return data_array

def file_validate(file_name):
    temps = set()
    gravs = set()
    metals = set()
    ratios = set()
    clouds = set()
    eddys = set()
    
    files = set()
    files.update(glob.glob("spectra*.tar.gz"))
    files.update(glob.glob("*.zip"))
    files.update(glob.glob("output*.tar.gz"))
    for f in files:
        if f.endswith(".zip"):
            temps.update([900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400])
            gravs.update([31, 100, 316, 1000, 3160])
            clouds.update([0, 1, 2, 3, 4, 8])
            metals.update([-0.5, 0.0, 0.5])
            ratios.update([1.0])
        elif f.startswith("spectra") and f.endswith(".tar.gz"):
            temps.update([200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400])
            gravs.update([1000])
            if "m+0.0" in f:
                metals.update([0.0])
            elif "m+0.5" in f:
                metals.update([0.5])
            elif "m-0.5" in f:
                metals.update([-0.5])
            if "co0.5" in f:
                ratios.update([0.5])
            elif "co1.5" in f:
                ratios.update([1.5])
            else:
                ratios.update([0.0])
            
            if "0.5.tar.gz" in f or "0.0.tar.gz" in f:
                gravs.update([10, 17, 31, 100, 316, 1000, 3160])
                if "+0.5" in f:
                    gravs.update([56, 178, 562])
                elif "-0.5" in f:
                    gravs.update([1780])
                else:
                    gravs.update([56, 178, 562, 1780])
        else:
            ratios.update([0.5, 1.0, 1.5,2.0, 2.5])
            gravs.update([10.0, 17.0, 31.0, 56.0, 100.0, 178.0, 316.0, 562.0, 1000.0, 1780.0, 3160.0])
            metals.update([-1.0, -0.5, 0.0, 0.5, 0.7, 1.0])
            if "1300.0_1400" in f:
                temps.update([1300.0, 1400.0, 1500.0])
            elif "1600.0_1800" in f:
                temps.update([1600.0, 1700.0, 1800.0])
            elif "1900.0_2100" in f:
                temps.update([1900.0, 2000.0, 2100.0])
            elif "275.0_325.0" in f:
                temps.update([275.0, 300.0, 325.0])
            elif "350.0_400.0" in f:
                temps.update([350.0, 375.0, 400.0])
            elif "425.0_475.0" in f:
                temps.update([425.0, 450.0, 475.0])
            elif "500.0_550.0" in f:
                temps.update([500.0, 525.0, 550.0])
            elif "575.0_650" in f:
                temps.update([575.0, 600.0, 650.0])
            elif "700.0_800" in f:
                temps.update([700.0, 750.0, 800.0])
            elif "850.0_950" in f:
                temps.update([850.0, 900.0, 950.0])
            elif "1000.0_1200" in f:
                temps.update([1000.0, 1100.0, 1200.0])
            else:
                temps.update([2200.0, 2300.0, 2400.0])
    if file_name.endswith(".spec"):
        name = file_name[1:-5].replace("g","_").replace("f", "_").replace("nc","_0").replace("+","").replace("m","").replace("co","").split("_")
        if int(name[0]) not in temps:
            print("ERROR - temperature value not found")
            print(f"Valid Temperatures: {sorted(temps)}")
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
        eddys.update([2.0, 4.0, 7.0, 8.0, 9.0])
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
        name = file_name[4:].replace("g", "_").replace("nc_m","_").replace("+","").replace("sp_","").replace("co", "").split("_")
        if len(name) == 3:
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

def plot_flux(wavelength, flux, label=None, units="erg/cm^2/s/Hz"):
    plt.plot(wavelength, flux, label=label)
    plt.xlabel('Wavelength (microns)')
    plt.ylabel(f'Flux ({units})')
    plt.title("Flux vs. Wavelength")
    plt.xlim(1.0, 14.0)
    plt.ylim(0, np.nanmax(flux) * 1.1)
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


# array1 = build_filename(1000, 17.0, 0.0, 0.5,"", 2.0)  # Elf Owl Test
# plot_flux(array1[:,0], array1[:,1], "LogZZ = 2")
# print("ELF OWL COMPLETED")
# plt.show()

array1 = build_filename(450, 17.0, 0.0, 0.5, "", 2.0)
print("Array:", array1)
