import numpy as np
import matplotlib.pyplot as plt


def madden_wavelength(filename = "WavelengthGrid_Microns_HighRes.txt"):
    with open(filename, 'r') as f:
        array = np.loadtxt(f)
    return array


def madden_plotter(wavelength = madden_wavelength(), flux = ""):
    plt.plot(wavelength, flux)
    plt.xlabel('Wavelength (microns)')
    plt.ylabel('Flux')
    plt.show()


def madden_zip_file_type(type):
    """
    Args:
        type(str): Holds the Resolution Type: Either Total, Reflectance, or Emission (CASE SENSITIVE)

    Returns: Appropriate ZipFile name for Opening

    """
    if type == "Total":
        zipfilename = "Spectra_Total_HighRes.zip"
    elif type == "Reflectance":
        zipfilename = "Spectra_ReflectanceOnly_HighRes.zip"
    elif type == "Emission":
        zipfilename = "Spectra_EmissionOnly_HighRes.zip"
    else:
        raise ValueError("Type must be Total, Reflectance, or Emission")
    return zipfilename


def madden_zip_file_path(surface = "", cloud_cover = "", surface_type = "", type = ""):
    path = ""
    if type == "Total":
        path += "Spectra_Total_HighRes\\"
    elif type == "Reflectance":
        path += "Spectra_ReflectanceOnly_HighRes\\"
    elif type == "Emission":
        path += "Spectra_EmissionOnly_HighRes\\"


    if surface == "Single":
        path += "Single_Surface\\"
    elif surface == "Multiple":
        path += "Mixed_Surface\\"
    else:
        raise ValueError("Surface must be Single or Multiple")


    if cloud_cover == "Cloudy":
        path += "Cloudy\\"
    elif cloud_cover == "Clear":
        path += "Clear\\"
    else:
        raise ValueError("Cloud cover must be Cloudy or Clear")


    if surface_type == "Trees" and surface == "Single":
        path += "Trees\\"
    elif surface_type == "Solid Clouds" and cloud_cover == "Cloudy" and surface == "Single":
        path += "Solid_Clouds\\"
    elif surface_type == "Seawater" and surface == "Single":
        path += "Seawater\\"
    elif surface_type == "Sand" and surface == "Single":
        path += "Sand\\"
    elif surface_type == "Grass" and surface == "Single":
        path += "Grass\\"
    elif surface_type == "Granite" and surface == "Single":
        path += "Granite\\"
    elif surface_type == "Coast" and surface == "Single":
        path += "Coast\\"
    elif surface_type == "Basalt" and surface == "Single":
        path += "Basalt\\"
    elif surface_type == "Flat" and cloud_cover == "Clear" and surface == "Single":
        path += "Flat_0.31"
    elif surface_type == "Basalt" and surface == "Multiple":
        path += "Basalt_Seawater\\"
    elif surface_type == "Earth" and surface == "Multiple":
        path += "Earth\\"
    elif surface_type == "Granite" and surface == "Multiple":
        path += "Granite_Seawater\\"
    elif surface_type == "Grass" and surface == "Multiple":
        path += "Grass_Seawater\\"
    elif surface_type == "Sand" and surface == "Multiple":
        path += "Sand_Seawater\\"
    elif surface_type == "Snow" and surface == "Multiple":
        path += "Snow_Seawater\\"
    elif surface_type == "Trees" and surface == "Multiple":
        path += "Trees_Seawater\\"
    else:
        if surface == "Single":
            raise ValueError("Single surface types must be Trees, Seawater, Sand, Grass, Granite, Coast, or Basalt. \n Solid Clouds surface type is only available for Cloudy Models, and Flat surface type is only available to Clear Models.")
        else:
            raise ValueError("Multiple surface types must be Basalt, Earth, Granite, Grass, Sand, Snow, or Trees.")

    return path


def madden_file_name():
    filename = ""
    return filename




print(madden_zip_file_path("Single", "Cloudy", "Basalt", "Emission"))


