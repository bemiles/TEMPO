import numpy as np
import matplotlib.pyplot as plt


def sky_transmission_NQ(pwv, airmass, sky_dir):

    #pwv - water vapor columm (mm), values = 1, 1.6, 3, 5
    #airmass - unitless           , values = 1, 1.5, 2

    pwv_str = str(int(pwv*10))
    airmass_str = str(int(airmass*10))
    
    transmission = np.genfromtxt(sky_dir + 'maunakea_sky_transmission_NQ/mktrans_nq_'+pwv_str+'_'+airmass_str+'_ph.dat')
    
    sky_t_lam = transmission[:,0] #wavelength (microns)
    sky_t = transmission[:,1] #transmission (fraction)

    return sky_t, sky_t_lam


def sky_transmission_JHKLM(pwv, airmass,sky_dir):

    #pwv - water vapor columm (mm), values = 1, 1.6, 3, 5
    #airmass - unitless           , values = 1, 1.5, 2

    pwv_str = str(int(pwv*10))
    airmass_str = str(int(airmass*10))
    

    transmission = np.genfromtxt(sky_dir + 'maunakea_sky_transmission_JHKLM/mktrans_zm_'+pwv_str+'_'+airmass_str+'_ph.dat')
    
    sky_t_lam = transmission[:,0] #wavelength (microns)
    sky_t = transmission[:,1] #transmission (fraction)

    return sky_t, sky_t_lam
