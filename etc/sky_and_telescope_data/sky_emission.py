import numpy as np
import matplotlib.pyplot as plt


#SKY DATA
def sky_emission_NQ(pwv, airmass, sky_dir):

    #pwv - water vapor columm (mm), values = 1, 1.6, 3, 5
    #airmass - unitless           , values = 1, 1.5, 2

    pwv_str = str(int(pwv*10))
    airmass_str = str(int(airmass*10))
    
    

    sky_data = np.genfromtxt(sky_dir + 'maunakea_sky_emission_NQ/mk_skybg_nq_'+pwv_str+'_'+airmass_str+'_ph.dat')

    sky_lam = sky_data[:,0]*10. #angstroms
    sky_flux = sky_data[:,1]    #phot/s/nm/arcsec^2/m^2

    return sky_flux, sky_lam



def sky_emission_JHKLM(pwv, airmass, sky_dir):

    #pwv - water vapor columm (mm), values = 1, 1.6, 3, 5
    #airmass - unitless           , values = 1, 1.5, 2

    pwv_str = str(int(pwv*10))
    airmass_str = str(int(airmass*10))
    
    

    sky_data = np.genfromtxt(sky_dir + 'maunakea_sky_emission_JHKLM/mk_skybg_zm_'+pwv_str+'_'+airmass_str+'_ph.dat')

    sky_lam = sky_data[:,0]*10. #angstroms
    sky_flux = sky_data[:,1]    #phot/s/nm/arcsec^2/m^2

    return sky_flux, sky_lam
