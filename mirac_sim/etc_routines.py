import numpy as n
from astropy.stats import sigma_clip
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

#constants
kb = 1.38064852e-23 #[m^2 kg s^-2 K^-1]
c = 2.998e8         #[m/s]
h = 6.62607004e-34  #[m^2 kg /s]
microns2m = 1e-6

def seeing_disk(x0,y0,x_grid, y_grid, r):

    dist = n.sqrt((x0 - x_grid)**2 + (y0 - y_grid)**2)

    good = n.where(dist < r)

    return good


def bin_array(array, w, no_elems, dpix, sig_clip = None, extra_array = None):

    binned_array = n.zeros(no_elems)
    binned_array_err = n.zeros(no_elems)
    if extra_array is not None:
        binned_extra_array = n.zeros(no_elems)
        

    for i in range(no_elems):

        array_subset = array[i*dpix : i*dpix + dpix]
        w_subset = w[i*dpix : i*dpix + dpix]
      
        if extra_array is not None:
            extra_subset = extra_array[i*dpix : i*dpix + dpix]

            

        good = n.where( (n.isnan(w_subset) == False) & (n.isnan(array_subset) == False))


        if len(good[0]) == 0:

            binned_array[i] = n.nan
            binned_array_err[i] = n.nan

            if extra_array is not None:
                binned_extra_array[i] = n.nan
                

        else:
           
            array_subset =  array_subset[good]
            w_subset =  w_subset[good]
            if extra_array is not None:
                extra_subset = extra_subset[good]
            
            if sig_clip is not None:
                array_subset = sigma_clip(array_subset, sigma = sig_clip)

                
                if extra_array is not None:
                    extra_subset =  extra_subset[~array_subset.mask]

                w_subset = w_subset[~array_subset.mask]
                array_subset = array_subset[~array_subset.mask]

                
            binned_array[i] = n.nansum(array_subset * w_subset) / n.nansum(w_subset)
            binned_array_err[i] = n.sqrt(1/ n.nansum(w_subset))
            if extra_array is not None:
                 binned_extra_array[i] = n.nansum(extra_subset * w_subset) / n.nansum(w_subset)


    if extra_array is not None:
        return  binned_array, binned_array_err, binned_extra_array

    else:
        return  binned_array, binned_array_err


def pass_through_transmission(data_flux, data_lam, filter_info, pass_filter = False):

    #Pass a spectrum through some kind of transmission curve
    #could be a filter or some QE cruve
    #the data has to have broader/the same coverage than the filter
    
    #the units you put in will the units that come out bitch!!!

    ##### filter transmission curve ###################
    filter_trans, filter_lam = filter_info
    
    good_values = n.where(filter_lam != 0)
    
    filter_trans = filter_trans[good_values]
    filter_lam = filter_lam[good_values]
    
    lam_min = filter_lam.min() 
    lam_max = filter_lam.max()
    
    #print(lam_min)
    #print(lam_max)

    #wavelengths containted in the filter
    relevant = n.where( (data_lam <= lam_max) & ( data_lam >= lam_min) )
    data_in_filter = data_flux[relevant]
    lam_in_filter =  data_lam[relevant]

    dlam_data = n.median(n.diff(lam_in_filter)) #median wavelenth spacing in  the model
    dlam_filter = n.median(n.diff(filter_lam)) #median wavelenth spacing in  the filter

    bins_filter = int(abs(dlam_data/dlam_filter))
    #print(bins_filter)


    if bins_filter <=  1:

        filter_trans_f = interp1d(filter_lam, filter_trans) #interpolation function of the transmission curve
        
        #interpolate the transmission curve onto the grid of that data
        filter_trans_interp_data = filter_trans_f(lam_in_filter)


    else:
        #if the resolution of the filter is higher than the data bin it down
        w = n.ones(len(filter_trans))
        elems_filter = int(len(filter_trans)/bins_filter) #total number of final elements

        filter_trans_bin, dummy, filter_lam_bin = bin_array(filter_trans, w, elems_filter, bins_filter, extra_array = filter_lam)
        filter_trans_f = interp1d(filter_lam_bin , filter_trans_bin, bounds_error = False, fill_value = n.nan)
        #interpolation function of the transmission curve

        filter_trans_interp_data = filter_trans_f(lam_in_filter)


    data_flux_out = filter_trans_interp_data * data_in_filter

    if pass_filter == False:
        return data_flux_out, lam_in_filter

    if pass_filter == True:
        return data_flux_out, lam_in_filter, filter_trans_interp_data

        


    
def get_flux_in_filter(data_flux, data_lam, filter_info, detector_info = False):

    #pass the spectra through the filters
    flux_out, lam_out, filter_transmission = pass_through_transmission(data_flux, data_lam, filter_info, pass_filter = True)

    transmission_tot = filter_transmission

    if detector_info != False:
        #pass the spectra through the detector qe
        flux_out, lam_out, detector_transmission = pass_through_transmission(flux_out, lam_out, detector_info, pass_filter = True)

        transmission_tot *= detector_transmission


    flux_tot = n.sum(flux_out)/n.sum(transmission_tot)

    return flux_tot
