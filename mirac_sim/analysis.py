import numpy as n
from astropy.stats import sigma_clip
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


#constants
kb = 1.38064852e-23 #[m^2 kg s^-2 K^-1]
c = 2.998e8         #[m/s]
h = 6.62607004e-34  #[m^2 kg /s]
microns2m = 1e-6


def mean_smooth(x,kernel):
    """
    # x - the array to smooth down
    # kernel - size of pixels to mean over for the new value should be odd
    """
    x_new = n.zeros(len(x))

    for i in range(len(x)):

        x1 = i - kernel/2
        x2 = i + kernel/2

        if x1 < 0:
            x1 = 0
        if x2 > len(x):
            x2 = len(x) 
    
        x_new[i] = n.mean(x[x1:x2])
    return x_new

def bin_array(x, bin_size, w = None):

   # bin_size - number of elements to bin by
   # x - the array
   # w - weights to apply when binning which are the error
    array_len = len(x)/bin_size
    if len(x)/bin_size % 2 != 0:
        array_len += 1

    x_bin = n.arange(0, array_len)
    array_bin = n.zeros(len(x_bin))
    error_bin = n.zeros(len(x_bin))

    if w is None:
        for i in x_bin:
            it = i*bin_size
            x1 = it - bin_size/2
            x2 = it + bin_size/2

            if bin_size % 2 != 0:
                x2 += 1
       
            if x1 < 0:
                x1 = 0
            if x2 >= len(x):
                x2 = len(x) - 1

            #print it, x1, x2, len(x[x1:x2])

            x_mean = n.nanmean( x[x1:x2] )
            array_bin[i] = x_mean

        return array_bin

    else:
   
        for i in x_bin:
            it = i*bin_size

            x1 = it - bin_size/2
            x2 = it + bin_size/2

            if bin_size % 2 != 0:
                x2 += 1

            if x1 < 0:
                x1 = 0
            if x2 >= len(x):
                x2 = len(x) - 1

            #print it, x1, x2, len(x[x1:x2]), len(x)

            good_vals = n.where(n.isnan(x[x1 : x2]) == False)
            #print good_vals

            if len(good_vals[0]) > 0:
                #print x1, x2
                x_mean = n.nansum(x[x1 : x2][good_vals] * w[x1 : x2][good_vals] ) / n.nansum(w[x1 : x2][good_vals] )
                x_err = ( 1 / n.nansum( w[x1 : x2][good_vals] )  ) **.5

            else:
                x_mean = n.nan
                x_err = n.nan
            
            array_bin[i] = x_mean
            error_bin[i] = x_err

        return array_bin, error_bin


def read_npyfiles(paths):

    data = [None] * len(paths)
    for i, path in enumerate(paths):

        data[i] = n.load(path)

    data = n.array(data)
    return data

def B_lam_RJ(T,lams):

    #Rayleigh-Jeans Law approximation

    ### inputs ####
    #T - temperature [Kelvin]
    #lams - microns

    ### output ####
    ## B_lam ### [W/ m^2 / m]
    B_lam = 2 * c * kb * T / ((lams * microns2m)**4)

    return B_lam

def B_lam(T,lams):

    #Planck's Law

    ### inputs ####
    #T - temperature [Kelvin]
    #lams - microns

    ### output ####
    ## B_lam ### [W/ m^2 / m]

    B_lam_a = (2 * h * c**2) / (lams * microns2m) **5
    B_lam_b = n.exp( h * c / (lams * microns2m) / kb / T ) - 1

    B_lam = B_lam_a / B_lam_b

    return B_lam

    

def bin_array2(array, w, no_elems, dpix, sig_clip = None, extra_array = None):

    binned_array = n.zeros(no_elems)
    binned_array_err = n.zeros(no_elems)
    if extra_array is not None:
        binned_extra_array = n.zeros(no_elems)
        

    for i in range(no_elems):

        array_subset = array[i*dpix : i*dpix + dpix]
        w_subset = w[i*dpix : i*dpix + dpix]
        # print i
        # print w_subset
        # raw_input()

        if extra_array is not None:
            extra_subset = extra_array[i*dpix : i*dpix + dpix]
            #print extra_subset
            #raw_input()
        
            

        good = n.where( (n.isnan(w_subset) == False) & (n.isnan(array_subset) == False))

        #print len(good[0])

        # print i
        # print good
        # print w_subset[good]
        # raw_input()

        if len(good[0]) == 0:

            binned_array[i] = n.nan
            binned_array_err[i] = n.nan

            if extra_array is not None:
                binned_extra_array[i] = n.nan
                

        else:
            #print i
            #print array_subset.shape

            array_subset =  array_subset[good]
            w_subset =  w_subset[good]
            if extra_array is not None:
                extra_subset = extra_subset[good]
            
            if sig_clip is not None:
                array_subset = sigma_clip(array_subset, sigma = sig_clip)

                #print array_subset.shape
                if extra_array is not None:
                    extra_subset =  extra_subset[~array_subset.mask]

                w_subset = w_subset[~array_subset.mask]
                array_subset = array_subset[~array_subset.mask]
                #print array_subset.shape
                #raw_input()

                
            binned_array[i] = n.nansum(array_subset * w_subset) / n.nansum(w_subset)
            binned_array_err[i] = n.sqrt(1/ n.nansum(w_subset))
            if extra_array is not None:
                 binned_extra_array[i] = n.nansum(extra_subset * w_subset) / n.nansum(w_subset)


    if extra_array is not None:
        return  binned_array, binned_array_err, binned_extra_array

    else:
        return  binned_array, binned_array_err


def bin_array3(array, array_err, w, no_elems, dpix, sig_clip = None, extra_array = None):

    binned_array = n.zeros(no_elems)
    binned_array_err = n.zeros(no_elems)
    if extra_array is not None:
        binned_extra_array = n.zeros(no_elems)
        

    for i in range(no_elems):

        array_subset = array[i*dpix : i*dpix + dpix]
        array_err_subset = array_err[i*dpix : i*dpix + dpix]
        w_subset = w[i*dpix : i*dpix + dpix]
        # print i
        # print w_subset
        # raw_input()

        if extra_array is not None:
            extra_subset = extra_array[i*dpix : i*dpix + dpix]
            #print extra_subset
            #raw_input()
        
            

        good = n.where( (n.isnan(w_subset) == False) & (n.isnan(array_subset) == False))

        #print len(good[0])

        # print i
        # print good
        # print w_subset[good]
        # raw_input()

        if len(good[0]) == 0:

            binned_array[i] = n.nan
            binned_array_err[i] = n.nan

            if extra_array is not None:
                binned_extra_array[i] = n.nan
                

        else:
            #print i
            #print array_subset.shape

            array_subset =  array_subset[good]
            array_err_subset =  array_err_subset[good]
            w_subset =  w_subset[good]
            if extra_array is not None:
                extra_subset = extra_subset[good]
            
            if sig_clip is not None:
                array_subset = sigma_clip(array_subset, sigma = sig_clip)

                #print array_subset.shape
                if extra_array is not None:
                    extra_subset =  extra_subset[~array_subset.mask]

                w_subset = w_subset[~array_subset.mask]
                array_err_subset = array_err_subset[~array_subset.mask]
                array_subset = array_subset[~array_subset.mask]
                #print array_subset.shape
                #raw_input()

                
            binned_array[i] = n.sum(array_subset * w_subset) / n.sum(w_subset)
            binned_array_err[i] = n.sqrt( n.sum( (w_subset**2) * (array_err_subset**2) ) / n.sum( w_subset) **2 )
            if extra_array is not None:
                 binned_extra_array[i] = n.nansum(extra_subset * w_subset) / n.nansum(w_subset)


    if extra_array is not None:
        return  binned_array, binned_array_err, binned_extra_array

    else:
        return  binned_array, binned_array_err



def flux_in_filter(model_flux, model_lam, filter_data, sky_transmission_path = False):

    # October 26 2017

    #the units you put in will the units that come out bitch!!!
    
    ##### filter transmission curve ###################
    filter_trans, filter_lam = filter_data
    
    lam_min = filter_lam.min() 
    lam_max = filter_lam.max()

    #wavelengths containted in the filter
    model_filter = n.where( ( model_lam <= lam_max) & ( model_lam >= lam_min) )
    model_flux_filter = model_flux[model_filter]
    model_lam_filter =  model_lam[model_filter]

    dlam_model = n.median(n.diff(model_lam)) #median wavelenth spacing in  the model
    dlam_filter = n.median(n.diff(filter_lam)) #median wavelenth spacing in  the filter

    bins_filter = abs(int(dlam_model/dlam_filter))

   

    

    if bins_filter <=  1:

        #print('yeah')


        filter_trans_f = interp1d(filter_lam, filter_trans) #interpolation function of the transmission curve
        
        #interpolate the transmission curve onto the grid of that data
        filter_trans_interp_model = filter_trans_f(model_lam_filter)

        
    else:

        #print('yeah')
        #print(bins_filter)
    
        
        #if the resolution of the filter is higher than the data bin it down
        w = n.ones(len(filter_trans))
        elems_filter = int(len(filter_trans)/bins_filter) #total number of final elements
        #print(elems_filter)

        filter_trans_bin, dummy, filter_lam_bin = bin_array2(filter_trans, w, elems_filter, bins_filter, extra_array = filter_lam)
        filter_trans_f = interp1d(filter_lam_bin , filter_trans_bin, bounds_error = False, fill_value = n.nan) #interpolation function of the transmission

        

         # print bins_filter
        # plt.figure('filter')
        # plt.subplot(211)
        # plt.plot(filter_lam,filter_trans)
        # plt.plot(filter_lam_bin,filter_trans_bin)

        # plt.subplot(212)
        # plt.plot(model_lam_filter, model_flux_filter)
        # plt.show()
        # raw_input()
        filter_trans_interp_model = filter_trans_f(model_lam_filter)

        #print(filter_trans_interp_model)

    

    


    if sky_transmission_path == False:

        # plt.figure()
        # plt.plot(model_flux_filter)
        # plt.show()

        # print(n.sum(filter_trans_interp_model))
        # print(n.sum( model_flux_filter))
        # print(n.sum(filter_trans_interp_model))
        
        int_model_flux = n.nansum(filter_trans_interp_model * model_flux_filter ) / n.nansum(filter_trans_interp_model)
        return int_model_flux

    else:
        
        ###### sky transmisson ########
        #calling in sky transmission data
        sky_data = n.genfromtxt(sky_transmission_path)
        sky_lam = sky_data[:,0]
        sky_t = sky_data[:,1]

        #finding out how much you should bin down the
        #sky model to the data.
        dlam_sky = n.median(n.diff(sky_lam))

        #integer values
        #print dlam_model
        #print dlam_sky
        #raw_input()
        bins_model = int(dlam_model/dlam_sky) #number of bins to apply to sky model

        if bins_model < 1:
            #model resolution is better than sky model
            sky_t_f_model = interp1d(sky_lam, sky_t) #interpolation function of the transmission
            sky_t_model = sky_t_f_model(model_lam_filter)

        else:
            w = n.ones(len(sky_t))
            elems_model = int(len(sky_t)/bins_model) #total number of final elements

            sky_t_bin_model, dummy, sky_lam_bin_model = bin_array2(sky_t, w, elems_model, bins_model, extra_array = sky_lam)
            sky_t_f_model = interp1d(sky_lam_bin_model, sky_t_bin_model) #interpolation function of the transmission
            sky_t_model = sky_t_f_model(model_lam_filter) #interopolated sky within the model

        #compute the integrated flux
        int_model_flux = n.nansum(filter_trans_interp_model * sky_t_model * model_flux_filter ) / n.nansum(filter_trans_interp_model * sky_t_model)

        return int_model_flux




###########
def counts_in_filter(model_flux, model_lam, filter_data, sky_transmission_path = False):

    # November 30th, 2017
    # everything needs to be in microns
    # does not provide exact counts it is proportional to counts
    model_dlams = n.diff(model_lam)
    model_dlams = n.append(model_dlams, n.diff(model_lam))

    #the units you put in will the units that come out bitch!
    ##### filter transmission curve ###################
    filter_trans, filter_lam = filter_data
    
    lam_min = filter_lam.min() 
    lam_max = filter_lam.max()

    #wavelengths containted in the filter
    model_filter = n.where( ( model_lam <= lam_max) & ( model_lam >= lam_min) )
    model_flux_filter = model_flux[model_filter]
    model_lam_filter =  model_lam[model_filter]
    model_dlams_filter = model_dlams[model_filter]

    dlam_model = n.median(n.diff(model_lam)) #median wavelenth spacing in  the model
    dlam_filter = n.median(n.diff(filter_lam)) #median wavelenth spacing in  the filter

    bins_filter = abs(int(dlam_model/dlam_filter))

    

    if bins_filter <=  1:

        filter_trans_f = interp1d(filter_lam, filter_trans) #interpolation function of the transmission curve
        #interpolate the transmission curve onto the grid of that data
        filter_trans_interp_model = filter_trans_f(model_lam_filter)

        
    else:
        #if the resolution of the filter is higher than the data bin it down
        w = n.ones(len(filter_trans))
        elems_filter = len(filter_trans)/bins_filter #total number of final elements

        filter_trans_bin, dummy, filter_lam_bin = bin_array2(filter_trans, w, elems_filter, bins_filter, extra_array = filter_lam)
        filter_trans_f = interp1d(filter_lam_bin , filter_trans_bin, bounds_error = False, fill_value = n.nan) #interpolation function of the transmission

        # print bins_filter
        # plt.figure('filter')
        # plt.subplot(211)
        # plt.plot(filter_lam,filter_trans)
        # plt.plot(filter_lam_bin,filter_trans_bin)

        # plt.subplot(212)
        # plt.plot(model_lam_filter, model_flux_filter)
        # plt.show()
        # raw_input()
        filter_trans_interp_model = filter_trans_f(model_lam_filter)



    if sky_transmission_path == False:
        #compute the integrated counts
        model_counts = n.nansum(filter_trans_interp_model * model_flux_filter * model_lam_filter * model_dlams_filter)
        return model_counts

    else:
        
        ###### sky transmisson ########
        #calling in sky transmission data
        sky_data = n.genfromtxt(sky_transmission_path)
        sky_lam = sky_data[:,0]
        sky_t = sky_data[:,1]

        #finding out how much you should bin down the
        #sky model to the data.
        dlam_sky = n.median(n.diff(sky_lam))

        #integer values
        #print dlam_model
        #print dlam_sky
        #raw_input()
        bins_model = int(dlam_model/dlam_sky) #number of bins to apply to sky model

        if bins_model < 1:
            #model resolution is better than sky model
            sky_t_f_model = interp1d(sky_lam, sky_t) #interpolation function of the transmission
            sky_t_model = sky_t_f_model(model_lam_filter)

        else:
            w = n.ones(len(sky_t))
            elems_model = len(sky_t)/bins_model #total number of final elements

            sky_t_bin_model, dummy, sky_lam_bin_model = bin_array2(sky_t, w, elems_model, bins_model, extra_array = sky_lam)
            sky_t_f_model = interp1d(sky_lam_bin_model, sky_t_bin_model) #interpolation function of the transmission
            sky_t_model = sky_t_f_model(model_lam_filter) #interopolated sky within the model

        #compute the integrated counts

        # plt.figure('filter')
        # plt.subplot(211)
        # plt.plot(filter_lam,filter_trans)
        # plt.plot(filter_lam_bin,filter_trans_bin)

        # plt.subplot(212)
        # plt.plot(model_lam_filter, model_flux_filter)
        # plt.show()
        # raw_input()
        model_counts = n.nansum(filter_trans_interp_model * sky_t_model * model_flux_filter * model_lam_filter * model_dlams_filter)

        return model_counts

######################

def interpolate_data(x,y,y_err, x_new):

    y_new = n.zeros(len(x_new))
    y_err_new = n.zeros(len(x_new))

    for i, val in enumerate(x_new) :

        dx = abs(val - x)
        order = n.argsort(dx)
        dx = dx[order]
        x_reordered = x[order]
        y_reordered = y[order]
        y_err_reordered = y_err[order]

        if x_reordered[0] > x_reordered[1]:

            x1 = x_reordered[0]
            x0 = x_reordered[1]

            y1 = y_reordered[0]
            y0 = y_reordered[1]

            y_err1 = y_err_reordered[0]
            y_err0 = y_err_reordered[1]

        if x_reordered[1] > x_reordered[0]:
            
            x1 = x_reordered[1]
            x0 = x_reordered[0]

            y1 = y_reordered[1]
            y0 = y_reordered[0]

            y_err1 = y_err_reordered[1]
            y_err0 = y_err_reordered[0]


        if val > x.max():
            #outside of the array
            y_new[i] = n.nan
            y_err_new[i] = n.nan
            continue
        
        if val < x.min():
            #outside of the array
            y_new[i] = n.nan
            y_err_new[i] = n.nan
            continue

        else:
            m = (y1 - y0) / (x1 - x0)
            
            y_new[i] = m * (val - x0) + y0

            A = (val - x0) / (x1 - x0)
            B = 1 - (val - x0) / (x1 - x0)

            y_err_new[i] = n.sqrt( ((A**2) * (y_err1 ** 2)) + ((B**2) * (y_err0 **2 )) )
        


    return x_new, y_new, y_err_new
