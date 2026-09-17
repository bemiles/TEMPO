import numpy as n
import matplotlib.pyplot as plt
import xlrd


#NOMIC data load sheet
book = xlrd.open_workbook_xls('/Users/zen/data/instrument_filters/nomic/JDSU Wideband Filter Curves.xls')
sh = book.sheet_by_index(0)

sheet_name = sh.name
sheet_rows = sh.nrows
sheet_cols = sh.ncols

t_filter_f8_699 = n.zeros(sheet_rows)
wave_filter_f8_699 = n.zeros(sheet_rows)

t_filter_f10_145 = n.zeros(sheet_rows)
wave_filter_f10_145 = n.zeros(sheet_rows)

t_filter_f10_228 = n.zeros(sheet_rows)
wave_filter_f10_228 = n.zeros(sheet_rows)

t_filter_f10_55 = n.zeros(sheet_rows)
wave_filter_f10_55 = n.zeros(sheet_rows)
 
for i in n.arange(2,sheet_rows):
    ###########################################
    cell_val = sh.cell_value(rowx=i, colx=12)
    if cell_val == '':
        cell_val = n.nan
        
    cell_val2 = sh.cell_value(rowx=i, colx=13)
    if cell_val2 == '':
        cell_val2 = n.nan
        
    wave_filter_f8_699[i] = float(cell_val)
    t_filter_f8_699[i] = float(cell_val2)
    
    
    ###########################################
    cell_val = sh.cell_value(rowx=i, colx=15)
    if cell_val == '':
        cell_val = n.nan
        
    cell_val2 = sh.cell_value(rowx=i, colx=16)
    if cell_val2 == '':
        cell_val2 = n.nan
    
    wave_filter_f10_145[i] = float(cell_val)
    t_filter_f10_145[i] = float(cell_val2)
    
    
    ###########################################
    cell_val = sh.cell_value(rowx=i, colx=18)
    if cell_val == '':
        cell_val = n.nan
        
    cell_val2 = sh.cell_value(rowx=i, colx=19)
    if cell_val2 == '':
        cell_val2 = n.nan
    
    wave_filter_f10_228[i] = float(cell_val)
    t_filter_f10_228[i] = float(cell_val2)
    
    
    ###########################################
    cell_val = sh.cell_value(rowx=i, colx= 21)
    if cell_val == '':
        cell_val = n.nan
        
    cell_val2 = sh.cell_value(rowx=i, colx=22)
    if cell_val2 == '':
        cell_val2 = n.nan
    
    wave_filter_f10_55[i] = float(cell_val)
    t_filter_f10_55[i] = float(cell_val2)
    
    #print(i, sh.cell_value(rowx=i, colx=21))
    
    
def nomic_f10_145_filter():
    
    return t_filter_f10_145, wave_filter_f10_145


#SKY DATA
def mir_sky_emission():

    mir_sky_data = n.genfromtxt('/Users/zen/data/sky_data/maunakea/mk_skybg_nq_16_15_ph.dat')

    mir_sky_lam = mir_sky_data[:,0]*10. #angstroms
    mir_sky_flux = mir_sky_data[:,1] #phot/s/nm/arcsec^2/m^2

    return mir_sky_flux, mir_sky_lam