#!/usr/bin/env duncenv
# coding: utf-8

import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
from datetime import timedelta
from statsmodels.stats.multitest import fdrcorrection
import multiprocessing as mp

import HS_FUNCTIONS as mf  # ensure this file is in your current working directory


path = '/specify path to your folder...'


####################################################
print('define long and short hotspells for all regions')
####################################################

# import excel table with hot spell durations/dates
# (same as final output from python script 02_HOTSPELLS.py)
hotspells = pd.read_excel(path + 'HS_dates.xlsx')

long_spell_list = []
for i in list(range(0, 12, 2)):
    cl = hotspells.loc[hotspells.iloc[:, i].between(12, 26), hotspells.columns[i:i+2]]
    long_spell_list.append(cl)
    
short_spell_list = []
for i in list(range(0, 12, 2)):
    cl = hotspells.loc[hotspells.iloc[:, i].between(4, 5), hotspells.columns[i:i+2]]
    short_spell_list.append(cl)

    
####################################################
print('define variable and parameters')
####################################################

var = xr.open_dataset(pth + 'filename.nc')['varname'] # import daily anomalies as xarray (time,lon,lat)
cluster_index = 1 # specify which of the 6 clusters in 'hotspells' you want to work with
varlab = var.name # string containing variable name, only important for file naming
numMC = 1000 # number of Monte Carlo simulated hotspell sets for the significance test


####################################################
print('compute and save long/short spell composite')
####################################################

comp_l = hsf.cluster_composite(hsf.spell_means(hsf.extract_daily_spell_data(var, long_spell_list[cluster_index])))
comp_l.to_netcdf(path + '/cl' + str(cluster_index+1) + '_' + varlab + '_long_comp.nc')
comp_s = hsf.cluster_composite(hsf.spell_means(hsf.extract_daily_spell_data(var, short_spell_list[cluster_index])))
comp_s.to_netcdf(path + '/cl' + str(cluster_index+1) + '_' + varlab + '_short_comp.nc')


####################################################
print('calculate significance for each spell-type composite and save mask')
####################################################
  
simdates_l = hsf.get_N_sim_dates(long_spell_list[cluster_index])
simdates_s = hsf.get_N_sim_dates(short_spell_list[cluster_index])

for simdates, comp, spelltype in zip([simdates_l,simdates_s],[comp_l,comp_s],['long','short']):
    def sim_comps_1000(sim):
        simcomp = hsf.cluster_composite(
            hsf.spell_means(
                hsf.extract_daily_spell_data(var, pd.DataFrame([[np.nan] * len(simdates.iloc[sim]), simdates.iloc[sim]]).T)
            )
        )
        return simcomp

    pool = mp.Pool(100) # creates a pool of 100 processes that can run tasks in parallel
    # but you should consider the capabilities of your hardware when deciding on the number of processes to use
    simcomps1000 = pool.map(sim_comps_1000, range(numMC))  
    pool.close()
    pool.join()

    # Calculate the rank and then adjusted p-value for the mth empirical field
    x = comp
    x_flat = x.values.flatten()[:, np.newaxis]
    a = np.stack(simcomps1000, axis=0)
    prod = x.shape[0] * x.shape[1]
    a_flat = a.reshape((len(a), prod)).T
    a_flat_sorted = np.sort(a_flat, axis=1)

    rankMat = []
    for i in range(len(x_flat)):
        rank_val = np.searchsorted(a_flat_sorted[i], x_flat[i]) + 1
        rankMat.append(int(rank_val))
    rankMat = xr.DataArray(np.array(rankMat).reshape(x.shape[0],x.shape[1]), coords=x.coords)
    pMat = (2*((numMC/2)-abs(rankMat-((numMC/2)+1)))/(numMC+1))
    pMat_adj = hsf.apply_fdr(pMat, significance=.05, alpha=.1)
    
    pMat_adj.to_netcdf(path + '/cl' + str(cluster_index + 1) + '_' + varlab + '_' + spelltype + '_sign.nc')
    print(spelltype + ' significance 2D mask saved')

