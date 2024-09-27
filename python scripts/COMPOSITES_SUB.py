#!/usr/bin/env duncenv
# coding: utf-8

import time
import numpy as np
from numpy.random import seed
import pandas as pd
import xarray as xr
import multiprocessing as mp

import HS_FUNCTIONS as mf  # ensure this file is in your current working directory


path = '/specify path to your folder...'


####################################################
print('define long and short hotspells for all regions')
####################################################

# import excel table with hot spell durations/dates
# same as output from python script HOTSPELLS.py
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
print('create 100 subsampled long-spell composites')
####################################################

def init_pool_processes():
    seed()

def composites_subsampled_100(var):
    lss = hsf.subsample_long_spells_emp(long_spell_list, cluster_index, deterministic=True, shuffle=False)
    comp = hsf.cluster_composite(hsf.spell_means(hsf.extract_daily_spell_data(var, lss)))
    return lss, comp

# Required by Windows:
if __name__ == '__main__':

    pool = mp.Pool(processes=100, initializer=init_pool_processes)

    comps100 = pool.map(composites_subsampled_100, [var] * 100)

    pool.close()
    pool.join()

print("Medoid index:", hsf.medoid100(comps100))
medoid = comps100[hsf.medoid100(comps100)][1].rename('ind' + str(hsf.medoid100(comps100)))
medoid.to_netcdf(path + '/cl' + str(cluster_index+1) + '_' + varlab + '_long_comp_medoid.nc')
comps100[hsf.medoid100(comps100)][0].to_csv(path + '/cl' + str(cluster_index+1) + '_' + varlab + '_medoid_dates.txt', sep='\t', index=False, header=False)


####################################################
print('calculate significance for each 100 composite')
####################################################

start_time = time.time()

pMat_adj_N = []
for m in range(1):
    print(m+1)
    
    # Create N sets of simulated dates that are then subsampled and composited 
    simdates = hsf.get_N_sim_dates(long_spell_list[cluster_index])
    def sim_comps_1000(sim):
        simcomp = hsf.cluster_composite(
            hsf.spell_means(
                hsf.extract_daily_spell_data(var,
                                         hsf.subsample_long_spells_emp(
                                             [pd.DataFrame([[np.nan] * len(simdates.iloc[sim]), simdates.iloc[sim]]).T]*6,
                                             cluster_index, deterministic=True, shuffle=False)
                                        )
            )
        )
        return simcomp

    pool = mp.Pool(100)
    simcomps1000 = pool.map(sim_comps_1000, range(numMC))  
    pool.close()
    pool.join()

    # Calculate the rank and then adjusted p-value for the mth empirical field
    x = comps100[m][1]
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
    pMat_adj = mf.apply_fdr(pMat, significance=.05, alpha=.1)
    
    pMat_adj_N.append(pMat_adj)

end_time = time.time()
elapsed_time = (end_time - start_time) / 60
print("Elapsed time:", elapsed_time, "minutes")

(xr.concat(pMat_adj_N, dim='i')).to_netcdf(path + '/cl' + str(cluster_index+1) + '_' + varlab + '_long_sign_all-i.nc')

