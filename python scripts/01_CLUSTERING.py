#!/usr/bin/env python
# coding: utf-8

import time
import warnings
import pandas as pd
import xarray as xr
import numpy as np
import scipy as scipy
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

path = '/specify path to your folder...'

# suppress the specific RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")


### read NetCDF file
# should be a time,lat,lon file with (standardised) temperature anomalies
da = xr.open_dataset(path + 'filename.nc')['variable_name'] # adapt to suit your case


##### CLUSTERING FUNCTION ######

def clustering(t_agg, trim, x_thresh, d_trunc, dist_metric, recluster=True):
    
    ### non-overlapping temporal aggregation (averaging)
    da_agg_y = [da.sel(time=da.time.dt.year.isin([y]))[trim[0]:trim[1],:,:].resample(time=t_agg).mean() for y in period]
    da_agg = xr.concat(da_agg_y, dim='time')

    ### binarise dataarray
    da_bin = (da_agg > da_agg.quantile(x_thresh, dim='time')).astype(int)
    df_bin = da_bin.drop_vars('quantile').to_dataframe('t2m').unstack('time')
    df_bin_clean = df_bin.dropna(axis=0)

    ### clustering (step 1)
    cl = linkage(df_bin_clean, method='average', metric=dist_metric)
    fcl = pd.DataFrame(fcluster(cl, t=.8, criterion='distance')) # set truncation to detect non-european regions, can also be changed
    fcl.columns = ['clusters']
    n = max(fcl1.clusters) # to get max number of clusters
    da_cl = pd.DataFrame(pd.concat([df_bin_clean, fcl.set_index(df_bin.index)], axis=1)['clusters']).to_xarray().clusters
    da_cl = da_cl.where(da_cl.values > 1, np.nan) - 1
    # dendrogram(cl, p=6, truncate_mode='level').plot() # to visualise top of dendrogram
    
    if recluster == True:
        ### remove 'periferic' regions (off europe)
        clusters_to_remove = []
        slices = [
            {"lon": slice(-25,-12), "lat": slice(60,71.75)},
            {"lon": slice(50, 53), "lat": slice(70, 71.75)},
            {"lon": slice(34, 53), "lat": slice(32.25, 44.25)},
            {"lon": slice(-25, 34), "lat": slice(32.25, 36.25)},
        ]
        da_rm = [da_cl.sel(**sl) for sl in slices]
        for r in da_rm:
            for i, n in enumerate(np.unique(r)[:-1]):
                if np.sum(r == n).item() >= np.sum(da_cl - xr.align(da_cl, r,
                                    join="left", fill_value=0)[1] == n).item():
                    clusters_to_remove.append(n)

        da_clrm = da_cl.where(~da_cl.isin(clusters_to_remove), np.nan)
        da_clrm = (da_clrm/da_clrm).expand_dims(time=np.arange(len(da_agg)))
        # new binary field with removed clusters
        da_binrm = (da_bin * da_clrm.assign_coords({'time': da_agg.time})).rename('t2m')
        df_binrm = da_binrm.drop_vars('quantile').to_dataframe('t2m').unstack('time')
        df_binrm_clean = df_binrm.dropna(axis=0)
        # dataarray with removed regions
        da_cl_ = xr.merge([da_cl.where(da_cl == i) for i in clusters_to_remove]).clusters

        ### clustering (step 2)
        cl = linkage(df_binrm_clean, method='average', metric='jaccard')
        fcl = pd.DataFrame(fcluster(cl, t=d_trunc, criterion='distance'))
        fcl.columns = ['clusters']
        n = max(fcl.clusters) # to get max number of clusters

        da_cl = pd.DataFrame(pd.concat([df_binrm, fcl.set_index(df_binrm_clean.index)], axis=1)['clusters']).to_xarray().clusters
        # can change numbering of clusters by preference, e.g. below:
        # da_cl = (n + 1) - da_cl
    
    return da_cl, n, da_cl_


### define parameters
period = range(1959,2023) # define period in years
t_agg, trim = ['21D', [24,-24]] # ['14D', [27,-28]] or ['7D', [27,-28]]
# 'trim' is for the removal of dates at edge of MJJAS season, for non-overlapping temporal aggregation to work / this part will be improved
x_thresh = .95 # .95, .90, .85 or any other percentile threshold
d_trunc = .875 # any desired level of event co-occurrence

### run clustering function
cl_output = clustering(t_agg, trim, x_thresh, d_trunc, 'jaccard', recluster=True)

### write 2D xarray of cluster regions to netCDF file
cl_output[0].to_netcdf(path + '/clusters_n' + str(cl_output[1]) + '_' + t_agg + '_p' + str(x_thresh*100) + '_d' + str(d_trunc)[2:] + '.nc')

