#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt

path = '/specify path to your folder...'

### read relevant files
t2m = xr.open_dataset(path + 'filename.nc')['variable_name'] # (standardised) temperature anomalies
clust = xr.open_dataset(path + 'clusters_filename.nc')['clusters'] # 2D field of regions (final output of CLUSTERING.py)
n_clust = np.max(np.unique(clust)[:-1]).astype(int) # get number of regions

def cluster_data(da, cl, fldmean=True):
    # get data for each cluster/region
    # compute field means for each cluster/region (if fldmean==True)
    # da and clust have to match (lon, lat, and spatial resolution)
    cls_da = []
    for i in range(1,n_clust+1):
        x = (da.where(clust.values == i)).rename('cl' + str(i))
        if fldmean == True:
            x = x.mean(dim=['lon','lat'])
        cls_da.append(x)
    return cls_da

def get_exceedances(cls_da, th, method='fm_stat', pra=None):
    # Check that method is specified and valid
    if method not in ['fm_stat', 'fm', 'frac']:
        raise ValueError("method must be one of 'fm_stat', 'fm', or 'frac'")

    # Check that pra is defined if method is 'frac'
    if method == 'frac' and F is None:
        raise ValueError("F must be specified if method is 'frac'")
    
    # Function logic based on the method
    if method == 'fm_stat':
        result = f"Processing with method fm_stat and th={th}"
    elif method == 'fm':
        result = f"Processing with method fm and th={th}"
    elif method == 'frac':
        result = f"Processing with method frac, th={th}, and F={F}"
    
    # Create binarised series for each cluster based on threshold exceedances
    cls_bin = []
    for ds in cls_da:
        if method == 'fm_stat':
            if isinstance(th, float):
                if 0 < th < 1:
                    threshold = ds.quantile(th)
                else:
                    raise ValueError(f"Quantile must be between 0 and 1, but got {th}.")
            elif isinstance(th, str):
                if th == '1':
                    threshold = ds.std()
                elif th == '1.5':
                    threshold = 1.5 * ds.std()
                else:
                    raise ValueError(f"Invalid string value for threshold: {th}. Must be '1' or '1.5'.")
            else:
                raise TypeError(f"Threshold must be of type int or str, but got {type(th)}.")
            
            above = (ds > threshold).astype(int)
        
        if method == 'fm':
            above = (ds > float(th)).astype(int)
            
        if method == 'frac':
            tot_g = int(np.sum(ds[0,:,:].notnull()).values)
            size  = np.array([np.sum((ds[t,:,:] > float(th)).values.astype(int)) for t in range(len(ds.time))])
            exc_g = size / tot_g
            
            ds_new = xr.DataArray(exc_g, coords=ds.time.coords, dims=ds.dims[0], name=ds.name)
            
            above = (ds_new > F).astype(int)
        
        cls_bin.append(above)
        
    return cls_bin

def get_hotspells(cls_bin, relax):
    cls_durations = []
    cls_event_dates = []
    cls_tot_events = []
    
    # Create hot spells with or without merging nearby exceedances
    for above in cls_bin:
        if relax == 0:
            pass
        if relax == 1:
            for i in range(1, len(above)-1):
                if above[i-1] == 1 and above[i+1] == 1 and above[i] == 0:
                    above[i] = 1
        if relax == 2:
            for i in range(1, len(above)-1):
                if above[i-1] == 1 and above[i+1] == 1 and above[i] == 0:
                    above[i] = 1
                if above[i-2] == 1 and above[i-1] == 0 and above[i] == 0 and above[i+1] == 1:
                    above[i] = 1
                    above[i-1] = 1

        if above[0] == 1:
            j = 1
            while j < len(above) and above[j] == 1:
                above[j] = 0
                j += 1
            above[0] = 0

        event_starts = above.diff(dim='time', n=1, label='upper') == 1
        event_ends = above.diff(dim='time', n=1, label='lower') == -1

        event_dates = [(start, end) for start, end in zip(event_starts.where(event_starts == True, drop=True).time.values,
                                                         event_ends.where(event_ends == True, drop=True).time.values)]
        # code lines to filter out errors, i.e. year-crossing events May-September (given specific ds.time format)
        event_dates_filtered = [(t0, t1) for t0, t1 in event_dates if not (
                    (dt.datetime.utcfromtimestamp(t0.astype(int)//1e9).month == 5 and dt.datetime.utcfromtimestamp(t1.astype(int)//1e9).month == 5) or
                    (dt.datetime.utcfromtimestamp(t0.astype(int)//1e9).month == 9 and dt.datetime.utcfromtimestamp(t1.astype(int)//1e9).month == 9) or
                    (dt.datetime.utcfromtimestamp(t0.astype(int)//1e9).month == 9 and dt.datetime.utcfromtimestamp(t1.astype(int)//1e9).month == 5) or
                    (dt.datetime.utcfromtimestamp(t0.astype(int)//1e9).month == 8 and dt.datetime.utcfromtimestamp(t1.astype(int)//1e9).month == 5) or
                    (dt.datetime.utcfromtimestamp(t0.astype(int)//1e9).month == 9 and dt.datetime.utcfromtimestamp(t1.astype(int)//1e9).month == 6)
                    )]
        formatted_dates = [f"{str(start_date)[:10]}/{str(end_date)[:10]}" for start_date, end_date in event_dates_filtered]
        cls_event_dates.append(formatted_dates)

        durations = [int((t1 - t0).astype('timedelta64[D]').astype(int) + 1) for t0, t1 in event_dates_filtered]
        cls_durations.append(durations)

        num_events = len(durations)
        cls_tot_events.append(num_events)
            
    return cls_durations, cls_event_dates, cls_tot_events


### compute hot spells
# comment/uncomment depending on which method you use
# change parameters in function according to your case

clsm_da = cluster_data(t2m, clust, fldmean=True)
# cls_da  = cluster_data(t2m, clust, fldmean=False)

BIN = get_exceedances(clsm_da, th='1', method='fm_stat')
HS = get_hotspells(BIN, relax=2)
print([sum(1 for x in HS[0][i] if 4 <= x <= 5) for i in range(n_clust)])
print([sum(1 for x in HS[0][i] if 12 <= x <= 26) for i in range(n_clust)])

# BIN = get_exceedances(clsm_da, th='1', method='fm')
# HS = get_hotspells(BIN, relax=2)
# print([sum(1 for x in HS[0][i] if 4 <= x <= 5) for i in range(n_clust)])
# print([sum(1 for x in HS[0][i] if 12 <= x <= 26) for i in range(n_clust)])

# BIN = get_exceedances(cls_da, th='1', method='frac', F=.66)
# HS = get_hotspells(BIN, relax=2)
# print([sum(1 for x in HS[0][i] if 4 <= x <= 5) for i in range(n_clust)])
# print([sum(1 for x in HS[0][i] if 12 <= x <= 26) for i in range(n_clust)])

# BIN = get_exceedances(cls_da, th='1', method='frac', F=.33)
# HS = get_hotspells(BIN, relax=2)
# print([sum(1 for x in HS[0][i] if 4 <= x <= 5) for i in range(n_clust)])
# print([sum(1 for x in HS[0][i] if 12 <= x <= 26) for i in range(n_clust)])


### store duration and date information as table
max_len = max(len(dat) for dat in HS[1])
dur_padded = [np.pad(dur, (0, max_len - len(dur)), 'constant', constant_values=-999) for dur in HS[0]]
dat_padded = [np.pad(dat, (0, max_len - len(dat)), 'constant', constant_values=-999) for dat in HS[1]]
data_dict = {f"duration{i+1}": dur_padded[i] for i in range(len(dur_padded))}
data_dict.update({f"dates{i+1}": dat_padded[i] for i in range(len(dat_padded))})
table = pd.DataFrame(data_dict)
order = [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11] # too manual, will need to be improved
table = pd.DataFrame({table.columns[i]: table.iloc[:, order[i]] for i in range(len(table.columns))})
columns = []
for i in order:
    columns.append(table.columns[i])
table.columns = columns
table.index = table.index+1
table.to_excel(path + 'HS_dates.xlsx', header=True)
