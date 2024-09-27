#!/usr/bin/env duncenv
# coding: utf-8

import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
from datetime import timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from statsmodels.stats.multitest import fdrcorrection


long_spell_list = []
for i in list(range(0, 12, 2)):
    cl = hotspells.loc[hotspells.iloc[:, i].between(12, 26), hotspells.columns[i:i+2]]
    long_spell_list.append(cl)
    
short_spell_list = []
for i in list(range(0, 12, 2)):
    cl = hotspells.loc[hotspells.iloc[:, i].between(4, 5), hotspells.columns[i:i+2]]
    short_spell_list.append(cl)


def expand_date_range(date_range):
    start, end = date_range.split('/')
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    date_range = pd.date_range(start=start_date, end=end_date)
    return date_range


def remove_first_last_two_dates(datetime_index):
    return datetime_index[2:-2]


def subsample_long_spells_emp(long_spell_list, cluster_index, deterministic=True, shuffle=True):
    long_spell_list_subsampled = []

    long_spells_days = long_spell_list[cluster_index].iloc[:,1].apply(expand_date_range).reset_index(drop=True)
    if deterministic and not shuffle:
        weights = np.ones(len(long_spell_list[cluster_index]))/len(long_spell_list[cluster_index])
        indices = np.random.choice(len(long_spell_list[cluster_index]), size=len(short_spell_list[cluster_index]), p=weights)
        selected_sets = np.array(long_spells_days)[indices]
        centre_dates = np.array([np.random.choice(datetime_set).astype('datetime64[D]') for datetime_set in selected_sets])
    elif not deterministic and not shuffle:
        all_dates = np.concatenate(long_spells_days).astype('datetime64[D]')
        centre_dates = np.random.choice(all_dates, len(short_spell_list[cluster_index]))
    elif not deterministic and shuffle:
        all_dates = np.concatenate(long_spells_days).astype('datetime64[D]')
        np.random.shuffle(all_dates)
        centre_dates = np.random.choice(all_dates, len(short_spell_list[cluster_index]))

    num_4day_spells = len(short_spell_list[cluster_index][short_spell_list[cluster_index].iloc[:,0] == 4])
    num_5day_spells = len(short_spell_list[cluster_index][short_spell_list[cluster_index].iloc[:,0] == 5])
    long_5days_expanded = []
    for date in centre_dates:
        if np.random.random() < (num_4day_spells / len(short_spell_list[cluster_index])):
            spell = pd.date_range(date-2,date+1)
        else:
            spell = pd.date_range(date-2,date+2)
        long_5days_expanded.append(spell)
    long_5days_expanded = pd.Series(long_5days_expanded)
    long_5days_ranges = long_5days_expanded.apply(lambda datetime_index: 
                        f"{datetime_index[0].strftime('%Y-%m-%d')}/{datetime_index[-1].strftime('%Y-%m-%d')}")
    long_spell_list_subsampled = pd.DataFrame([[len(spell) for spell in long_5days_expanded], long_5days_ranges]).T
    return long_spell_list_subsampled


def extract_daily_spell_data(dataset, spell_list):
    subsets = []
    for i in range(0, len(spell_list)):
        subset = dataset.sel(time=slice(pd.to_datetime(spell_list.iloc[:, 1].str.split('/').str[0], format='%Y-%m-%d').iloc[i],
                        pd.to_datetime(spell_list.iloc[:, 1].str.split('/').str[1], format='%Y-%m-%d').iloc[i]+timedelta(1)))
        subsets.append(subset)
    return subsets


def spell_means(subsets):
    spell_means = [df.mean(dim='time') for df in subsets]
    return spell_means


def cluster_composite(spell_means):
    composite = xr.concat(spell_means, dim='time').mean(dim='time')
    return composite


def medoid100(composites):
    data = []
    for ds in composites:
        flattened = ds[1].values.flatten()
        # Remove NaN values before appending
        flattened = flattened[~np.isnan(flattened)]
        data.append(flattened)
    n_clusters = 1
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=0)
    kmedoids.fit(data)
    medoid_index = kmedoids.medoid_indices_[0]
    return medoid_index


def get_N_sim_dates(spell_list_cl):
    
    doyVec = pd.to_datetime(spell_list_cl.iloc[:, 1].str.split('/').str[0], format='%Y-%m-%d').dt.dayofyear # doy for start of spell
    pdf_doy = sns.kdeplot(doyVec, color='black', alpha=0.6, bw_adjust=.3, cut=1.5, linewidth=3); plt.close()
    doyVec_pdf = pd.DataFrame(pdf_doy.lines[0].get_xydata(), columns=['doyVec','p'])
    doyVec_pdf['doyVec'] = round(doyVec_pdf['doyVec'], 0)
    doyVec_pdf = doyVec_pdf.groupby('doyVec').mean()
    durations = spell_list_cl.iloc[:, 0]
    pdf_duration = sns.kdeplot(durations, color='red', alpha=0.6, bw_adjust=.3, cut=0, linewidth=3); plt.close()
    durations_pdf = pd.DataFrame(pdf_duration.lines[0].get_xydata(), columns=['duration','p'])
    durations_pdf['duration'] = round(durations_pdf['duration'], 0)
    durations_pdf = durations_pdf.groupby('duration').mean()

    sim_doyVec = []
    numMC = 1000
    for n in range(numMC):
        samples = []
        for i in range(len(doyVec)):
            random_doyVec = np.random.choice(doyVec_pdf.index, p=(doyVec_pdf['p'] + (( 1 - doyVec_pdf['p'].sum()) / len(doyVec_pdf['p']))) )
            samples.append(random_doyVec)
        sim_doyVec.append(samples)
    sim_doyVec = pd.DataFrame(sim_doyVec)
    years = pd.DataFrame(np.random.randint(1959,2023,size=(numMC, len(doyVec))))
    sim_dates = []
    for r in range(numMC):
        for c in range(len(doyVec)):
            sim_dates.append(dt.datetime.strptime(str(years[c][r]) + ' ' + str(int(sim_doyVec[c][r])), '%Y %j').strftime('%Y-%m-%d'))
    sim_dates = pd.DataFrame(np.array(sim_dates).reshape(numMC,len(doyVec)))
    end_dates = sim_dates.applymap(lambda x: pd.to_datetime(x) + pd.Timedelta(days=np.random.choice(durations_pdf.index,
                        p=(durations_pdf['p'] + (( 1 - durations_pdf['p'].sum()) / len(durations_pdf['p']))))) )
    end_dates = end_dates.applymap(lambda x: x.strftime('%Y-%m-%d')).astype('object')
    sim_date_ranges = sim_dates.applymap(str) + '/' + end_dates.applymap(str)
    
    return sim_date_ranges


def apply_fdr(pMat, significance, alpha):
    
    p2D = pMat.to_dataframe('p-value')
    p2D_sorted = p2D.sort_values(p2D.columns[0], ascending = True)
    fdr = fdrcorrection(p2D_sorted['p-value'], alpha=alpha, method='indep', is_sorted=True)
    p2D_sorted['p-adjusted'] = fdr[1]
    pMat_adj = p2D_sorted['p-adjusted'].to_xarray()
    pMat_adj.values[pMat_adj >= significance] = np.nan
    pMat_adj.values[pMat_adj < significance] = 1
    
    return pMat_adj
