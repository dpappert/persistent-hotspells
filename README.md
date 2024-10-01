# Persistent hot spells

This repository contains code relating to the study in the (preprint) journal article in Weather and Climate Dynamics (WCD, Copernicus Publications), titled '[Long vs. Short: Understanding the dynamics of persistent summer hot spells in Europe](https://doi.org/10.5194/egusphere-2024-2980)'. Additional information can be found in the [Supplement](https://egusphere.copernicus.org/preprints/2024/egusphere-2024-2980/egusphere-2024-2980-supplement.pdf).

It contains the Python code to perform the following:

* regional clustering based on temperature anomalies
* compute hot spells for different regions
* compute long/short spell composites and calculate statistical significance of the anomaly fields

## Referencing
If you use this code in your publication, please cite: Pappert, D., Tuel, A., Coumou, D., Vrac, M., and Martius, O.: Long vs. Short: Understanding the dynamics of persistent summer hot spells in Europe, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2024-2980, 2024.

Please report any issues on the GitHub portal.
 
 
#### *NB: THIS PAGE IS BEING REGULARLY IMPROVED/UPDATED UNTIL PUBLICATION* <br />


## Supporting information about python scripts:

### 01_CLUSTERING.py

**Input:** daily gridded dataset of standardised temperature anomalies (with the land-sea mask already applied for a regionalisation over land).
Path and file name have to specified.

**Output:** a 2D (lon,lat) NetCDF file with numbered clusters/regions.

### 02_HOTSPELLS.py

**Input:** the same gridded temperature dataset used for the clustering & the 2D NetCDF file with the numbered clusters. Path and file name have to specified.

**Output:** a table containing the a) durations and b) date ranges of detected hotspells for each region in the cluster xarray.
