# Persistent hot spells

This repository contains code relating to the study in the journal article in Weather and Climate Dynamics (WCD, Copernicus Publications), titled '[Long vs. short: understanding the dynamics of persistent summer hot spells in Europe](https://wcd.copernicus.org/articles/6/769/2025/)'. Additional information can be found in the [Supplement](https://wcd.copernicus.org/articles/6/769/2025/wcd-6-769-2025-supplement.pdf).

It contains the Python code to perform the following:

* regional clustering based on temperature anomalies
* compute hot spells for different regions
* compute long/short spell composites and calculate statistical significance of the anomaly fields

## Referencing
If you use this code in your publication, please cite the corresponding article: Pappert, D., Tuel, A., Coumou, D., Vrac, M., and Martius, O.: Long vs. short: understanding the dynamics of persistent summer hot spells in Europe, Weather Clim. Dynam., 6, 769–788, https://doi.org/10.5194/wcd-6-769-2025, 2025.

Please report any issues on the GitHub portal.


## Supporting information about python scripts:

### 01_CLUSTERING.py

**Input:** daily gridded dataset of standardised temperature anomalies (with the land-sea mask already applied for a regionalisation over land).
Path and file name have to specified.

**Output:** a 2D (lon,lat) NetCDF file with numbered clusters/regions.

### 02_HOTSPELLS.py

**Input:** the same gridded temperature dataset used for the clustering & the 2D NetCDF file with the numbered clusters. Path and file name have to specified.

**Output:** a table containing the a) durations and b) date ranges of detected hot spells for each region in the cluster xarray.

### 03_COMPOSITES.py

**Input:** i) the output of 02_HOTSPELLS.py, i.e. the .csv table of hot spell events for each cluster (duration, date ranges); ii) a gridded NetCDF file containing anomalies of a specific variable (time, lon, lat).

**Output:** for a specified region, the script calculates the full spell anomaly composites of long and short hot spells for the imported variable AND the corresponding mask of grid cells that are statistically significant at the specified confidence level (2D: lon,lat).

### 03_COMPOSITES_SUB.py

**Input:** i) the output of 02_HOTSPELLS.py, i.e. the .csv table of hot spell events for each cluster (duration, date ranges); ii) a gridded NetCDF file containing anomalies of a specific variable (time, lon, lat).

**Output:** for a specified region, the script calculates the subsampled anomaly composites of long hot spells (based on number of short spells) AND the corresponding mask of grid cells that are statistically significant at the specified confidence level (2D: lon,lat).
