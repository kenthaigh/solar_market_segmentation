## Segmentation of the Australian solar market

### Motivation

Using data on demographics and solar irradiation, this script finds clusters of geographic areas
(by postcode) using KMeans. These segments are anlaysed to find those which had a high take up of
of small scale solar (ie solar panels) in 2016 and then compared to the number of installations in
2017. Within high take up segments, areas with the potential for high growth are also identified.

### Model
The clustering in this analysis is undertaken using the KMeans algorithm from Python's scikit-learn. 

### Project structure
The main project script is solar_market_segmentation.py. This script will load data from the source_data folder, run the data cleaning and clustering model, and output interesting findings to the console.

The files energymatters_spider.py	and parse_energymatters.py are used for webscraping and data parsing. They cannot be run directly as is. For more infomation see the data section below. 

### Data
The data in the project is from the following sources:
1. Census data
* Data from the 2016 Census by postcode is is downloaded from https://datapacks.censusdata.abs.gov.au/datapacks/.
* Data is has been extracted from the zip file locally. Only the relevant extracted .csv files are included in this repository.
2. Solar installation data
* Data is downloaded from http://www.cleanenergyregulator.gov.au/DocumentAssets/Pages/Postcode-data-for-small-scale-installations---SGU-Solar.aspx
3. Solar irradiation data
* Data is collected from the energymatters.com.au on pages for individual locations (eg. https://www.energymatters.com.au/solar-location/kalgoorlie-6430/). 
* The relevant data is extracted using the Python scrapy script energy_spider.py. The scrapy project and script and executed from the command line. Not all project files are included in this repository. For more information on Python scrapy visit https://scrapy.org/.


#### System and software
Ubuntu 16.04LTX Python 3.5.2
