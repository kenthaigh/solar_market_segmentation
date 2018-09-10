## Segmentation of the Australian solar market

### Motivation
The uptake of small scale solar (ie. household solar panels) has been growing in recent years alongside increases in electricity prices and decreasing installation costs. This analysis identifies geographical areas that may be expected to show further uptake of solar panels. This information is useful for energy market operator planning as well as for sales forecasting by solar panel retailers.

### Model
The clustering in this analysis is undertaken using the KMeans algorithm from Python's scikit-learn. The data analysed includes demographic information and solar irradiation data by postcode. The variables selected for consideration are those which are could be considered as potentially related to the uptake of household solar panels. These include the capacity to pay, education status, age, number/percentage of properties that could benefit from solar panels, and amount of sun energy in that location. The clusters produced are compared to actual solar panel installation data for analysis.

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
Ubuntu 16.04LTS Python 3.5.2
