

###############################
'''
Script to extract data from .jl files created by scrapy routine.

Versions
Ubuntu 16.04 LTS
Python 3.5.2
'''
###############################

# IMPORT LIBRARIES
import pandas as pd
import re
import os

# source_data = [file path to source data directory - not jl files not uploaded]
# output_data = [file path to where you want to save output]

jl_list = []
for file in os.listdir(file_path):
    if file.endswith(".jl"):
        jl_list = jl_list + [file]

jl_concat = pd.Series()
for file in jl_list:
    temp_file = file_path + file
    temp_series = pd.Series(tuple(open(temp_file, 'r')))
    jl_concat = jl_concat.append(temp_series)

df_energymatters = jl_concat.str.split('irra', expand=True)

# Extract digits
postcodes = df_energymatters.iloc[:,0].apply(lambda x: re.findall(r'\d+', x))
# Collapse lists
postcodes = postcodes.apply(lambda x: ''.join(x))
# Extract last 4 digits (this filters out additional digits where there is more than one of a suburb name
# (eg https://www.energymatters.com.au/solar-location/balmoral-4-4171)
postcodes = postcodes.apply(lambda x: x[-4:])
# Add string to match format in census data
postcodes = 'POA' + postcodes.apply(lambda x: str(x))

# Extract float characters (digits and fullstops) if not null
irradiation = df_energymatters.iloc[:,1].apply(lambda x: re.findall(r"\d+\.\d+", x) if x is not None else None)
# Convert to float if not null or empty list
irradiation = irradiation.apply(lambda x: float(x[0]) if x is not None and len(x)>0 else None)

irradiation_df = pd.DataFrame({'POA_CODE_2016': postcodes,
                               'irradiation': irradiation})
irradiation_df = irradiation_df.groupby('POA_CODE_2016')['irradiation'].mean().to_frame()
irradiation_df.reset_index(inplace=True)
irradiation_df.to_csv(output_data + r'/irradiation.csv', index=False)
