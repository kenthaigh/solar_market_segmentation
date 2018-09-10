
###############################
'''
Solar market segmentation

Using data on demographics and solar irradiation, this script finds clusters of geographic areas
(by postcode) using KMeans. These segments are anlaysed to find those which had a high take up of
of small scale solar (ie solar panels) in 2016 and then compared to the number of installations in
2017. Within high take up segments, areas with the potential for high growth are also identified.

Versions
Ubuntu 16.04 LTS
Python 3.5.2
'''
###############################

# IMPORT LIBRARIES
import pandas as pd
import zipfile36 as zipfile
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from scipy.spatial import distance

# SETUP PROJECT DIRECTORIES
# project_dir = r'_____' # [path to top level directory]
source_data_dir = project_dir + r'/source_data'

# # UNZIP DATA FILES
# # Census data
# zip_ref = zipfile.ZipFile(source_data_dir + r'/2016_GCP_POA_for_AUS_short-header.zip', 'r')
# zip_ref.extractall(source_data_dir)
# zip_ref.close()

# IMPORT DATA
# Solar installation data
solar_install = pd.read_csv(source_data_dir + r'/small_scale_solar_ installations_by_postcode.csv')

df_solar = solar_install[['Small Unit Installation Postcode',
                          'Previous Years (2001-2016) - Installations Quantity',
                          'Installations Quantity Total' ]]
df_solar.rename(columns={'Small Unit Installation Postcode':                    'POA_CODE_2016',
                         'Previous Years (2001-2016) - Installations Quantity': 'solar_units_2016',
                         'Installations Quantity Total':                        'solar_units_2017'},
                inplace=True)
df_solar['solar_units_change']     = df_solar.solar_units_2017 - df_solar.solar_units_2016
df_solar['solar_units_change_pct'] = (df_solar.solar_units_2017 / df_solar.solar_units_2016) - 1

# Solar irradiation data
df_irradiation = pd.read_csv(source_data_dir + r'/irradiation.csv')

# Census data
census_dict = {'2016Census_G59_AUS_POA': ['One_method_Train_P',
                                          'One_method_Bus_P',
                                          'One_met_Tram_incl_lt_rail_P',
                                          'One_method_Bicycle_P',
                                          'One_method_Walked_only_P',
                                          'One_method_Car_as_driver_P',
                                          'One_method_Car_as_passenger_P',
                                          'Tot_P'],
               '2016Census_G32_AUS_POA': ['OPDs_Separate_house_Dwellings',
                                          'OPDs_SD_r_t_h_th_Tot_Dwgs',
                                          'Total_PDs_Dwellings'],
               '2016Census_G02_AUS_POA': ['Median_age_persons',
                                          'Median_mortgage_repay_monthly',
                                          'Median_rent_weekly',
                                          'Median_tot_hhd_inc_weekly',
                                          'Average_household_size'],
               '2016Census_G30_AUS_POA': ['Num_MVs_per_dweling_Tot',
                                          'Total_dwelings'],
               '2016Census_G33_AUS_POA': ['O_OR_DS_Sep_house',
                                          'O_OR_DS_SemiD_ro_or_tce_h_th',
                                          'O_OR_Total',
                                          'O_MTG_DS_Sep_house',
                                          'O_MTG_DS_SemiD_ro_or_tce_h_th',
                                          'O_MTG_Total'],
               '2016Census_G40_AUS_POA': ['Percnt_LabForc_prticipation_P',
                                          'Percnt_Employment_to_populn_P',
                                          'Non_sch_quals_PostGrad_Dgre_P',
                                          'Non_sch_quals_Bchelr_Degree_M']
               }

for i in census_dict.keys():
    temp_file = pd.read_csv(source_data_dir + r'/' + i + r'.csv')
    responses = census_dict[i]
    responses = responses + ['POA_CODE_2016']
    temp_file = temp_file[responses]

    try:
        df
    except NameError:
        df = temp_file
    else:
    df = pd.merge(df, temp_file,
                  how='outer',
                  on='POA_CODE_2016')



df.rename(columns={'OPDs_Separate_house_Dwellings': 'house_count',
                   'OPDs_SD_r_t_h_th_Tot_Dwgs':     'semi_detached_count',
                   'Total_PDs_Dwellings':           'private_dwellings_count',
                   'POA_CODE_2016':                 'POA_CODE_2016',
                   'Median_age_persons':            'age_median',
                   'Median_mortgage_repay_monthly': 'mortgage_repay_median',
                   'Median_rent_weekly':            'rent_median',
                   'Median_tot_hhd_inc_weekly':     'household_income_median',
                   'Average_household_size':        'household_size_avg',
                   'One_method_Train_P':            'train_only_count',
                   'One_method_Bus_P':              'bus_only_count',
                   'One_met_Tram_incl_lt_rail_P':   'tram_only_count',
                   'One_method_Bicycle_P':          'bike_only_count',
                   'One_method_Walked_only_P':      'walk_only_count',
                   'One_method_Car_as_driver_P':    'car_drive_only_count',
                   'One_method_Car_as_passenger_P': 'car_passenger_only_count',
                   'Tot_P':                         'population',
                   'Percnt_LabForc_prticipation_P': 'labour_participation_pct',
                   'Percnt_Employment_to_populn_P': 'employment_pct',
                   'Non_sch_quals_PostGrad_Dgre_P': 'postgrad_count',
                   'Non_sch_quals_Bchelr_Degree_M': 'grad_count',
                   'Num_MVs_per_dweling_Tot':       'motor_vehicles_count',
                   'Total_dwelings':                'all_dwellings_count',
                   'O_OR_DS_Sep_house':             'house_owned_count',
                   'O_OR_DS_SemiD_ro_or_tce_h_th':  'semi_detached_owned_count',
                   'O_OR_Total':                    'all_owned_count',
                   'O_MTG_DS_Sep_house':            'house_mortgage_count',
                   'O_MTG_DS_SemiD_ro_or_tce_h_th': 'semi_detached_mortgage_count',
                   'O_MTG_Total':                   'all_mortgage_count'},
          inplace=True)


# Create calculated fields (eg. percentages, aggregations)
df['house_semi_count']       = df.house_count + df.semi_detached_count
df['house_semi_pct']         = df.house_semi_count / df.private_dwellings_count
df['mortgage_to_income_pct'] = df.mortgage_repay_median * 12 / df.household_income_median *52
df['only_public_pct']        = (df.train_only_count + df.bus_only_count + df.tram_only_count + \
                                df.bike_only_count + df.walk_only_count) / df.population
df['only_car_pct']           = (df.car_drive_only_count + df.car_passenger_only_count) / df.population
df['grad_pct']               = df.grad_count / df.population
df['postgrad_pct']           = df.postgrad_count / df.population
df['motor_vehicles_pct']     = df.motor_vehicles_count / df.private_dwellings_count
df['owned_mortgage_pct']     = (df.house_owned_count + df.house_mortgage_count) / df.private_dwellings_count

# Merge with irradiation data
df = pd.merge(df, df_irradiation,
              how='left',
              on='POA_CODE_2016')

# Drop columns not wanted for segmentation
X = df.drop(['private_dwellings_count',
             'train_only_count',
             'bus_only_count',
             'tram_only_count',
             'bike_only_count',
             'walk_only_count',
             'car_drive_only_count',
             'car_passenger_only_count',
             'all_dwellings_count',
             'all_owned_count',
             'all_mortgage_count']
            , axis=1)
# Remove na's
X = X.dropna()
# Save a list of the postcode
postcode_idx = X.POA_CODE_2016
postcode_idx.reset_index(drop=True, inplace=True)
# Then drop postcode from the matrix for segmentation
X = X.drop('POA_CODE_2016', axis=1)
# Save a list of the fields
field_idx = X.dtypes
# Convert to numpy
X = np.array(X)

# Create pre-processing and clustering pipeline...
pipe = make_pipeline(StandardScaler(), Normalizer(), KMeans())
# ...and now create segments
kmeans_fit_predict = pipe.set_params(kmeans__random_state=23).fit_predict(X)

# Create a dataframe with clusters, postcodes and solar installation data
output_clusters = pd.Series(kmeans_fit_predict)
clusters_postcodes = pd.DataFrame({'postcode': postcode_idx,
                                   'cluster':  output_clusters})
clusters_postcodes.postcode = clusters_postcodes.postcode.str.replace("POA", "")
clusters_postcodes.postcode = clusters_postcodes.postcode.apply(lambda x: int(x))

output_by_postcode = pd.merge(clusters_postcodes, df_solar,
                              how='left',
                              left_on='postcode',
                              right_on='POA_CODE_2016')

units_2016_df = output_by_postcode.groupby('cluster')['solar_units_2016'].mean()
units_2016_df = units_2016_df.reset_index()
change_df = output_by_postcode.groupby('cluster')['solar_units_change'].mean()
change_df = change_df.reset_index()

summary_output = pd.merge(units_2016_df, change_df, on='cluster')

# Find the cluster with the most units installed as at 2016
print("Cluster with most units in 2016: cluster " +  str(summary_output.iloc[summary_output.idxmax().solar_units_2016].cluster))
# If we assume that take-up of solar panels has not reached saturation, we can suppose that this
# cluster will show strong sales in 2017. Let's have a look
print("Cluster with most units installed in 2017: cluster " + str(summary_output.iloc[summary_output.idxmax().solar_units_change].cluster))
#...yes, this cluster had the most sales in 2017.

# However, the number of installations in this cluster is widely distributed:
output_by_postcode[output_by_postcode.cluster==4].solar_units_change.hist(bins=30)

# We could consider looking at the poscodes in this cluster that had weaker sales. There could be
# potential for growth in these similiar areas as they 'catch-up' to other parts of the segment that
# have already have a high number of installations.


# ASIDE - let's see how close each cluster is to our 'high-value' cluster - 4. We might want to
# see if there is potential for growth in the clusters that are closest to number 4.
#
# Get centriod for cluster 4
ref_point = pipe.named_steps['kmeans'].cluster_centers_[4]
# Caclulate the euclidian distance between 4 and each other clusters.
rel_dist = []
for i in range(8):
    temp_point = pipe.named_steps['kmeans'].cluster_centers_[i]
    temp_dist  = distance.euclidean(ref_point, temp_point)
    rel_dist.append([i, temp_dist])
rel_dist = pd.DataFrame(rel_dist)
rel_dist.columns = ['cluster', 'rel_dist']

print("These are the distances between cluster 4 and all clusters")
print(rel_dist)
