
###############################
'''PROJECT DESCRIPTION
____

Versions
Ubuntu 16.04 LTS
Python 3.5.2
'''
###############################

# IMPORT LIBRARIES
import pandas as pd
import zipfile36 as zipfile
import numpy as np
from sklearn.cluster import KMeans # Need to install


# SETUP PROJECT DIRECTORIES
project_dir = r'/home/kent/PycharmProjects/KPMG_task' # [path to top level directory]
source_data_dir = project_dir + r'/source_data'
output_data_dir = project_dir + r'/output_data'

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
df_solar['solar_units_change'] = df_solar.solar_units_2017 - df_solar.solar_units_2016

# Solar irradiation data
df_irradiation = pd.read_csv(source_data_dir + r'/energymatters/irradiation.csv')

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
    temp_file = pd.read_csv(source_data_dir + r'/2016 Census GCP Postal Areas for AUST/' + i + r'.csv')
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

df = pd.merge(df, df_irradiation,
              how='left',
              on='POA_CODE_2016')


X = df.drop(['POA_CODE_2016',
             'private_dwellings_count',
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
X = X.dropna()
field_idx = X.dtypes
postcode_idx = df.POA_CODE_2016
X = np.array(X)

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
# make_pipeline(StandardScaler(), Normalizer(), KMeans())
# Pipeline(memory=None,
#          steps=[('binarizer', Binarizer(copy=True, threshold=0.0)),
#                 ('multinomialnb', MultinomialNB(alpha=1.0,
#                                                 class_prior=None,
#                                                 fit_prior=True))])

# Create pre-processing and clustering pipeline
pipe = make_pipeline(StandardScaler(), Normalizer(), KMeans())
kmeans_fit_predict = pipe.fit_predict(X)
# kmeans_fit_predict = pipe.fit_predict(np.array(df.iloc[:, :3]))

test_cluster = pd.Series(kmeans_fit_predict)


df_testing = pd.DataFrame({'postcode': df.POA_CODE_2016,
                           'cluster':  test_cluster})
df_testing.postcode = df_testing.postcode.str.replace("POA", "")
df_testing.postcode = df_testing.postcode.apply(lambda x: int(x))

test_merge = pd.merge(df_testing, df_solar,
         how='left',
         left_on='postcode',
         right_on='POA_CODE_2016')

test_merge.groupby('cluster')['solar_units_2016'].mean()
test_merge.groupby('cluster')['solar_units_change'].mean()

cluster_4 = test_merge[test_merge.cluster==4].sort_values('solar_units_2016')
cluster_4_a = cluster_4.iloc[:150, :]
cluster_4_b = cluster_4.iloc[150:, :]
print(cluster_4_a.solar_units_change.mean())
print(cluster_4_b.solar_units_change.mean())
