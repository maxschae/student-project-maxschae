'''
Auxiliary file for replication of 
Brückner and Ciccone (2011) "Rain and the Window of Opportunity"
'''

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt



def get_panel_dataset():
    dataset = "data/database.dta"
    dataset_elections = "data/african_elections.xls"
    df_database = pd.read_stata(dataset)
    df_elections = pd.read_excel(dataset_elections)

    df_elections = df_elections[df_elections.year.between(1980,2004)]
    df_elections = df_elections[['ccode', 'year']]
    df_elections = df_elections.drop_duplicates()
    df_elections['election'] = 1

    df_merge = pd.merge(df_database, df_elections, how='left', on=['ccode','year'])

    index = list()
    for i in range(len(df_merge)):
        index.append((df_merge.iloc[i]['ccode'], df_merge.iloc[i]['year']))
    index = pd.MultiIndex.from_tuples(index, names=('country_index', 'year_index'))
    df_merge.index = index
    df = df_merge.dropna(axis=0, subset=['polity2'])
    df = df.fillna(value={'election':0})
    return df



def plot_reduced_form(country_list, election):
    '''
    Note that dum_rain_20 is already lagged two periods
    and hence the year must be adjusted 
    so to correctly indicate year of drought
    '''
    
    df = get_panel_dataset()

    fig = plt.figure(figsize=[16,9])
    plt.suptitle('Figure 1: Reduced-form relationship between instrument and outcome', y=0.94, fontsize=16)
    colors = ['#c33764', '#7d2648', '#9d3367', '#863169', '#742f6a', '#592c6c', '#3c296e', '#3c296e', '#1d2671']

    for i,country_code in enumerate(country_list):
        plt.subplot(3,3,i+1)
        plt.plot(df.loc[(country_code,), 'polity2'], color=colors[i], label=df.loc[(country_code,1981), 'country'])
        plt.ylim(bottom=df['polity2'].min(), top=10)
        plt.legend(loc='best')
        if i==0 or i==3 or i==6:
            plt.ylabel('Polity 2 score')
        if i==6 or i==7 or i==8:
            plt.xlabel('Year')
            
        subset = df.loc[(country_code,),:]
        subset = subset.loc[subset['dum_rain_20'] == 1]
        for yr in subset.index:
            plt.axvline(x=(yr-2), ymin=0, ymax=1, color='#9F9F9F', linestyle='--') 
        
        if election:
            subset = subset.loc[subset['election'] == 1]
            for yr in subset.index:
                plt.axvline(x=yr, ymin=0, ymax=1, color='#76A8FC', linestyle='-.')



def get_map_data(year):
    shape_file = "data/Africa.shp"
    geo_df = gpd.read_file(shape_file)
    geo_df['countryisocode'] = geo_df['CODE']    
    country_dict = {'TAN':'TZA', 'SIL':'SLE', 'ANG':'AGO', 'ZAM':'ZMB', 'SUD':'SDN', 'GUB':'GNB', 'GAM':'GMB', 
                    'MAL':'MLI', 'MAU':'MRT', 'NIG':'NER', 'CDI':'CIV', 'BUF':'BFA', 'LIB':'LBR', 'TOG':'TGO', 
                    'CAM':'CMR', 'NIR':'NGA', 'CAR':'CAF', 'CHA':'TCD', 'ZAI':'ZAR', 'BUR':'BDI', 'ZIM':'ZWE', 
                    'MAA':'MWI', 'SOU':'ZAF', 'LES':'LSO', 'BOT':'BWA', 'SWA':'SWZ', 'MAD':'MDG', 'CNG':'COG'}
    col_location = geo_df.columns.get_loc('countryisocode')
    for i in range(0,len(geo_df),1):
        for key, value in country_dict.items():
            if geo_df.iloc[i, col_location] == key:
                geo_df.iloc[i, col_location] = value

    df = get_panel_dataset()
    df_temp = df.copy()
    df_temp['polity_change_l'] = df_temp['polity2l'] - df_temp['polity2l2']
    df_temp.loc[df_temp['year']==1981, 'polity_change_l'] = 'nan'
    merged_df = pd.merge(geo_df, df_temp, how='left', on='countryisocode')
    
    '''
    fill values of missing countries such that they are
    easily spotted as "missing" on maps
    '''
    values = {'polity_change_l': -2, 'recession_l2': -0.3, 'dum_rain_20': -0.3, 'year': year,
             'agri_gdp_av': 0, 'agri_gdpshare': 0, 'gpcp':0}
    merged_df = merged_df.fillna(value=values)    
    return merged_df



def draw_settings_map(year):
    
    map_df = get_map_data(year)
    map_df_temp = map_df[map_df['year']==year]

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(20,7))
    
    map_df_temp.plot(column='agri_gdpshare', cmap='Greens', ax=ax1, alpha=1, edgecolor='.3', linewidth=.3)
    ax1.set_title('GDP share of agriculture in {:}'.format(year), fontsize=12)
    ax1.annotate('Countries in white: no data available.', xy=(0.1, .08), horizontalalignment='left', 
                verticalalignment='top', xycoords='figure fraction', fontsize=8, color='#696969')
    ax1.set_axis_off()
    vmin, vmax = 0, map_df['agri_gdpshare'].max()
    sm = plt.cm.ScalarMappable(cmap='Greens', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm, fraction=0.035, pad=0.005, ax=ax1)
    
    map_df_temp.plot(column='gpcp', cmap='PuBu', ax=ax2, alpha=1, edgecolor='.3', linewidth=.3)
    ax2.set_title('Rainfall in {:}'.format(year), fontsize=12)
    ax2.set_axis_off()
    vmin, vmax = 0, map_df['gpcp'].max()
    sm = plt.cm.ScalarMappable(cmap='PuBu', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm, fraction=0.035, pad=0.005, ax=ax2)



def draw_story_map(year):
    '''
    Note that year adjustment only necessary due to nested lag nature
    e.g. 'dum_rain_20' is drought dummy already lagged two years
    '''
    year = year+1
    
    map_df = get_map_data(year)    
    map_df_temp = map_df[map_df['year']==year]

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(20,7))    
    
    map_df_temp.plot(column='dum_rain_20', cmap='Blues', ax=ax1, alpha=1, edgecolor='.3', linewidth=.3)
    ax1.set_title('Drought in %s' %str(year-2), fontsize=12)
    ax1.annotate('Countries in white: no data available.', xy=(0.1, .08), horizontalalignment='left', 
                    verticalalignment='top', xycoords='figure fraction', fontsize=8, color='#696969')
    ax1.set_axis_off()
    
    map_df_temp.plot(column='recession_l2', cmap='Purples', ax=ax2, alpha=1, edgecolor='.3', linewidth=.3)
    ax2.set_title('Recession in %s' %str(year-2), fontsize=12)
    ax2.set_axis_off()
    
    map_df_temp.plot(column='polity_change_l', cmap='PuBuGn', ax=ax3, alpha=1, edgecolor='.3', linewidth=.3)
    ax3.set_title('Change of combined polity score %s to %s' %(str(year-2),str(year-1)), fontsize=12)
    ax3.set_axis_off()
    vmin, vmax = 0, map_df_temp['polity_change_l'].max()
    sm = plt.cm.ScalarMappable(cmap='PuBuGn', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm, fraction=0.035, pad=0.005, ax=ax3)
