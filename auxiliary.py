'''
Auxiliary file for replication of 
Brückner and Ciccone (2011) "Rain and the Democratic Window of Opportunity"
'''

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col


###


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


def plot_hists():
    import warnings
    warnings.filterwarnings('ignore');

    df = get_panel_dataset()
    columns = ['polity2', 'lgdp', 'agri_gdpshare', 'polity_change', 'lgpcp_l2']
    labels = ['Polity2', 'Log GDP per capita', 'Agriculture GDP share', 'Change in polity2 score', 'Log rainfall']
    colors = ['#f4416b', '#0F0F0F', '#8ce222', '#b042f4', '#1f77b4']

    fig = plt.figure(figsize=[16,8])
    for i,column in enumerate(columns):
        plt.subplot(2,3,i+1)
        if column == 'polity_change':
            sns.distplot(df[columns[i]], bins=40, kde=False, axlabel=labels[i], color=colors[i])
        else:
            sns.distplot(df[columns[i]], bins=20, kde=False, axlabel=labels[i], color=colors[i])


def plot_reduced_form(country_list, election):
    '''
    Note that dum_rain_20 is already lagged two periods
    and hence the year must be adjusted 
    so to correctly indicate year of drought
    '''
    
    df = get_panel_dataset()

    fig = plt.figure(figsize=[16,9])
    if election == False:
        plt.suptitle('Figure 1: Reduced-form relationship between instrument and outcome', y=0.94, fontsize=16)
    elif election == True:
        plt.suptitle('Figure A.1: Reduced-form relationship between instrument and outcome with elections', y=0.94, fontsize=16)
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
    '''
    include rows for Namibia's missing data for visualization purpose only
    '''
    df_nam = pd.DataFrame(index=range(0,10,1),columns=df_temp.columns)
    year_nam = 1981
    for j in range(len(df_nam)):
        df_nam['countryisocode'] = 'NAM'
        df_nam['ccode'] = 565
        df_nam.iloc[j, df_nam.columns.get_loc('year')] = year_nam
        year_nam += 1
    df_temp = df_temp.append(df_nam, ignore_index=True)
    df_temp['polity_change_l'] = df_temp['polity2l'] - df_temp['polity2l2']
    df_temp.loc[df_temp['year']==1981, 'polity_change_l'] = np.NaN
    for i in range(len(df_temp)):
        if (df_temp.iloc[i, df_temp.columns.get_loc('year')] == 1991 and df_temp.iloc[i, df_temp.columns.get_loc('ccode')] == 565):
            df_temp.iloc[i, df_temp.columns.get_loc('polity_change_l')] = np.NaN
    merged_df = pd.merge(geo_df, df_temp, how='left', on='countryisocode')
    
    '''
    fill values of missing countries such that they are
    easily spotted as "missing" on maps
    '''
    values = {'polity_change_l': -2, 'recession_l2': -0.3, 'dum_rain_20': -0.3, 'year': year,
             'agri_gdp_av': 0, 'agri_gdpshare': 0, 'gpcp':0, 'polity2': -11}
    merged_df = merged_df.fillna(value=values)    
    return merged_df



def draw_settings_map(plot, year):
    
    map_df = get_map_data(year)
    map_df_temp = map_df[map_df['year']==year]
    
    if plot==1:
        fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(20,7))
        map_df_temp.plot(column='agri_gdp_av', cmap='Greens', ax=ax1, alpha=1, edgecolor='.6', linewidth=.3)
        ax1.set_title('Average agriculture GDP share in percent (1980-2004)', fontsize=12)
        ax1.annotate('Countries in white: no data available.', xy=(0.1, .08), horizontalalignment='left', 
                    verticalalignment='top', xycoords='figure fraction', fontsize=8, color='#696969')
        ax1.set_axis_off()
        vmin, vmax = 0, map_df['agri_gdp_av'].max()
        sm = plt.cm.ScalarMappable(cmap='Greens', norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        cbaxes = fig.add_axes([0.14, 0.25, 0.01, 0.55])
        cbar = fig.colorbar(sm, fraction=0.035, pad=0.005, ax=ax1, cax=cbaxes)
        
        map_df_temp.plot(column='gpcp', cmap='PuBu', ax=ax2, alpha=1, edgecolor='.6', linewidth=.3)
        ax2.set_title('Rainfall in millimetre in {:}'.format(year), fontsize=12)
        ax2.set_axis_off()
        vmin, vmax = 0, map_df['gpcp'].max()
        sm = plt.cm.ScalarMappable(cmap='PuBu', norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        cbaxes = fig.add_axes([0.56, 0.25, 0.01, 0.55])
        cbar = fig.colorbar(sm, fraction=0.035, pad=0.005, ax=ax2, cax=cbaxes)
    
    elif plot==2:
        fig, ax1 = plt.subplots(ncols=1, sharex=True, sharey=True, figsize=(12,6))
        
        map_df_temp.plot(column='polity2', cmap='Oranges', ax=ax1, alpha=1, edgecolor='.6', linewidth=.3)
        ax1.set_title('Combined polity score in {:}'.format(year), fontsize=12)
        ax1.set_axis_off()
        vmin, vmax = map_df['polity2'].min(), map_df['polity2'].max()
        sm = plt.cm.ScalarMappable(cmap='Oranges', norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        cbaxes = fig.add_axes([0.27, 0.25, 0.01, 0.55])
        cbar = fig.colorbar(sm, fraction=0.035, pad=0.005, ax=ax1, cax=cbaxes)
        


def draw_story_map(year):
    '''
    Note that year adjustment only necessary due to nested lag nature
    e.g. 'dum_rain_20' is drought dummy already lagged two years
    '''
    year = year+1    
    map_df = get_map_data(year)    
    map_df_temp = map_df[map_df['year']==year]

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(20,7))    
    
    map_df_temp.plot(column='dum_rain_20', cmap='Blues', ax=ax1, alpha=1, edgecolor='.6', linewidth=.3)
    ax1.set_title('Drought in %s' %str(year-2), fontsize=12)
    ax1.annotate('Countries in white: no data available.', xy=(0.1, .08), horizontalalignment='left', 
                    verticalalignment='top', xycoords='figure fraction', fontsize=8, color='#696969')
    ax1.set_axis_off()
    
    map_df_temp.plot(column='recession_l2', cmap='Purples', ax=ax2, alpha=1, edgecolor='.6', linewidth=.3)
    ax2.set_title('Recession in %s' %str(year-2), fontsize=12)
    ax2.set_axis_off()
    
    map_df_temp.plot(column='polity_change_l', cmap='BuPu', ax=ax3, alpha=1, edgecolor='.6', linewidth=.3)
    ax3.set_title('Change of combined polity score %s to %s' %(str(year-2),str(year-1)), fontsize=12)
    ax3.set_axis_off()
    vmin, vmax = 0, map_df_temp['polity_change_l'].max()
    sm = plt.cm.ScalarMappable(cmap='BuPu', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm, fraction=0.035, pad=0.005, ax=ax3)



def quartile_means(by, col):   
    agri_q1 = df[by].quantile(q=0.25)-0.0001
    agri_q2 = df[by].quantile(q=0.5)-0.0001
    agri_q3 = df[by].quantile(q=0.75)-0.0001
    agri_q4 = df[by].max()

    df_1 = df.copy()
    df_1 = df_1[df_1[by]<agri_q1]
    df_2 = df.copy()
    df_2 = df_2[df_2[by]>agri_q1]
    df_2 = df_2[df_2[by]<agri_q2]
    df_3 = df.copy()
    df_3 = df_3[df_3[by]>agri_q2]
    df_3 = df_3[df_3[by]<agri_q3]
    df_4 = df.copy()
    df_4 = df_4[df_4[by]>agri_q3]
    df_4 = df_4[df_4[by]<agri_q4]
    
    return df_1[col].mean(), df_2[col].mean(), df_3[col].mean(), df_4[col].mean()



def ecdf(column):
    df = get_panel_dataset()
    df_temp = df[df['dum_rain_20']!=1]
    treatment = df_temp[column].values    
        
    cdfx = np.sort(np.unique(treatment))
    x_values = np.linspace(start=min(cdfx), stop=max(cdfx), num=len(cdfx))
    y_values = []
    for i in x_values:
        temp = treatment[treatment <= i]
        y_value = temp.size / treatment.size
        y_values.append(y_value)
    
    df_temp = df[df['dum_rain_20']==1]
    treatment = df_temp[column].values
    
    cdfx = np.sort(np.unique(treatment))
    x_values_drought = np.linspace(start=min(cdfx), stop=max(cdfx), num=len(cdfx))
    y_values_drought = []
    for i in x_values_drought:
        temp = treatment[treatment <= i]
        y_value = temp.size / treatment.size
        y_values_drought.append(y_value)
        
    return x_values, y_values, x_values_drought, y_values_drought



def acr_weighting_fct(column):
    x_values, y_values, x_values_drought, y_values_drought = ecdf(column)
    df_wf = pd.DataFrame(index=range(0,len(x_values),1))
    df_wf_d = pd.DataFrame(index=range(0,len(x_values_drought),1))

    df_wf['x_values'] = x_values.round(2)
    df_wf['y_values'] = y_values
    df_wf_d['x_values_drought'] = x_values_drought.round(2)
    df_wf_d['y_values_drought'] = y_values_drought

    df_wf_merge = df_wf.merge(df_wf_d, left_on='x_values', right_on='x_values_drought')
    df_acr_wf = df_wf_merge.copy()
    df_acr_wf['acr_weighting'] =  (1 - df_acr_wf['y_values']) - (1 - df_acr_wf['y_values_drought'])

    return df_acr_wf


def plot_ecdfs(column):
    df_acr_wf = acr_weighting_fct(column)
    x_values, y_values, x_values_drought, y_values_drought = ecdf(column)

    fig = plt.figure(figsize=(8,5))
    plt.suptitle('Figure 3: Empirical CDFs with instrument flipped on and off', y=0.97, fontsize=14)
    plt.plot(df_acr_wf['x_values'], df_acr_wf['y_values'], label='instrument off')
    plt.plot(df_acr_wf['x_values'], df_acr_wf['y_values_drought'], label='instrument on')
    plt.xlabel('log GDP per capita')
    plt.ylabel('Empirical CDFs')
    plt.legend(loc='best')
    plt.show()


def plot_acr_weighting_fct(column):
    df_acr_wf = acr_weighting_fct(column)
    fig = plt.figure(figsize=(8,5))
    plt.suptitle('Figure 2: Average causal response weighting function', y=0.97, fontsize=14)
    plt.plot(df_acr_wf['x_values'],df_acr_wf['acr_weighting'], c='#7a99c4')
    plt.xlabel('log GDP per capita')
    plt.ylabel('Difference in CDFs');
    plt.show()





'''

In the code section below estimations and output are produced

'''




def table_first_stage(show):
    df = get_panel_dataset()

    spec_1 = 'lgdp_l2 ~ lgpcp_l2 + C(ccode) + C(ccode) : I(year) + C(year)'
    spec_2 = 'lgdp_l2 ~ lgpcp_l2 + lgpcp_l3 + C(ccode) + C(ccode) : I(year) + C(year)'
    spec_3 = 'lgdp_l2 ~ lgpcp_l2 + lgpcp_l3 + lgpcp_l4 + C(ccode) + C(ccode) : I(year) + C(year)'
    spec_4 = 'lgdp_l2 ~ lgpcp_l2 + lgpcp_l2_polity2l2 + polity2l2 + C(ccode) + C(ccode) : I(year) + C(year)'

    spec_5 = 'recession_l2 ~ lgpcp_l2 + C(ccode) + C(ccode) : I(year) + C(year)'
    spec_6 = 'recession_l2 ~ lgpcp_l2 + lgpcp_l3 + C(ccode) + C(ccode) : I(year) + C(year)'
    spec_7 = 'recession_l2 ~ lgpcp_l2 + lgpcp_l3 + lgpcp_l4 + C(ccode) + C(ccode) : I(year) + C(year)'
    spec_8 = 'recession_l2 ~ lgpcp_l2 + lgpcp_l2_polity2l2 + polity2l2 + C(ccode) + C(ccode) : I(year) + C(year)'


    rslt_1 = smf.ols(formula=spec_1, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['ccode']})
    rslt_2 = smf.ols(formula=spec_2, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['ccode']})
    rslt_3 = smf.ols(formula=spec_3, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['ccode']})
    rslt_4 = smf.ols(formula=spec_4, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['ccode']})

    rslt_5 = smf.ols(formula=spec_5, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['ccode']})
    rslt_6 = smf.ols(formula=spec_6, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['ccode']})
    rslt_7 = smf.ols(formula=spec_7, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['ccode']})
    rslt_8 = smf.ols(formula=spec_8, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['ccode']})

    output_1 = summary_col([rslt_1,rslt_2,rslt_3,rslt_4], model_names=['(1)','(2)','(3)','(4)'],
                         stars=True, float_format='%0.3f', 
                         regressor_order=['lgpcp_l2','lgpcp_l3','lgpcp_l4','polity2l2','lgpcp_l2_polity2l2'],
                         info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),'R2':lambda x: "{:.2f}".format(x.rsquared)})

    output_2 = summary_col([rslt_5,rslt_6,rslt_7,rslt_8], model_names=['(5)','(6)','(7)','(8)'],
                         stars=True, float_format='%0.3f', 
                         regressor_order=['lgpcp_l2','lgpcp_l3','lgpcp_l4','polity2l2','lgpcp_l2_polity2l2'],
                         info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),'R2':lambda x: "{:.2f}".format(x.rsquared)})
    if show == True:
        print(output_1)
        print(output_2)
    elif show == False:
        for table in ['table1_raw.tex', 'table2_raw.tex']:
            begintex1 = "\\documentclass{report}"
            begintex2 = "\\begin{document}"
            endtex = "\end{document}"
            folder = 'tex_files/'
            f = open(folder+table, 'w')
            f.write(begintex1)
            f.write(begintex2)
            if table.startswith('table1'):
                f.write(output_1.as_latex())
            else:
                f.write(output_2.as_latex())
            f.write(endtex)
            f.close()



def table_second_stage(show):
    df = get_panel_dataset()
    df_int = df.copy()
    df_trans = df.copy()
    df_int = df_int.dropna(subset=['exconst_change'])
    df_trans = df_trans.dropna(subset=['trans_democ'])

    spec_first_stage = 'lgdp_l2 ~ lgpcp_l2 + C(ccode) + C(ccode) : I(year) + C(year)'
    rslt = smf.ols(formula=spec_first_stage, data=df).fit()
    rslt_int = smf.ols(formula=spec_first_stage, data=df_int).fit()
    rslt_trans = smf.ols(formula=spec_first_stage, data=df_trans).fit()
    df['lgdp_l_hat'] = rslt.predict()
    df_int['lgdp_l_hat'] = rslt_int.predict()
    df_trans['lgdp_l_hat'] = rslt_trans.predict()    

    spec_second_stage_1 = 'polity_change ~ lgdp_l_hat + C(ccode) + C(ccode) : I(year) + C(year)'
    spec_second_stage_2 = 'exconst_change ~ lgdp_l_hat + C(ccode) + C(ccode) : I(year) + C(year)'
    spec_second_stage_3 = 'polcomp_change ~ lgdp_l_hat + C(ccode) + C(ccode) : I(year) + C(year)'
    spec_second_stage_4 = 'exrec_change ~ lgdp_l_hat + C(ccode) + C(ccode) : I(year) + C(year)'
    spec_second_stage_5 = 'trans_democ ~ lgdp_l_hat + C(ccode) + C(ccode) : I(year) + C(year)'

    rslt_1 = smf.ols(formula=spec_second_stage_1,data=df).fit(cov_type='cluster',cov_kwds={'groups':df['ccode']})
    rslt_2 = smf.ols(formula=spec_second_stage_2,data=df_int).fit(cov_type='cluster',cov_kwds={'groups':df_int['ccode']})
    rslt_3 = smf.ols(formula=spec_second_stage_3,data=df_int).fit(cov_type='cluster',cov_kwds={'groups':df_int['ccode']})
    rslt_4 = smf.ols(formula=spec_second_stage_4,data=df_int).fit(cov_type='cluster',cov_kwds={'groups':df_int['ccode']})
    rslt_5 = smf.ols(formula=spec_second_stage_5,data=df_trans).fit(cov_type='cluster',cov_kwds={'groups':df_trans['ccode']})
   
    output = summary_col([rslt_1,rslt_2,rslt_3,rslt_4,rslt_5], model_names=['(1)','(2)','(3)','(4)','(5)'],
                             stars=True, float_format='%0.3f',
                             regressor_order=['lgdp_l_hat'],
                             info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),'R2':lambda x: "{:.2f}".format(x.rsquared)})
    if show == True:
        print(output)
    elif show == False:
        begintex1 = "\\documentclass{report}"
        begintex2 = "\\begin{document}"
        endtex = "\end{document}"
        f = open('tex_files/table3_raw.tex', 'w')
        f.write(begintex1)
        f.write(begintex2)
        f.write(output.as_latex())
        f.write(endtex)
        f.close()



def table_x_basic(show):
    df = get_panel_dataset()
    agri_q2 = df['agri_gdp_av'].quantile(q=0.5)-0.0001

    df_1 = df.copy()
    df_1 = df_1[df['agri_gdp_av']<agri_q2]
    df_1_int = df_1.dropna(subset=['trans_democ'])
    df_2 = df.copy()
    df_2 = df_2[df['agri_gdp_av']>agri_q2]
    df_2_int = df_2.dropna(subset=['trans_democ'])

    spec_first_stage = 'lgdp_l2 ~ lgpcp_l2 + lgpcp_l3 + C(ccode) + C(ccode) : I(year) + C(year)'
    spec_reduc_form_1 = 'polity_change ~ lgpcp_l + lgpcp_l2 + C(ccode) + C(ccode) : I(year) + C(year)'
    spec_reduc_form_2 = 'trans_democ ~ lgpcp_l + lgpcp_l2 + C(ccode) + C(ccode) : I(year) + C(year)'
    
    rslt_1 = smf.ols(formula=spec_first_stage,data=df_1).fit(cov_type='cluster',cov_kwds={'groups':df_1['ccode']})
    rslt_2 = smf.ols(formula=spec_reduc_form_1,data=df_1).fit(cov_type='cluster',cov_kwds={'groups':df_1['ccode']})
    rslt_3 = smf.ols(formula=spec_reduc_form_2,data=df_1_int).fit(cov_type='cluster',cov_kwds={'groups':df_1_int['ccode']})    
    rslt_4 = smf.ols(formula=spec_first_stage,data=df_2).fit(cov_type='cluster',cov_kwds={'groups':df_2['ccode']})
    rslt_5 = smf.ols(formula=spec_reduc_form_1,data=df_2).fit(cov_type='cluster',cov_kwds={'groups':df_2['ccode']})
    rslt_6 = smf.ols(formula=spec_reduc_form_2,data=df_2_int).fit(cov_type='cluster',cov_kwds={'groups':df_2_int['ccode']})
    
    output = summary_col([rslt_1,rslt_2,rslt_3,rslt_4,rslt_5,rslt_6], model_names=['(1)','(2)','(3)','(4)','(5)','(6)'],
                         stars=True, float_format='%0.3f', 
                         regressor_order=['lgpcp_l', 'lgpcp_l2', 'lgpcp_l3'],
                         info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),'R2':lambda x: "{:.2f}".format(x.rsquared)})
    if show == True:
        print(output)
    elif show == False:
        begintex1 = "\\documentclass{report}"
        begintex2 = "\\begin{document}"
        endtex = "\end{document}"
        f = open('tex_files/table4_raw.tex', 'w')
        f.write(begintex1)
        f.write(begintex2)
        f.write(output.as_latex())
        f.write(endtex)
        f.close()



def table_x_extension(show):
    '''
    Function estimates reduced-form relationship for sub-samples based 
    on agricultural GDP shares by means of OLS with standard errors
    clustered at country level. Output is Tex-file used to produce table below.
    '''
    df = get_panel_dataset()
    agri_q1 = df['agri_gdp_av'].quantile(q=0.25)-0.0001
    agri_q2 = df['agri_gdp_av'].quantile(q=0.5)-0.0001
    agri_q3 = df['agri_gdp_av'].quantile(q=0.75)-0.0001
    agri_q4 = df['agri_gdp_av'].max()

    df_1 = df.copy()
    df_1 = df_1[df_1['agri_gdp_av']<agri_q1]
    df_1_int = df_1.dropna(subset=['trans_democ'])
    df_2 = df.copy()
    df_2 = df_2[df_2['agri_gdp_av']>agri_q1]
    df_2 = df_2[df_2['agri_gdp_av']<agri_q2]
    df_2_int = df_2.dropna(subset=['trans_democ'])
    df_3 = df.copy()
    df_3 = df_3[df_3['agri_gdp_av']>agri_q2]
    df_3 = df_3[df_3['agri_gdp_av']<agri_q3]
    df_3_int = df_3.dropna(subset=['trans_democ'])
    df_4 = df.copy()
    df_4 = df_4[df_4['agri_gdp_av']>agri_q3]
    df_4 = df_4[df_4['agri_gdp_av']<agri_q4]
    df_4_int = df_4.dropna(subset=['trans_democ'])

    spec_first_stage = 'lgdp_l2 ~ lgpcp_l2 + lgpcp_l3 + C(ccode) + C(ccode) : I(year) + C(year)'
    spec_reduc_form_1 = 'polity_change ~ lgpcp_l + lgpcp_l2 + C(ccode) + C(ccode) : I(year) + C(year)'
    spec_reduc_form_2 = 'trans_democ ~ lgpcp_l + lgpcp_l2 + C(ccode) + C(ccode) : I(year) + C(year)'
    
    rslt_1 = smf.ols(formula=spec_first_stage,data=df_1).fit(cov_type='cluster',cov_kwds={'groups':df_1['ccode']})
    rslt_2 = smf.ols(formula=spec_reduc_form_1,data=df_1).fit(cov_type='cluster',cov_kwds={'groups':df_1['ccode']})
    rslt_3 = smf.ols(formula=spec_reduc_form_2,data=df_1_int).fit(cov_type='cluster',cov_kwds={'groups':df_1_int['ccode']})    
    rslt_4 = smf.ols(formula=spec_first_stage,data=df_2).fit(cov_type='cluster',cov_kwds={'groups':df_2['ccode']})
    rslt_5 = smf.ols(formula=spec_reduc_form_1,data=df_2).fit(cov_type='cluster',cov_kwds={'groups':df_2['ccode']})
    rslt_6 = smf.ols(formula=spec_reduc_form_2,data=df_2_int).fit(cov_type='cluster',cov_kwds={'groups':df_2_int['ccode']})
    
    output_1 = summary_col([rslt_1,rslt_2,rslt_3,rslt_4,rslt_5,rslt_6], model_names=['(1)','(2)','(3)','(4)','(5)','(6)'],
                         stars=True, float_format='%0.3f', 
                         regressor_order=['lgpcp_l', 'lgpcp_l2', 'lgpcp_l3'],
                         info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),'R2':lambda x: "{:.2f}".format(x.rsquared)})
    
    rslt_7 = smf.ols(formula=spec_first_stage,data=df_3).fit(cov_type='cluster',cov_kwds={'groups':df_3['ccode']})
    rslt_8 = smf.ols(formula=spec_reduc_form_1,data=df_3).fit(cov_type='cluster',cov_kwds={'groups':df_3['ccode']})
    rslt_9 = smf.ols(formula=spec_reduc_form_2,data=df_3_int).fit(cov_type='cluster',cov_kwds={'groups':df_3_int['ccode']})    
    rslt_10 = smf.ols(formula=spec_first_stage,data=df_4).fit(cov_type='cluster',cov_kwds={'groups':df_4['ccode']})
    rslt_11 = smf.ols(formula=spec_reduc_form_1,data=df_4).fit(cov_type='cluster',cov_kwds={'groups':df_4['ccode']})
    rslt_12 = smf.ols(formula=spec_reduc_form_2,data=df_4_int).fit(cov_type='cluster',cov_kwds={'groups':df_4_int['ccode']})
    
    output_2 = summary_col([rslt_7,rslt_8,rslt_9,rslt_10,rslt_11,rslt_12], model_names=['(7)','(8)','(9)','(10)','(11)','(12)'],
                         stars=True, float_format='%0.3f', 
                         regressor_order=['lgpcp_l', 'lgpcp_l2', 'lgpcp_l3'],
                         info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),'R2':lambda x: "{:.2f}".format(x.rsquared)})
    if show == True:
        print(output_1)
        print(output_2)
    elif show == False:
        begintex1 = "\\documentclass{report}"
        begintex2 = "\\begin{document}"
        endtex = "\end{document}"
        f = open('tex_files/table5_raw.tex', 'w')
        f.write(begintex1)
        f.write(begintex2)
        f.write(output_1.as_latex())
        f.write(output_2.as_latex())
        f.write(endtex)
        f.close()



def table_elections(show):
    df = get_panel_dataset()
    df_int = df.copy()
    df_int = df_int.dropna(subset=['exconst_change'])
    df_dt = df.copy()
    df_dt = df_dt.dropna(subset=['trans_democ'])
    
    spec_first_stage = 'lgdp_l2 ~ lgpcp_l2 + C(ccode) + C(ccode) : I(year) + C(year)'                           
    rslt = smf.ols(formula=spec_first_stage, data=df).fit()
    rslt_int = smf.ols(formula=spec_first_stage, data=df_int).fit()
    rslt_dt = smf.ols(formula=spec_first_stage, data=df_dt).fit()

    df['lgdp_l_hat'] = rslt.predict()
    df_int['lgdp_l_hat'] = rslt_int.predict()
    df_dt['lgdp_l_hat'] = rslt_dt.predict()
    df['lgdp_l_hat_elect'] = df['lgdp_l_hat'] * df['election']
    df_int['lgdp_l_hat_elect'] = df_int['lgdp_l_hat'] * df_int['election']
    df_dt['lgdp_l_hat_elect'] = df_dt['lgdp_l_hat'] * df_dt['election']

    spec_second_stage_1 = 'polity_change ~ lgdp_l_hat + C(ccode) + C(ccode) : I(year) + C(year)'
    spec_second_stage_2 = 'polity_change ~ lgdp_l_hat + election + C(ccode) + C(ccode) : I(year) + C(year)'
    spec_second_stage_3 = 'polity_change ~ lgdp_l_hat + lgdp_l_hat_elect + election + C(ccode) + C(ccode) : I(year) + C(year)'
    spec_second_stage_4 = 'exconst_change ~ lgdp_l_hat + lgdp_l_hat_elect + election + C(ccode) + C(ccode) : I(year) + C(year)'
    spec_second_stage_5 = 'polcomp_change ~ lgdp_l_hat + lgdp_l_hat_elect + election + C(ccode) + C(ccode) : I(year) + C(year)'
    spec_second_stage_6 = 'exrec_change ~ lgdp_l_hat + lgdp_l_hat_elect + election + C(ccode) + C(ccode) : I(year) + C(year)'
    spec_second_stage_7 = 'trans_democ ~ lgdp_l_hat + lgdp_l_hat_elect + election + C(ccode) + C(ccode) : I(year) + C(year)'

    rslt_1 = smf.ols(formula=spec_second_stage_1,data=df).fit(cov_type='cluster',cov_kwds={'groups':df['ccode']})
    rslt_2 = smf.ols(formula=spec_second_stage_2,data=df).fit(cov_type='cluster',cov_kwds={'groups':df['ccode']})
    rslt_3 = smf.ols(formula=spec_second_stage_3,data=df).fit(cov_type='cluster',cov_kwds={'groups':df['ccode']})
    rslt_4 = smf.ols(formula=spec_second_stage_4,data=df_int).fit(cov_type='cluster',cov_kwds={'groups':df_int['ccode']})
    rslt_5 = smf.ols(formula=spec_second_stage_5,data=df_int).fit(cov_type='cluster',cov_kwds={'groups':df_int['ccode']})
    rslt_6 = smf.ols(formula=spec_second_stage_6,data=df_int).fit(cov_type='cluster',cov_kwds={'groups':df_int['ccode']})
    rslt_7 = smf.ols(formula=spec_second_stage_7,data=df_dt).fit(cov_type='cluster',cov_kwds={'groups':df_dt['ccode']})

    output_1 = summary_col([rslt_1,rslt_2,rslt_3], model_names=['(1)','(2)','(3)'],
                            stars=True, float_format='%0.3f',
                            regressor_order=['lgdp_l_hat','lgdp_l_hat_elect','election'],
                            info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),'R2':lambda x: "{:.2f}".format(x.rsquared)})

    output_2 = summary_col([rslt_4,rslt_5,rslt_6,rslt_7], model_names=['(1)','(2)','(3)','(4)'],
                            stars=True, float_format='%0.3f',
                            regressor_order=['lgdp_l_hat','lgdp_l_hat_elect','election'],
                            info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),'R2':lambda x: "{:.2f}".format(x.rsquared)})
    if show == True:
        print(output_1)
        print(output_2)
    elif show == False:
        begintex1 = "\\documentclass{report}"
        begintex2 = "\\begin{document}"
        endtex = "\end{document}"
        f = open('tex_files/table6_raw.tex', 'w')
        f.write(begintex1)
        f.write(begintex2)
        f.write(output_1.as_latex())
        f.write(output_2.as_latex())
        f.write(endtex)
        f.close()


def table_mil_gov(show):
    df = get_panel_dataset()
    spec_first_stage = 'lgdp_l2 ~ lgpcp_l2 + C(ccode) + C(ccode) : I(year) + C(year)'
    rslt = smf.ols(formula=spec_first_stage, data=df).fit()
    df_temp = df.copy()
    df_temp['lgdp_l_hat'] = rslt.predict()
    df_temp['lgdp_l_hat_elect'] = df_temp['lgdp_l_hat'] * df_temp['election']

    df_temp['lgov_change'] = df_temp['lgov'] - df_temp['lgov_l']
    df_temp['lmil_change'] = df_temp['lmil'] - df_temp['lmil_l']
    df_temp.loc[df['year']==1981, 'lgov_change'] = np.NaN
    df_temp.loc[df['year']==1981, 'lmil_change'] = np.NaN

    df_gov = df_temp.dropna(subset=['lgov_change'])
    df_mil = df_temp.dropna(subset=['lmil_change'])

    spec_second_stage_1 = 'lgov_change ~ lgdp_l_hat + C(ccode) + C(ccode) : I(year) + C(year)'
    spec_second_stage_2 = 'lgov_change ~ lgdp_l_hat + lgdp_l_hat_elect + election + C(ccode) + C(ccode) : I(year) + C(year)'
    spec_second_stage_3 = 'lmil_change ~ lgdp_l_hat + C(ccode) + C(ccode) : I(year) + C(year)'
    spec_second_stage_4 = 'lmil_change ~ lgdp_l_hat + lgdp_l_hat_elect + election + C(ccode) + C(ccode) : I(year) + C(year)'

    rslt_1 = smf.ols(formula=spec_second_stage_1, data=df_gov).fit(cov_type='cluster', cov_kwds={'groups':df_gov['ccode']})
    rslt_2 = smf.ols(formula=spec_second_stage_2, data=df_gov).fit(cov_type='cluster', cov_kwds={'groups':df_gov['ccode']})
    rslt_3 = smf.ols(formula=spec_second_stage_3, data=df_mil).fit(cov_type='cluster', cov_kwds={'groups':df_mil['ccode']})
    rslt_4 = smf.ols(formula=spec_second_stage_4, data=df_mil).fit(cov_type='cluster', cov_kwds={'groups':df_mil['ccode']})

    output = summary_col([rslt_1,rslt_2,rslt_3,rslt_4], model_names=['(1)','(2)','(3)','(4)'],
                             stars=True, float_format='%0.3f', 
                             regressor_order=['lgdp_l_hat','lgdp_l_hat_elect','election'],
                             info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),'R2':lambda x: "{:.2f}".format(x.rsquared)})
    if show == True:
        print(output)
    elif show == False:
        begintex1 = "\\documentclass{report}"
        begintex2 = "\\begin{document}"
        endtex = "\end{document}"
        f = open('tex_files/table8_raw.tex', 'w')
        f.write(begintex1)
        f.write(begintex2)
        f.write(output.as_latex())
        f.write(endtex)
        f.close()



def table_mean_revert(show):
    df = get_panel_dataset()
    df_temp = df.copy()
    df_temp['lgpcp_change'] = df_temp['lgpcp_l'] - df_temp['lgpcp_l2']
    spec_1 = 'lgpcp_l ~ lgpcp_l2 + C(ccode)'
    spec_2 = 'lgpcp_l ~ lgpcp_l2 + C(ccode) + C(ccode) : I(year)'  

    rslt_1 = smf.ols(formula=spec_1, data=df).fit(cov_type='cluster',cov_kwds={'groups':df['ccode']})
    rslt_2 = smf.ols(formula=spec_2, data=df).fit(cov_type='cluster',cov_kwds={'groups':df['ccode']})

    output = summary_col([rslt_1,rslt_2], model_names=['(1)','(2)'],
                                 stars=True, float_format='%0.3f',
                                 regressor_order=['lgpcp_l2'],
                                 info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),'R2':lambda x: "{:.2f}".format(x.rsquared)})
    if show == True:
        print(output)
    elif show == False:
        begintex1 = "\\documentclass{report}"
        begintex2 = "\\begin{document}"
        endtex = "\end{document}"
        f = open('tex_files/table9_raw.tex', 'w')
        f.write(begintex1)
        f.write(begintex2)
        f.write(output.as_latex())
        f.write(endtex)
        f.close()














