from enum import unique
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

raw_files = os.listdir('orbis_data/') # get list of raw input files 

df = pd.DataFrame([])

# loop merging over available raw files from a directory:
# csv would be faster but the Orbis database gives xlsx
for file in enumerate(raw_files):
    print('Reading file: ' + file[1] + ' ...')
    xlsx_file = pd.ExcelFile('orbis_data/' + file[1])
    file_df   = pd.read_excel(xlsx_file, 'Results')
    df        = pd.concat([df, file_df], ignore_index=True)
    
# a dictionary of new variable names
new_colnames = {'Company name Latin alphabet': 'comp_name', 
                    'Country ISO code':                   'country',
                    'NACE Rev. 2, core code (4 digits)':  'nace',
                    'Last avail. year':                   'last_year', 
                    'Number of employees\nLast avail. yr':'n_employees',
                    'Number of employees\nYear - 1': 'n_employees_lag',
                    'Status date': 'status_date',
                    'BvD sectors': 'sector',
                    'National industry classification': 'nat_industry_class',
                    'Type of entity': 'entity',
                    'P/L for period [=Net income]\nth EUR Year - 1': 'pnl',
                    'P/L for period [=Net income]\nth EUR Year - 2': 'pnl_lag',
                    'Total assets\nth EUR Year - 1': 'assets',
                    'Total assets\nth EUR Year - 2': 'assets_lag',
                    'Fixed assets\nth EUR Year - 1': 'fixed_assets',
                    'Fixed assets\nth EUR Year - 2': 'fixed_assets_lag',
                    'Current assets\nth EUR Year - 1': 'curr_assets',
                    'Current assets\nth EUR Year - 2': 'curr_assets_lag',
                    'Capital\nth EUR Year - 1': 'capital',
                    'Long term debt\nth EUR Year - 1': 'lt_debt',
                    'Current liabilities\nth EUR Year - 1': 'curr_liab',
                    'Cash & cash equivalent\nth EUR Year - 1': 'cash',
                    'Sales\nth EUR Year - 1': 'revenue',
                    'Sales\nth EUR Year - 2': 'revenue_lag',
                    'Operating P/L [=EBIT]\nth EUR Year - 1': 'ebit',
                    'Operating P/L [=EBIT]\nth EUR Year - 2': 'ebit_lag',
                    'P/L for period [=Net income]\nth EUR Year - 1.1': 'pnl2',
                    'P/L for period [=Net income]\nth EUR Year - 2.1': 'pnl2_lag',
                    'Net Income/Starting Line\nth EUR Year - 1': 'cashflow',
                    'Net Income/Starting Line\nth EUR Year - 2': 'cashflow_lag'} 

# ebit_ratio1:  ebit/assets
# ebit_ratio2: ebit/capital
# ebit_ratio3: ebit/revenue
# pnl_ratio1: pnl/assets
# liab_ratio1:  liab / assets
# curr_assets_ratio1: curr_assets / curr_liab
# revenue_ratio1: sales / assets
# capital_ratio1: capital / assets
# pnl_ratio2: pnl / curr_liab
# pnl_ratio3: pnl / sales
# liab_ratio2: (liabs * 365) / pnl
# pnl_ratio4: pnl / liab
# curr_liab_ratio1: curr_liab / fixed_assets
# log(assets)
# liab_ratio3: (liab - cash) / sales
# curr_liab_ratio2: (curr_liab * 365) / (sales - pnl)
# revenue_ratio2: (sales - pnl) / curr_liab
# revenue_ratio3: (sales - pnl) / liab
# curr_assets_ratio2: curr_assets / liab
# capital_ratio2: capital / fixed_assets
# liab_ratio4: lt_liab / capital
# revenue_ratio4: sales / curr_liab
# revenue_ratio5: sales / fixed_assets
# curr_assets_ratio3: (curr_assets * 365)/sales
# capital_ratio3: capital / curr_liab

df.rename(columns=new_colnames, inplace=True)
df[df.isin(['n.a.', '-'])] = np.NaN
df.drop(['Unnamed: 0'], axis=1, inplace = True)
df['Inactive'] = np.where(df['Inactive'] == 'Yes', 1,0)


# deleting variables with more than 25% NAs
df.drop(df.columns[df.isna().sum() / df.shape[0] > 0.25], axis=1, inplace=True)
df.drop(['Status', 'comp_name'], axis=1, inplace=True)

# new variables:
df['ebit_ratio1'] = df['ebit'] / np.where(df['assets'] == 0.001, np.NaN, df['assets'])
df['ebit_ratio2'] = df['ebit'] / np.where(df['capital'] == 0, 0.001, df['capital'])
df['ebit_ratio3'] = df['ebit'] / np.where(df['revenue'] == 0, 0.001, df['revenue'])
df['pnl_ratio1'] = df['pnl'] / np.where(df['assets'] == 0, 0.001, df['assets'])
df['liab_ratio1'] = (df['lt_debt'] + df['curr_liab']) / np.where(df['assets'] == 0, 0.001, df['assets'])
df['curr_assets_ratio1'] = df['curr_assets'] / np.where(df['curr_liab'] == 0, 0.001, df['curr_liab'])
df['revenue_ratio1'] = df['revenue'] / np.where(df['assets'] == 0, 0.001, df['assets'])
df['capital_ratio1'] = df['capital'] / np.where(df['assets'] == 0, 0.001, df['assets'])
df['pnl_ratio2'] = df['pnl'] / np.where(df['curr_liab'] == 0, 0.001, df['curr_liab'])
df['pnl_ratio3'] = df['pnl'] / np.where(df['revenue'] == 0, 0.001, df['revenue'])
df['liab_ratio2'] = (df['lt_debt'] + df['curr_liab'])*365 / np.where(df['pnl'] == 0, 0.001, df['pnl'])
df['pnl_ratio4'] = df['pnl'] / np.where((df['lt_debt'] + df['curr_liab']) == 0, 0.001, (df['lt_debt'] + df['curr_liab']))
df['curr_liab_ratio1'] = df['curr_liab'] / np.where(df['fixed_assets'] == 0, 0.001, df['fixed_assets'])
df['assets_log'] = np.log(np.where(df['assets'] == 0, 0.001, pd.to_numeric(df['assets'])))
df['liab_ratio3'] = (df['lt_debt'] + df['curr_liab'] - df['cash']) / np.where(df['revenue'] == 0, 0.001, df['revenue'])
df['curr_liab_ratio2'] = df['curr_liab'] * 365 / np.where((df['revenue'] - df['pnl']) == 0, 0.001, df['revenue'] - df['pnl'])
df['revenue_ratio2'] = (np.where(df['revenue'] == 0, 0.001, df['revenue']) - np.where(df['pnl'] == 0, 0.001, df['pnl'])) / np.where(df['curr_liab'] == 0, 0.001, df['curr_liab'])
df['revenue_ratio3'] = (np.where(df['revenue'] == 0, 0.001, df['revenue']) - np.where(df['pnl'] == 0, 0.001, df['pnl'])) / (np.where(df['curr_liab'] == 0, 0.001, df['curr_liab']) + np.where(df['lt_debt'] == 0, 0.001, df['lt_debt']))
df['curr_assets_ratio2'] = df['curr_assets'] / (np.where(df['curr_liab'] == 0, 0.001, df['curr_liab']) + np.where(df['lt_debt'] == 0, 0.001, df['lt_debt']))
df['capital_ratio2'] = df['capital'] / np.where(df['fixed_assets'] == 0, np.NaN, df['fixed_assets'])
df['liab_ratio4'] = df['lt_debt'] / np.where(df['capital'] == 0, np.NaN, df['capital'])
df['revenue_ratio4'] = df['revenue'] / np.where(df['curr_liab'] == 0, np.NaN, df['curr_liab'])
df['revenue_ratio5'] = df['revenue'] / np.where(df['fixed_assets'] == 0, np.NaN, df['fixed_assets'])
df['curr_assets_ratio3'] = df['curr_assets'] * 365 / np.where(df['revenue'] == 0, np.NaN, df['revenue'])

# period change variables
df['revenue_ch'] = df['revenue'] / np.where(df['revenue_lag'] == 0, np.NaN, df['revenue_lag'])
df['pnl_ch'] = df['pnl'] / np.where(df['pnl_lag'] == 0, np.NaN, df['pnl_lag'])
df['assets_ch'] = df['assets'] / pd.to_numeric(np.where(df['assets_lag'] == 0, np.NaN, df['assets_lag']))
df['ebit_ch'] = df['ebit'] / np.where(df['ebit_lag'] == 0, np.NaN, df['ebit_lag'])
df['capital_ratio3'] = df['capital'] / np.where(df['curr_liab'] == 0, 1, df['curr_liab'])

# one-hot encoding sector variable
sector_df = pd.DataFrame({'sector': df.sector.unique(),
                          'sector_id': ['sector_' + str(i) for i in range(df.sector.unique().shape[0])]})

for id in range(sector_df.shape[0]):
    df[sector_df.sector_id[id]] = np.where(df.sector == sector_df.sector[id], 1,0) 

# one-hot encoding country variables
for country_id in df.country.unique():
    df[country_id] = np.where(df.country == country_id,1,0)


