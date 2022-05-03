import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

raw_files = os.listdir('orbis_data/orbis_raw') # get list of raw input files 

df = pd.DataFrame([])

# loop merging over available raw files from a directory:
for file in enumerate(raw_files):
    print('Reading file:' + file[1])
    xlsx_file = pd.ExcelFile('orbis_data/orbis_raw/' + file[1])
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

df.rename(columns=new_colnames, inplace=True)
df[df == 'n.a.'] = np.NaN

# feature engineering:
df['revenue_change'] = df['revenue'] / np.where(df['revenue_lag'] == 0, np.NaN, df['revenue_lag'])
df['Inactive'] = np.where(df['Inactive'] == 'Yes', 1,0)
df['ratio1'] = df['capital'] / np.where(df['curr_liab'] == 0, 1, df['curr_liab'])

# modeling:
model_df = df[['Inactive', 'ratio1', 'revenue_change']].dropna()

model_df = model_df.groupby('Inactive').sample(n = model_df['Inactive'].value_counts()[1])

ydat = model_df['Inactive'].astype(float)
xdat = model_df[['ratio1', 'revenue_change']].astype(float)

model = sm.Logit(endog = ydat, exog = xdat).fit()

# preliminary EDA
df.isna().sum() / df.shape[0] # share of NAs per column

df.set_index('country')['revenue'].mean(level = 'country')
df[['country', 'revenue']].groupby('country').agg(['mean', lambda x: x.size, 'size', np.mean])

df['country'].value_counts()

df.mean(level = 'country')

