DATA_PATH = "../data"

def bab(i):
    import numpy as np
    import datetime
    import pandas as pd
    import matplotlib.pyplot as plt
    import datetime
    #import wrds
    import utils
    import seaborn as sns

    pd.options.mode.chained_assignment = None
    #data = pd.read_parquet('slow_part.parquet')
    data = pd.read_parquet(f'{DATA_PATH}/slow_part.parquet')
    import re
    #with open('Siccodes12.txt', 'r') as file:
    with open(f'{DATA_PATH}/Siccodes12.txt', 'r') as file:
        lines = file.readlines()



    ff12_mapping = []
    current_ff12 = None

    for line in lines:
        
        category_match = re.match(r'^\s*(\d+)\s+\w+', line)
        if category_match:
            current_ff12 = int(category_match.group(1))
        else:
            
            interval_match = re.match(r'^\s*(\d+)-(\d+)', line)
            if interval_match:
                start = int(interval_match.group(1))
                end = int(interval_match.group(2))
                ff12_mapping.append((start, end, current_ff12))

    def map_siccd_to_ff12(siccd):
        for start, end, ff12 in ff12_mapping:
            if start <= siccd <= end:
                return ff12
        return None


    data['FF12'] = data['siccd'].apply(map_siccd_to_ff12)
    #data['FF12'].fillna(12, inplace=True)
    data['FF12'] = data['FF12'].fillna(12)



    data = data[data['FF12'] == i]

    df = data
    average_beta_permno = df.groupby('permno')['beta'].mean()

    df['month'] = df['date'].dt.to_period('M')
    average_beta_month = df.groupby('month')['beta'].mean()
    average_beta_month_sorted = average_beta_month.sort_index()

    def annualize_return(monthly_return):
        return ((1 + monthly_return)**12 - 1)

    def sharpe_ratio(mean_return, std_dev, risk_free_rate):
        return (mean_return - risk_free_rate) / std_dev

    data = data.dropna(subset=['beta']).copy()
    data['beta_Q'] = data.groupby('date')['beta'].transform(lambda x: pd.qcut(x, 10, labels=False, duplicates='drop'))

    data['vw_Q'] = data['mcap'] / data.groupby(['date', 'beta_Q'])['mcap'].transform('sum')

    data['beta_ret_vw'] = data['vw_Q'] * data['Rn']
    ret_vw = data.groupby(['date', 'beta_Q'])['beta_ret_vw'].sum().reset_index()
    vw_ret_mean = ret_vw.groupby('beta_Q')['beta_ret_vw'].mean()
    vw_ret_std = ret_vw.groupby('beta_Q')['beta_ret_vw'].std()

    ret_ew = data.groupby(['date', 'beta_Q'])['Rn'].mean().reset_index()
    ew_ret_mean = ret_ew.groupby('beta_Q')['Rn'].mean()
    ew_ret_std = ret_ew.groupby('beta_Q')['Rn'].std()

    vw_ret_mean_annual = annualize_return(vw_ret_mean)
    ew_ret_mean_annual = annualize_return(ew_ret_mean)

    vw_ret_std_annual = vw_ret_std * np.sqrt(12)
    ew_ret_std_annual = ew_ret_std * np.sqrt(12)
    risk_free_rate_annual = annualize_return(data['rf'].mean())

    vw_sharpe_ratios = sharpe_ratio(vw_ret_mean_annual, vw_ret_std_annual,risk_free_rate_annual)
    ew_sharpe_ratios = sharpe_ratio(ew_ret_mean_annual, ew_ret_std_annual,risk_free_rate_annual)

    deciles = ['D' + str(x) for x in range(1, 11)]






    # *c)*

    # In[ ]:


    # Weights
    data['z'] = data.groupby('date')['beta'].transform(lambda x: x.rank())
    data['z_'] = data['z']-data.groupby('date')['z'].transform('mean')
    data['k'] = np.abs(data['z_'])
    data['k'] = 2/data.groupby('date')['k'].transform('sum')
    data['w_H'] = data['k'] * data['z_'] * (data['z_']>0)
    data['w_L'] = -data['k'] * data['z_'] * (data['z_']<0)




    # d)

    # In[21]:


    # Weighted returns and beta
    data['beta_H'] = data['w_H'] * data['beta']
    data['beta_L'] = data['w_L'] * data['beta']
    data['R_H'] = data['w_H'] * data['Rn']
    data['R_L'] = data['w_L'] * data['Rn']
    data['R_H_e'] = data['w_H'] * data['Rn_e']
    data['R_L_e'] = data['w_L'] * data['Rn_e']
    BAB = data.groupby('date')[['R_H','R_L','R_H_e','R_L_e','beta_H','beta_L']].sum().reset_index()
    df_Rf = data[['rf','date','Rm']]
    df_Rf['date'] = df['date'].dt.strftime('%Y-%m')

    BAB['BAB2'] = BAB['R_L_e']/BAB['beta_L'] - BAB['R_H_e']/BAB['beta_H']
    BAB_final = BAB.drop(columns=['R_H','R_L','R_H_e','R_L_e','beta_H','beta_L'])
    BAB_final['date'] = BAB_final['date'].dt.strftime('%Y-%m')
    BAB_final['BAB'] = BAB_final['BAB2']
    BAB_final = BAB_final.merge(df_Rf, on='date', how='left')
    BAB_final = BAB_final.drop(columns=['BAB2','rf','Rm'])
    BAB_final = BAB_final.drop_duplicates(subset=['BAB'])


    return BAB_final


def mom(i):

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from scipy import stats

    from utils import plot_metrics

    # Load the data
    import re
    #data = pd.read_parquet('stock_data.parquet')
    data = pd.read_parquet(f'{DATA_PATH}/stock_data.parquet')
    #with open('Siccodes12.txt', 'r') as file:
    with open(f'{DATA_PATH}/Siccodes12.txt', 'r') as file:
        lines = file.readlines()


    ff12_mapping = []
    current_ff12 = None

    for line in lines:
        
        category_match = re.match(r'^\s*(\d+)\s+\w+', line)
        if category_match:
            current_ff12 = int(category_match.group(1))
        else:
            
            interval_match = re.match(r'^\s*(\d+)-(\d+)', line)
            if interval_match:
                start = int(interval_match.group(1))
                end = int(interval_match.group(2))
                ff12_mapping.append((start, end, current_ff12))

    def map_siccd_to_ff12(siccd):
        for start, end, ff12 in ff12_mapping:
            if start <= siccd <= end:
                return ff12
        return None


    data['FF12'] = data['siccd'].apply(map_siccd_to_ff12)
    #data['FF12'].fillna(12, inplace=True)
    data['FF12'] = data['FF12'].fillna(12)
    data = data[data['FF12'] == i]

    data = data[data.permno.isin(data.permno.unique()[:100])]

    
    data.shape


    # In[30]:


    from tqdm.auto import tqdm  # for notebooks
    tqdm.pandas()

    # Ensure date is in datetime format
    data['date'] = pd.to_datetime(data['date'])

    # Calculate cumulative returns for each stock over the past 11 months
    data['cum_Rn'] = data.groupby('permno')['Rn'].rolling(11).progress_apply(lambda x: np.prod(1 + x) - 1).reset_index(level=0, drop=True)

    # Remove rows with NaN cumulative returns
    data = data.dropna(subset=['cum_Rn'])

    # Sort stocks into deciles based on cumulative return
    data['decile'] = data.groupby('date')['cum_Rn'].transform(lambda x: pd.qcut(x, 10, labels=False, duplicates = 'drop') + 1)


    


    # In[31]:


    # Compute equal-weighted returns for each decile
    ew_returns = data.groupby(['date', 'decile'])['Rn'].mean().unstack()



    # In[32]:


    # Compute value-weighted returns for each decile
    vw_returns = data.groupby(['date', 'decile']).apply(lambda x: np.average(x['Rn'], weights=x['mcap'])).unstack()

    


    # *b) The momentum strategy is then the portfolio that goes long the three highest deciles and short the three lowest decile portfolios. Compute and compare the mean, stan- dard deviation, and Sharpe ratios of the long and short legs of the strategy as well as of the strategy itself. Test if the strategy has an average return that is statistically significantly different from zero. Repeat both tests for equal and value-weighted portfolios.*

    # In[33]:


    # Long-Short Momentum Portfolio
    long_ew = ew_returns.loc[:, 8:10].sum(axis=1)
    short_ew = - ew_returns.loc[:, 0:2].sum(axis=1)
    mom_ew = long_ew + short_ew

    

    # Statistical significance testing
    t_stat_equal, p_value_equal = stats.ttest_1samp(mom_ew, 0)
    print(f"Equal-Weighted Long-Short Strategy: t-stat={t_stat_equal}, p-value={p_value_equal}")


    # In[34]:


    # Step 4: Long-Short Momentum Portfolio
    long_vw = vw_returns.loc[:, 8:10].sum(axis=1)
    short_vw = - vw_returns.loc[:, 0:2].sum(axis=1)
    mom_vw = long_vw + short_vw

    

    # Statistical significance testing
    t_stat_value, p_value_value = stats.ttest_1samp(mom_vw, 0)
    print(f"Value-Weighted Long-Short Strategy: t-stat={t_stat_value}, p-value={p_value_value}")


    # In[35]:


    mom_df = mom_vw.reset_index().rename(columns={0: 'MoM'})
    mom_df['date'] = mom_df['date'].dt.to_period('M')
    return mom_df


def iv(i):


    import numpy as np
    import datetime
    import pandas as pd
    import matplotlib.pyplot as plt
    import datetime
    #import wrds
    import seaborn as sns
    from statsmodels.regression.rolling import RollingOLS
    import statsmodels.api as sm
    from scipy.stats import ttest_1samp
    from scipy.stats import mstats

    sns.set_theme(style='whitegrid')


    from utils import plot_strategy_performance


    # ---
    # # Load the data

    # In[7]:


    # Complete data
    #data = pd.read_parquet('stock_data.parquet')
    data = pd.read_parquet(f'{DATA_PATH}/stock_data.parquet')
    import re
    #with open('Siccodes12.txt', 'r') as file:
    with open(f'{DATA_PATH}/Siccodes12.txt', 'r') as file:
        lines = file.readlines()


    ff12_mapping = []
    current_ff12 = None

    for line in lines:
        
        category_match = re.match(r'^\s*(\d+)\s+\w+', line)
        if category_match:
            current_ff12 = int(category_match.group(1))
        else:
            
            interval_match = re.match(r'^\s*(\d+)-(\d+)', line)
            if interval_match:
                start = int(interval_match.group(1))
                end = int(interval_match.group(2))
                ff12_mapping.append((start, end, current_ff12))

    def map_siccd_to_ff12(siccd):
        for start, end, ff12 in ff12_mapping:
            if start <= siccd <= end:
                return ff12
        return None


    data['FF12'] = data['siccd'].apply(map_siccd_to_ff12)
    #data['FF12'].fillna(12, inplace=True)
    data['FF12'] = data['FF12'].fillna(12)
    data = data[data['FF12'] == i]

    data = data[data.permno.isin(data.permno.unique()[:100])]

    
    data.shape


    # In[9]:


    data['date'] = pd.to_datetime(data['date'])  
    


    # ---
    # # Idiosyncratic Volatility Strategy (IV)
    # 
    # (a) Compute the time-varying estimate for each stock’s idiosyncratic volatility $ σ^{idio}_{t,n} $ obtained as the volatility of the residual in the monthly rolling 5-year regressions of
    # stock-specific excess returns on the excess market return. Require at least 36 months
    # of observations for each stock. Winsorize the volatility at 5 and 95 %.

    # In[10]:


    # Remove any potential NaN values
    data = data.dropna(subset=['mcap_l','Rn_e','Rm_e']).copy()

    # Remove rare stocks with less than 60 observations
    data['N'] = data.groupby(['permno'])['date'].transform('count')
    data = data[data['N']>60].copy()

    # Calculate the rolling covariance matrix
    cov_nm = data.set_index('date').groupby('permno')[['Rn_e', 'Rm_e']].rolling(60, min_periods=36).cov()

    # Extract the required components
    cov_ee = cov_nm.iloc[0::2, 0].droplevel(2)
    cov_em = cov_nm.iloc[0::2, 1].droplevel(2)
    cov_mm = cov_nm.iloc[1::2, 1].droplevel(2)

    # Calculate idio vol
    beta = cov_em / cov_mm
    idio_variance = cov_ee - beta**2 * cov_mm
    idio_volatility = np.sqrt(idio_variance).dropna()

    # Add the idiosyncratic volatility to the original dataframe
    data = data.set_index(['permno', 'date'])
    data['sigma_idio'] = idio_volatility

    # Reset index to merge correctly
    data = data.reset_index()

    # Winsorize the idiosyncratic volatility at 5% and 95%
    data['sigma_idio'] = data['sigma_idio'].clip(data['sigma_idio'].quantile(0.05),data['sigma_idio'].quantile(0.95))


    # In[11]:


    


    # - Look up on the volatilities

    # In[12]:


    # Select five unique stocks for plotting
    selected_stocks = data['permno'].unique()[:25]

    # Filter data for the selected stocks
    filtered_data = data[data['permno'].isin(selected_stocks)]

    


    # (b) At every month t, sort all stocks into deciles based on their idiosyncratic volatility
    # (estimated using the most recent rolling window). Then compute monthly returns for
    # 10 decile portfolios that equal weight all stocks in each decile. Plot the average annu-
    # alized portfolio mean, standard deviation, and Sharpe ratios across the 10 deciles in
    # three barplots. Repeat for value-weighted decile portfolios. Summarize your findings.
    # Is the evidence consistent with the CAPM?

    # In[13]:


    # Sort stocks into deciles based on idiosyncratic volatility at each month
    data = data.dropna(subset=['sigma_idio']).copy()
    data['sigma_idio_decile'] = data.groupby('date')['sigma_idio'].transform(lambda x: pd.qcut(x, 10, labels=False, duplicates='drop'))

    # Compute market weights within quintiles
    data['vw_Q'] = data['mcap'] / data.groupby(['date', 'sigma_idio_decile'])['mcap'].transform('sum')

    # Compute value-weighted returns for each decile
    data['idio_ret_vw'] = data['vw_Q'] * data['Rn_e']
    ret_vw = data.groupby(['date', 'sigma_idio_decile'])['idio_ret_vw'].sum().reset_index()
    vw_ret_mean = ret_vw.groupby('sigma_idio_decile')['idio_ret_vw'].mean()

    # Compute equal-weighted returns for each decile
    ret_ew = data.groupby(['date', 'sigma_idio_decile'])['Rn_e'].mean().reset_index()
    ew_ret_mean = ret_ew.groupby('sigma_idio_decile')['Rn_e'].mean()


    # In[14]:


    # Calculate annualized portfolio statistics
    def calculate_stats(returns):
        mean_return = returns.mean() * 12
        std_dev = returns.std() * np.sqrt(12)
        sharpe_ratio = mean_return / std_dev
        return mean_return, std_dev, sharpe_ratio

    # Compute value-weighted portfolio statistics
    vw_stats = ret_vw.groupby('sigma_idio_decile').apply(lambda x: calculate_stats(x['idio_ret_vw']))

    # Compute equal-weighted portfolio statistics
    ew_stats = ret_ew.groupby('sigma_idio_decile').apply(lambda x: calculate_stats(x['Rn_e']))


    # In[15]:


   


    # In[16]:


    


    # (c) Now we construct the idiosyncratic volatility factor. At every month t, we go long
    # the three highest decile volatility portfolios and we go short the three lowest decile
    # volatility portfolios. Compute and compare the mean, standard deviation, and Sharpe
    # ratios of the long and short legs of the strategy as well as of the strategy itself. Test if
    # the strategy has an average return that is statistically significantly different from zero.
    # Repeat both tests for equal and value-weighted portfolios. How do your results differ
    # from Ang, Hodrick, Xing, and Zhang (2006; table VI page 285) and what may be the
    # explanation for the difference?

    # In[17]:


    #  Select stocks for long and short legs
    long_leg_data = data[data['sigma_idio_decile'] >= 7]
    short_leg_data = data[data['sigma_idio_decile'] < 3]

    # Compute portfolio returns for long and short legs
    long_leg_returns_vw = long_leg_data.groupby('date')['idio_ret_vw'].sum()
    short_leg_returns_vw = short_leg_data.groupby('date')['idio_ret_vw'].sum()
    long_leg_returns_ew = long_leg_data.groupby('date')['Rn_e'].mean()
    short_leg_returns_ew = short_leg_data.groupby('date')['Rn_e'].mean()

    # Calculate factor returns
    factor_returns_vw = long_leg_returns_vw - short_leg_returns_vw
    factor_returns_ew = long_leg_returns_ew - short_leg_returns_ew

    # Factor returns
    factor_mean_return_vw = factor_returns_vw.mean() * 12
    factor_std_dev_vw = factor_returns_vw.std() * np.sqrt(12)
    factor_sharpe_ratio_vw = factor_mean_return_vw / factor_std_dev_vw

    factor_mean_return_ew = factor_returns_ew.mean() * 12
    factor_std_dev_ew = factor_returns_ew.std() * np.sqrt(12)
    factor_sharpe_ratio_ew = factor_mean_return_ew / factor_std_dev_ew

    # Long and Short legs
    long_mean_return_vw = long_leg_returns_vw.mean() * 12
    long_std_dev_vw = long_leg_returns_vw.std() * np.sqrt(12)
    long_sharpe_ratio_vw = long_mean_return_vw / long_std_dev_vw

    short_mean_return_vw = short_leg_returns_vw.mean() * 12
    short_std_dev_vw = short_leg_returns_vw.std() * np.sqrt(12)
    short_sharpe_ratio_vw = short_mean_return_vw / short_std_dev_vw

    long_mean_return_ew = long_leg_returns_ew.mean() * 12
    long_std_dev_ew = long_leg_returns_ew.std() * np.sqrt(12)
    long_sharpe_ratio_ew = long_mean_return_ew / long_std_dev_ew

    short_mean_return_ew = short_leg_returns_ew.mean() * 12
    short_std_dev_ew = short_leg_returns_ew.std() * np.sqrt(12)
    short_sharpe_ratio_ew = short_mean_return_ew / short_std_dev_ew


    # Significance test
    t_stat_vw, p_value_vw = ttest_1samp(factor_returns_vw, 0)
    t_stat_ew, p_value_ew = ttest_1samp(factor_returns_ew, 0)





    stats_data = pd.DataFrame({
        'Metric': ['Mean Return', 'Standard Deviation', 'Sharpe Ratio'],
        'Factor Returns (Value-Weighted)': [factor_mean_return_vw, factor_std_dev_vw, factor_sharpe_ratio_vw],
        'Factor Returns (Equal-Weighted)': [factor_mean_return_ew, factor_std_dev_ew, factor_sharpe_ratio_ew],
        'Long Leg (Value-Weighted)': [long_mean_return_vw, long_std_dev_vw, long_sharpe_ratio_vw],
        'Short Leg (Value-Weighted)': [short_mean_return_vw, short_std_dev_vw, short_sharpe_ratio_vw],
        'Long Leg (Equal-Weighted)': [long_mean_return_ew, long_std_dev_ew, long_sharpe_ratio_ew],
        'Short Leg (Equal-Weighted)': [short_mean_return_ew, short_std_dev_ew, short_sharpe_ratio_ew]
    })
    stats_data


    # In[20]:


    # Stats metrics
    metrics = ['Mean Return', 'Standard Deviation', 'Sharpe Ratio']

    
    # In[22]:


    # Load data for next section 
    IV = factor_returns_vw.reset_index().rename(columns={'idio_ret_vw': 'IV'})
    IV['date'] = IV['date'].dt.to_period('M')

    return IV














