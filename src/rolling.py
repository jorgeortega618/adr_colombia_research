import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import pandas as pd
import numpy as np

def compute_rolling_betas(returns, windows=[126, 252, 504], cov_type='nonrobust'):
    """
    Computes rolling regressions for each ADR against S&P and COP=X.
    cov_type can be 'nonrobust' or 'HAC' (Newey-West).
    """
    adrs = ['EC', 'AVAL', 'CIB']
    # Exogenous variables
    X = returns[['^GSPC', 'COP=X']].copy()
    X = sm.add_constant(X)
    
    results = {}
    
    for adr in adrs:
        y = returns[adr]
        adr_results = {}
        for w in windows:
            if w > len(y):
                continue
                
            model = RollingOLS(endog=y, exog=X, window=w)
            if cov_type == 'HAC':
                # HAC standard errors - maxlags heuristic ~ 0.75 * window^(1/3) -> for 252 ~ 4.7
                maxlags = int(0.75 * (w ** (1/3)))
                res = model.fit(cov_type='HAC', cov_kwds={'maxlags': maxlags}, use_t=True)
            else:
                res = model.fit()
                
            params = res.params.copy()
            params.rename(columns={'const': 'alpha', '^GSPC': 'beta_sp500', 'COP=X': 'gamma_fx'}, inplace=True)
            adr_results[w] = params
            
        results[adr] = adr_results
        
    return results

def compute_rolling_correlation(returns, windows=[126, 252, 504]):
    """Computes rolling correlation between ADRs and S&P/FX"""
    adrs = ['EC', 'AVAL', 'CIB']
    results = {}
    for adr in adrs:
        adr_results = {}
        for w in windows:
            corr_sp = returns[adr].rolling(window=w).corr(returns['^GSPC'])
            corr_fx = returns[adr].rolling(window=w).corr(returns['COP=X'])
            adr_results[w] = pd.DataFrame({'corr_sp500': corr_sp, 'corr_fx': corr_fx})
        results[adr] = adr_results
    return results

def get_beta_summary(rolling_betas):
    """
    Produces a compact summary table of beta distributions and COVID sub-periods.
    """
    covid_start = '2020-02-01'
    covid_end = '2021-12-31'
    
    summary_data = []
    
    for adr, windows in rolling_betas.items():
        for w, df in windows.items():
            df_cur = df.dropna()
            if df_cur.empty: 
                continue
                
            beta = df_cur['beta_sp500']
            
            # Sub-period masking via string slice works if index is sorted DatetimeIndex
            idx_covid = (beta.index >= pd.to_datetime(covid_start)) & (beta.index <= pd.to_datetime(covid_end))
            covid_beta = beta.loc[idx_covid].mean() if idx_covid.any() else np.nan
            non_covid_beta = beta.loc[~idx_covid].mean() if (~idx_covid).any() else np.nan
            
            summary_data.append({
                'ADR': adr,
                'Window': w,
                'Beta Mean': beta.mean(),
                'Beta SD': beta.std(),
                'Beta Min': beta.min(),
                'Beta Max': beta.max(),
                'COVID Beta Mean': covid_beta,
                'Non-COVID Beta Mean': non_covid_beta,
                'Gamma Mean': df_cur['gamma_fx'].mean()
            })
            
    return pd.DataFrame(summary_data).round(4)
