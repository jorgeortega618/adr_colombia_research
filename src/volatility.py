import pandas as pd
import numpy as np
from arch import arch_model

def grid_search_garch(returns, dists=['normal', 't', 'ged']):
    """
    Runs bounded grid search for Ecopetrol residual ARCH issue across all ADRs.
    Selects optimal via BIC. 
    """
    # Exogenous variables for mean equation
    X = returns[['^GSPC', 'COP=X']].dropna()
    common_idx = returns.index.intersection(X.index)
    
    y = returns.loc[common_idx]
    X = X.loc[common_idx]
    
    models_to_test = [
        {'vol': 'Garch', 'p': 1, 'q': 1, 'o': 0},
        {'vol': 'Garch', 'p': 2, 'q': 1, 'o': 0},
        {'vol': 'Garch', 'p': 1, 'q': 2, 'o': 0},
        {'vol': 'Garch', 'p': 2, 'q': 2, 'o': 0},
        {'vol': 'EGARCH', 'p': 1, 'q': 1, 'o': 1},
        {'vol': 'Garch', 'p': 1, 'q': 1, 'o': 1} # GJR-GARCH
    ]
    
    results = []
    best_models = {}
    
    adrs = ['EC', 'AVAL', 'CIB']
    
    for adr in adrs:
        adr_y = y[adr]
        best_bic = np.inf
        best_res = None
        best_name = ""
        best_dist = ""
        
        for dist in dists:
            for spec in models_to_test:
                model_name = f"{spec['vol']}({spec['p']},{spec['o']},{spec['q']})"
                if spec['vol'] == 'Garch' and spec['o'] == 1:
                    model_name = f"GJR-GARCH({spec['p']},{spec['q']})"
                    
                try:
                    am = arch_model(adr_y, x=X, mean='ARX', lags=1, 
                                    vol=spec['vol'], p=spec['p'], o=spec['o'], q=spec['q'], 
                                    dist=dist)
                    
                    res = am.fit(disp='off', show_warning=False)
                    bic = res.bic
                    aic = res.aic
                    
                    # Approximated persistence
                    try:
                        if spec['vol'].lower() == 'egarch':
                            persistence = res.params.get('beta[1]', 0)
                        elif spec['o'] > 0:
                            persistence = res.params.get('beta[1]', 0) + res.params.get('alpha[1]', 0) + (0.5 * res.params.get('gamma[1]', 0))
                        else:
                            persistence = sum([val for key, val in res.params.items() if 'alpha[' in key or 'beta[' in key])
                    except:
                        persistence = np.nan
                        
                    results.append({
                        'ADR': adr,
                        'Model': model_name,
                        'Dist': dist,
                        'AIC': aic,
                        'BIC': bic,
                        'Persistence': persistence,
                        'Nu': res.params.get('nu', np.nan)
                    })
                    
                    if bic < best_bic:
                        best_bic = bic
                        best_res = res
                        best_name = model_name
                        best_dist = dist
                        
                except Exception as e:
                    # Model failed to converge
                    continue
        
        if best_res is not None:
             best_models[adr] = {
                 'name': best_name,
                 'dist': best_dist,
                 'cond_vol': best_res.conditional_volatility,
                 'summary': best_res.summary()
             }
             
    # Find the top models per ADR ordered by BIC
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res = df_res.sort_values(['ADR', 'BIC']).reset_index(drop=True)
        # Select best model strictly for summary table
        top_models = df_res.groupby('ADR').first().reset_index()
    else:
        top_models = pd.DataFrame()
        
    return df_res, top_models, best_models

def extract_volatility_series(best_models):
    res = {}
    for adr, model_info in best_models.items():
        res[adr] = model_info['cond_vol']
    return res
