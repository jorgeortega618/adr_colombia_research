import argparse
import os

from src.data import load_and_prep_data
from src.rolling import compute_rolling_betas, compute_rolling_correlation, get_beta_summary
from src.breaks import process_all_breaks, get_breaks_table
from src.volatility import grid_search_garch, extract_volatility_series
from src.report import export_tables, export_figures

def main():
    parser = argparse.ArgumentParser(description="Robustness Checks for Colombian ADRs")
    parser.add_argument('--freq', type=str, default='daily', choices=['daily', 'weekly'], help='Data frequency')
    parser.add_argument('--windows', type=int, nargs='+', default=[126, 252, 504], help='Rolling windows in days')
    parser.add_argument('--pen', type=int, nargs='+', default=[5, 10, 20], help='PELT penalty parameters')
    parser.add_argument('--out', type=str, default='./outputs/', help='Output directory base')
    
    args = parser.parse_args()
    
    print("1. Loading Data...")
    filepath = './adr_prices_daily.csv'
    returns = load_and_prep_data(filepath, freq=args.freq)
    
    print("2. Computing Rolling Betas (Non-Robust and HAC)...")
    rolling_betas = compute_rolling_betas(returns, windows=args.windows, cov_type='nonrobust')
    rolling_betas_hac = compute_rolling_betas(returns, windows=args.windows, cov_type='HAC')
    rolling_corrs = compute_rolling_correlation(returns, windows=args.windows)
    
    beta_summary = get_beta_summary(rolling_betas)
    
    print("3. Break Detection (Rolling Params)...")
    break_series_dict = {}
    adrs = ['EC', 'AVAL', 'CIB']
    for adr in adrs:
        for w in args.windows:
            if w in rolling_betas[adr]:
                df_roll = rolling_betas[adr][w]
                break_series_dict[f"{adr}_beta_sp500_{w}"] = df_roll['beta_sp500']
                break_series_dict[f"{adr}_gamma_fx_{w}"] = df_roll['gamma_fx']
            
            if w in rolling_corrs[adr]:
                df_corr = rolling_corrs[adr][w]
                break_series_dict[f"{adr}_corr_sp500_{w}"] = df_corr['corr_sp500']

    breaks_res = process_all_breaks(break_series_dict, margin=30, penalties=args.pen)
    
    print("4. Volatility Model Grid Search...")
    all_vol_res, top_models_df, best_models_dict = grid_search_garch(returns)
    cond_vols = extract_volatility_series(best_models_dict)
    
    print("5. Break Detection (Volatility)...")
    vol_break_series_dict = {f"{adr}_cond_vol": cond_vols[adr] for adr in adrs if adr in cond_vols}
    vol_breaks_res = process_all_breaks(vol_break_series_dict, margin=30, penalties=args.pen)
    
    breaks_res.update(vol_breaks_res)
    breaks_table = get_breaks_table(breaks_res)
    
    print("6. Exporting Deliverables...")
    export_tables(beta_summary, breaks_table, top_models_df, output_dir=os.path.join(args.out, 'tables'))
    export_figures(rolling_betas, breaks_res, cond_vols, vol_breaks_res, args.windows, output_dir=os.path.join(args.out, 'figures'))

    print(f"Pipeline Complete! All outputs exported to {args.out}")

if __name__ == "__main__":
    main()
