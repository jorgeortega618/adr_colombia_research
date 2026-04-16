import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def export_tables(beta_summary, breaks_table, vol_top_models, output_dir='./outputs/tables/'):
    os.makedirs(output_dir, exist_ok=True)
    
    beta_summary.to_csv(os.path.join(output_dir, 'rolling_beta_summary.csv'), index=False)
    breaks_table.to_csv(os.path.join(output_dir, 'stable_breaks.csv'), index=False)
    vol_top_models.to_csv(os.path.join(output_dir, 'volatility_optimal_models.csv'), index=False)
    
    print(f"Tables exported successfully to {output_dir}")

def export_figures(rolling_betas, breaks_res, cond_vols, vol_breaks_res, windows, output_dir='./outputs/figures/'):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    adrs = ['EC', 'AVAL', 'CIB']
    target_window = windows[1] if len(windows) > 1 else windows[0]

    for adr in adrs:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        if target_window not in rolling_betas[adr]: continue
        df_roll = rolling_betas[adr][target_window]
        
        # Plot Beta SP500
        axes[0].plot(df_roll.index, df_roll['beta_sp500'], label=f'Cond. Beta SP500 (w={target_window})', color='blue')
        axes[0].set_title(f'{adr} - Time-Varying S&P 500 Exposure (Beta)')
        axes[0].set_ylabel('Beta')
        
        beta_key = f"{adr}_beta_sp500_{target_window}"
        if beta_key in breaks_res:
            stable_bkps = breaks_res[beta_key]['stable_breaks']
            for b_str in stable_bkps:
                b_ts = pd.to_datetime(b_str)
                axes[0].axvline(x=b_ts, color='red', linestyle='--', alpha=0.7)
                
        # Plot Gamma FX
        axes[1].plot(df_roll.index, df_roll['gamma_fx'], label=f'Cond. Gamma FX (w={target_window})', color='purple')
        axes[1].set_title(f'{adr} - Time-Varying USD/COP Exposure (Gamma)')
        axes[1].set_ylabel('Gamma')
        
        gamma_key = f"{adr}_gamma_fx_{target_window}"
        if gamma_key in breaks_res:
            stable_bkps = breaks_res[gamma_key]['stable_breaks']
            for b_str in stable_bkps:
                b_ts = pd.to_datetime(b_str)
                axes[1].axvline(x=b_ts, color='red', linestyle='--', alpha=0.7)
                
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f'{adr}_rolling_params_breaks.png'), dpi=300)
        plt.close(fig)
        
    for adr in adrs:
        fig, ax = plt.subplots(figsize=(12, 6))
        volumetric = cond_vols[adr]
        ax.plot(volumetric.index, volumetric, label='Conditional Volatility', color='darkorange')
        ax.set_title(f'{adr} - Optimal GARCH Conditional Volatility')
        ax.set_ylabel('Volatility')
        
        vol_key = f"{adr}_cond_vol"
        if vol_key in vol_breaks_res:
            stable_bkps = vol_breaks_res[vol_key]['stable_breaks']
            for i, b_str in enumerate(stable_bkps):
                b_ts = pd.to_datetime(b_str)
                label = 'Stable Vol Break' if i == 0 else None
                ax.axvline(x=b_ts, color='black', linestyle='-.', alpha=0.8, label=label)
        
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f'{adr}_conditional_volatility.png'), dpi=300)
        plt.close(fig)
        
    print(f"Figures exported successfully to {output_dir}")
