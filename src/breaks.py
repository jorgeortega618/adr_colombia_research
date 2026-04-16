import ruptures as rpt
import pandas as pd
import numpy as np

def detect_breaks_pelt(series, penalties=[5, 10, 20]):
    """
    Applies PELT on a 1D series (numpy array or pandas series) across varying penalties.
    Returns a dict mapping penalty -> list of breakpoint indices.
    """
    y = np.array(series.dropna())
    results = {}
    if len(y) < 10:
        return {p: [] for p in penalties}
        
    model = rpt.Pelt(model="rbf").fit(y)
    
    for pen in penalties:
        try:
            bkps = model.predict(pen=pen)
            if bkps and bkps[-1] == len(y):
                bkps = bkps[:-1]
            results[pen] = bkps
        except:
            results[pen] = []
            
    return results

def compute_stable_breaks(pelt_results, margin=30, agreement_threshold=2/3):
    """
    Finds breaks that appear consistently across penalty settings.
    margin: +/- trading days to consider it the "same" break
    agreement_threshold: fraction of penalty settings that must agree
    """
    all_bkps = []
    num_penalties = len(pelt_results)
    
    for pen, bkps in pelt_results.items():
        all_bkps.extend([(b, pen) for b in bkps])
        
    if not all_bkps:
        return []
        
    all_bkps.sort(key=lambda x: x[0])
    
    stable_breaks_clusters = []
    current_cluster = [all_bkps[0]]
    
    for i in range(1, len(all_bkps)):
        b, pen = all_bkps[i]
        cluster_mean = np.mean([x[0] for x in current_cluster])
        if abs(b - cluster_mean) <= margin:
            current_cluster.append((b, pen))
        else:
            stable_breaks_clusters.append(current_cluster)
            current_cluster = [(b, pen)]
            
    stable_breaks_clusters.append(current_cluster)
    
    final_stable_breaks = []
    for cluster in stable_breaks_clusters:
        unique_pens = set([x[1] for x in cluster])
        if len(unique_pens) / num_penalties >= agreement_threshold:
            median_bkp = int(np.median([x[0] for x in cluster]))
            final_stable_breaks.append(median_bkp)
            
    return final_stable_breaks

def process_all_breaks(data_dict, margin=30, penalties=[5, 10, 20]):
    """
    data_dict: dict of 'name' -> pd.Series.
    For each series, run PELT and find stable breaks.
    """
    final_res = {}
    for name, series in data_dict.items():
        df_cur = series.dropna()
        if df_cur.empty: 
            continue
        pelt_res = detect_breaks_pelt(df_cur, penalties)
        stable = compute_stable_breaks(pelt_res, margin=margin)
        
        dates_stable = [df_cur.index[b].strftime('%Y-%m-%d') for b in stable if b < len(df_cur)]
        
        dates_pelt = {}
        for pen, bkps in pelt_res.items():
            dates_pelt[pen] = [df_cur.index[b].strftime('%Y-%m-%d') for b in bkps if b < len(df_cur)]
            
        final_res[name] = {
            'pelt_by_pen': dates_pelt,
            'stable_breaks': dates_stable
        }
        
    return final_res

def get_breaks_table(process_results):
    rows = []
    for name, res in process_results.items():
        row = {'Series Name': name}
        for pen, dates in res['pelt_by_pen'].items():
            row[f'Pen={pen}'] = ', '.join(dates) if dates else 'None'
        row['Stable Breaks (+/- 30d)'] = ', '.join(res['stable_breaks']) if res['stable_breaks'] else 'None'
        rows.append(row)
        
    df = pd.DataFrame(rows)
    # Ensure column order
    base_cols = [c for c in df.columns if c not in ['Series Name', 'Stable Breaks (+/- 30d)']]
    return df[['Series Name'] + sorted(base_cols) + ['Stable Breaks (+/- 30d)']]
