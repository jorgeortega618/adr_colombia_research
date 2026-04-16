import yfinance as yf
import pandas as pd
import datetime
import os

def update_data():
    tickers = ['EC', 'AVAL', 'CIB', '^GSPC', 'COP=X']
    start_date = '2016-02-01'
    end_date = datetime.date.today().strftime('%Y-%m-%d')
    
    print(f"Downloading data from {start_date} to {end_date}...")
    df = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close' if tuple([int(i) for i in pd.__version__.split('.')[:2]]) < (2, 2) else 'Price']
    
    # Handle MultiIndex columns depending on yfinance return version
    if isinstance(df.columns, pd.MultiIndex):
        if 'Close' in df.columns.get_level_values(0):
            df = df['Close']
        else:
            # Drop the top level which is probably Price
            df.columns = df.columns.get_level_values(1)

    df = df[['EC', 'AVAL', 'CIB', '^GSPC', 'COP=X']]
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Date', 'Date': 'Date'}, inplace=True)
    
    df['Date'] = df['Date'].dt.strftime('%d/%m/%Y')
    
    output_file = 'adr_prices_daily.csv'
    df.to_csv(output_file, sep=';', index=False)
    print(f"Data successfully updated to {output_file}")

if __name__ == "__main__":
    update_data()
