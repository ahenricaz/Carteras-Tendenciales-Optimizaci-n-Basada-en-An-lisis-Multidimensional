import os
import pandas as pd
import numpy as np
import yfinance as yf

# Ruta de la carpeta donde están los datos históricos de los tickers 
data_folder = "data/tickers_data"
output_folder = "data/metrics_advanced"
os.makedirs(output_folder, exist_ok=True)

# Ruta del archivo que contiene la industria por ticker
industry_file = "data/tickers_tech_med_clean.csv"

# Leer el archivo de industria
industry_data = pd.read_csv(industry_file)
industry_data = industry_data[['Ticker', 'Industry']].dropna()

# Listar todos los archivos de tickers
files = [f for f in os.listdir(data_folder) if f.endswith("_historical_data.csv")]

# Función para calcular métricas avanzadas
def calculate_advanced_metrics(data, ticker):
    # Agregar columnas derivadas
    data['Daily Return'] = data['Adj Close'].pct_change()
    data['Cumulative Return'] = (1 + data['Daily Return']).cumprod()

    # Agrupar por año
    data['Year'] = pd.to_datetime(data['Date']).dt.year
    metrics = []

    # Calcular métricas anuales
    for year, group in data.groupby('Year'):
        annual_return = group['Daily Return'].mean() * 252
        annual_volatility = group['Daily Return'].std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else np.nan
        max_drawdown = ((group['Cumulative Return'] / group['Cumulative Return'].cummax()) - 1).min()

        metrics.append({
            'Ticker': ticker,
            'Year': year,
            'Annual Return': annual_return,
            'Annual Volatility': annual_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown
        })

    metrics_df = pd.DataFrame(metrics)

    # Añadir métricas estáticas (repetidas por año)
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        metrics_df['Market Cap'] = info.get('marketCap', np.nan)
        metrics_df['Beta'] = info.get('beta', np.nan)
        metrics_df['P/E Ratio'] = info.get('trailingPE', np.nan)
        metrics_df['Dividend Yield'] = info.get('dividendYield', np.nan)
    except Exception as e:
        print(f"Error fetching static metrics for {ticker}: {e}")
        metrics_df['Market Cap'] = np.nan
        metrics_df['Beta'] = np.nan
        metrics_df['P/E Ratio'] = np.nan
        metrics_df['Dividend Yield'] = np.nan

    # Añadir la industria desde el archivo proporcionado
    industry_row = industry_data[industry_data['Ticker'] == ticker]
    if not industry_row.empty:
        metrics_df['Industry'] = industry_row['Industry'].values[0]
    else:
        metrics_df['Industry'] = 'Unknown'

    return metrics_df

# Calcular métricas para cada archivo
all_metrics = []
for file in files:
    ticker = file.split("_")[0]  # Extraer el ticker del nombre del archivo
    file_path = os.path.join(data_folder, file)
    print(f"Procesando métricas avanzadas para {ticker}")

    try:
        # Leer el archivo de datos
        data = pd.read_csv(file_path)
        if data.empty:
            print(f"El archivo {file} está vacío. Se omite.")
            continue

        # Calcular métricas avanzadas
        ticker_metrics = calculate_advanced_metrics(data, ticker)
        all_metrics.append(ticker_metrics)

    except Exception as e:
        print(f"Error procesando {ticker}: {e}")

# Concatenar métricas de todos los tickers
if all_metrics:
    final_metrics = pd.concat(all_metrics, ignore_index=True)

    # Guardar métricas avanzadas en un único archivo CSV
    output_path = os.path.join(output_folder, "advanced_metrics_all_tickers.csv")
    final_metrics.to_csv(output_path, index=False)
    print(f"Métricas avanzadas guardadas en: {output_path}")
else:
    print("No se generaron métricas avanzadas. Verifica los datos.")

print("Proceso completado.")
