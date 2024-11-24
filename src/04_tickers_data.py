import os
import pandas as pd
import yfinance as yf

# Cargar el archivo que contiene los tickers
file_path = "data/tickers_tech_med_clean.csv"
tickers_df = pd.read_csv(file_path)

# Extraer los tickers únicos
tickers = tickers_df['Ticker'].dropna().unique()

# Añadir S&P 500 como benchmark
benchmark_ticker = "^GSPC"  # Ticker del S&P 500 en Yahoo Finance
tickers = list(tickers) + [benchmark_ticker]

# Crear una carpeta para guardar los datos
output_folder = "data/tickers_data"
os.makedirs(output_folder, exist_ok=True)

# Descargar datos históricos y guardar únicamente las columnas elementales
def download_ticker_data(ticker, start_date="2010-01-01"):
    try:
        # Descargar datos históricos
        stock_data = yf.download(ticker, start=start_date)
        
        # Validar si el DataFrame contiene datos
        if stock_data.empty:
            print(f"Ticker {ticker}: No se encontraron datos. Se omite.")
            return

        # Añadir columna "Ticker" al DataFrame
        stock_data["Ticker"] = ticker

        # Reiniciar el índice para convertir el índice en columna "Date"
        stock_data.reset_index(inplace=True)

        # Filtrar columnas elementales
        elemental_columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume', 'Ticker']
        elemental_data = stock_data[elemental_columns]

        # Guardar datos en un archivo CSV limpio
        ticker_name = "SP500" if ticker == benchmark_ticker else ticker
        output_path = os.path.join(output_folder, f"{ticker_name}_historical_data.csv")
        elemental_data.to_csv(output_path, index=False)
        print(f"Datos elementales para {ticker_name} guardados en: {output_path}")

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")

# Descargar datos para todos los tickers y el benchmark
for ticker in tickers:
    print(f"Descargando datos para: {ticker}")
    download_ticker_data(ticker)

print("Proceso finalizado.")
