import os
import pandas as pd

# Ruta de la carpeta donde están los archivos de los tickers
tickers_data_folder = "data/tickers_data"

# Comprobar si la carpeta existe
if not os.path.exists(tickers_data_folder):
    print(f"La carpeta {tickers_data_folder} no existe. Verifica la ruta.")
else:
    # Recorrer todos los archivos de la carpeta
    for filename in os.listdir(tickers_data_folder):
        # Verificar si el archivo es un archivo CSV
        if filename.endswith(".csv"):
            file_path = os.path.join(tickers_data_folder, filename)
            try:
                # Cargar el archivo en un DataFrame
                df = pd.read_csv(file_path)

                # Verificar que el DataFrame tiene al menos dos filas
                if len(df) > 1:
                    # Asegurar que la primera fila es eliminada
                    df = df.iloc[1:].reset_index(drop=True)

                    # Procesar la columna 'Date'
                    if 'Date' in df.columns:
                        # Convertir a string, extraer los primeros 10 caracteres y convertir de vuelta a datetime
                        df['Date'] = df['Date'].astype(str).str[:10]
                        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')

                        # Eliminar filas con fechas inválidas
                        invalid_dates = df['Date'].isna().sum()
                        if invalid_dates > 0:
                            print(f"Advertencia: {invalid_dates} filas con fechas no válidas en {filename}. Eliminando.")
                            df = df.dropna(subset=['Date'])

                    # Normalizar las columnas: Asegurar que nombres de columnas son consistentes
                    df.columns = [col.strip() for col in df.columns]

                    # Verificar si hay columnas con valores numéricos y rellenar NaN si es necesario
                    for col in df.columns:
                        if col not in ['Date', 'Ticker']:  # Excluir 'Date' y 'Ticker' del procesamiento numérico
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

                    # Asegurar que la columna 'Ticker' está correctamente asignada
                    if 'Ticker' not in df.columns or df['Ticker'].isnull().all():
                        ticker_name = filename.split('_')[0]  # Suponer que el nombre del archivo empieza con el ticker
                        df['Ticker'] = ticker_name

                    # Guardar el archivo sobrescribiendo el original
                    df.to_csv(file_path, index=False)
                    print(f"Archivo procesado y actualizado: {filename}")
                else:
                    print(f"El archivo {filename} tiene menos de dos filas. No se modificará.")
            except Exception as e:
                print(f"Error procesando el archivo {filename}: {e}")
