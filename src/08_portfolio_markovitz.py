import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Descarga datos históricos ajustados para los tickers especificados en el rango de fechas dado.
# Retorna los precios de cierre ajustados en formato DataFrame.
def download_data(tickers, start_date, end_date):
    """
    Descarga los precios históricos ajustados de cierre para una lista de tickers desde Yahoo Finance.

    Parámetros:
        tickers (list): Lista de símbolos de las acciones.
        start_date (str): Fecha de inicio en formato 'YYYY-MM-DD'.
        end_date (str): Fecha de fin en formato 'YYYY-MM-DD'.

    Retorna:
        pd.DataFrame: DataFrame con precios ajustados de cierre.
    """
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

# Calcula los pesos óptimos de la cartera para minimizar la varianza para un retorno esperado dado.
def optimize_portfolio(mean_returns, covariance, target_return):
    """
    Optimiza una cartera utilizando el modelo de Markowitz para minimizar el riesgo (varianza)
    dado un retorno esperado objetivo.

    Parámetros:
        mean_returns (np.ndarray): Retornos medios esperados de cada activo.
        covariance (np.ndarray): Matriz de covarianza de los activos.
        target_return (float): Retorno esperado objetivo de la cartera.

    Retorna:
        np.ndarray: Pesos óptimos de los activos en la cartera.
    """
    num_assets = len(mean_returns)

    # Función objetivo: Minimizar la varianza de la cartera
    def portfolio_variance(weights):
        return weights.T @ covariance @ weights

    # Restricción de suma de pesos igual a 1
    def weight_constraint(weights):
        return np.sum(weights) - 1

    # Restricción de retorno mínimo
    def return_constraint(weights):
        return weights.T @ mean_returns - target_return

    # Definición de restricciones y límites
    constraints = (
        {'type': 'eq', 'fun': weight_constraint},
        {'type': 'eq', 'fun': return_constraint}
    )
    bounds = [(0, 1) for _ in range(num_assets)]

    # Pesos iniciales distribuidos uniformemente
    initial_weights = np.ones(num_assets) / num_assets

    # Resolución de la optimización
    optimal = minimize(portfolio_variance, initial_weights, constraints=constraints, bounds=bounds)
    if not optimal.success:
        raise ValueError("Optimización no exitosa: " + optimal.message)

    return optimal.x

# Calcula la cartera óptima utilizando el modelo de Markowitz.
def markowitz_portfolio(data):
    """
    Calcula los pesos óptimos, el rendimiento y el riesgo de la cartera óptima.

    Parámetros:
        data (pd.DataFrame): DataFrame con precios ajustados de cierre de los activos.

    Retorna:
        tuple: Pesos óptimos (np.ndarray), rendimiento esperado (float), riesgo esperado (float).
    """
    # Filtrar columnas con datos válidos para todo el año
    data = data.dropna(axis=1, how='any')

    # Cálculo de los retornos anuales reales y la matriz de covarianza
    annual_returns = (data.iloc[-1] / data.iloc[0] - 1).dropna()
    mean_returns = annual_returns.values
    covariance = data.pct_change().dropna().cov().values * 252

    # Definir el retorno objetivo como el promedio de los retornos reales
    target_return = np.mean(mean_returns)

    # Optimizar los pesos utilizando Markowitz
    optimal_weights = optimize_portfolio(mean_returns, covariance, target_return)

    # Calcular el rendimiento y riesgo de la cartera óptima
    portfolio_return = optimal_weights.T @ mean_returns
    portfolio_risk = np.sqrt(optimal_weights.T @ covariance @ optimal_weights)

    return optimal_weights, portfolio_return, portfolio_risk, data.columns

# Main
if __name__ == "__main__":
    # Obtener la lista de empresas del S&P 500 desde Wikipedia
    sp500_tickers = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]['Symbol'].tolist()
    start_date = "2015-01-01"
    end_date = "2023-12-31"

    # Ruta para guardar los resultados
    output_path = "/Users/alee/Carteras-Tendenciales-Optimizaci-n-Basada-en-An-lisis-Multidimensional/src/results/markovitz"

    # Descargar datos históricos
    data = download_data(sp500_tickers, start_date, end_date)

    # Almacenar resultados anuales
    annual_results = []
    for year in range(2015, 2024):
        yearly_data = data[f"{year}-01-01":f"{year}-12-31"]
        if not yearly_data.empty:
            try:
                weights, portfolio_return, portfolio_risk, tickers_used = markowitz_portfolio(yearly_data)

                # Guardar composición de la cartera en un archivo CSV
                portfolio_df = pd.DataFrame({
                    'Ticker': tickers_used,
                    'Weight': weights
                })
                portfolio_df.to_csv(f"{output_path}/portfolio_{year}.csv", index=False)

                # Agregar resultados anuales
                annual_results.append({
                    'Year': year,
                    'Real Portfolio Return': portfolio_return,
                    'Portfolio Risk': portfolio_risk
                })
            except ValueError as e:
                print(f"Error en el año {year}: {e}")

    # Guardar resultados generales en un archivo CSV
    results_df = pd.DataFrame(annual_results)
    results_df.to_csv(f"{output_path}/annual_portfolios.csv", index=False)

    print("Resultados guardados en la ruta especificada.")
