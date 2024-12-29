import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis
import os
import warnings
import logging
warnings.filterwarnings('ignore')

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_industry_score(industry, year, trends):
    """
    Obtiene el puntaje de tendencia para una industria específica en un año dado.

    :param industry: Nombre de la industria.
    :param year: Año para el cual se busca el puntaje.
    :param trends: Diccionario con datos de tendencias organizados por año.
    :return: Puntaje de tendencia de la industria, 0.0 si no se encuentra.
    """
    year_trends = trends.get(year, {'industries': [], 'scores': []})
    try:
        idx = year_trends['industries'].index(industry)
        return year_trends['scores'][idx]
    except (ValueError, KeyError):
        return 0.0

# Función para obtener tasas libres de riesgo históricas con yfinance
def load_risk_free_rates():
    """
    Descarga las tasas libres de riesgo históricas usando yfinance (bonos a 3 meses de EE.UU.).

    :return: Diccionario con tasas por año.
    """
    try:
        logging.info("Descargando tasas libres de riesgo.")
        risk_free_data = yf.download("^IRX", start="2010-01-01", end="2024-01-01", progress=False)
        risk_free_data["Year"] = risk_free_data.index.year
        risk_free_data["Rate"] = risk_free_data["Adj Close"] / 100  # Convertir a formato porcentual
        annual_rates = risk_free_data.groupby("Year")["Rate"].mean().to_dict()
        logging.info("Tasas libres de riesgo descargadas exitosamente.")
        return annual_rates
    except Exception as e:
        logging.error(f"Error obteniendo tasas libres de riesgo: {e}")
        return {}

def calculate_real_return(weights, ticker_data, year):
    """
    Calcula el retorno real de la cartera basado en los datos de los tickers y los pesos.

    :param weights: Diccionario con los pesos de los activos.
    :param ticker_data: Diccionario con datos históricos de los activos.
    :param year: Año para calcular el retorno real.
    :return: Retorno real de la cartera.
    """
    portfolio_return = 0

    for ticker, weight in weights.items():
        if ticker in ticker_data:
            data = ticker_data[ticker]
            year_data = data[data['Date'].dt.year == year]

            if not year_data.empty:
                start_price = year_data.iloc[0]['Adj Close']
                end_price = year_data.iloc[-1]['Adj Close']
                ticker_return = (end_price / start_price) - 1
                portfolio_return += ticker_return * weight

    return portfolio_return

def calculate_sharpe_ratio(annual_return, volatility, year, risk_free_rates):
    """
    Calcula el ratio de Sharpe ajustado a la tasa libre de riesgo histórica.

    :param annual_return: Retorno anual del activo.
    :param volatility: Volatilidad anual del activo.
    :param year: Año de análisis.
    :param risk_free_rates: Diccionario con tasas libres de riesgo por año.
    :return: Ratio de Sharpe.
    """
    risk_free_rate = risk_free_rates.get(year, 0.02)  # Valor por defecto si el año no está presente
    return (annual_return - risk_free_rate) / volatility if volatility != 0 else 0

def align_monthly_returns(tickers_metrics):
    """
    Alinea los retornos mensuales de todos los tickers para garantizar
    que tengan la misma longitud, rellenando los valores faltantes con 0.

    :param tickers_metrics: Diccionario con métricas de los activos.
    :return: DataFrame con los retornos mensuales alineados por ticker.
    """
    monthly_returns = {ticker: metrics['monthly_returns'] for ticker, metrics in tickers_metrics.items()}
    aligned_returns = pd.DataFrame(monthly_returns).fillna(0)
    return aligned_returns

def optimize_portfolio(tickers_metrics, year, risk_free_rates, max_companies=30):
    """
    Realiza la optimización de la cartera bajo restricciones de tamaño y peso.

    Este método utiliza programación cuadrática para minimizar la volatilidad
    de la cartera mientras maximiza un índice compuesto basado en tendencias
    y otras métricas personalizadas.

    :param tickers_metrics: Diccionario con métricas de los activos.
    :param year: Año de análisis para la optimización.
    :param risk_free_rates: Diccionario con tasas libres de riesgo por año.
    :param max_companies: Límite superior en el número de empresas en la cartera.
    :return: Diccionario con los pesos óptimos de los activos, o None si falla la optimización.
    """
    tickers = list(tickers_metrics.keys())
    n = len(tickers)

    if n < 5:
        logging.warning(f"Insuficientes tickers ({n}) para optimizar en el año {year}.")
        return None

    # Limitar a un máximo de empresas
    if n > max_companies:
        tickers = tickers[:max_companies]
        tickers_metrics = {ticker: tickers_metrics[ticker] for ticker in tickers}
        logging.info(f"Reduciendo a las {max_companies} principales empresas para el año {year}.")

    n = len(tickers)

    init_weights = np.array([1 / n] * n)

    max_weight = min(0.3, 3 / n)
    bounds = [(0.01, max_weight) for _ in range(n)]

    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

    def objective(weights):
        """
        Calcula la función objetivo para la optimización de la cartera.

        La función combina varios factores, como la volatilidad de la cartera, 
        las tendencias y un puntaje combinado para determinar la calidad de la cartera. 
        Además, penaliza configuraciones que no cumplan la restricción de suma de pesos.

        :param weights: Vector de pesos para los activos en la cartera.
        :return: Valor de la función objetivo a minimizar.
        """
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        trend_score = sum(tickers_metrics[ticker]['trend_score'] * w for ticker, w in zip(tickers, weights))
        combined_score = sum(tickers_metrics[ticker]['combined_score'] * w for ticker, w in zip(tickers, weights))
        penalty = 100 * max(0, abs(np.sum(weights) - 1) - 0.01)
    
        return port_volatility - (combined_score + trend_score) + penalty

    # Alinear retornos mensuales
    aligned_returns = align_monthly_returns(tickers_metrics)

    # Calcular matriz de covarianza
    cov_matrix = aligned_returns.cov().values

    result = minimize(
        objective,
        init_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-6}
    )

    if not result.success:
        logging.warning("Intentando optimización alternativa con 'trust-constr'")
        result = minimize(
            objective,
            init_weights,
            method='trust-constr',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 2000}
        )

    if result.success:
        weights = result.x / result.x.sum()
        logging.info(f"Optimización exitosa para el año {year}.")
        return dict(zip(tickers, weights))
    
    if not result.success:
        logging.error(f"Optimización fallida: {result.message}")
        print(f"Pesos finales: {result.x}")
        return None

def find_optimal_portfolio_multicriteria(tickers_metrics, year, ticker_data, risk_free_rates, min_companies, max_companies):
    """
    Encuentra la cartera óptima para un rango de tamaños de cartera utilizando un enfoque multicriterio.

    Este enfoque considera factores como el ratio de Sharpe, la volatilidad, 
    la diversificación y las tendencias para evaluar la calidad de la cartera.

    :param tickers_metrics: Diccionario con métricas de los activos.
    :param year: Año de análisis.
    :param ticker_data: Diccionario con datos históricos de los activos.
    :param risk_free_rates: Diccionario con tasas libres de riesgo por año.
    :param min_companies: Límite inferior en el número de empresas en la cartera.
    :param max_companies: Límite superior en el número de empresas en la cartera.
    :return: La mejor cartera, el número óptimo de empresas y el índice de calidad asociado.
    """
    best_portfolio = None
    best_num_companies = None
    best_quality_index = -np.inf

    for num_companies in range(min_companies, max_companies + 1):
        logging.info(f"Probando cartera con un máximo de {num_companies} empresas para el año {year}.")
        portfolio_weights = optimize_portfolio(tickers_metrics, year, risk_free_rates, max_companies=num_companies)

        if portfolio_weights:
            # Calcular rendimiento real y volatilidad
            annual_return = calculate_real_return(portfolio_weights, ticker_data, year)
            portfolio_volatility = np.sqrt(
                sum(
                    tickers_metrics[t]['volatility'] ** 2 * w ** 2 for t, w in portfolio_weights.items()
                )
            )
            portfolio_sharpe = calculate_sharpe_ratio(annual_return, portfolio_volatility, year, risk_free_rates)

            # Calcular diversificación (entropía de los pesos)
            diversification = -np.sum([w * np.log(w) for w in portfolio_weights.values() if w > 0])

            # Calcular el puntaje de tendencia ponderado
            trend_score = sum(
                tickers_metrics[ticker]['trend_score'] * w for ticker, w in portfolio_weights.items()
            )

            # Crear un índice compuesto
            quality_index = (
                0.4 * portfolio_sharpe  # Sharpe ratio
                - 0.3 * portfolio_volatility  # Penalizar volatilidad
                + 0.2 * diversification  # Fomentar diversificación
                + 0.1 * trend_score  # Incorporar tendencia
            )

            logging.info(f"Índice de calidad para {num_companies} empresas: {quality_index:.4f}")

            # Actualizar la mejor cartera si el índice de calidad es mayor
            if quality_index > best_quality_index:
                best_quality_index = quality_index
                best_portfolio = portfolio_weights
                best_num_companies = num_companies

    if best_portfolio:
        logging.info(
            f"Mejor cartera para el año {year} tiene {best_num_companies} empresas con índice de calidad {best_quality_index:.4f}."
        )
    else:
        logging.warning(f"No se encontró una cartera óptima para el año {year}.")

    return best_portfolio, best_num_companies, best_quality_index

def calculate_metrics(ticker_data, advanced_metrics, ticker, year, trends, tickers_info, risk_free_rates):
    """
    Calcula métricas clave para un activo específico en un año dado.

    Este método genera métricas relevantes para la optimización de la cartera, 
    incluyendo retorno esperado, volatilidad, ratio de Sharpe, drawdowns y 
    puntuaciones basadas en tendencias.

    :param ticker_data: DataFrame con datos históricos del activo.
    :param advanced_metrics: DataFrame con métricas adicionales de los activos.
    :param ticker: Identificador del activo (ticker).
    :param year: Año para el análisis.
    :param trends: Diccionario con datos de tendencias organizados por año.
    :param tickers_info: DataFrame con información complementaria sobre los activos (e.g., industria).
    :param risk_free_rates: Diccionario con tasas libres de riesgo por año.
    :return: Diccionario con las métricas calculadas para el activo o None si no se pueden calcular.
    """    
    try:
        # Validaciones iniciales
        year_data = ticker_data[ticker_data['Date'].dt.year == year].copy()
        if len(year_data) < 60:
            return None
        
        # Cálculos básicos
        monthly_data = year_data.resample('M', on='Date')['Adj Close'].last()
        monthly_returns = monthly_data.pct_change().dropna()
        if len(monthly_returns) < 6:
            return None
        
        annual_return = (1 + monthly_returns).prod() - 1
        daily_returns = year_data['Adj Close'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe = calculate_sharpe_ratio(annual_return, volatility, year, risk_free_rates)

        rolling_max = year_data['Adj Close'].expanding(min_periods=1).max()
        drawdowns = (year_data['Adj Close'] - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()

        # Precios de inicio y fin de año
        if not year_data.empty:
            start_price = year_data.iloc[0]['Adj Close']
            end_price = year_data.iloc[-1]['Adj Close']
            real_return = (end_price / start_price) - 1
        else:
            start_price = None
            end_price = None
            real_return = None

        # Datos adicionales
        industry_match = tickers_info[tickers_info['Ticker'] == ticker]
        industry = industry_match['Industry'].iloc[0] if not industry_match.empty else "Unknown"
        trend_score = get_industry_score(industry, year, trends)
        
        # Métricas de mercado
        advanced_ticker = advanced_metrics[(advanced_metrics['Ticker'] == ticker) & (advanced_metrics['Year'] == year)]
        market_cap = float(advanced_ticker['Market Cap'].iloc[0]) if not advanced_ticker.empty else 0
        
        combined_score = (
            0.3 * sharpe +
            0.3 * trend_score +
            0.25 * (1 + max_drawdown) +
            0.15 * (market_cap / 1e12)
        )
        # Validación final
        if any(np.isnan([annual_return, volatility, sharpe, max_drawdown, combined_score])):
            return None
        
        return {
            'expected_return': annual_return,
            'volatility': volatility,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'trend_score': trend_score,
            'combined_score': combined_score,
            'industry': industry,
            'monthly_returns': monthly_returns,
            'start_price': start_price,
            'end_price': end_price,
            'real_return': real_return  # Asegúrate de que esto no sea None
        }
    except Exception as e:
        print(f"Error en métricas para {ticker}: {str(e)}")
        return None

def save_yearly_portfolio(weights, tickers_metrics, year, results_path, num_companies):
    """
    Guarda un archivo CSV con la composición de la cartera para un año específico.

    El archivo incluye los pesos de los activos, métricas clave y los precios 
    iniciales y finales del año.

    :param weights: Diccionario con los pesos de los activos en la cartera.
    :param tickers_metrics: Diccionario con métricas de los activos.
    :param year: Año de análisis.
    :param results_path: Ruta donde se guardará el archivo.
    :param num_companies: Número de empresas en la cartera.
    """
    portfolio = []
    for ticker, weight in weights.items():
        if weight > 0:
            metrics = tickers_metrics[ticker]
            portfolio.append({
                'Ticker': ticker,
                'Weight': weight,
                'Expected Return': metrics['expected_return'],
                'Volatility': metrics['volatility'],
                'Sharpe Ratio': metrics['sharpe'],
                'Max Drawdown': metrics['max_drawdown'],
                'Start Price': metrics['start_price'],  # Añadir precio inicial
                'End Price': metrics['end_price'],      # Añadir precio final
                'Real Return': metrics.get('real_return', None)
            })
    portfolio_df = pd.DataFrame(portfolio)
    # Adaptar el nombre del archivo para incluir el número de compañías
    portfolio_path = os.path.join(results_path, f'portfolio_{year}_{num_companies}_companies.csv')
    portfolio_df.to_csv(portfolio_path, index=False)

def calculate_future_performance(weights, ticker_data, start_year, end_year):
    """
    Calcula la rentabilidad futura de la cartera para un rango de años.

    Este método evalúa la rentabilidad anual y acumulada de la cartera con 
    base en los pesos actuales de los activos y sus datos históricos.

    :param weights: Diccionario con los pesos de los activos de la cartera.
    :param ticker_data: Diccionario con datos históricos de los activos.
    :param start_year: Año en que se creó la cartera.
    :param end_year: Último año para evaluar la rentabilidad.
    :return: Lista de diccionarios con la rentabilidad anual y acumulada por año.
    """
    future_performance = []

    # Obtener precios iniciales desde el año de creación de la cartera
    initial_prices = {}
    for ticker in weights:
        if ticker in ticker_data:
            data = ticker_data[ticker]
            start_data = data[data['Date'].dt.year == start_year]
            if not start_data.empty:
                initial_prices[ticker] = start_data.iloc[0]['Adj Close']

    cumulative_return = 1  # Inicializar la rentabilidad acumulada en 1

    for year in range(start_year, end_year + 1):
        portfolio_return = 0

        for ticker, weight in weights.items():
            if ticker in ticker_data:
                data = ticker_data[ticker]
                year_data = data[data['Date'].dt.year == year]

                if not year_data.empty:
                    start_price = year_data.iloc[0]['Adj Close']
                    end_price = year_data.iloc[-1]['Adj Close']

                    # Calcular el retorno anual de cada acción basado en los precios del año actual
                    ticker_return = (end_price / start_price) - 1

                    # Ajustar el retorno por el peso de la acción
                    portfolio_return += ticker_return * weight

        # Actualizar la rentabilidad acumulada correctamente
        cumulative_return *= (1 + portfolio_return)

        future_performance.append({
            'Portfolio Start Year': start_year,
            'Future Year': year,
            'Annual Return': portfolio_return,
            'Cumulative Return': cumulative_return - 1
        })

    return future_performance

def save_general_summary(results, results_path, label):
    """
    Guarda un archivo CSV con un resumen general de los resultados de las carteras.

    Este archivo contiene métricas clave como el retorno real, la volatilidad 
    y el ratio de Sharpe para cada año.

    :param results: Lista de diccionarios con los resultados anuales de las carteras.
    :param results_path: Ruta donde se guardará el archivo.
    :param label: Etiqueta para identificar el archivo (por ejemplo, "optimal").
    """
    results_df = pd.DataFrame(results)
    consolidated_path = os.path.join(results_path, f'portfolio_results_{label}.csv')
    results_df.to_csv(consolidated_path, index=False)
    logging.info(f"Resumen general guardado en {consolidated_path}.")

def save_combined_future_performance(all_future_performances, results_path, label):
    """
    Guarda un archivo CSV consolidado con la rentabilidad futura de todas las carteras.

    :param all_future_performances: Lista de listas con las rentabilidades futuras por cartera.
    :param results_path: Ruta donde se guardará el archivo.
    :param label: Etiqueta para identificar el archivo (por ejemplo, "optimal").
    """
    combined_performance = []
    for performance in all_future_performances:
        combined_performance.extend(performance)

    combined_df = pd.DataFrame(combined_performance)
    combined_path = os.path.join(results_path, f'combined_future_performance_{label}.csv')
    combined_df.to_csv(combined_path, index=False)
    
def main():
    """
    Orquesta el flujo de trabajo para la creación y optimización de carteras.

    Este método ejecuta la carga de datos, el cálculo de métricas, la búsqueda
    de carteras óptimas y la exportación de resultados. Asegura que todas las 
    operaciones se realicen para un rango de años y que se guarden los resultados.
    """
    logging.info("Iniciando proceso de optimización dinámica...")
    data_path = "/Users/alee/Carteras-Tendenciales-Optimizaci-n-Basada-en-An-lisis-Multidimensional/src/data/tickers_data"
    results_path = "/Users/alee/Carteras-Tendenciales-Optimizaci-n-Basada-en-An-lisis-Multidimensional/src/results/portfolio"

    trends_df = pd.read_csv('/Users/alee/Carteras-Tendenciales-Optimizaci-n-Basada-en-An-lisis-Multidimensional/src/data/top_industry_trends_by_year.csv')
    advanced_metrics = pd.read_csv('/Users/alee/Carteras-Tendenciales-Optimizaci-n-Basada-en-An-lisis-Multidimensional/src/data/metrics_advanced/advanced_metrics_all_tickers.csv')
    tickers_info = pd.read_csv('/Users/alee/Carteras-Tendenciales-Optimizaci-n-Basada-en-An-lisis-Multidimensional/src/data/tickers_tech_med_clean.csv')
    
    trends = {int(year): {
        'industries': group['industry'].tolist(),
        'scores': group['composite_score'].tolist()
    } for year, group in trends_df.groupby('year')}
    
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Cargar tasas libres de riesgo usando yfinance
    risk_free_rates = load_risk_free_rates()

    # Cargar datos de tickers
    ticker_data = {}
    for f in os.listdir(data_path):
        if f.endswith('_historical_data.csv'):
            ticker = f.split('_')[0]
            df = pd.read_csv(os.path.join(data_path, f))
            df['Date'] = pd.to_datetime(df['Date'])
            ticker_data[ticker] = df

    logging.info(f"Datos cargados: {len(ticker_data)} tickers")

    results = []
    all_future_performances = []

    for year in range(2015, 2024):
        logging.info(f"Analizando año {year}.")
        tickers_metrics = {}

        # Calcular métricas para los tickers
        for ticker, data in ticker_data.items():
            metrics = calculate_metrics(data, advanced_metrics, ticker, year, trends, tickers_info, risk_free_rates)
            if metrics:
                tickers_metrics[ticker] = metrics

        if len(tickers_metrics) < 10:
            logging.warning(f"Insuficientes tickers para el año {year}.")
            continue

        # Buscar la cartera óptima
        best_portfolio, best_num_companies, best_sharpe = find_optimal_portfolio(
            tickers_metrics, year, ticker_data, risk_free_rates, min_companies=10, max_companies=40
        )

        if best_portfolio:
            # Guardar la mejor cartera para el año con el número de empresas optimizado
            save_yearly_portfolio(best_portfolio, tickers_metrics, year, results_path, best_num_companies)

            real_return = calculate_real_return(best_portfolio, ticker_data, year)

            portfolio_volatility = np.sqrt(
                sum(
                    tickers_metrics[t]['volatility'] ** 2 * w ** 2 for t, w in best_portfolio.items()
                )
            )

            results.append({
                'Year': year,
                'Portfolio Real Return': real_return,
                'Portfolio Volatility': portfolio_volatility,
                'Portfolio Sharpe': best_sharpe,
                'Risk Free Rate': risk_free_rates.get(year, 0.02),
                'Num Companies': best_num_companies  # Añadir el número de empresas optimizado
            })

            # Calcular y guardar rentabilidad futura
            future_performance = calculate_future_performance(best_portfolio, ticker_data, year, 2023)
            all_future_performances.append(future_performance)

    # Guardar resultados generales
    save_general_summary(results, results_path, "optimal")

    # Guardar resultados combinados de rentabilidad futura
    save_combined_future_performance(all_future_performances, results_path, "optimal")

if __name__ == "__main__":
    main()
