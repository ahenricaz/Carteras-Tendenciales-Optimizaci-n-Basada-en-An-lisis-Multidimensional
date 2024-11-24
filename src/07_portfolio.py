import pandas as pd
import numpy as np
from scipy.optimize import minimize
import os
import warnings
warnings.filterwarnings('ignore')

def get_industry_score(industry, year, trends):
    """Obtiene el score de tendencia de la industria"""
    year_trends = trends.get(year, {'industries': [], 'scores': []})
    try:
        idx = year_trends['industries'].index(industry)
        return year_trends['scores'][idx]
    except (ValueError, KeyError):
        return 0.0

def optimize_portfolio(tickers_metrics, year):
    """Optimiza la cartera con restricciones modificadas"""
    tickers = list(tickers_metrics.keys())
    n = len(tickers)
    
    if n < 5:
        print(f"Insuficientes tickers ({n}) para optimizar")
        return None
        
    print(f"\nOptimizando cartera para {year} con {n} tickers")
    
    init_weights = np.array([1/n] * n)
    
    # Ajustar límite máximo por posición basado en número de tickers
    max_weight = min(0.3, 3/n)  # El máximo será el menor entre 30% y 3/n
    bounds = tuple((0, max_weight) for _ in range(n))
    
    # Restricción de suma = 1 con tolerancia
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'ineq', 'fun': lambda x: x.sum() - 0.99},  # Suma >= 99%
        {'type': 'ineq', 'fun': lambda x: 1.01 - x.sum()}   # Suma <= 101%
    ]
    
    def objective(weights):
        try:
            port_return = sum(tickers_metrics[ticker]['expected_return'] * w for ticker, w in zip(tickers, weights))
            port_vol = sum(tickers_metrics[ticker]['volatility'] * w for ticker, w in zip(tickers, weights))
            trend_score = sum(tickers_metrics[ticker]['trend_score'] * w for ticker, w in zip(tickers, weights))
            combined_score = sum(tickers_metrics[ticker]['combined_score'] * w for ticker, w in zip(tickers, weights))
            
            # Penalizar pesos negativos y exceso de 1
            weight_penalty = 100 * max(0, abs(sum(weights) - 1) - 0.01)
            
            return -(combined_score + trend_score - port_vol) + weight_penalty
        except Exception as e:
            print(f"Error en función objetivo: {str(e)}")
            return np.inf
    
    try:
        result = minimize(
            objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if result.success:
            # Normalizar pesos para asegurar suma = 1
            weights = result.x / result.x.sum()
            print("Optimización exitosa")
            return dict(zip(tickers, weights))
        else:
            print(f"Optimización falló: {result.message}")
            
            # Intentar con método alternativo
            print("Intentando optimización alternativa...")
            result_alt = minimize(
                objective,
                init_weights,
                method='trust-constr',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 2000}
            )
            
            if result_alt.success:
                weights = result_alt.x / result_alt.x.sum()
                print("Optimización alternativa exitosa")
                return dict(zip(tickers, weights))
                
    except Exception as e:
        print(f"Error en optimización: {str(e)}")
    
    return None

def calculate_metrics(ticker_data, advanced_metrics, ticker, year, trends, tickers_info):
    try:
        # Obtener datos anuales
        year_data = ticker_data[ticker_data['Date'].dt.year == year].copy()
        if len(year_data) < 60:
            return None
            
        # Calcular retornos mensuales para mejor estimación
        monthly_data = year_data.resample('M', on='Date')['Adj Close'].last()
        monthly_returns = monthly_data.pct_change().dropna()
        
        if len(monthly_returns) < 6:
            return None
            
        # Calcular retorno anual usando retornos mensuales compuestos
        annual_return = (1 + monthly_returns).prod() - 1
        
        # Volatilidad basada en retornos diarios para mayor precisión
        daily_returns = year_data['Adj Close'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252)
        
        # Sharpe ratio ajustado por riesgo
        risk_free_rate = 0.02  # Tasa libre de riesgo aproximada
        sharpe = (annual_return - risk_free_rate) / volatility if volatility != 0 else 0
        
        # Maximum Drawdown
        rolling_max = year_data['Adj Close'].expanding(min_periods=1).max()
        drawdowns = (year_data['Adj Close'] - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Datos fundamentales
        advanced_ticker = advanced_metrics[
            (advanced_metrics['Ticker'] == ticker) & 
            (advanced_metrics['Year'] == year)
        ]
        
        if advanced_ticker.empty:
            return None
            
        industry_match = tickers_info[tickers_info['Ticker'] == ticker]
        if industry_match.empty:
            return None
        
        industry = industry_match['Industry'].iloc[0]
        trend_score = get_industry_score(industry, year, trends)
        
        # Métricas de mercado
        market_cap = float(advanced_ticker['Market Cap'].iloc[0]) if 'Market Cap' in advanced_ticker.columns else 0
        beta = float(advanced_ticker['Beta'].iloc[0]) if 'Beta' in advanced_ticker.columns else 1.0
        
        # Score combinado con ajuste por tendencia
        combined_score = (
            0.25 * sharpe +
            0.25 * trend_score +
            0.20 * (1 + max_drawdown) +
            0.15 * (1/beta if beta > 0 else 0) +  # Menor beta = mejor
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
            'beta': beta,
            'monthly_returns': monthly_returns
        }
        
    except Exception as e:
        print(f"Error en {ticker}: {str(e)}")
        return None

def backtest_portfolio(portfolio_weights, ticker_data, year):
    try:
        # Crear DataFrame para retornos mensuales
        monthly_returns = {}
        
        for ticker, weight in portfolio_weights.items():
            ticker_df = ticker_data[ticker]
            year_data = ticker_df[ticker_df['Date'].dt.year == year].copy()
            year_data.set_index('Date', inplace=True)
            year_data['monthly'] = year_data.index.to_period('M')
            
            # Calcular retornos mensuales
            monthly_data = year_data.groupby('monthly')['Adj Close'].last()
            returns = monthly_data.pct_change().dropna()
            monthly_returns[ticker] = returns * weight
        
        if monthly_returns:
            # Combinar retornos de todos los activos
            returns_df = pd.DataFrame(monthly_returns)
            portfolio_returns = returns_df.sum(axis=1)
            
            # Calcular retorno anual
            annual_return = (1 + portfolio_returns).prod() - 1
            
            # Calcular volatilidad diaria
            daily_returns = {}
            for ticker, weight in portfolio_weights.items():
                ticker_df = ticker_data[ticker]
                year_data = ticker_df[ticker_df['Date'].dt.year == year].copy()
                returns = year_data['Adj Close'].pct_change().dropna()
                daily_returns[ticker] = returns * weight
            
            daily_df = pd.DataFrame(daily_returns)
            portfolio_daily = daily_df.sum(axis=1)
            volatility = portfolio_daily.std() * np.sqrt(252)
            
            # Calcular Sharpe Ratio
            risk_free_rate = 0.02
            sharpe = (annual_return - risk_free_rate) / volatility if volatility != 0 else 0
            
            return {
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe': sharpe
            }
            
    except Exception as e:
        print(f"Error en backtest: {str(e)}")
        return None

    return None

def main():
    print("Iniciando proceso...")
    data_path = "/Users/alee/Carteras-Tendenciales-Optimizaci-n-Basada-en-An-lisis-Multidimensional/src/data/tickers_data"
    results_path = "/Users/alee/Carteras-Tendenciales-Optimizaci-n-Basada-en-An-lisis-Multidimensional/src/results"
    
    if not os.path.exists(results_path):
        os.makedirs(results_path)   
         
    # Cargar datos
    ticker_data = {}
    for f in os.listdir(data_path):
        if f.endswith('_historical_data.csv'):
            ticker = f.split('_')[0]
            df = pd.read_csv(os.path.join(data_path, f))
            df['Date'] = pd.to_datetime(df['Date'])
            ticker_data[ticker] = df
    
    print(f"Datos cargados: {len(ticker_data)} tickers")
    
    trends_df = pd.read_csv('data/top_industry_trends_by_year.csv')
    advanced_metrics = pd.read_csv('data/metrics_advanced/advanced_metrics_all_tickers.csv')
    tickers_info = pd.read_csv('data/tickers_tech_med_clean.csv')
    
    trends = {int(year): {
        'industries': group['industry'].tolist(),
        'scores': group['composite_score'].tolist()
    } for year, group in trends_df.groupby('year')}
    
    results = {}
    detailed_results = []
    
    for year in range(2015, 2024):
        print(f"\nAnalizando año {year}")
        print("-" * 50)
        
        tickers_metrics = {}
        for ticker in ticker_data:
            metrics = calculate_metrics(ticker_data[ticker], advanced_metrics, ticker, year, trends, tickers_info)
            if metrics:
                tickers_metrics[ticker] = metrics
        
        print(f"\nTickers con métricas válidas: {len(tickers_metrics)}")
        
        if len(tickers_metrics) < 5:
            print(f"Insuficientes tickers para {year}")
            continue
        
        portfolio_weights = optimize_portfolio(tickers_metrics, year)
        if portfolio_weights:
            backtest_results = backtest_portfolio(portfolio_weights, ticker_data, year)
            
            if backtest_results:
                results[year] = {
                    'weights': portfolio_weights,
                    'expected_return': sum(tickers_metrics[t]['expected_return'] * w for t, w in portfolio_weights.items()),
                    'real_return': backtest_results['annual_return'],
                    'volatility': backtest_results['volatility'],
                    'sharpe': backtest_results['sharpe']
                }
                
                # Guardar información detallada
                for ticker, weight in portfolio_weights.items():
                    detailed_results.append({
                        'Year': year,
                        'Ticker': ticker,
                        'Industry': tickers_metrics[ticker]['industry'],
                        'Weight': weight,
                        'Expected_Return': tickers_metrics[ticker]['expected_return'],
                        'Volatility': tickers_metrics[ticker]['volatility'],
                        'Sharpe': tickers_metrics[ticker]['sharpe'],
                        'Trend_Score': tickers_metrics[ticker]['trend_score'],
                        'Beta': tickers_metrics[ticker]['beta'],
                        'Portfolio_Expected_Return': results[year]['expected_return'],
                        'Portfolio_Real_Return': results[year]['real_return'],
                        'Portfolio_Sharpe': results[year]['sharpe']
                    })
    
    # Guardar resultados
    if detailed_results:
        df_results = pd.DataFrame(detailed_results)
        df_results.to_csv(os.path.join(results_path, 'portfolio_results.csv'), index=False)
        print(f"\nResultados guardados en {os.path.join(results_path, 'portfolio_results.csv')}")
    
    return results

if __name__ == "__main__":
    results = main()