import pandas as pd
import numpy as np

# Leer y preparar los datos
df = pd.read_csv('detailed_trends_tech_med.csv', sep=';')

def get_top_industries_by_year(df, year):
    # Filtrar por año
    year_data = df[df['year'] == year].copy()
    
    # Agrupar por industria y calcular métricas agregadas
    industry_metrics = year_data.groupby('industry').agg({
        'total_citations': 'sum',
        'citation_velocity': 'mean',
        'impact_score': 'mean',
        'normalized_score': 'mean',
        'author_diversity': 'mean',
        'total_papers': 'sum',
        'keyword_relevance': 'max'  # Tomar la máxima relevancia de las keywords asociadas
    }).reset_index()
    
    # Crear score compuesto para industrias
    industry_metrics['composite_score'] = (
        0.25 * (industry_metrics['normalized_score'] / industry_metrics['normalized_score'].max()) +
        0.20 * (industry_metrics['total_citations'] / industry_metrics['total_citations'].max()) +
        0.20 * (industry_metrics['impact_score'] / industry_metrics['impact_score'].max()) +
        0.15 * (industry_metrics['citation_velocity'] / industry_metrics['citation_velocity'].max()) +
        0.10 * (industry_metrics['author_diversity'] / industry_metrics['author_diversity'].max()) +
        0.10 * (industry_metrics['total_papers'] / industry_metrics['total_papers'].max())
    )
    
    # Si no es el primer año, considerar tendencia histórica
    if year > 2010:
        prev_year_data = df[df['year'] == year-1].copy()
        prev_industry_metrics = prev_year_data.groupby('industry')['normalized_score'].mean().reset_index()
        prev_scores = dict(zip(prev_industry_metrics['industry'], prev_industry_metrics['normalized_score']))
        
        # Ajustar score basado en tendencia histórica
        industry_metrics['historical_factor'] = industry_metrics['industry'].map(lambda x: prev_scores.get(x, 0))
        industry_metrics['composite_score'] = industry_metrics['composite_score'] * 0.7 + \
                                            (industry_metrics['historical_factor'] / industry_metrics['historical_factor'].max()) * 0.3
    
    # Obtener top 5 industrias
    top_industries = industry_metrics.nlargest(5, 'composite_score')
    
    # Añadir las keywords más relevantes para cada industria
    def get_top_keywords(industry):
        industry_keywords = year_data[year_data['industry'] == industry]
        top_3_keywords = industry_keywords.nlargest(3, 'normalized_score')['keyword'].tolist()
        return ', '.join(top_3_keywords)
    
    top_industries['top_keywords'] = top_industries['industry'].apply(get_top_keywords)
    
    # Preparar resultado final
    result = pd.DataFrame({
        'year': year,
        'rank': range(1, 6),
        'industry': top_industries['industry'],
        'composite_score': top_industries['composite_score'],
        'total_citations': top_industries['total_citations'],
        'impact_score': top_industries['impact_score'],
        'normalized_score': top_industries['normalized_score'],
        'top_keywords': top_industries['top_keywords']
    })
    
    return result

# Procesar cada año
years = sorted(df['year'].unique())
all_top_industries = []

for year in years:
    top_industries = get_top_industries_by_year(df, year)
    all_top_industries.append(top_industries)

# Combinar resultados
final_trends = pd.concat(all_top_industries)

# Formatear y guardar resultados
final_trends = final_trends.round(4)
final_trends.to_csv('top_industry_trends_by_year.csv', index=False)

# Mostrar un ejemplo de los resultados
print("\nEjemplo de resultados para 2023:")
print(final_trends[final_trends['year'] == 2023].to_string(index=False))