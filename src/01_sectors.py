import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import ngrams
import nltk
import requests
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import logging
import time
from datetime import datetime
from scipy.stats import linregress
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class SectorKeywordExtractor:
    def __init__(self, company_data_path: str):
        """
        Initialize the keyword extractor with enhanced sector analysis
        """
        logger.info("Initializing Enhanced SectorKeywordExtractor...")
        
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        
        # Convertir stopwords a lista y añadir palabras específicas
        self.stop_words = list(stopwords.words('english'))
        domain_stops = ['company', 'business', 'industry', 'sector', 'market']
        self.stop_words.extend(domain_stops)
        
        logger.info("Loading and preprocessing company data...")
        self.company_data = pd.read_csv(company_data_path)
        self.sector_docs = self._prepare_sector_documents()
        self.word2vec_model = None
        
        # Cargar palabras clave predefinidas por sector
        self.sector_specific_keywords = self._load_sector_keywords()
        
        # Palabras a filtrar (expandida)
        self.filter_words = {
            # Términos genéricos de tecnología
            'general', 'system', 'tech', 'solution', 'platform', 'technology', 'technologies',
            'application', 'device', 'product', 'management', 'service', 'services',
            'data', 'digital', 'smart', 'advanced', 'innovative', 'solutions', 'systems',
            'analysis', 'analytics', 'development', 'processing', 'infrastructure',
            
            # Términos comerciales genéricos
            'business', 'enterprise', 'commercial', 'industrial', 'professional',
            'solutions', 'provider', 'platform', 'service', 'services',
            
            # Modificadores sin contexto
            'next', 'generation', 'modern', 'new', 'integrated', 'enhanced',
            'automated', 'strategic', 'custom', 'specialized', 'advanced'
        }
        
        # Mapeo de palabras clave a industrias específicas
        self.industry_mapping = self._create_industry_mapping(company_data_path)
        
    def _create_industry_mapping(self, data_path: str) -> Dict[str, str]:
        """
        Crear un mapeo detallado de keywords a industrias específicas
        """
        df = pd.read_csv(data_path)
        mapping = {}
        
        for _, row in df.iterrows():
            industry = row['Industry']
            sector = row['Sector']
            
            # Procesar el nombre de la industria
            words = set(word.lower() for word in re.findall(r'\w+', industry))
            
            # Crear mapeo para cada palabra significativa
            for word in words:
                if len(word) > 3 and word not in self.filter_words:
                    # Guardar tanto la industria como el sector
                    mapping[word] = {'industry': industry, 'sector': sector}
        
        return mapping
        
    def _load_sector_keywords(self) -> Dict[str, Set[str]]:
        """
        Cargar keywords predefinidas por sector
        """
        return {
            'Computer and Technology': {
                'artificial intelligence', 'machine learning', 'cloud computing', 
                'cybersecurity', 'blockchain', 'data analytics', '5g', 'iot',
                'quantum computing', 'edge computing', 'neural networks'
            },
            'Medical': {
                'biotechnology', 'telemedicine', 'genomics', 'digital health',
                'precision medicine', 'immunotherapy', 'medical devices',
                'drug discovery', 'clinical trials', 'personalized medicine'
            },
            'Finance': {
                'fintech', 'blockchain', 'cryptocurrency', 'digital banking',
                'payment systems', 'insurtech', 'regtech', 'wealth management',
                'algorithmic trading', 'open banking'
            },
            'Energy': {
                'renewable energy', 'solar power', 'wind energy', 'energy storage',
                'smart grid', 'clean tech', 'hydrogen', 'sustainability',
                'carbon capture', 'electric vehicles'
            },
            'Retail-Wholesale': {
                'e-commerce', 'online marketplaces', 'retail tech', 'omnichannel retail', 
                'last-mile delivery', 'subscription retail', 'fashion retail', 
                'luxury goods', 'health and wellness retail', 'consumer electronics'
            },
            'Construction': {
                'green building', 'sustainable construction', 'smart cities', 
                'modular construction', 'prefabricated homes', 'civil engineering', 
                'infrastructure development', 'construction robotics', '3d printing', 
                'construction project management', 'eco-friendly materials'
            },
            'Transportation': {
                'autonomous vehicles', 'electric vehicles', 'hyperloop', 
                'ride-sharing', 'urban mobility', 'logistics tech', 
                'last-mile logistics', 'cold chain logistics', 
                'drone delivery', 'smart transportation systems', 
                'freight forwarding', 'public transit'
            },
            'Aerospace': {
                'space exploration', 'satellites', 'drone technology', 'reusable rockets', 
                'hypersonic technology', 'space tourism', 'missile systems', 
                'space debris management', 'aerospace ai', 'unmanned aerial vehicles', 
                'satellite communication', 'astrophysics'
            },
            'Utilities': {
                'renewable energy', 'smart grid', 'energy storage', 'wastewater management', 
                'bioenergy', 'telecommunications utilities', 'carbon capture', 
                'district heating', 'smart metering', 'clean energy integration', 
                'stormwater management', 'battery recycling'
            }
        }

    def _clean_text(self, text: str) -> str:
        """
        Limpieza mejorada de texto
        """
        # Convertir a minúsculas y eliminar caracteres especiales
        text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
        
        # Tokenizar
        tokens = word_tokenize(text)
        
        # Filtrar stopwords y palabras cortas
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        
        # Unir tokens
        return ' '.join(tokens)

    def _clean_and_contextualize_keyword(self, keyword: str, sector: str) -> Tuple[str, bool, str]:
            """
            Clean keyword and return industry context separately
            """
            words = keyword.lower().split()
            
            if any(word in self.filter_words for word in words):
                return None, False, None
                
            matched_industries = set()
            for word in words:
                if word in self.industry_mapping:
                    industry_info = self.industry_mapping[word]
                    if industry_info['sector'] == sector:
                        matched_industries.add(industry_info['industry'])
            
            if matched_industries:
                industry_context = next(iter(matched_industries))
                return keyword, True, industry_context
            
            sector_keywords = self._load_sector_keywords()
            if sector in sector_keywords and any(word in sector_keywords[sector] for word in words):
                return keyword, True, None
                
            return None, False, None

    def _prepare_sector_documents(self) -> Dict[str, str]:
        """
        Preparar documentos por sector con mejor estructuración
        """
        sector_docs = defaultdict(list)
        
        # Agrupar por sector
        for sector, group in self.company_data.groupby('Sector'):
            # Combinar descripción e industria
            sector_text = []
            for _, row in group.iterrows():
                industry = str(row['Industry'])
                description = str(row.get('Description', ''))
                
                # Limpiar y añadir texto
                clean_industry = self._clean_text(industry)
                clean_description = self._clean_text(description)
                
                sector_text.extend([clean_industry, clean_description])
            
            sector_docs[sector] = ' '.join(sector_text)
            
        return dict(sector_docs)

    def _extract_key_phrases(self, text: str) -> List[str]:
        """
        Extraer frases clave usando POS tagging
        """
        tokens = word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        
        phrases = []
        current_phrase = []
        
        for word, tag in pos_tags:
            # Patrones para identificar términos técnicos y conceptos
            if tag.startswith(('NN', 'JJ')):  # Sustantivos y adjetivos
                current_phrase.append(word)
            else:
                if current_phrase:
                    phrases.append(' '.join(current_phrase))
                    current_phrase = []
        
        if current_phrase:
            phrases.append(' '.join(current_phrase))
            
        return phrases

    def _calculate_sector_specificity(self, term: str, sector: str) -> float:
        """
        Calcular especificidad del término para el sector
        """
        sector_freq = self.sector_docs[sector].count(term)
        other_sectors_freq = sum(doc.count(term) for s, doc in self.sector_docs.items() if s != sector)
        
        if other_sectors_freq == 0:
            return sector_freq
        
        return sector_freq / (other_sectors_freq + 1)

    def _train_word2vec(self, texts: List[str]):
        """
        Entrenar modelo Word2Vec mejorado
        """
        logger.info("Training enhanced Word2Vec model...")
        sentences = [word_tokenize(text.lower()) for text in texts]
        
        # Parámetros mejorados para capturar mejor las relaciones semánticas
        self.word2vec_model = Word2Vec(
            sentences,
            vector_size=200,  # Aumentado para mejor representación
            window=10,        # Ventana más grande para capturar más contexto
            min_count=2,
            workers=4,
            epochs=20         # Más épocas para mejor entrenamiento
        )

    def _get_additional_sector_keywords(self, sector: str, n_needed: int) -> List[Tuple[str, float]]:
        """
        Obtener keywords adicionales del sector cuando se necesitan más después del filtrado
        """
        # Obtener todas las industrias del sector
        sector_industries = [
            industry for industry, info in self.industry_mapping.items()
            if info['sector'] == sector
        ]
        
        additional_keywords = []
        # Usar las industrias directamente como keywords con un score base
        base_score = 0.5  # Score más bajo que los keywords principales
        
        for industry in sector_industries:
            if len(additional_keywords) >= n_needed:
                break
            
            # Solo añadir si no está ya en los keywords principales
            keyword = f"{industry} [{self.industry_mapping[industry]['industry']}]"
            additional_keywords.append((keyword, base_score))
        
        return additional_keywords[:n_needed]
    
    def extract_keywords(self, n_keywords: int = 20) -> Dict[str, List[Tuple[str, float, str]]]:
        logger.info("Starting enhanced keyword extraction process...")
        
        tfidf = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 3),
            stop_words=list(self.stop_words)
        )
        
        tfidf_matrix = tfidf.fit_transform(list(self.sector_docs.values()))
        feature_names = np.array(tfidf.get_feature_names_out())
        
        self._train_word2vec(list(self.sector_docs.values()))
        
        sector_keywords = {}
        
        for sector_idx, (sector, doc) in enumerate(tqdm(self.sector_docs.items(), desc="Processing sectors")):
            logger.info(f"Processing sector: {sector}")
            
            sector_scores = tfidf_matrix[sector_idx].toarray()[0]
            key_phrases = self._extract_key_phrases(doc)
            keyword_scores = {}
            
            for term, score in zip(feature_names, sector_scores):
                if score > 0:
                    keyword_scores[term] = score
            
            for phrase in key_phrases:
                if phrase in keyword_scores:
                    keyword_scores[phrase] *= 1.5
            
            for term in list(keyword_scores.keys()):
                specificity = self._calculate_sector_specificity(term, sector)
                keyword_scores[term] *= (1 + specificity)
            
            tech_terms = {'ai', 'automation', 'digital', 'innovation', 'smart', 'technology'}
            for term in keyword_scores:
                if any(tech in term.lower() for tech in tech_terms):
                    keyword_scores[term] *= 1.3
            
            for predefined_term in self.sector_specific_keywords.get(sector, []):
                if predefined_term in keyword_scores:
                    keyword_scores[predefined_term] *= 2.0
            
            max_score = max(keyword_scores.values())
            keyword_scores = {k: v/max_score for k, v in keyword_scores.items()}
            
            best_keywords = sorted(
                keyword_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:n_keywords]
            
            sector_keywords[sector] = best_keywords
        
        cleaned_sector_keywords = {}
        for sector, keywords in sector_keywords.items():
            cleaned_keywords = []
            
            for keyword, score in keywords:
                cleaned_keyword, is_valid, industry = self._clean_and_contextualize_keyword(keyword, sector)
                if is_valid and cleaned_keyword:
                    cleaned_keywords.append((cleaned_keyword, score, industry))
            
            if len(cleaned_keywords) < n_keywords:
                additional_keywords = self._get_additional_sector_keywords(sector, 
                                                                        n_keywords - len(cleaned_keywords))
                additional_with_industry = [
                    (kw.split('[')[0].strip(), score, kw.split('[')[1].rstrip(']'))
                    for kw, score in additional_keywords
                ]
                cleaned_keywords.extend(additional_with_industry)
            
            cleaned_sector_keywords[sector] = cleaned_keywords[:n_keywords]
        
        return cleaned_sector_keywords
        
class TrendAnalyzer:
    def __init__(self, years: List[int] = None):
        """
        Initialize trend analyzer with improved metrics
        """
        logger.info("Initializing Enhanced TrendAnalyzer...")
        self.years = years or [
            2010, 2011, 2012, 2013, 2014, 2015, 2016, 
            2017, 2018, 2019, 2020, 2021 ,2022, 2023,
            2024
        ]
        
    def _extract_papers(self, query: str, year: int, limit: int = 100) -> List[Dict]:
        """
        Extract papers with enhanced filtering
        """
        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "year": str(year),
            "limit": limit,
            "fields": "title,abstract,citationCount,influentialCitationCount,year,venue,authors,url"
        }
        
        try:
            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                papers = response.json().get('data', [])
                # Filtrar papers por relevancia
                filtered_papers = [
                    p for p in papers
                    if p.get('venue') and  # Solo papers de venues reconocidas
                    p.get('abstract') and   # Debe tener abstract
                    len(p.get('authors', [])) > 0  # Debe tener autores
                ]
                return filtered_papers
            return []
        except Exception as e:
            logger.error(f"Error extracting papers: {e}")
            return []
            
    def _calculate_trend_metrics(self, papers: List[Dict], term: str) -> Dict:
        """
        Calculate enhanced trend metrics
        """
        if not papers:
            return {}
            
        # Métricas básicas
        citations = [p.get('citationCount', 0) for p in papers]
        influential_citations = [p.get('influentialCitationCount', 0) for p in papers]
        
        # Calcular métricas avanzadas
        metrics = {
            'total_papers': len(papers),
            'total_citations': sum(citations),
            'avg_citations': np.mean(citations) if citations else 0,
            'influential_citations': sum(influential_citations),
            'impact_factor': sum(influential_citations) / len(papers) if papers else 0,
            'h_index': self._calculate_h_index(citations),
            'citation_velocity': self._calculate_citation_velocity(papers),
            'author_diversity': len(set(author['authorId'] for p in papers for author in p.get('authors', [])))
        }
        
        # Análisis temporal
        if len(papers) > 1:
            years = [p.get('year', 0) for p in papers]
            citation_years = list(zip(years, citations))
            citation_years.sort()
            
            x = [float(year) for year, _ in citation_years]
            y = [float(c) for _, c in citation_years]
            
            if len(x) > 1 and len(set(x)) > 1:
                slope, _, r_value, _, _ = linregress(x, y)
                metrics.update({
                    'growth_rate': slope,
                    'trend_strength': abs(r_value),
                    'momentum': slope * abs(r_value)
                })
        
        return metrics
    
    def _calculate_h_index(self, citations: List[int]) -> int:
        """
        Calculate h-index for the trend
        """
        if not citations:
            return 0
        sorted_citations = sorted(citations, reverse=True)
        h = 0
        for i, c in enumerate(sorted_citations, 1):
            if c >= i:
                h = i
            else:
                break
        return h
    
    def _calculate_citation_velocity(self, papers: List[Dict]) -> float:
        """
        Calculate citation velocity (citations per year)
        """
        if not papers:
            return 0
        
        current_year = datetime.now().year
        total_citations = sum(p.get('citationCount', 0) for p in papers)
        years_active = current_year - min(p.get('year', current_year) for p in papers) + 1
        
        return total_citations / years_active if years_active > 0 else 0
      
    def analyze_trends(self, sector_keywords: Dict[str, List[Tuple[str, float, str]]]) -> pd.DataFrame:
        """
        Analyze trends with separate industry column
        """
        trend_data = []
        
        total_iterations = sum(len(keywords) * len(self.years) for keywords in sector_keywords.values())
        
        with tqdm(total=total_iterations, desc="Analyzing trends") as pbar:
            for sector, keywords in sector_keywords.items():
                for keyword, keyword_score, industry in keywords:
                    sector_trend_data = []
                    for year in self.years:
                        papers = self._extract_papers(keyword, year)
                        metrics = self._calculate_trend_metrics(papers, keyword)
                        
                        if metrics:
                            metrics.update({
                                'keyword_relevance': keyword_score,
                                'year': year,
                                'industry': industry
                            })
                            sector_trend_data.append(metrics)
                        
                        time.sleep(1)
                        pbar.update(1)
                    
                    if sector_trend_data:
                        trend_data.extend([{
                            'sector': sector,
                            'keyword': keyword,
                            **data
                        } for data in sector_trend_data])
        
        df = pd.DataFrame(trend_data)
        
        if not df.empty:
            # Normalización y cálculo de scores
            numeric_columns = ['total_citations', 'influential_citations', 'h_index', 
                             'citation_velocity', 'author_diversity']
            
            # Normalizar métricas
            df_normalized = df.copy()
            for col in numeric_columns:
                if col in df.columns:
                    df_normalized[f'{col}_norm'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            
            # Calcular scores compuestos
            df['impact_score'] = (
                0.3 * df_normalized['total_citations_norm'] +
                0.3 * df_normalized['influential_citations_norm'] +
                0.2 * df_normalized['h_index_norm'] +
                0.1 * df_normalized['citation_velocity_norm'] +
                0.1 * df_normalized['author_diversity_norm']
            )
            
            # Ajustar por relevancia del keyword
            df['relevance_adjusted_score'] = df['impact_score'] * df['keyword_relevance']
            
            # Calcular momentum y tendencia
            df['trend_momentum'] = df.groupby('keyword')['relevance_adjusted_score'].diff()
            df['trend_acceleration'] = df.groupby('keyword')['trend_momentum'].diff()
            
            # Calcular score final
            df['final_score'] = (
                0.4 * df['relevance_adjusted_score'] +
                0.3 * df['trend_momentum'].fillna(0) +
                0.3 * df['trend_acceleration'].fillna(0)
            )
            
            # Normalizar score final
            df['normalized_score'] = (df['final_score'] - df['final_score'].min()) / \
                                   (df['final_score'].max() - df['final_score'].min())
            
            # Añadir métricas de sector
            df['sector_avg_score'] = df.groupby('sector')['normalized_score'].transform('mean')
            df['sector_trend_strength'] = df.groupby('sector')['normalized_score'].transform(lambda x: x.std())
            
            # Obtener top tendencias por sector y año
            df = df.sort_values(['sector', 'year', 'normalized_score'], ascending=[True, True, False])
            df = df.groupby(['sector', 'year']).head(20).reset_index(drop=True)
        
        logger.info("Trend analysis completed")
        return df

def analyze_and_visualize_trends(df: pd.DataFrame, output_dir: str = '.'):
    """
    Analizar y visualizar tendencias con métricas avanzadas
    """
    logger.info("Starting trend analysis and visualization...")
    
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Crear resumen de tendencias
    trend_summary = pd.DataFrame({
        'total_papers': df.groupby(['sector', 'year'])['total_papers'].sum(),
        'avg_impact': df.groupby(['sector', 'year'])['impact_score'].mean(),
        'top_keywords': df.groupby(['sector', 'year'])['keyword'].agg(
            lambda x: ', '.join(x.value_counts().head(5).index)
        )
    }).reset_index()
    
    # 2. Análisis temporal
    temporal_analysis = df.groupby(['year', 'sector'])['normalized_score'].agg([
        'mean', 'std', 'min', 'max'
    ]).reset_index()
    
    # 3. Análisis de correlación entre métricas
    correlation_matrix = df[[
        'total_citations', 'influential_citations', 'h_index',
        'citation_velocity', 'normalized_score'
    ]].corr()
    
    # 4. Generar visualizaciones
    plt.style.use('seaborn')
    
    # 4.1 Evolución temporal por sector
    plt.figure(figsize=(15, 8))
    for sector in df['sector'].unique():
        sector_data = temporal_analysis[temporal_analysis['sector'] == sector]
        # Convertir a arrays de numpy antes de plotear
        years = sector_data['year'].to_numpy()
        means = sector_data['mean'].to_numpy()
        plt.plot(years, means, label=sector, marker='o')
    
    plt.title('Evolución de Tendencias por Sector')
    plt.xlabel('Año')
    plt.ylabel('Score Normalizado (Promedio)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trend_evolution.png'))
    plt.close()
    
    # 4.2 Heatmap de correlaciones
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='YlOrRd')
    plt.title('Correlación entre Métricas')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metric_correlation.png'))
    plt.close()
    
    # 4.3 Top keywords por sector
    top_keywords_df = df.groupby(['sector', 'keyword'])['normalized_score'].mean().reset_index()
    top_keywords_df = top_keywords_df.sort_values(['sector', 'normalized_score'], ascending=[True, False])
    top_keywords_viz = top_keywords_df.groupby('sector').head(10)
    
    plt.figure(figsize=(15, len(df['sector'].unique()) * 2))
    sectors = df['sector'].unique()
    for i, sector in enumerate(sectors, 1):
        plt.subplot(len(sectors), 1, i)
        sector_data = top_keywords_viz[top_keywords_viz['sector'] == sector]
        # Convertir a arrays de numpy
        scores = sector_data['normalized_score'].to_numpy()
        keywords = sector_data['keyword'].to_numpy()
        plt.barh(range(len(keywords)), scores)
        plt.yticks(range(len(keywords)), keywords)
        plt.title(f'Top Keywords - {sector}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_keywords.png'))
    plt.close()
    
    # 5. Análisis de tendencias emergentes
    emerging_trends = pd.DataFrame({
        'sector': df['sector'].unique(),
        'top_emerging': [
            df[df['sector'] == sector]
            .sort_values('trend_momentum', ascending=False)
            ['keyword'].head(5).tolist()
            for sector in df['sector'].unique()
        ]
    })
    
    # 6. Guardar resultados
    trend_summary.to_csv(os.path.join(output_dir, 'trend_summary.csv'), index=False)
    temporal_analysis.to_csv(os.path.join(output_dir, 'temporal_analysis.csv'), index=False)
    correlation_matrix.to_csv(os.path.join(output_dir, 'metric_correlations.csv'))
    emerging_trends.to_csv(os.path.join(output_dir, 'emerging_trends.csv'), index=False)
    
    # 7. Generar reporte de insights
    with open(os.path.join(output_dir, 'trend_insights.txt'), 'w') as f:
        f.write("ANÁLISIS DE TENDENCIAS\n")
        f.write("=====================\n\n")
        
        # 7.1 Tendencias globales
        f.write("Tendencias Globales:\n")
        for sector in df['sector'].unique():
            sector_data = df[df['sector'] == sector]
            top_trends = sector_data.nlargest(5, 'normalized_score')
            f.write(f"\n{sector}:\n")
            for _, trend in top_trends.iterrows():
                f.write(f"- {trend['keyword']} (Score: {trend['normalized_score']:.3f})\n")
        
        # 7.2 Sectores más dinámicos
        sector_dynamics = df.groupby('sector')['trend_momentum'].mean().sort_values(ascending=False)
        f.write("\nSectores más Dinámicos:\n")
        for sector, momentum in sector_dynamics.items():
            f.write(f"- {sector}: {momentum:.3f}\n")
        
        # 7.3 Análisis de velocidad de adopción
        f.write("\nVelocidad de Adopción por Sector:\n")
        adoption_speed = df.groupby('sector')['citation_velocity'].mean().sort_values(ascending=False)
        for sector, speed in adoption_speed.items():
            f.write(f"- {sector}: {speed:.2f} citas/año\n")
        
        # 7.4 Resumen estadístico
        f.write("\nResumen Estadístico por Sector:\n")
        stats = df.groupby('sector')['normalized_score'].agg(['mean', 'std', 'max']).round(3)
        for sector in stats.index:
            f.write(f"\n{sector}:\n")
            f.write(f"  Media: {stats.loc[sector, 'mean']:.3f}\n")
            f.write(f"  Desv. Est.: {stats.loc[sector, 'std']:.3f}\n")
            f.write(f"  Max Score: {stats.loc[sector, 'max']:.3f}\n")

    # Add industry-based trend analysis
    industry_analysis = df.groupby(['sector', 'industry', 'year'])['normalized_score'].agg([
        'mean', 'std', 'count'
    ]).reset_index()
    
    # Save industry analysis
    industry_analysis.to_csv(os.path.join(output_dir, 'industry_analysis.csv'), index=False)
    
    # Add industry visualization
    plt.figure(figsize=(15, 10))
    for sector in df['sector'].unique():
        sector_data = df[df['sector'] == sector]
        for industry in sector_data['industry'].unique():
            if pd.notna(industry):
                industry_data = sector_data[sector_data['industry'] == industry]
                grouped_data = industry_data.groupby('year')['normalized_score'].mean()
                # Convertir a numpy arrays antes de graficar
                years = np.array(grouped_data.index)
                scores = np.array(grouped_data.values)
                plt.plot(years, scores, label=f"{sector} - {industry}", alpha=0.7)
    
    plt.title('Evolución de Tendencias por Industria')
    plt.xlabel('Año')
    plt.ylabel('Puntuación Normalizada')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'industry_trends.png'))
    plt.close()

    return trend_summary, temporal_analysis, correlation_matrix, industry_analysis

def main():
    logger.info("Starting main execution...")
    
    try:
        # Inicializar extractor y obtener keywords
        logger.info("Initializing keyword extractor...")
        extractor = SectorKeywordExtractor('company_data.csv')
        
        logger.info("Extracting sector keywords...")
        sector_keywords = extractor.extract_keywords(n_keywords=20)
        
        # Analizar tendencias
        logger.info("Starting trend analysis...")
        analyzer = TrendAnalyzer()
        trends_df = analyzer.analyze_trends(sector_keywords)
        
        # Analizar y visualizar resultados
        logger.info("Analyzing and visualizing trends...")
        analyze_and_visualize_trends(trends_df, output_dir='trend_analysis_results')
        
        # Guardar resultados detallados
        trends_df.to_csv('trend_analysis_results/detailed_trends.csv', index=False)
        
        logger.info("Execution completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()