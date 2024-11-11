import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import yfinance as yf
import time
from typing import Dict, List, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

class ResearchTrendsAnalyzer:
    def __init__(self, company_data_path: str):
        """
        Inicializa el analizador con datos de empresas y prepara NLTK
        """
        self.company_data = pd.read_csv(company_data_path)
        self._prepare_nltk()
        self.sector_keywords = self._extract_sector_keywords()
        
    def _prepare_nltk(self):
        """
        Descarga y prepara recursos necesarios de NLTK
        """
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def _extract_sector_keywords(self) -> Dict[str, Set[str]]:
        """
        Extrae keywords relevantes para cada sector desde los datos de empresas
        """
        sector_keywords = defaultdict(set)
        
        for _, row in self.company_data.iterrows():
            sector = row['Sector']
            industry = row['Industry']
            company_name = row['Company Name']
            
            # Procesar cada campo
        for text in [sector, industry, company_name]:
            if isinstance(text, str):  # Verifica si text es una cadena de texto
                words = word_tokenize(text.lower())
                words = [self.lemmatizer.lemmatize(word) for word in words 
                        if word.isalnum() and word not in self.stop_words]
                sector_keywords[sector].update(words)

        
        return dict(sector_keywords)

    def fetch_research_papers(self, years_back: int = 5) -> List[Dict]:
        """
        Obtiene artículos de investigación de múltiples fuentes
        """
        papers = []
        papers.extend(self._fetch_from_arxiv(years_back))
        papers.extend(self._fetch_from_springer(years_back))
        return papers

    def _fetch_from_arxiv(self, years_back: int) -> List[Dict]:
        """
        Obtiene artículos de arXiv
        """
        base_url = "http://export.arxiv.org/api/query?"
        papers = []
        
        # Categorías relevantes de arXiv
        categories = ['q-fin', 'econ']
        
        for category in categories:
            start = 0
            total_results = 1000  # Límite para evitar sobrecarga
            
            while start < total_results:
                query = (
                    f"cat:{category}.*"
                    f"+AND+submittedDate:[{datetime.now().year-years_back}0101 TO {datetime.now().year}1231]"
                )
                params = {
                    'search_query': query,
                    'start': start,
                    'max_results': 100
                }
                
                response = requests.get(base_url, params=params)
                if response.status_code != 200:
                    break
                
                root = ET.fromstring(response.text)
                
                # Procesar resultados
                for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                    paper = {
                        'title': entry.find('{http://www.w3.org/2005/Atom}title').text,
                        'abstract': entry.find('{http://www.w3.org/2005/Atom}summary').text,
                        'published': entry.find('{http://www.w3.org/2005/Atom}published').text,
                        'source': 'arxiv'
                    }
                    papers.append(paper)
                
                start += 100
                time.sleep(3)  # Respetar límites de la API
                
        return papers

    def _fetch_from_springer(self, years_back: int) -> List[Dict]:
        """
        Obtiene artículos de Springer Nature (API abierta)
        """
        base_url = "http://api.springer.com/metadata/json"
        papers = []
        
        subjects = ['Economics', 'Finance', 'Business']
        
        for subject in subjects:
            params = {
                'q': f'subject:"{subject}"',
                's': 1,
                'p': 100
            }
            
            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                data = response.json()
                
                for record in data.get('records', []):
                    paper = {
                        'title': record.get('title', ''),
                        'abstract': record.get('abstract', ''),
                        'published': record.get('publicationDate', ''),
                        'source': 'springer'
                    }
                    papers.append(paper)
            
            time.sleep(1)  # Respetar límites de la API
            
        return papers

    def analyze_trends(self, papers: List[Dict]) -> Dict:
        """
        Analiza tendencias en los papers y las relaciona con sectores
        """
        # Preparar textos para análisis
        texts = [f"{p['title']} {p['abstract']}" for p in papers]
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # Análisis TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            stop_words='english'
        )
        tfidf_matrix = vectorizer.fit_transform(processed_texts)
        
        # Análisis LDA para temas
        lda = LatentDirichletAllocation(n_components=10, random_state=42)
        lda_output = lda.fit_transform(tfidf_matrix)
        
        # Extraer keywords por tema
        feature_names = vectorizer.get_feature_names()
        trends = self._extract_trends(lda, feature_names, papers)
        
        # Relacionar tendencias con sectores
        sector_trends = self._map_trends_to_sectors(trends)
        
        return sector_trends

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocesa texto para análisis
        """
        # Tokenización y limpieza
        words = word_tokenize(text.lower())
        words = [self.lemmatizer.lemmatize(word) for word in words 
                if word.isalnum() and word not in self.stop_words]
        return ' '.join(words)

    def _extract_trends(self, lda, feature_names: List[str], papers: List[Dict]) -> List[Dict]:
        """
        Extrae tendencias principales de los temas LDA
        """
        trends = []
        
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[:-10-1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            
            # Encontrar papers más relevantes para este tema
            relevant_papers = self._find_relevant_papers(topic_idx, lda, papers)
            
            trend = {
                'keywords': top_words,
                'papers': relevant_papers,
                'strength': float(topic.sum())  # Importancia del tema
            }
            trends.append(trend)
            
        return trends

    def _find_relevant_papers(self, topic_idx: int, lda, papers: List[Dict]) -> List[Dict]:
        """
        Encuentra papers más relevantes para un tema
        """
        paper_scores = lda.transform(self.vectorizer.transform([p['abstract'] for p in papers]))
        relevant_indices = paper_scores[:, topic_idx].argsort()[-5:][::-1]
        
        return [papers[i] for i in relevant_indices]

    def _map_trends_to_sectors(self, trends: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Mapea tendencias identificadas a sectores específicos
        """
        sector_trends = defaultdict(list)
        
        for trend in trends:
            # Calcular similitud con cada sector
            sector_scores = {}
            for sector, keywords in self.sector_keywords.items():
                score = self._calculate_trend_sector_similarity(trend['keywords'], keywords)
                if score > 0.1:  # Umbral de similitud
                    sector_scores[sector] = score
            
            # Asignar tendencia a sectores relevantes
            for sector, score in sector_scores.items():
                sector_trends[sector].append({
                    'keywords': trend['keywords'],
                    'papers': trend['papers'],
                    'relevance_score': score,
                    'strength': trend['strength']
                })
        
        return dict(sector_trends)

    def _calculate_trend_sector_similarity(self, trend_keywords: List[str], 
                                        sector_keywords: Set[str]) -> float:
        """
        Calcula similitud entre keywords de tendencia y sector
        """
        trend_set = set(trend_keywords)
        intersection = trend_set.intersection(sector_keywords)
        return len(intersection) / (len(trend_set) + len(sector_keywords) - len(intersection))

def main(company_data_path: str = 'company_data.csv'):
    # Inicializar analizador
    analyzer = ResearchTrendsAnalyzer(company_data_path)
    
    # Obtener papers
    print("Obteniendo artículos de investigación...")
    papers = analyzer.fetch_research_papers(years_back=5)
    
    # Analizar tendencias
    print("Analizando tendencias...")
    sector_trends = analyzer.analyze_trends(papers)
    
    # Guardar resultados
    results_df = pd.DataFrame([
        {
            'sector': sector,
            'trends': trends,
            'num_trends': len(trends),
            'avg_strength': np.mean([t['strength'] for t in trends])
        }
        for sector, trends in sector_trends.items()
    ])
    
    results_df.to_csv('research_trends_results.csv', index=False)
    return results_df

if __name__ == "__main__":
    results = main()
    print("\nAnálisis de tendencias completado. Resultados guardados en 'research_trends_results.csv'")