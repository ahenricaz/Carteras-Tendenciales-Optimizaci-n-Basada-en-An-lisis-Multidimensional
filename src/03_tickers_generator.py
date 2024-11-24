import pandas as pd

def generate_clean_tickers_csv():
    """
    Genera un CSV limpio con los tickers verificados organizados por industria.
    """
    # Definir los tickers verificados por industria
    verified_tickers = {
        'Artificial Intelligence': {
            'Companies': ['GOOGL', 'MSFT', 'NVDA', 'IBM', 'AMD', 'META', 'AAPL', 'PLTR', 'AMZN'],
            'Description': 'Companies focused on AI development and implementation'
        },
        'Cloud Computing': {
            'Companies': ['MSFT', 'AMZN', 'GOOGL', 'ORCL', 'CRM', 'NOW', 'NET', 'DDOG', 'SNOW', 'VMW'],
            'Description': 'Cloud infrastructure and services providers'
        },
        'Cybersecurity': {
            'Companies': ['CRWD', 'PANW', 'FTNT', 'OKTA', 'ZS', 'CHKP', 'CYBR', 'RPD', 'TENB'],
            'Description': 'Cybersecurity solutions and services'
        },
        'Semiconductor': {
            'Companies': ['NVDA', 'AMD', 'TSM', 'INTC', 'QCOM', 'AVGO', 'ASML', 'AMAT', 'LRCX', 'MU'],
            'Description': 'Semiconductor design and manufacturing'
        },
        'Healthcare Technology': {
            'Companies': ['VEEV', 'CERN', 'TDOC', 'HCAT', 'ONEM', 'AMWL', 'PHR', 'ACCD'],
            'Description': 'Healthcare software and technology solutions'
        },
        'Biotechnology': {
            'Companies': ['MRNA', 'BNTX', 'REGN', 'VRTX', 'GILD', 'BIIB', 'AMGN', 'SGEN', 'INCY', 'ALNY'],
            'Description': 'Biotechnology research and development'
        },
        'Medical Devices': {
            'Companies': ['MDT', 'SYK', 'BSX', 'EW', 'ZBH', 'BAX', 'ISRG', 'HOLX', 'ABMD', 'NVRO'],
            'Description': 'Medical devices and equipment'
        },
        'Digital Health': {
            'Companies': ['TDOC', 'CERN', 'ACCD', 'AMWL', 'PHR', 'ONEM', 'OPRX'],
            'Description': 'Digital health and telemedicine'
        }
    }

    # Crear lista para el DataFrame
    records = []
    
    for industry, data in verified_tickers.items():
        for ticker in data['Companies']:
            records.append({
                'Industry': industry,
                'Ticker': ticker,
                'Type': 'Company',
                'Description': data['Description']
            })
    
    # Crear DataFrame
    df = pd.DataFrame(records)
    
    # Ordenar por Industry y Ticker
    df = df.sort_values(['Industry', 'Ticker'])
    
    # Guardar a CSV
    output_file = 'tickers_tech_med_clean_2020.csv'
    df.to_csv(output_file, index=False)
    
    # Mostrar resumen
    print("Resumen del archivo generado:")
    print(f"\nTotal de tickers: {len(df)}")
    print("\nTickers por industria:")
    industry_counts = df.groupby('Industry')['Ticker'].count().sort_values(ascending=False)
    print(industry_counts)
    
    print("\nPrimeras entradas de cada industria:")
    for industry in df['Industry'].unique():
        print(f"\n{industry}:")
        print(df[df['Industry'] == industry]['Ticker'].head().tolist())
    
    print(f"\nArchivo guardado como: {output_file}")
    
    return df

if __name__ == "__main__":
    tickers_df = generate_clean_tickers_csv()