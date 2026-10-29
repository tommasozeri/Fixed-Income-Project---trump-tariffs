import requests
import polars as pl


## Generic function to fetch data from FRED using the FRED API
def fetch_fred_data(series_id, start_date, end_date, api_key):
    """
    Fetch data from FRED API for a specific series and return as Polars DataFrame
    """
    if api_key is None:
        print(f"⚠️  Cannot fetch {series_id}: FRED API key not available")
        return None
        
    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': start_date,
        'observation_end': end_date
    }
    
    response = requests.get("https://api.stlouisfed.org/fred/series/observations", params=params)
    
    if response.status_code == 200:
        data = response.json()
        observations = data['observations']
        
        # Convert to Polars DataFrame
        df = pl.DataFrame(observations)
        
        # Convert date column and filter out invalid values
        df = df.with_columns([
            pl.col('date').str.to_date('%Y-%m-%d').alias('date'),
            pl.col('value').cast(pl.Float64, strict=False).alias('value')
        ]).filter(
            pl.col('value').is_not_null()
        ).select(['date', 'value']).rename({'value': series_id})
        
        return df
    else:
        print(f"Error fetching {series_id}: {response.status_code}")
        if response.status_code == 400:
            print("This might be due to an invalid API key. Please check your FRED API key.")
        return None


## Functions to retrieve yield curve information
def fetch_yield_curve_data(
        treasury_series = {
            '3M': 'DGS3MO',    # 3-Month Treasury Constant Maturity Rate
            '6M': 'DGS6MO',    # 6-Month Treasury Constant Maturity Rate
            '1Y': 'DGS1',      # 1-Year Treasury Constant Maturity Rate
            '2Y': 'DGS2',      # 2-Year Treasury Constant Maturity Rate
            '3Y': 'DGS3',      # 3-Year Treasury Constant Maturity Rate
            '5Y': 'DGS5',      # 5-Year Treasury Constant Maturity Rate
            '7Y': 'DGS7',      # 7-Year Treasury Constant Maturity Rate
            '10Y': 'DGS10',    # 10-Year Treasury Constant Maturity Rate
            '20Y': 'DGS20',    # 20-Year Treasury Constant Maturity Rate
            '30Y': 'DGS30',    # 30-Year Treasury Constant Maturity Rate
        },
        start_date='2015-01-01', 
        end_date='2025-08-01',
        FRED_API_KEY=None
    ):
    """
    Fetch all Treasury yield curve data and return as Polars DataFrame
    """
    if FRED_API_KEY is None:
        print("⚠️  FRED API key not loaded. Please check data/fred/api_key.py file.")
        print("The file should contain: My_FRED_API_KEY = 'your_actual_api_key_here'")
        return None
    
    yield_data = None
    successful_fetches = 0
    
    for maturity, series_id in treasury_series.items():
        print(f"Fetching {maturity} Treasury data ({series_id})...")
        
        df = fetch_fred_data(series_id, start_date, end_date, FRED_API_KEY)
        
        if df is not None:
            successful_fetches += 1
            if yield_data is None:
                yield_data = df
            else:
                # Use Polars join with coalesce strategy to handle overlapping columns
                yield_data = yield_data.join(df, on='date', how='full', coalesce=True)
    
    if yield_data is not None and successful_fetches > 0:
        # Sort by date
        yield_data = yield_data.sort('date')
        
        # Rename columns to maturity names
        column_mapping = {series_id: maturity for maturity, series_id in treasury_series.items()}
        yield_data = yield_data.rename(column_mapping)
        
        print(f"\n✅ Successfully fetched {successful_fetches}/{len(treasury_series)} yield series")
        print(f"Data range: {yield_data['date'].min()} to {yield_data['date'].max()}")
        print(f"Data shape: {yield_data.shape}")
        
    elif successful_fetches == 0:
        print("\n❌ Failed to fetch any yield curve data. Please check your API key.")
        return None
        
    return yield_data

