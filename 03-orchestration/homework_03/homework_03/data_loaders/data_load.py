if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader

@data_loader
def load_data(*args, **kwargs):
    import pandas as pd
    cols = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'PULocationID', 'DOLocationID']
    df = pd.read_parquet("data.parquet",columns=cols, engine="pyarrow")
    print(f"Loaded records: {len(df)}")
    return df