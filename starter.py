import pickle
import pandas as pd
import sys

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

if __name__ == '__main__':
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    filename = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'

    df = read_data(filename)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    mean_pred = y_pred.mean()
    print(f'Predicted mean duration: {mean_pred:.2f} minutes')
    df_result = pd.DataFrame()
    df_result['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result['predictions'] = y_pred


    output_file = f'predictions_{month}_{year}.parquet'
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )
    print(f'Wrote {len(df_result)} rows to {output_file}')


