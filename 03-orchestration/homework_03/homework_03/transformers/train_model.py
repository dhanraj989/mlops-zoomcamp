if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer
def transform(df, *args, **kwargs):
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LinearRegression
    from scipy.sparse import vstack
    import pandas as pd

    categorical = ['PULocationID', 'DOLocationID']
    y = df['duration'].values

    dv = DictVectorizer()

    # Simulate fitting by collecting unique pairs (safe on memory)
    unique_df = df[categorical].drop_duplicates()
    dicts_unique = unique_df.to_dict(orient='records')
    dv.fit(dicts_unique)

    # Now transform all data in batches
    def batch_transform(data, batch_size=500_000):
        matrices = []
        for i in range(0, len(data), batch_size):
            batch = data.iloc[i:i+batch_size]
            dicts = batch[categorical].to_dict(orient='records')
            X_batch = dv.transform(dicts)
            matrices.append(X_batch)
        return vstack(matrices)

    X = batch_transform(df)

    model = LinearRegression()
    model.fit(X, y)

    print(f"Intercept: {model.intercept_}")
    return (dv, model)