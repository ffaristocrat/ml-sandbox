import pandas as pd
import numpy as np
import sklearn as skv
import sklearn.cluster as sc


from gensim.models.doc2vec import Doc2Vec

ALL_COLUMNS = [
    "num", "subnum", "thread_num", "op", "timestamp", "timestamp_expired",
    "preview_orig", "preview_w", "preview_h", "media_filename", "media_w",
    "media_h", "media_size", "media_hash", "media_orig", "spoiler", "deleted",
    "capcode", "email", "name", "trip", "title", "comment", "sticky", "locked",
    "poster_hash", "poster_country", "exif"
]


COLUMNS_TO_KEEP = [
    "num", "thread_num", "op", "timestamp", "media_w",
    "media_h", "trip", "comment", "poster_country",
]


def extract_meta(filename):
    dtypes = {
        'thread_num': np.uint32,
        'op': np.uint8,
        "media_w": np.uint32,
        "media_h": np.uint32,
        "poster_country": np.object,
        "trip": np.object,
    }

    df = pd.read_csv(
        'pol.csv',
        header=None,
        names=ALL_COLUMNS,
        doublequote=False,
        quotechar='"',
        na_values=['N'],
        escapechar='\\',
        usecols=COLUMNS_TO_KEEP,
        index_col=0,
        dtype=dtypes,
        parse_dates=['timestamp'],
        date_parser=lambda x: pd.to_datetime(x, unit='s')
    )

    df.trip = (~df.trip.isna()).astype(np.uint8)
    df['image'] = (df.media_w > 0).astype(np.uint8)
    df['landscape'] = (df.media_w > df.media_h).astype(np.uint8)
    del df['media_w']
    del df['media_h']

    df['country_US'] = (df.poster_country == 'US').astype(np.uint8)
    df['country_CA'] = (df.poster_country == 'CA').astype(np.uint8)
    df['country_GB'] = (df.poster_country.isin(['GB', 'UK'])).astype(np.uint8)
    df['country_RU'] = (df.poster_country == 'RU').astype(np.uint8)
    df['country_AU'] = (df.poster_country == 'AU').astype(np.uint8)
    df['country_EU'] = (df.poster_country.isin([
        'BE', 'BG', 'CZ', 'DK', 'DE', 'EE', 'IE', 'EL', 'ES', 'FR', 'HR', 'IT',
        'CY', 'LV', 'LT', 'LU', 'HU', 'MT', 'NL', 'AT', 'PL', 'PT', 'RO', 'SI',
        'SK', 'FI', 'SE',
    ])).astype(np.uint8)
    df['country_null'] = (df.poster_country.isna()).astype(np.uint8)
    del df['poster_country']

    df['hour'] = df['timestamp'].dt.hour
    df['hour_0_3'] = df.hour.isin([0, 1, 2, 3]).astype(np.uint8)
    df['hour_4_7'] = df.hour.isin([4, 5, 6, 7]).astype(np.uint8)
    df['hour_8_11'] = df.hour.isin([8, 9, 10, 11]).astype(np.uint8)
    df['hour_12_15'] = df.hour.isin([12, 13, 14, 15]).astype(np.uint8)
    df['hour_16_19'] = df.hour.isin([16, 17, 18, 19]).astype(np.uint8)
    df['hour_20_23'] = df.hour.isin([20, 21, 22, 23]).astype(np.uint8)
    del df['hour']
    del df['timestamp']

    df.to_csv(f'meta-{filename}')


def load_model(board):
    model = Doc2Vec.load(f'{board}-doc2vec.model')
    return model


def cluster_threads(board):
    df = pd.read_csv(
        f'{board}-doc2vec.vectors',
        skiprows=1, index_col=0, delim_whitespace=True, header=None)
    
    df.sample(frac=0.1, replace=True)
    df['thread_id'] = df.index.str.replace('\*dt_', '')
    
    matrix = df.astype(np.float64).values
    df['cluster'] = sc.AgglomerativeClustering(
        n_clusters=8, linkage='ward').fit_predict(matrix)
    
    return df[['thread_id', 'cluster']]


def main():
    board = 'pol'
    df = cluster_threads(board)
    df.to_csv(f'{board}-cluster.csv', index=False, header=True)


main()
