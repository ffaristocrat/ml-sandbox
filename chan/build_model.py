import sqlite3
import logging
import os.path
import re
import csv

import pandas as pd
import numpy as np
import sklearn as skv

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import deaccent
from sklearn.cluster import KMeans, MiniBatchKMeans

from chan.utils import Benchmark, clean_string

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.INFO)


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


def load_archive(board, conn):
    filename = f'{board}.csv'
    
    create = f"""
        DROP TABLE IF EXISTS {board};
        CREATE TABLE {board} (
            num INTEGER,
            thread_num INTEGER,
            op INTEGER,
            timestamp TEXT,
            media_w INTEGER,
            media_h INTEGER,
            trip TEXT,
            comment TEXT,
            poster_country TEXT
        );
    """
    conn.executescript(create)

    insert = f"""
        INSERT INTO {board}
            ({','.join(COLUMNS_TO_KEEP)})
        VALUES
            ({','.join(['?'] * len(COLUMNS_TO_KEEP))});
    """
    
    with open(filename, newline='') as f:
        reader = csv.DictReader(
            f,
            fieldnames=ALL_COLUMNS,
            delimiter=',',
            quoting=csv.QUOTE_MINIMAL,
            doublequote=False,
            quotechar='"',
            escapechar='\\',
        )
        
        def yield_line():
            for line in reader:
                line['comment'] = clean_string(line['comment'])
                yield [
                    None if v == 'N' else v
                    for k, v in line.items()
                    if k in COLUMNS_TO_KEEP
                ]
        
        conn.executemany(insert, yield_line())
    
    index = f"""
        CREATE INDEX
            idx_{board}_thread_num
        ON {board}
            (thread_num);
        CREATE INDEX
            idx_{board}_num
        ON {board}
            (num);
        CREATE INDEX
            idx_{board}_thread_num_num
        ON {board}
            (thread_num, num);
    """
    
    conn.executescript(index)


def extract_meta(board):
    dtypes = {
        'thread_num': np.uint32,
        'op': np.uint8,
        "media_w": np.uint32,
        "media_h": np.uint32,
        "poster_country": np.object,
        "trip": np.object,
    }

    df = pd.read_csv(
        f'{board}.csv',
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

    df.to_csv(f'{board}.meta')


class FileThreads(object):
    def __init__(self, board):
        self.board = board
    
    def __iter__(self):
        with open(f'{self.board}.threads', 'r') as f:
            for line in f.readlines():
                thread_num, comment = line.split('\t')
                yield TaggedDocument(comment.split(), [thread_num])


def export_threads(board, conn, minsize=3):
    sql = f"SELECT thread_num, comment, op FROM {board} " \
          "ORDER BY thread_num, num"
    current_thread = None
    document = ""
    
    with open(f'{board}.threads', 'wt') as f:
        for thread_num, comment, op in conn.execute(sql):
            if op == 1:
                if document and current_thread:
                    words = ' '.join([
                        word for word in document.strip().split()
                        if len(word) >= minsize and word not in STOPWORDS
                    ])
                    
                    print(f'{current_thread}\t{words}', file=f)
                document = ""
            elif comment:
                document += ' ' + deaccent(str(comment))
            current_thread = thread_num


def build_doc2vec_model(board):
    documents = FileThreads(board)
    model = Doc2Vec(vector_size=100, window=2, min_count=5, workers=4)
    model.build_vocab(documents=documents)
    
    model.train(
        documents=documents,
        total_examples=model.corpus_count,
        epochs=model.iter,
    )
    
    model.save(f'{board}-doc2vec.model')
    model.docvecs.save_word2vec_format(f'{board}-doc2vec.vectors')


def load_sample_vectors(board, frac: float) -> pd.DataFrame:
    df = pd.read_csv(
        f'{board}-doc2vec.vectors',
        skiprows=1, index_col=0, delim_whitespace=True, header=None)
    df['thread_id'] = df.index.str.replace('\*dt_', '')
    df.set_index('thread_id', inplace=True)

    df = df.sample(frac=frac)
    print(f'{len(df)} records')

    return df


def cluster_threads(df):
    matrix = df.astype(np.float64).values
    df['cluster'] = MiniBatchKMeans(n_clusters=8).fit_predict(matrix)
    
    return df[['cluster']]


def main():
    board = 'pol'
    database = 'chan.db'

    with sqlite3.Connection(database) as conn:
        if not os.path.isfile(f'{board}.threads'):
            with Benchmark('load_archive'):
                load_archive(board, conn)
    
            with Benchmark('export_threads'):
                export_threads(board, conn)

    with Benchmark('build_doc2vec_model'):
        build_doc2vec_model(board)

    with Benchmark('load_sample_vectors'):
        df = load_sample_vectors(board, 1)

    with Benchmark('cluster_threads'):
        df = cluster_threads(df)

    with Benchmark('save_clusters'):
        df.to_csv(f'{board}-cluster.csv', index=True, header=True)


main()
