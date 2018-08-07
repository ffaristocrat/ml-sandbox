import sqlite3
import logging
import os.path
import csv
import datetime
from datetime import datetime
from typing import List, Dict

import pandas as pd

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from chan.utils import Benchmark, tokenize_string, cluster

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

DATABASE_COLUMNS = [
    "num", "thread_num", "op", "timestamp", "comment",
]

META_COLUMNS = [
    "num", "thread_num", "op", "timestamp", "media_w", "media_h", "trip",
    "poster_country",
]

COUNTRIES = {
    'US': ['US'],
    'CA': ['CA'],
    'RU': ['RU'],
    'AU': ['AU'],
    'UK': ['UK', 'GB'],
    'EU': [
        'BE', 'BG', 'CZ', 'DK', 'DE', 'EE', 'IE', 'EL', 'ES', 'FR',
        'HR', 'IT', 'CY', 'LV', 'LT', 'LU', 'HU', 'MT', 'NL', 'AT',
        'PL', 'PT', 'RO', 'SI', 'SK', 'FI', 'SE',
    ]
}

HOURS = {
    '0_3': list(range(0, 4)),
    '4_7': list(range(4, 8)),
    '8_11': list(range(8, 12)),
    '12_15': list(range(12, 16)),
    '16_19': list(range(16, 20)),
    '20_23': list(range(20, 24)),
}


def yield_line(board, parse_func):
    with open(f'{board}.csv', newline='') as f:
        reader = csv.DictReader(
            f,
            fieldnames=ALL_COLUMNS,
            delimiter=',',
            quoting=csv.QUOTE_MINIMAL,
            doublequote=False,
            quotechar='"',
            escapechar='\\',
        )
    
        for line in reader:
            yield parse_func(line)


def parse_for_database(line: Dict) -> List:
    return [
        None if v == 'N' else v
        for k, v in line.items()
        if k in DATABASE_COLUMNS
    ]


def parse_for_meta(line: Dict) -> Dict:
    return {
        k: None if v == 'N' else v
        for k, v in line.items()
        if k in META_COLUMNS
    }


def load_archive(board, conn):
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
            ({','.join(DATABASE_COLUMNS)})
        VALUES
            ({','.join(['?'] * len(DATABASE_COLUMNS))});
    """

    conn.executemany(insert, yield_line(board, parse_for_database))

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
    with open(f'{board}.meta', 'wt') as f:
        writer = None

        for df in yield_line(board, parse_for_meta):
            df['trip'] = 1 if df['trip'] else 0
            df['image'] = 1 if int(df['media_w']) > 0 else 0
            df['landscape'] = \
                1 if int(df['media_w']) > int(df['media_h']) else 0

            for k, v in COUNTRIES.items():
                df[f'country_{k}'] = 1 if df['poster_country'] in v else 0
            df['country_null'] = 1 if not df['poster_country'] else 0

            df['timestamp'] = datetime.fromtimestamp(int(df['timestamp']))
            df['hour'] = df['timestamp'].hour

            for k, v in HOURS.items():
                df['hour_{k{'] = 1 if df['hour'] in v else 0

            if not writer:
                writer = csv.DictWriter(f, fieldnames=df.keys())
                writer.writeheader()

            writer.writerow(df)


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
                if current_thread:
                    tokens = tokenize_string(document, minsize)
                    if tokens:
                        print(f'{current_thread}\t{tokens}', file=f)
                document = ""
            elif comment:
                document += ' ' + str(comment)
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


def main():
    board = 'pol'
    database = 'chan.db'

    if not os.path.isfile(f'{board}.meta'):
        with Benchmark('extract_meta'):
            extract_meta(board)

    with sqlite3.Connection(database) as conn:
        if not os.path.isfile(f'{board}.threads'):
            with Benchmark('load_archive'):
                load_archive(board, conn)

            with Benchmark('export_threads'):
                export_threads(board, conn)

    if not os.path.isfile(f'{board}.vectors'):
        with Benchmark('build_doc2vec_model'):
            build_doc2vec_model(board)

    with Benchmark('load_sample_vectors'):
        df = load_sample_vectors(board, 1)

    with Benchmark('cluster'):
        df = cluster(df)

    with Benchmark('save_clusters'):
        df.to_csv(f'{board}-cluster.csv', index=True, header=True)


main()
