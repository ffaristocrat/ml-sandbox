import sqlite3
import logging
import os.path as op
import csv
import datetime
from datetime import datetime
from typing import List, Dict, Callable

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


def yield_line(board: str, parse_func: Callable, input_dir: str='.'):
    filename = op.join(input_dir, f'{board}.csv')
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


def load_archive(board: str, conn: sqlite3.Connection, input_dir: str='.'):
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

    conn.executemany(
        insert, yield_line(board, parse_for_database, input_dir=input_dir))

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


def extract_meta(board: str, input_dir: str='.'):
    filename = op.join(input_dir, f'{board}.meta')
    with open(filename, 'wt') as f:
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
    def __init__(self, board: str, input_dir: str='.'):
        self.board = board
        self.input_dir = input_dir
    
    def __iter__(self):
        filename = op.join(self.input_dir, f'{self.board}.threads')
        with open(filename, 'r') as f:
            for line in f.readlines():
                thread_num, comment = line.split('\t')
                yield TaggedDocument(comment.split(), [thread_num])


def export_threads(board: str, conn: sqlite3.Connection, minsize: int=3,
                   input_dir: str='.'):
    sql = f"SELECT thread_num, comment, op FROM {board} " \
          "ORDER BY thread_num, num"
    current_thread = None
    document = ""
    filename = op.join(input_dir, f'{board}.threads')

    with open(filename, 'wt') as f:
        for thread_num, comment, orig in conn.execute(sql):
            if orig == 1:
                if current_thread:
                    tokens = tokenize_string(document, minsize)
                    if tokens:
                        print(f'{current_thread}\t{tokens}', file=f)
                document = ""
            elif comment:
                document += ' ' + str(comment)
            current_thread = thread_num


def build_doc2vec_model(board: str, vectors: int, input_dir: str='.'):
    documents = FileThreads(board, input_dir=input_dir)
    model = Doc2Vec(vector_size=vectors, window=2, min_count=5, workers=4)
    model.build_vocab(documents=documents)

    model.train(
        documents=documents,
        total_examples=model.corpus_count,
        epochs=model.iter,
    )

    filename = op.join(input_dir, f'{board}.model')
    model.save(filename)
    filename = op.join(input_dir, f'{board}.vectors')
    model.docvecs.save_word2vec_format(filename)


def load_sample_vectors(board: str, frac: float=1.0, input_dir: str='.'
                        ) -> pd.DataFrame:
    filename = op.join(input_dir, f'{board}.vectors')
    df = pd.read_csv(
        filename, skiprows=1, index_col=0, delim_whitespace=True, header=None)
    df['thread_id'] = df.index.str.replace('\*dt_', '')
    df.set_index('thread_id', inplace=True)

    df = df.sample(frac=frac)
    print(f'{len(df)} records')

    return df


def load_clusters(board: str, conn: sqlite3.Connection, input_dir: str='.'):
    create = f"""
        DROP TABLE IF EXISTS {board}_clusters;
        CREATE TABLE {board}_clusters (
            thread_num INTEGER,
            cluster INTEGER
        );
    """
    conn.executescript(create)

    insert = f"""
        INSERT INTO {board}_clusters
            (thread_num, cluster)
        VALUES
            (?, ?);
    """

    filename = op.join(input_dir, f'{board}.clusters')
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        conn.executemany(insert, reader)

    index = f"""
        CREATE INDEX
            idx_{board}_clusters_thread_num
        ON {board}_clusters
            (thread_num);
        CREATE INDEX
            idx_{board}_clusters
        ON {board}_clusters
            (cluster);
        CREATE INDEX
            idx_{board}_clusters_thread_num_clusters
        ON {board}_clusters
            (thread_num, cluster);
    """

    conn.executescript(index)


def main():
    board = 'pol'
    database = 'chan.db'
    vectors = 200
    input_dir = '.'

    filename = op.join(input_dir, f'{board}.meta')
    if not op.isfile(filename):
        with Benchmark('extract_meta'):
            extract_meta(board, input_dir=input_dir)

    filename = op.join(input_dir, f'{board}.threads')
    if not op.isfile(filename):
        with sqlite3.Connection(database) as conn:
            with Benchmark('load_archive'):
                load_archive(board, conn, input_dir=input_dir)

            with Benchmark('export_threads'):
                export_threads(board, conn, input_dir=input_dir)

    filename = op.join(input_dir, f'{board}.vectors')
    if not op.isfile(filename):
        with Benchmark('build_doc2vec_model'):
            build_doc2vec_model(board, vectors, input_dir=input_dir)

    filename = op.join(input_dir, f'{board}.clusters')
    if not op.isfile(filename):
        with Benchmark('load_sample_vectors'):
            df = load_sample_vectors(board, frac=1.0, input_dir=input_dir)

        with Benchmark('cluster'):
            df = cluster(df, 8)

        with Benchmark('save_clusters'):
            df.to_csv(filename, header=False)

    with Benchmark('load_clusters'):
        with sqlite3.Connection(database) as conn:
            load_clusters(board, conn, input_dir=input_dir)


if __name__ == '__main__':
    main()
