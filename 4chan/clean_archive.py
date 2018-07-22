import re
import csv
import sqlite3
import logging

import numpy as np
import pandas as pd

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import deaccent

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.INFO)

re_reply_to = re.compile(r'>>([0-9]+)(\n|$)')
re_quote_line = re.compile(r'>.+?(\n|$)')
re_echoes = re.compile(r'\(\(\(|\)\)\)')
re_parentheses = re.compile(r'(|\)\()')
re_punc_to_space = re.compile(r'[\n\r,/:"“”\]\[}{()!\t*&^@~-]')
re_punc_to_none = re.compile(r'[;<>]')
re_question = re.compile(r'\?')
re_period = re.compile(r'\.')
re_pol_board = re.compile(r'/pol/')
re_b_board = re.compile(r'/b/')
re_4chan_board = re.compile(r'/.+/?')
re_youtube_link = re.compile(r"http(s|)://.youtube\.com[^\s]+[\s]?")
re_link = re.compile(r"http(s|)://[^\s]+[\s]?")
re_numbers = re.compile(r'([0-9]+)')
re_ellipsis = re.compile(r'\.\.\.')


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


def clean_string(string):
    if string == 'N':
        return None

    string = string.lower()
    string = re.sub(re_reply_to, '', string)
    string = re.sub(re_quote_line, '', string)
    string = re.sub(re_punc_to_none, '', string)
    string = re.sub(re_ellipsis, ' <ELLIPSIS> ', string)
    string = re.sub(re_echoes, ' <ECHOES> ', string)
    string = re.sub(re_youtube_link, ' <YOUTUBE> ', string)
    string = re.sub(re_link, ' <LINK> ', string)
    string = re.sub(re_pol_board, ' <POLBOARD> ', string)
    string = re.sub(re_b_board, ' <RANDOMBOARD> ', string)
    string = re.sub(re_4chan_board, ' <FOURCHANBOARD> ', string)
    string = re.sub(re_numbers, ' <NUMBER> ', string)
    string = re.sub(re_period, ' <PERIOD> ', string)
    string = re.sub(re_question, ' <QUESTION> ', string)
    string = re.sub(re_punc_to_space, ' ', string)

    string = ' '.join(string.strip().split())

    return string if string else None


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
    print(insert)
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


class BoardThreads(object):
    def __init__(self, board, conn):
        self.board = board
        self.conn = conn
        self.minsize = 3
    
    def __iter__(self):
        sql = f"SELECT thread_num, comment, op FROM {self.board} " \
               "ORDER BY thread_num, num"
        current_thread = None
        document = ""

        for thread_num, comment, op in self.conn.execute(sql):
            if op == 1:
                if document and current_thread:
                    words = [
                        word for word in document.strip().split()
                        if len(word) <= self.minsize
                        and word not in STOPWORDS
                    ]
                    
                    yield TaggedDocument(
                        words,
                        [int(current_thread)])
                document = ""
            elif comment:
                document += ' ' + deaccent(str(comment))
            current_thread = thread_num


def build_model(board, conn):
    model = Doc2Vec(vector_size=100, window=2, min_count=5, workers=4)
    documents = BoardThreads(board, conn)
    model.build_vocab(documents=documents)
    model.train(
        documents=documents,
        total_examples=model.corpus_count,
        epochs=model.iter,
    )

    model.save(f'{board}-doc2vec.model')


def main():
    board = 'pol'
    database = '4chan.db'
    with sqlite3.Connection(database) as conn:
        # load_archive(board, conn)
        
        build_model(board, conn)

    # print('extract meta')
    # extract_meta(filename)


if __name__ == '__main__':
    main()
