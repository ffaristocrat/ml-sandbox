import sqlite3
import logging
import os.path as op
import csv
import datetime

from datetime import datetime
from typing import List, Dict, Callable

import pandas as pd

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.word2vec import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from gensim.parsing.preprocessing import STOPWORDS

from chan.utils import benchmark, clean_string

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


def yield_line(filename: str, parse_func: Callable):
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


class ReadThreads(object):
    def __init__(self, board: str, input_dir: str='.',
                 file_type: str='threads', return_func: Callable=None):
        self.board = board
        self.input_dir = input_dir
        self.returner = return_func
        self.file_type = file_type

    def __iter__(self):
        filename = op.join(self.input_dir, f'{self.board}.{self.file_type}')
        with open(filename, 'r') as f:
            for line in f.readlines():
                thread_num, comment = line.split('\t')
                yield self.returner(thread_num, comment)


class Chanalysis:
    def __init__(self, board: str, database: str, input_dir: str='.'):
        self.board = board
        self.database = database
        self.input_dir = input_dir

    @staticmethod
    def _parse_for_database(line: Dict) -> List:
        return [
            None if v == 'N' else v
            for k, v in line.items()
            if k in DATABASE_COLUMNS
        ]

    @staticmethod
    def _parse_for_meta(line: Dict) -> Dict:
        return {
            k: None if v == 'N' else v
            for k, v in line.items()
            if k in META_COLUMNS
        }

    @benchmark
    def load_archive(self):
        create_sql = f"""
            DROP TABLE IF EXISTS
                {self.board};
            CREATE TABLE {self.board} (
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

        insert_sql = f"""
            INSERT INTO {self.board}
                ({','.join(DATABASE_COLUMNS)})
            VALUES
                ({','.join(['?'] * len(DATABASE_COLUMNS))});
        """

        index_sql = f"""
            CREATE INDEX
                idx_{self.board}_thread_num
            ON {self.board}
                (thread_num);
            CREATE INDEX
                idx_{self.board}_num
            ON {self.board}
                (num);
            CREATE INDEX
                idx_{self.board}_thread_num_num
            ON {self.board}
                (thread_num, num);
        """

        filename = op.join(self.input_dir, f'{self.board}.csv')

        with sqlite3.Connection(self.database) as conn:
            conn.executescript(create_sql)
            conn.executemany(
                insert_sql, yield_line(filename, self._parse_for_database))
            conn.executescript(index_sql)

    @benchmark
    def load_phrases(self):
        create_sql = f"""
            DROP TABLE IF EXISTS
                {self.board}_phrases;
            CREATE TABLE {self.board}_phrases (
                thread_num INTEGER,
                comment TEXT
            );
        """

        insert_sql = f"""
            INSERT INTO {self.board}_phrases
                (thread_num, comment)
            VALUES
                (?, ?);
        """

        index_sql = f"""
            CREATE INDEX
                idx_{self.board}_phrases_thread_num
            ON {self.board}_phrases
                (thread_num);
        """

        with sqlite3.Connection(self.database) as conn:
            conn.executescript(create_sql)
            conn.executemany(insert_sql, ReadThreads(
                self.board, input_dir=self.input_dir, file_type='phrases',
                return_func=lambda x, y: (x, y))
            )
            conn.executescript(index_sql)

    @benchmark
    def extract_meta(self):
        filename = op.join(self.input_dir, f'{self.board}.meta')
        with open(filename, 'wt') as f:
            writer = None
    
            for df in yield_line(self.board, self._parse_for_meta):
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
    
    @benchmark
    def export_threads(self, sample: float=1.0):
        with sqlite3.Connection(self.database) as conn:
            sql = f"SELECT COUNT(DISTINCT thread_num) FROM {self.board}"

            rows = int(conn.execute(sql).fetchone()[0] * sample)
            print(f'Exporting {rows} rows')

            if sample < 1.0:
                sql = f"""
                    SELECT
                        thread_num, comment, op
                    FROM
                        {self.board}
                    WHERE
                        thread_num IN (
                            SELECT DISTINCT
                                thread_num
                            FROM
                                {self.board}
                            ORDER BY
                                RANDOM()
                            LIMIT {rows}
                        )
                    ORDER BY
                        thread_num, num
                """
            else:
                sql = f"""
                    SELECT
                        thread_num, comment, op
                    FROM
                        {self.board}
                    ORDER BY
                        thread_num, num
                """
                
            current_thread = None
            document = ""
            filename = op.join(self.input_dir, f'{self.board}.threads')

            with open(filename, 'wt') as f:
                cursor = conn.execute(sql)
                for thread_num, comment, orig in cursor:
                    if orig == 1:
                        if current_thread:
                            tokens = clean_string(document)
                            if tokens:
                                print(f'{current_thread}\t{tokens}', file=f)
                        document = ""
                    elif comment:
                        document += ' ' + str(comment)
                    current_thread = thread_num

    @benchmark
    def build_phraser(self, threshold: int=None):
        tokens = ReadThreads(
            self.board, self.input_dir, return_func=lambda x, y: y.split())
        bigram = Phrases(tokens, min_count=5, threshold=threshold)
        trigram = Phrases(bigram[tokens], threshold=threshold)

        bigram_mod = Phraser(bigram)
        trigram_mod = Phraser(trigram)

        filename = op.join(self.input_dir, f'{self.board}.bigrams')
        bigram_mod.save(filename)
        filename = op.join(self.input_dir, f'{self.board}.trigrams')
        trigram_mod.save(filename)

        return trigram_mod

    @benchmark
    def build_phrases(self):
        threads = ReadThreads(
            self.board, self.input_dir,
            return_func=lambda x, y: (x, y.split()))
        filename = op.join(self.input_dir, f'{self.board}.trigrams')
        trigram_mod = Phraser.load(filename)

        filename = op.join(self.input_dir, f'{self.board}.phrases')
        with open(filename, 'wt') as f:
            for num, thread in threads:
                line = ' '.join([
                    word for word in trigram_mod[thread]
                    if word not in STOPWORDS and
                    len(word) >= 3
                ])
                print(f'{num}\t{line}', file=f)

    @benchmark
    def build_dictionary(self):
        documents = ReadThreads(
            self.board, input_dir=self.input_dir, file_type='phrases',
            return_func=lambda x, y: y.split())
        dictionary = Dictionary(documents)
        dictionary.save(f'{self.board}.dictionary')
        
        return dictionary
    
    @benchmark
    def build_lda_model(self, topics: int=20):
        ignore_words = [
            'like', 'know', 'fuck', 'fucking', 'want', 'shit', 'know', 'sure',
            'isn', 'CHANBOARD', 'think', 'people', 'good', 'time', 'going',
            'WEBLINK', 'got', 'way', ''
        ]
        filename = op.join(self.input_dir, f'{self.board}.dictionary')
        dictionary: Dictionary = Dictionary.load(filename)
        documents = ReadThreads(
            self.board, input_dir=self.input_dir, file_type='phrases',
            return_func=lambda x, y: dictionary.doc2bow(
                [w for w in y.split() if w not in ignore_words]
            )
        )

        lda = LdaMulticore(
            documents, id2word=dictionary, num_topics=topics, iterations=2)

        filename = op.join(self.input_dir, f'{self.board}.lda')
        lda.save(filename)

        return lda

    @benchmark
    def build_doc2vec_model(self, vectors: int=200):
        filename = op.join(self.input_dir, f'{self.board}.phraser')
        phraser = Phraser.load(filename)
        documents = ReadThreads(
            self.board, input_dir=self.input_dir, file_type='phrases',
            return_func=lambda x, y: TaggedDocument(phraser[y.split()], [x]))
        model = Doc2Vec(vector_size=vectors, window=2, min_count=5, workers=3)
        model.build_vocab(documents=documents)
        
        model.train(
            documents=documents,
            total_examples=model.corpus_count,
            epochs=model.iter,
        )
        
        filename = op.join(self.input_dir, f'{self.board}.doc2vec')
        model.save(filename)

        return model
    
    @benchmark
    def build_word2vec_model(self, vectors: int=200):
        sentences = ReadThreads(
            self.board, self.input_dir, 'phrases',
            return_func=lambda x, y: y.split())
        model = Word2Vec(
            sentences=sentences,
            size=vectors, window=5, min_count=5, workers=3)
        
        filename = op.join(self.input_dir, f'{self.board}.word2vec')
        model.wv.save(filename)
        
        return model

    @benchmark
    def load_sample_vectors(self, frac: float = 1.0) -> pd.DataFrame:
        filename = op.join(self.input_dir, f'{self.board}.doc2vectors')
        df = pd.read_csv(
            filename, skiprows=1, index_col=0,
            delim_whitespace=True, header=None)
        df['thread_id'] = df.index.str.replace('\*dt_', '')
        df.set_index('thread_id', inplace=True)
        
        df = df.sample(frac=frac)
        print(f'{len(df)} records')
        
        return df

    @benchmark
    def load_clusters(self):
        with sqlite3.Connection(self.database) as conn:
            sql = f"""
                DROP TABLE IF EXISTS {self.board}_clusters;
                CREATE TABLE {self.board}_clusters (
                    thread_num INTEGER,
                    cluster INTEGER
                );
            """
            conn.executescript(sql)
            
            insert = f"""
                INSERT INTO {self.board}_clusters
                    (thread_num, cluster)
                VALUES
                    (?, ?);
            """
            
            filename = op.join(self.input_dir, f'{self.board}.clusters')
            with open(filename, newline='') as f:
                reader = csv.reader(f)
                conn.executemany(insert, reader)

                sql = f"""
                CREATE INDEX
                    idx_{self.board}_clusters_thread_num
                ON {self.board}_clusters
                    (thread_num);
                CREATE INDEX
                    idx_{self.board}_clusters
                ON {self.board}_clusters
                    (cluster);
                CREATE INDEX
                    idx_{self.board}_clusters_thread_num_clusters
                ON {self.board}_clusters
                    (thread_num, cluster);
            """
            
            conn.executescript(sql)
    

def main():
    board = 'pol'
    database = 'chan.db'
    vectors = 200
    input_dir = '.'
    
    chan = Chanalysis(board=board, database=database, input_dir=input_dir)

    chan.export_threads(sample=1.00)
    chan.build_phraser(threshold=50)
    chan.build_phrases()
    chan.build_dictionary()
    chan.build_word2vec_model(vectors=vectors)


if __name__ == '__main__':
    main()
