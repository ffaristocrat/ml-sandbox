import sqlite3
import logging
import os.path

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import deaccent

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.INFO)


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
    vocab_file = f'{board}.vocab'

    model = Doc2Vec(vector_size=100, window=2, min_count=5, workers=4)
    documents = BoardThreads(board, conn)

    if not os.path.isfile(vocab_file):
        model.build_vocab(documents=documents)
        model.vocabulary.save(vocab_file)
    else:
        model.vocabulary.load(vocab_file)

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
        build_model(board, conn)


if __name__ == '__main__':
    main()
