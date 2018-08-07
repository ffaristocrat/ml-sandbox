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
                        if len(word) >= minsize
                        and word not in STOPWORDS
                    ])

                    print(f'{current_thread}\t{words}', file=f)
                document = ""
            elif comment:
                document += ' ' + deaccent(str(comment))
            current_thread = thread_num


def build_model(board):
    vocab_file = f'{board}.vocab'
    model_file = f'{board}-doc2vec.model'

    documents = FileThreads(board)
    
    if os.path.isfile(model_file):
        model = Doc2Vec.load(model_file)
    elif os.path.isfile(vocab_file):
        model = Doc2Vec.load(vocab_file)
    else:
        model = Doc2Vec(vector_size=100, window=2, min_count=5, workers=4)
        model.build_vocab(documents=documents)
        model.save(vocab_file)

    model.train(
        documents=documents,
        total_examples=model.corpus_count,
        epochs=model.iter,
    )
    
    model.save(f'{board}-doc2vec.model')
    model.docvecs.save_word2vec_format(f'{board}-doc2vec.vectors')


def main():
    board = 'pol'
    database = '4chan.db'
    with sqlite3.Connection(database) as conn:
        if not os.path.isfile(f'{board}.threads'):
            export_threads(board, conn)
    
    build_model(board)


if __name__ == '__main__':
    main()
