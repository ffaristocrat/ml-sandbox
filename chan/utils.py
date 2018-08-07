import timeit
import re

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import deaccent


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
re_chan_board = re.compile(r'/.+/?')
re_youtube_link = re.compile(r"http(s|)://.youtube\.com[^\s]+[\s]?")
re_link = re.compile(r"http(s|)://[^\s]+[\s]?")
re_numbers = re.compile(r'([0-9]+)')
re_ellipsis = re.compile(r'\.\.\.')


def tokenize_string(string, minsize: int=3):
    # Empty strings
    if not string or string == 'N':
        return None

    string = deaccent(string).lower()

    # Remove quote text
    string = re.sub(re_reply_to, '', string)
    string = re.sub(re_quote_line, '', string)
    
    # Punctuation to remove completely
    string = re.sub(re_punc_to_none, '', string)
    
    # Substitute in this order
    string = re.sub(re_ellipsis, ' <ELLIPSIS> ', string)
    string = re.sub(re_echoes, ' <ECHOES> ', string)
    string = re.sub(re_youtube_link, ' <YOUTUBE> ', string)
    string = re.sub(re_link, ' <LINK> ', string)
    string = re.sub(re_pol_board, ' <POLBOARD> ', string)
    string = re.sub(re_b_board, ' <RANDOMBOARD> ', string)
    string = re.sub(re_chan_board, ' <FOURCHANBOARD> ', string)
    string = re.sub(re_numbers, ' <NUMBER> ', string)
    string = re.sub(re_period, ' <PERIOD> ', string)
    string = re.sub(re_question, ' <QUESTION> ', string)

    # Replace all other punc to spaces and remove whitespace in between
    string = re.sub(re_punc_to_space, ' ', string)
    
    string = ' '.join([
        word for word in [w.strip() for w in string.split()]
        if len(word) >= minsize and word not in STOPWORDS
    ])

    return string if string else None


class Benchmark(object):
    def __init__(self, msg: str):
        self.msg = msg
    
    def __enter__(self):
        print(f'Starting {self.msg}')
        self.start = timeit.default_timer()
        return self
    
    def __exit__(self, *args):
        t = timeit.default_timer() - self.start
        print(self.make_msg(t))
        self.time = t
    
    def make_msg(self, t):
        m, s = divmod(t, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        if d:
            msg = f'{self.msg}: ' \
                  f'{d} days, {h} hours, {m} minutes, {int(s)} seconds'
        elif h:
            msg = f'{self.msg}: {h} hours, {m} minutes, {int(s)} seconds'
        elif m:
            msg = f'{self.msg}: {m:0} minutes, {int(s)} seconds'
        else:
            msg = f'{self.msg}: {s:.3f} seconds'
        
        return msg
    
    def current(self):
        t = timeit.default_timer() - self.start
        print(self.make_msg(t))
        return t


def cluster(df):
    matrix = df.astype(np.float64).values
    df['cluster'] = MiniBatchKMeans(n_clusters=8).fit_predict(matrix)
    
    return df[['cluster']]


