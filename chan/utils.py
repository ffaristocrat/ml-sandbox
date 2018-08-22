import timeit
import re
from functools import wraps

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from gensim.parsing.preprocessing import strip_punctuation
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


def clean_string(string):
    # Empty strings
    if not string or string == 'N':
        return None

    string = deaccent(string).lower()

    # Remove quote text
    string = re.sub(re_reply_to, '', string)
    string = re.sub(re_quote_line, '', string)

    string = re.sub(re_youtube_link, ' YOUTUBELINK ', string)
    string = re.sub(re_link, ' WEBLINK ', string)
    string = re.sub(re_pol_board, ' pol ', string)
    string = re.sub(re_b_board, ' RANDOMBOARD ', string)
    string = re.sub(re_chan_board, ' CHANBOARD ', string)

    string = strip_punctuation(string)

    # Punctuation to remove completely
    # string = re.sub(re_punc_to_none, '', string)

    # Substitute in this order
    # string = re.sub(re_ellipsis, ' <ELLIPSIS> ', string)
    # string = re.sub(re_echoes, ' <ECHOES> ', string)
    # string = re.sub(re_pol_board, ' <POLBOARD> ', string)
    # string = re.sub(re_numbers, ' <NUMBER> ', string)
    # string = re.sub(re_period, ' <PERIOD> ', string)
    # string = re.sub(re_question, ' <QUESTION> ', string)

    # Replace all other punc to spaces and remove whitespace in between
    # string = re.sub(re_punc_to_space, ' ', string)

    string = ' '.join([word for word in [w.strip() for w in string.split()]])

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
        print(make_msg(self.msg, t))
        self.time = t
 
    def current(self):
        t = timeit.default_timer() - self.start
        print(make_msg(self.msg, t))
        return t


def make_msg(msg, t):
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if d:
        msg = f'{msg}: {d:.0f} days, ' \
              f'{h:.0f} hours, {m:.0f} minutes, {s:.0f} seconds'
    elif h:
        msg = f'{msg}: ' \
              f'{h:.0f} hours, {m:.0f} minutes, {s:.0f} seconds'
    elif m:
        msg = f'{msg}: {m:.0f} minutes, {s:.0f} seconds'
    else:
        msg = f'{msg}: {s:.3f} seconds'

    return msg


def benchmark(method):
    @wraps(method)
    def f(*args, **kwargs):
        start = timeit.default_timer()
        print(f'Starting {method.__name__}')
        result = method(*args, **kwargs)
        t = timeit.default_timer() - start
        print(make_msg(method.__name__, t))
        return result

    return f


def cluster(df, n_clusters: int=8):
    matrix = df.astype(np.float64).values
    df['cluster'] = MiniBatchKMeans(n_clusters=n_clusters).fit_predict(matrix)
    
    return df[['cluster']]


