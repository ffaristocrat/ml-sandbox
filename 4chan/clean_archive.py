from collections import defaultdict
import re
import csv

import numpy as np
import pandas as pd


re_reply_to = re.compile(r'>>([0-9]+)(\n|$)')
re_quote_line = re.compile(r'>.+?(\n|$)')
re_echoes = re.compile(r'\(\(\(|\)\)\)')
re_parentheses = re.compile(r'(|\)\()')
re_punc_to_space = re.compile(r'[\n\r,/:"\]\[}{()!\t*&^@~]')
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


COLUMNS = [
    "num", "subnum", "thread_num", "op", "timestamp", "timestamp_expired",
    "preview_orig", "preview_w", "preview_h", "media_filename", "media_w",
    "media_h", "media_size", "media_hash", "media_orig", "spoiler", "deleted",
    "capcode", "email", "name", "trip", "title", "comment", "sticky", "locked",
    "poster_hash", "poster_country", "exif"
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


def process_comment_text(filename):
    words = defaultdict(int)

    with open(filename, newline='') as f, \
            open(f'comments-{filename}', 'wt') as c:

        reader = csv.reader(
            f,
            delimiter=',',
            quoting=csv.QUOTE_MINIMAL,
            doublequote=False,
            quotechar='"',
            escapechar='\\',
        )

        for i, line in enumerate(reader):
            clean_comment = clean_string(line.pop(22))
            if clean_comment:
                for word in clean_comment.split():
                    words[word] += 1
                print(clean_comment, file=c)

    with open(f'words-{filename}', 'w') as f:
        for w, c in words.items():
            print(f'{w}, {c}', file=f)


def extract_meta(filename):
    usecols = [
        "num", "thread_num", "op", "timestamp", "media_w", "media_h", "trip",
        "poster_country",
    ]

    dtypes = {
        'thread_num': np.uint32,
        'op': np.uint8,
        "media_w": np.uint32,
        "media_h": np.uint32,
        "poster_country": np.object,
        "trip": np.object,
    }

    print('reading file')
    df = pd.read_csv(
        'pol.csv',
        header=None,
        names=COLUMNS,
        doublequote=False,
        quotechar='"',
        na_values=['N'],
        escapechar='\\',
        usecols=usecols,
        index_col=0,
        dtype=dtypes,
        parse_dates=['timestamp'],
        date_parser=lambda x: pd.to_datetime(x, unit='s')
    )

    print('tripcode & landscape')
    df.trip = (~df.trip.isna()).astype(np.uint8)
    df['image'] = (df.media_w > 0).astype(np.uint8)
    df['landscape'] = (df.media_w > df.media_h).astype(np.uint8)
    del df['media_w']
    del df['media_h']

    print('country')
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

    print('hour')
    df['hour'] = df['timestamp'].dt.hour
    df['hour_0_3'] = df.hour.isin([0, 1, 2, 3]).astype(np.uint8)
    df['hour_4_7'] = df.hour.isin([4, 5, 6, 7]).astype(np.uint8)
    df['hour_8_11'] = df.hour.isin([8, 9, 10, 11]).astype(np.uint8)
    df['hour_12_15'] = df.hour.isin([12, 13, 14, 15]).astype(np.uint8)
    df['hour_16_19'] = df.hour.isin([16, 17, 18, 19]).astype(np.uint8)
    df['hour_20_23'] = df.hour.isin([20, 21, 22, 23]).astype(np.uint8)
    del df['hour']
    del df['timestamp']

    print('writing')
    df.to_csv(f'meta-{filename}')


def main():
    filename = 'pol.csv'
    print('process comment text')
    process_comment_text(filename)

    print('extract meta')
    extract_meta(filename)


if __name__ == '__main__':
    main()
