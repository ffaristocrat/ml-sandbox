import re
import csv
import sqlite3
import logging

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


def main():
    board = 'pol'
    database = '4chan.db'
    with sqlite3.Connection(database) as conn:
        load_archive(board, conn)


if __name__ == '__main__':
    main()
