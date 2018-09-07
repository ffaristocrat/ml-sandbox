import os
from typing import List, Tuple

import pandas as pd

from selenium import webdriver

chromedriver = '/usr/local/Caskroom/chromedriver'
driver = webdriver.Chrome(executable_path=chromedriver)

domain = 'https://infoweb-newsbank-com.dclibrary.idm.oclc.org'


def login(user_id: int):
    base_url = f'{domain}/resources/?p=WORLDNEWS'
    driver.get(base_url)
    field = driver.find_element_by_css_selector(
        'input[type="text"][name="user"]')
    field.send_keys(user_id)
    field.submit()


def run_search(name):
    first, _, last = name.partition(' ')

    url = f'{domain}/resources/?p=WORLDNEWS&b=results&action=search' \
          f'&t=product%3AAWNB%21Access%2BWorld%2BNews/country%3AUSA%21USA/' \
          f'stp%3ANewspaper%21Newspaper&fld0=Author' \
          f'&val0=%22{first}%20{last}%22%20or%20%22{last}%2C%20{first}%22' \
          f'&bln1=AND&fld1=YMD_date&val1=&sort=YMD_date%3AA'
    driver.get(url)


def get_article_urls(name) -> pd.DataFrame:
    links = []

    while True:
        elements = driver.find_elements_by_css_selector('a.nb-doc-link')
        links.extend([e.get_attribute('href') for e in elements])

        try:
            next_page = driver.find_element_by_css_selector('li.pager-next')
        except BaseException:
            break

        next_page.click()

    df = pd.DataFrame(links, columns=['urls'])

    df['name'] = name

    return df


def parse_article(row) -> Tuple:
    driver.get(row['urls'])
    body = driver.find_element_by_css_selector('div.body').text
    byline = driver.find_element_by_css_selector('span.val').text

    return row['urls'], body, byline


def read_names(filename) -> List[str]:
    names = [n for n in open(filename).read().split('\n') if n]
    print(names)
    return names


def main():
    user_id = os.environ.get('USER_ID')
    filename = 'senior_admin_officials.txt'

    login(user_id)

    df = pd.DataFrame()
    names = read_names(filename)

    for name in names:
        print(name)
        run_search(name)
        df = df.append(get_article_urls(name))

    df.to_csv('urls.txt', index=False)
    df = pd.read_csv('urls.txt', header=0)

    df2 = df.apply(lambda x: parse_article(x), axis=1, result_type='expand')
    df2.to_csv('articles.txt', headers=True, index=False)

    driver.close()


if __name__ == '__main__':
    main()
