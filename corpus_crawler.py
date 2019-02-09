# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     corpus_crawler
   Description :
   date：          2018/12/19
-------------------------------------------------
   Change Activity:
                   2018/12/19:
-------------------------------------------------
"""

import lxml
import requests
from bs4 import BeautifulSoup


def get_head(path='../spider/headers.txt'):
    with open(path, 'r') as f:
        headers = eval(f.read())
    return headers[3]

def get_tag(url):
    respon = requests.get(url=url, headers=get_head())
    soup = BeautifulSoup(respon.text, 'lxml')
    tag = soup.find_all(name='div',
                        class_='js_selection_area',
                        id='main_content')[0]
    return tag

def save_txt(url):
    tag = get_tag(url)
    with open('speech_corpus.txt', 'w', encoding='utf-8') as f:
        for child in tag.children:
            if child.string not in [None, '\n']:
                f.write(child.string + '\n')


if __name__ == '__main__':
    url = "http://finance.ifeng.com/a/20181218/16627248_0.shtml"
    save_txt(url)






































