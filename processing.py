# -*- coding:utf-8 -*-
import os
from pyltp import Segmentor, Postagger, Parser, NamedEntityRecognizer, SementicRoleLabeller
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import heapq
import re
import emoji
from extractor import Extractor

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 5000)
pd.set_option('max_colwidth', 30)
pd.set_option('display.width', 1000)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

# 一、数据处理
# 导入数据
df = pd.read_csv('F:\pycharm project data\\taobao\phone\\comment1.csv', encoding='utf-8-sig')
# 提取评论数据
co_df = df[['content']]
co_df = co_df.loc[co_df['content'] != '15天内买家未作出评价', ['content']]
co_df = co_df.loc[co_df['content'] != '评价方未及时做出评价,系统默认好评!', ['content']]
comment_list = co_df['content'].tolist()

if __name__ == '__main__':
    myextractor = Extractor()
    myextractor.get_seginfo(comment_list)















































