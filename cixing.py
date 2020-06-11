# -*- coding:utf-8 -*-
import os
from pyltp import Segmentor, Postagger, Parser, NamedEntityRecognizer, SementicRoleLabeller
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import heapq
import re
import emoji

class Sentence_Parser:
    def __init__(self):
        LTP_DIR = 'F:\project support\ltp_data_v3.4.0'
        # 分词
        self.segmentor = Segmentor()
        self.segmentor.load(os.path.join(LTP_DIR, 'cws.model'))

        # 词性标注
        self.postagger = Postagger()
        self.postagger.load(os.path.join(LTP_DIR, 'pos.model'))

        # 依存句法分析
        self.parser = Parser()
        self.parser.load(os.path.join(LTP_DIR, 'parser.model'))

        # 命名实体识别（人名、地名、机构名等）
        self.recognizer = NamedEntityRecognizer()
        self.recognizer.load(os.path.join(LTP_DIR, 'ner.model'))

        # 词义角色标注（施事、受事、时间、地点）
        self.labeller = SementicRoleLabeller()
        self.labeller.load(os.path.join(LTP_DIR, 'pisrl_win.model'))


    def format_labelrole(self, words, postags):
        """
        词义角色标注
        """
        arcs = self.parser.parse(words, postags)
        roles = self.labeller.label(words, postags, arcs)
        roles_dict = {}
        for role in roles:
            roles_dict[role.index] = {arg.name: [arg.name, arg.range.start, arg.range.end] for arg in role.arguments}
        # for item in roles_dict.items():
        #     print(item)
        return roles_dict


    def bulid_parser_child_dict(self, words, postags, arcs):
        """
        句法分析---为句子中的每个词语维护一个保存句法依存子节点的字典
        """
        child_dict_list = []
        format_parse_list = []
        for index in range(len(words)):
            child_dict = dict()
            for arc_index in range(len(arcs)):
                if arcs[arc_index].head == index + 1:
                    if arcs[arc_index].relation not in child_dict:
                        child_dict[arcs[arc_index].relation] = []
                        child_dict[arcs[arc_index].relation].append(arc_index)
                    else:
                        child_dict[arcs[arc_index].relation].append(arc_index)
            child_dict_list.append(child_dict)
        rely_id = [arc.head for arc in arcs]
        # print(rely_id)
        relation = [arc.relation for arc in arcs]
        # for i in range(len(relation)):
        #     print(words[i], '_', postags[i], '_', i, '_', relation[i])
        heads = ['Root' if id == 0 else words[id-1] for id in rely_id]
        # print(heads)
        for i in range(len(words)):
            a = [relation[i], words[i], i, postags[i], heads[i], rely_id[i]-1, postags[rely_id[i]-1]]
            format_parse_list.append(a)
        return child_dict_list, format_parse_list


    def parser_main(self, sentence):
        """
        parser主函数
        """
        words = list(self.segmentor.segment(sentence))
        postags = list(self.postagger.postag(words))
        arcs = self.parser.parse(words, postags)
        child_dict_list, format_parse_list = self.bulid_parser_child_dict(words, postags, arcs)
        roles_dict = self.format_labelrole(words, postags)
        return words, postags, child_dict_list, roles_dict, format_parse_list

    def select(self, words, postags):
        """
        筛选出名词和形容词
        """
        co_model = Word2Vec.load('coseg_text.model')
        n_list0 = []
        a_list = []
        for i in range(len(postags)):
            if postags[i] == 'n':
                if len(words[i]) >= 2:
                    n_list0.append(words[i])
            if postags[i] == 'a':
                # if len(words[i]) >= 2:
                a_list.append(words[i])
        n_list0 = list(set(n_list0))
        a_list = list(set(a_list))
        # print(n_list0)
        # print(a_list)
        si_p = []
        for n in n_list0:
            try:
                s = co_model.similarity(n, '手机')
                si_p.append(s)
            except Exception as e:
                si_p.append(0)
        index_list = list(map(si_p.index, heapq.nlargest(int(0.8*len(si_p)), si_p))) #取出和手机相关度最高的n
        n_list = []
        for index in index_list:
            n_list.append(n_list0[index])
        # print(n_list)
        return n_list, a_list


    def simlarity(self, n_list0, a_list):
        """
        计算相似度,进行正逆向匹配，筛选出名词和形容词的最佳搭配
        """
        n_list0 = n_list0
        a_list = a_list
        co_model = Word2Vec.load('coseg_text.model')
        si_p = []
        for n in n_list0:
            try:
                s = co_model.similarity(n, '手机')
                si_p.append(s)
            except Exception as e:
                si_p.append(0)
        index_list = list(map(si_p.index, heapq.nlargest(int(0.8*len(si_p)), si_p))) #取出和手机相关度最高的n
        n_list = []
        for index in index_list:
            n_list.append(n_list0[index])

        # 名词正向匹配
        comment1_df = pd.DataFrame(columns=['comment_tag', 'similarity'], index=[np.arange(100)])
        index = 0
        for i in range(len(n_list)):
            f_si = 0
            for j in range(len(a_list)):
                try:
                    si = co_model.similarity(n_list[i], a_list[j])
                    if si >= f_si:
                        f_si = si
                        comment_tag = n_list[i] + a_list[j]
                    else:
                        f_si = f_si
                except Exception as e:
                    print('语料库中缺少该词', e)
            comment1_df.loc[index, ] = [comment_tag, f_si]
            index += 1
        comment1_df = comment1_df.sort_values(by='similarity', ascending=False, ignore_index=True)
        comment1_df.dropna(subset=['comment_tag'], inplace=True)
        # comment1_df = comment1_df.iloc[0: int(0.2*len(comment_df)), ]

        # 形容词匹配逆向匹配
        comment2_df = pd.DataFrame(columns=['comment_tag', 'similarity'], index=[np.arange(100)])
        index = 0
        for i in range(len(a_list)):
            f_si = 0
            for j in range(len(n_list)):
                try:
                    si = co_model.similarity(n_list[j], a_list[i])
                    if si >= f_si:
                        f_si = si
                        comment_tag = n_list[j] + a_list[i]
                    else:
                        f_si = f_si
                except Exception as e:
                    print('语料库中缺少该词', e)
            comment2_df.loc[index, ] = [comment_tag, f_si]
            index += 1
            comment2_df = comment2_df.sort_values(by='similarity', ascending=False, ignore_index=True)
            comment1_df.dropna(subset=['comment_tag'], inplace=True)
        comment_df = pd.merge(comment1_df, comment2_df, on='comment_tag', how='inner')
        comment_df.dropna(subset=['comment_tag'], inplace=True)
        return comment_df

    def cleandata(self, x):
        """
        对数据进行清洗，替换一些不规则的标点符号
        """
        pat = re.compile("[^\u4e00-\u9fa5^.^a-z^A-Z^0-9]")  # 只保留中英文，去掉符号
        x = x.replace(' ', ',')
        emoji.demojize(x)  # 去掉表情表情符号
        x = re.sub(pat, ',', x)
        return x


