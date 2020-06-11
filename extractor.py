# -*- coding:utf-8 -*-
import os
from pyltp import Segmentor, Postagger, Parser, NamedEntityRecognizer, SementicRoleLabeller
from gensim.models import Word2Vec
from cixing import Sentence_Parser
import pandas as pd
import numpy as np
import heapq
import re
import emoji

class Extractor:
    def __init__(self):
        self.co_model = Word2Vec.load('coseg_text.model')
        self.parser = Sentence_Parser()

    def get_seginfo(self, comment_list):
        for c in range(len(comment_list)):
            if len(comment_list[c]) <= 200:
                sentence = comment_list[c]
            else:
                sentence = comment_list[c][0: 201]
            if sentence != '':
                sentence = self.parser.cleandata(sentence)
                words, postags, child_dict_list, roles_dict, format_parse_list = self.parser.parser_main(sentence)
                n_list, a_list = self.parser.select(words, postags)

                tags = []
                for j in range(len(a_list)):
                    # print(child_dict_list[j])
                    p = words.index(a_list[j])
                    if child_dict_list[p]:
                        # print(child_dict_list[p])
                        # 构成的是主谓关系
                        if 'SBV' in child_dict_list[p]:
                            # print(child_dict_list[p])
                            si_p = []
                            for po in child_dict_list[p]['SBV']:
                                try:
                                    si = self.co_model.similarity(words[po], '手机')
                                    si_p.append(si)
                                except Exception as e:
                                    si_p.append(0)
                                id = list(map(si_p.index, heapq.nlargest(1, si_p)))  # 和该形容词最高的名词

                            s = child_dict_list[p]['SBV'][id[0]]
                            w1 = words[s] + a_list[j]
                            if child_dict_list[s]:
                                # print(child_dict_list[s])
                                if 'ATT' in child_dict_list[s]:
                                    if postags[child_dict_list[s]['ATT'][0]] == 'n':
                                        w2 = words[child_dict_list[s]['ATT'][0]] + w1
                                        tags.append(w2)
                                    else:
                                        tags.append(w1)
                            else:
                                tags.append(w1)

                        if 'ATT' in child_dict_list[p]:
                            # print(child_dict_list[p])
                            s = child_dict_list[p]['ATT'][0]
                            if 'SBV' in child_dict_list[s]:
                                w3 = words[child_dict_list[s]['SBV'][0]]
                                w4 = w3 + a_list[j]
                                id1 = words.index(w3)
                                if child_dict_list[id1]:
                                    if 'ATT' in child_dict_list[id1]:
                                        if postags[child_dict_list[id1]['ATT'][0]] == 'n':
                                            w5 = words[child_dict_list[id1]['ATT'][0]] + w4
                                            tags.append(w5)
                                else:
                                    tags.append(w4)

                with open('F:\pycharm project data\\taobao\phone\\tags.txt', 'a') as t:
                    t.writelines(' '.join(tags))
                    t.writelines('\n')
                    # f.close()
                print(tags)


                # 获取相关的名词和用户组
                n_list = list(set(n_list))
                if n_list:
                    with open('F:\pycharm project data\\taobao\phone\\noun.txt', 'a') as f:
                        f.writelines(' '.join(n_list))
                        f.writelines('\n')
                        # f.close()
                si_p = []
                u_list = ['小孩子', '作业', '高中', '初中', '儿童', '学校', '小孩', '老师', '网瘾', '中学生', '小学', '女儿', '小学生', '孩子', '闺女', '儿子', '学生', '网课', '小朋友',
                            '同事', '表弟', '亲戚', '姐妹', '表哥', '邻居', '同学', '朋友', '盆友', '链接',
                            '姥姥', '老太太', '老人', '岳母', '父亲', '老娘', '小姨', '老丈人', '舅舅', '岳父', '亲人', '老妈子', '老头儿', '婆婆', '老太', '老头子', '父母', '家婆', '老父亲', '老爹', '长辈', '大人', '外爷', '爷爷', '我爸', '老头', '老妈', '老爷子', '爸妈', '奶奶', '老伴', '老爸', '母亲', '老人家', '妈妈', '公公', '爸爸', '丈母娘', '姥爷', '家里人', '家人',
                            '老奶奶', '小伙子', '阿姨', '娘娘', '小姑子', '姐姐', '老妹', '婶婶', '大姐', '外孙', '小屁孩', '孙子', '姨妈', '棉袄', '伯母', '孝心',
                            '媳妇', '妹妹', '男朋友', '对象', '生日', '女朋友', '男票', '老婆', '弟弟', '情人节', '爹妈', '麻麻', '老公', '外甥', '老弟'
                ]
                # print(n_list)
                # print(n_list)
                for n in range(len(n_list)):
                    for u in range(len(u_list)):
                        try:
                            s = self.co_model.similarity(n_list[n], u_list[u])
                            si_p.append(s)
                        except Exception as e:
                                si_p.append(0)
                index_list = list(map(si_p.index, heapq.nlargest(1, si_p)))  # 取出和手机相关度最高的n
                # print(index_list)
                user_list = []
                for index in index_list:
                    index = int(index/len(u_list))
                    user_list.append(n_list[index])
                # print(user_list)
                with open('F:\pycharm project data\\taobao\phone\\user.txt', 'a') as u:
                    u.writelines(user_list)
                    u.writelines('\n')
                    # f.close()
            t.close()
            f.close()
            u.close()












































