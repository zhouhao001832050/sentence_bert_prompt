# -*- coding:UTF-8 -*-
import re
import json
import jieba
import random
import pandas as pd
from tqdm import tqdm

class JaccardProcessor:
    """jaccard score processor"""
    def __init__(self, data_dir):
        self.data_dir = data_dir


    def getJaccardSimilarity(self, str1, str2):
        terms1 = jieba.cut(str1)
        terms2 = jieba.cut(str2)
        grams1 = set(terms1)
        grams2 = set(terms2)
        temp = 0
        for i in grams1:
            if i in grams2:
                temp += 1
        demoninator = len(grams2) + len(grams1) - temp # 并集
        jaccard_coefficient = float(temp/demoninator)   # 交集
        return jaccard_coefficient


    def get_data_fomat(self, input_path, mode):
        df = pd.read_excel(input_path)
        left, right = df["原始词"].to_list(),df["标准词"].to_list()
        data = list(zip(left, right))
        length = len(data)
        negative = []
        for i in tqdm(range(length)):
            left = data[i][0]
            candidates = []
            for j in range(length):
                if j != i:
                    right = data[j][1]
                    tempt = self.getJaccardSimilarity(left, right)
                    candidates.append((right, tempt))
            ranked = sorted(candidates, key=lambda x: x[1], reverse = True)[:5]
            negative.append((left, ranked[0][0]))
        positive = [(x[0], x[1], 0) for x in data]
        negative = [(x[0], x[1], 1) for x in negative]
        res = (positive + negative)
        random.shuffle(res)
        all_data = []
        with open(f"dataset/jaccard/{mode}.json", "w") as f_w:
            for element in res:
                d = {}
                corpus, entity, label = element[0], element[1], element[2]
                d["corpus"] = corpus
                d["entity"] = entity
                d["label"] = label
                f_w.write(json.dumps(d, ensure_ascii=False) + "\n")




class LongestSameStrProcessor:
    """longest score processor"""
    def __init__(self, data_dir):
        self.data_dir = data_dir
    


class DiceSimilarityProcessor:
    pass


class MinDistanceProcessor:
    pass


class NormalLevenProcessor:
    pass


class Bm25Processor:
    pass

if __name__ == "__main__":
    train_data_dir = "source_data/train.xlsx"
    dev_data_dir = "source_data/dev.xlsx"
    processor = JaccardProcessor(data_dir="")
    processor.get_data_fomat(train_data_dir, "train")
    # processor.get_data_fomat(dev_data_dir, "dev")