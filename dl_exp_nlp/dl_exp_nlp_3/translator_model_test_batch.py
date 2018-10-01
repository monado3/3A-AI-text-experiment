#!/usr/bin/env python3

import sys

import sentence_data
from translator_model_batch import TranslatorModel

dataset = sentence_data.SentenceData("dataset/data_1000.txt")

model = TranslatorModel(dataset.english_word_size(),
                        dataset.japanese_word_size())

model.load_model("trained_model/translator_batch_10.model")

# 入力された文章を単語に分割する
sentence = input("input an english sentence : ").split(' ')
# 単語IDのリストに変換する
sentence_id = []
for word in sentence:
    if not word:
        # 単語が空だったら飛ばす
        continue
    word = word.lower()
    word_id = dataset.english_word_id(word)
    if word_id is None:
        sys.stderr.write("Error : Unknown word " + word + "\n")
        sys.exit()
    else:
        sentence_id.append(word_id)

model.reset_state()
# リストのリストを与えると，リストのリストが返ってくることに注意
japanese = model.test([sentence_id])[0]
for word_id in japanese:
    print(dataset.japanese_word(word_id), end='')
print()
