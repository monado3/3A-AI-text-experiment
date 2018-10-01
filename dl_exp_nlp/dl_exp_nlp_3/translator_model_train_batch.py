#!/usr/bin/env python3

import sentence_data
from chainer import cuda, optimizers
from translator_model_batch import TranslatorModel

dataset = sentence_data.SentenceData("dataset/data_1000.txt")

model = TranslatorModel(dataset.english_word_size(),
                        dataset.japanese_word_size())

optimizer = optimizers.Adam()
optimizer.setup(model)

batch_size = 64

epoch_num = 10
for epoch in range(epoch_num):
    print("{0} / {1} epoch start.".format(epoch + 1, epoch_num))

    sum_loss = 0.0
    for batch_start in range(0, dataset.sentences_size(), batch_size):
        batch_end = min(batch_start + batch_size, dataset.sentences_size())
        model.reset_state()
        model.zerograds()
        loss = model.loss(dataset.english_sentences()[
                          batch_start:batch_end], dataset.japanese_sentences()[batch_start:batch_end])
        loss.backward()
        loss.unchain_backward()
        optimizer.update()
        sum_loss += float(cuda.to_cpu(loss.data))

        if batch_end % (batch_size * 4) == 0:
            print("{0} / {1} sentences finished.".format(batch_end,
                                                         dataset.sentences_size()))

    print("mean loss = {0}.".format(sum_loss / dataset.sentences_size()))

    # 1 epoch 毎にファイルに書き出す
    modelfile = "trained_model/translator_batch_" + str(epoch + 1) + ".model"
    model.save_model(modelfile)
