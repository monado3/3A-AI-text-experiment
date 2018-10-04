import os.path as op
from glob import glob

import chainer
import numpy as np
from PIL import Image

MINI_CIFAR_DIR = op.join(op.dirname(op.abspath(__file__)), 'mini_cifar')


class MyCifarDataset(chainer.dataset.DatasetMixin):
    def __init__(self, train_dir):
        # TODO self._pathsに全ての画像のpathをリスト状に保存する
        self._paths = []
        self._labels = {
            'airplane': 0,
            'automobile': 1,
            'bird': 2,
            'cat': 3,
            'deer': 4,
            'dog': 5,
            'frog': 6,
            'horse': 7,
            'ship': 8,
            'truck': 9
        }
        self._paths = glob(op.join(MINI_CIFAR_DIR, train_dir, '*'), recursive=True)

    def __len__(self):
        """ データセットの数を返す関数 """
        return len(self._paths)

    def get_example(self, i):
        """    chainerに入力するための画像ファイルと，そのラベルを出力する．"""

        # TODO self._pathsからi番目のpathを取得して画像として読み込む
        path = self._paths[i]
        image = np.array(Image.open(path).convert('RGB').resize((224, 224)), np.float32)
        mean = [123.68, 116.779, 103.939]
        image -= mean

        # 画像データは3次元配列で入っており，その軸は(幅，高さ，チャンネル)の順である
        # TODO chainerは(チャンネル，幅，高さ)で入力することを求めるのでtransposeを行なう
        image_for_chainer = image.transpose((2, 1, 0))

        # pathから画像のラベルを取得することが出来る．
        # ラベルはairplaneのような文字列型ではなく，0のような整数である必要がある．
        label = self.get_label_from_path(path)
        return image_for_chainer, label

    def get_label_from_path(self, path):
        # TODO pathからlabel(0~9)を推測して返す関数を実装せよ
        return self._labels[op.dirname(path)]
