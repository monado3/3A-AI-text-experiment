import numpy
import chainer
from chainer import Variable, serializers, Chain, cuda
import chainer.functions as F
import chainer.links as L

from sentence_data import EOS_ID

# GPU上で計算を行う場合は，この変数を非Noneの整数にする
gpu_id = None

if gpu_id is not None:
    xp = cuda.cupy
else:
    xp = numpy


# Encoder-Decoderモデルを用いた翻訳モデルの定義
class TranslatorModel(chainer.Chain):
    def __init__(self, source_vocabulary_size, target_vocabulary_size, embed_size=100):
        self.lstm_layer_size = 1
        super(TranslatorModel, self).__init__()
        with self.init_scope():
            self.W_x_hi = L.EmbedID(source_vocabulary_size, embed_size)
            self.W_y_hi = L.EmbedID(target_vocabulary_size, embed_size)
            # NStepLSTMは多層LSTMを扱えるため，最初の引数がレイヤー数を表す（今回は1）
            # 最後の0.5はドロップアウト率
            # 意味は山崎先生の説明を思い出すか，各自で検索すること
            self.W_lstm_enc = L.NStepLSTM(self.lstm_layer_size,
                                     embed_size, embed_size, 0.5)
            self.W_lstm_dec = L.NStepLSTM(self.lstm_layer_size,
                                     embed_size, embed_size, 0.5)
            self.W_hr_y = L.Linear(embed_size, target_vocabulary_size)
        # 隠れ層の次元数を保存
        self.embed_size = embed_size
        if gpu_id is not None:
            cuda.get_device_from_id(gpu_id).use()
            self.to_gpu(gpu_id)
        self.reset_state()

    def reset_state(self):
        # 隠れ層の状態をリセットする
        # NStepLSTMはNoneを渡すことで勝手にゼロベクトルとして扱ってくれる
        self.hr = None
        # メモリーセルの状態も初期化する
        self.c = None

    def encode(self, sentences):
        # 各文章データをchainerのVariableに変換する
        # 例えば，もとが[[1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]] だった場合，
        # sentences_varは，[(3次元ベクトル), (4次元ベクトル), (5次元ベクトル)] のようなもの
        sentences_var = [Variable(xp.array(sentence, dtype=xp.int32))
                         for sentence in sentences]
        # hi_listは行列のリストのようなもの
        # 例えば，sentences_varが，[(3次元ベクトル), (4次元ベクトル), (5次元ベクトル)] で，
        # 隠れ層の大きさ（self.embed_size）が100のとき，
        # hi_listは，[(3×100 行列), (4×100 行列), (5×100 行列)] のようなものになる
        hi_list = [self.W_x_hi(sentence) for sentence in sentences_var]
        # NStepLSTMは隠れ層の状態を同時に入力に取る
        # hr，cは次の隠れ層の状態
        # 隠れ層の大きさが100で，バッチサイズが128のとき，
        # hr及びcは(1×128×100)のテンソルになる
        # 最初の1はLSTMのレイヤーの数（self.lstm_layer_size）
        # すなわち(レイヤー数×バッチサイズ×隠れ層の大きさ)
        # hr_listは，各入力に対するlstmの出力がベクトルになったもののリスト
        # hi_listが，[(3×100 行列), (4×100 行列), (5×100 行列)]のとき，
        # hr_listも，[(3×100 行列), (4×100 行列), (5×100 行列)]
        # ただし，NStepLSTMの入力次元と出力次元が異なる場合，hr_listの次元数が
        # hi_listの次元数と異なる場合もある．
        self.hr, self.c, hr_list = self.W_lstm_enc(self.hr, self.c, hi_list)

    def decode_train(self, sentences):
        # 各文章データをchainerのVariableに変換する
        # ただし，先頭にEOSを追加する
        # 理由については教科書の理論編を参照すること
        sentences_var = [Variable(xp.array([EOS_ID] + sentence,
                                           dtype=xp.int32))
                         for sentence in sentences]
        hi_list = [self.W_y_hi(sentence) for sentence in sentences_var]
        self.hr, self.c, hr_list = self.W_lstm_dec(self.hr, self.c, hi_list)
        # 各出力をまとめてyに変換する
        # hr_listが，[(3×100 行列), (4×100 行列), (5×100 行列)]で，
        # 例えば，出力語彙数（target_vocabulary_size）が2000のとき，
        # y_listは，[(3×2000 行列), (4×2000 行列), (5×2000 行列)]になる
        y_list = [self.W_hr_y(hr) for hr in hr_list]
        return y_list

    def decode_test(self, word_id_list):
        # test時には，入力が前のデータに依存するため，まとめてデコードすることができない
        # そのため，(1単語×バッチサイズ)という形式のデータを入力に取る
        # 隠れ層の大きさ（self.embed_size）が100のとき，
        # hi_listは，(1×100 行列)がバッチサイズ分だけ並んだリストになる
        hi_list = [self.W_y_hi(Variable(xp.array([word_id], dtype=xp.int32)))
                   for word_id in word_id_list]
        # hr_listも，(1×100 行列)がバッチサイズ分だけ並んだリストになる
        self.hr, self.c, hr_list = self.W_lstm_dec(self.hr, self.c, hi_list)
        # 例えば，出力語彙数（target_vocabulary_size）が2000のとき，
        # y_listは，(1×2000 行列)がバッチサイズ分だけ並んだリストになる
        y_list = [self.W_hr_y(hr) for hr in hr_list]
        return y_list

    # 入力データと正解データから，lossを計算する
    def loss(self, source_sentences, target_sentences):
        # まとめてエンコードする
        self.encode(source_sentences)
        # メモリーセルの状態は引き継がない
        self.c = None
        # まとめてデコードする
        # テスト時は正解データをデコーダの入力にすることに注意
        y_list = self.decode_train(target_sentences)
        accum_loss = None
        for y, target_sentence in zip(y_list, target_sentences):
            target_var = Variable(
                xp.array(target_sentence + [EOS_ID], dtype=xp.int32))
            # 非バッチ化のプログラムと比較するため，ロスの合計を求めたい
            # まとめて計算した場合ロスは平均値になるので
            # 合計を計算するために，単語数（+ EOS）を掛ける
            loss = (F.softmax_cross_entropy(y, target_var)
                    * (len(target_sentence) + 1))
            accum_loss = loss if accum_loss is None else accum_loss + loss
        return accum_loss

    # 現在のモデルを用いて，入力単語列から，翻訳結果の出力単語列を返す
    def test(self, source_sentences):
        batch_size = len(source_sentences)
        results = [[] for i in range(batch_size)]
        self.encode(source_sentences)
        # メモリーセルの状態は引き継がない
        self.c = None
        ends_sentence = [False for i in range(batch_size)]
        id_list = [EOS_ID for i in range(batch_size)]
        # 最大30単語で打ち切る
        for i in range(30):
            y_list = self.decode_test(id_list)
            # もっともそれらしい単語を得る
            id_list = [cuda.to_cpu(F.argmax(y).data) for y in y_list]
            # EOSが出力されたかどうかをチェックする
            for i, word_id in enumerate(id_list):
                ends_sentence[i] |= (word_id == EOS_ID)
            if all(ends_sentence):
                # すべてのデータについてEOSが出力済みなら終了
                break
            for result, word_id, end in zip(results, id_list, ends_sentence):
                if not end:
                    result.append(word_id)
        return results

    # モデルを読み込む
    def load_model(self, file_name):
        serializers.load_npz(file_name, self)

    # モデルを書き出す
    def save_model(self, file_name):
        serializers.save_npz(file_name, self)
