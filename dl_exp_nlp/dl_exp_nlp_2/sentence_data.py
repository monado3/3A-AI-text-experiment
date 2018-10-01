import random

# EOSとは End Of Sentence の略であり，文の終わりを意味する
# EOSの単語IDを0と定義する
EOS_ID = 0
UNKNOWN_WORD_ID = 1


class SentenceData:
    def __init__(self, file_name):
        with open(file_name, "r") as f:
            self.en_word_to_id = {"<EOS>": EOS_ID, "<UNKNOWN>": UNKNOWN_WORD_ID}
            self.en_word_list = ["<EOS>", "<UNKNOWN>"]
            self.jp_word_to_id = {"<EOS>": EOS_ID, "<UNKNOWN>": UNKNOWN_WORD_ID}
            self.jp_word_list = ["<EOS>", "<UNKNOWN>"]
            self.en_sentences = []
            self.jp_sentences = []
            line = f.readline().rstrip("\n")
            while line:
                sentences = line.split("\t")
                english = sentences[0].split(" ")
                japanese = sentences[1].split(" ")

                # 単語IDのリスト
                en_sentence = []
                en_sentence_unk_added = []
                for word in english:
                    word = word.lower()
                    id = 0
                    id_unk_added = 0
                    if word in self.en_word_to_id:
                        id = self.en_word_to_id[word]
                        id_unk_added = id
                    else:
                        id = len(self.en_word_list)
                        self.en_word_list.append(word)
                        self.en_word_to_id[word] = id
                        # 初めて出てきた単語をUNKNOWNに変換
                        id_unk_added = UNKNOWN_WORD_ID
                    en_sentence.append(id)
                    en_sentence_unk_added.append(id_unk_added)

                # 単語IDのリスト
                jp_sentence = []
                jp_sentence_unk_added = []
                for word in japanese:
                    id = 0
                    id_unk_added = 0
                    if word in self.jp_word_to_id:
                        id = self.jp_word_to_id[word]
                        id_unk_added = id
                    else:
                        id = len(self.jp_word_list)
                        self.jp_word_list.append(word)
                        self.jp_word_to_id[word] = id
                        # 初めて出てきた単語をUNKNOWNに変換
                        id_unk_added = UNKNOWN_WORD_ID
                    jp_sentence.append(id)
                    jp_sentence_unk_added.append(id_unk_added)
                self.en_sentences.append(en_sentence)
                self.jp_sentences.append(jp_sentence)
                # 元のデータと，UNKNOWN化したデータを両方登録
                self.en_sentences.append(en_sentence_unk_added)
                self.jp_sentences.append(jp_sentence_unk_added)
                line = f.readline().rstrip("\n")

        # 上記方式だとUNKNOWNがデータの前の方に集中してしまう
        # また，元のデータとUNKNOWN化したデータが連続してしまう
        # データ間の相関関係をなくすため，シャッフルする
        zipped_sentences = list(zip(self.en_sentences, self.jp_sentences))
        random.shuffle(zipped_sentences)
        self.en_sentence = [sentences for sentences, _ in zipped_sentences]
        self.jp_sentence = [sentences for _, sentences in zipped_sentences]

    def sentences_size(self):
        return len(self.en_sentences)

    def japanese_word_size(self):
        return len(self.jp_word_list)

    def english_word_size(self):
        return len(self.en_word_list)

    def japanese_sentences(self):
        return self.jp_sentences

    def english_sentences(self):
        return self.en_sentences

    def japanese_word_id(self, word):
        if word in self.jp_word_to_id:
            return self.jp_word_to_id[word]
        else:
            return None

    def english_word_id(self, word):
        if word in self.en_word_to_id:
            return self.en_word_to_id[word]
        else:
            return None

    def japanese_word(self, id):
        return self.jp_word_list[id]

    def english_word(self, id):
        return self.en_word_list[id]
