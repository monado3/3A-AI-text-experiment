50文のパラレルデータを用いてchainer で構築したNN で声質変換を行います。

00setup.sh: pip で 必要なライブラリのインストール
01feature_analysis.py: 特徴量抽出
02timewarping.py: 動的時間伸縮を行い引き伸ばした学習データを保存
03makelist.sh: 学習データとテストデータをわける
04train.py: ニューラルネットワークを学習
05convert.py: 学習したニューラルネットワークを使ってテストデータを変換する

上記のスクリプトを順次実行してください。
.sh の拡張子は bash ????.sh
.py の拡張子は python ????.py

で実行します
