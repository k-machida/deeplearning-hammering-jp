# ディープラーニングによる打音検査
MATLABによるディープラーニングを使った打音検査のデモです。

![HammeringTest_s](https://user-images.githubusercontent.com/17340750/85642916-b2a61080-b6cd-11ea-8146-d4c851ca85ec.gif)

# 内容物
* demo_live.m : デモスクリプト本体
* samplePeaks.m : 打音を取得する関数
* sampleBG.m : 背景音を取得する関数
* speechSpectrograms.m : スペクトログラム変換関数
* weightedClassificationLayer.m : 重みづけされた分類層


# デモの流れ
demo_live.mをMATLABで開き、セクション毎に実行してください
* Step.１	最初に正常な打音と異常な打音をそれぞれ取得します。大きめのピークを検出し、自動で音声データを保存します。表示されるウィンドウを閉じると取得終了となります。
* Step.２	背景音を取得します。表示されるウィンドウを閉じると取得終了となります。
* Step.３	ネットワークを構築し、学習を実行します。
* Step.４	マイクからの音声に対しリアルタイムに処理し、検証を行います。

※
Step.1、Step.1の実行で取得されるデータは指定フォルダ（livedata）内に逐次追加されます。
一からデモを試しなおす場合はフォルダ内のデータを消去してください。


# 環境設定
* PCにマイクを接続し、各OSの設定から接続したマイクがアクティブになっていることを確認してください。


# デモで使用しているToolbox
* Signal Processing Toolbox：信号の前処理（ピーク検出）
* DSP System Toolbox       ：Audio Toolboxに必要
* Audio Toolbox            ：音声データの録音、オーディオ用のデータストア
* Deep Learning Toolbox    ：ディープニューラルネットワークの構築と学習
* （オプション）Parallel Computing Toolbox：GPUによる学習・推論の高速化



Copyright 2019 The MathWorks, Inc.
