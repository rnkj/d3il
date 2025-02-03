# Quick Start

## インストール
conda をインストールしている前提とする（ここでは miniconda でテストした）

```console
bash install.sh
```
`New Environment Name:`の表示には，`d3il`と入力してエンター

インストール後，
```console
$ source ~/miniconda3/bin/activate
(base) $ conda activate d3il
(d3il) $ 
```
と表示してあれば可

### Trouble shooting
> [!WARNING]
> Linuxでグローバル・ローカル環境にNumPyがインストール済みの場合，NumPy2系とNumPy1系が衝突する場合がある．
> 
> インストール後に
> ```
> (d3il) $ python -c "import numpy; print(numpy.__version__)"
> ```
> がバージョン1系であることをチェックしておく．テスト環境では"1.26.4"だった．  
> グローバル・ローカル環境にpip経由でnumpyをインストールしていると衝突する模様．
> 
> バージョン2系の場合，pinocchioが動作しない．
> ```console
> (d3il) $ pip list | grep numpy
> (d3il) $ conda list | grep numpy
> ```
> でnumpyのバージョンを確認の上，
> ```console
> (d3il) $ pip uninstall numpy
> ```
> でpipのnumpyをアンインストールする．

## 追加モジュールのインストール
conda の仮想環境 d3il が有効化されているものとする．
```console
git submodule update --init --update

# EIPL
pip install -r third_party/eipl/requirements.txt
pip install -e third_party/eipl

# [TODO] minGRU を追加
```

## Reproduce the results
Rollout中に`CUDA error: out of memory`が頻発するので，trainingとrolloutを分けた．

- Train state-based MLP on the Pushing task
```console
# training
bash scripts/aligning/bc_benchmark.sh

# rollout
# model-type は eval | last のどちらか．
#   - eval: 検証データに対する学習誤差が最も小さいモデル
#   - last: 学習終了時に保存されるモデル
python run_sim.py --logdir logs/aligning/sweeps/bc/YY-mm-dd/HH-MM-SS \
    --model-type last \
    --simulation.render True
```

- Train image-based ACT on the sorting task
```console
# training
bash scripts/sorting_4_vision/act_benchmark.sh

# rollout
# model-type は eval | last どちらも最新のモデル
python run_sim.py --logdir logs/sorting/sweeps/act_vision/YY=mm-dd/HH-MM-SS \
    --model-type last \
    --simulation.render True

# リモート計算機上では先頭に`MUJOCO_GL=egl`をつけておく
MUJOCO_GL=egl python run_sim.py --logdir logs/sorting/sweeps/act_vision/YY=mm-dd/HH-MM-SS \
    --model-type last
```

> [!NOTE]
> オプションの詳細は `python run_sim.py -h` から確認．
> * options:
>     - logdir (required): 学習モデルのあるフォルダ
>     - multirun-index: Multirunした場合のフォルダのインデックス番号（Alphabet順にソート）
>     - model-type {eval, last}: 検証データ最優か最終モデルか
> * simulation options: `configs/XXXXX.yaml` の `simulation`を設定する
>     - seed: 乱数シード
>     - device: 計算デバイス (CPU/CUDA)
>     - render / no-render: render でディスプレイに描画する．通常はFalse (no-render)．
>     - n-cores: 並列実行するシミュレーションの個数 (CPUのコア数に合わせるのがベター)
>     - n-contexts: タスクの初期条件の数（タスクによって異なる，[元論文](https://arxiv.org/abs/2402.14606)のTable 4を参照）
>     - n-trajectories-per-context: 1つの初期条件でテストを繰り返す回数．確率論的モデルはサンプリングごとに生成動作が異なる．

> [!IMPORTANT]
> image-based の場合，eval がタスク成功率での評価だったが，学習中にほとんどタスクが成功しない・評価が１回のみと機能していない． 
> しかも，評価回数を増やすと並列処理がGPU, CPUのメモリを潰してしまい，その内 out of memory を起こす．
> state-based の場合と同様に検証データに対する学習誤差で評価するように変更が必要

## Training RNN-based model
（検証中）上手くいっていない。。。

[rnn/](https://github.com/rnkj/d3il/tree/dev_rnn/rnn)以下を参照．

agentとmodelは[READMEのTrain your models](https://github.com/rnkj/d3il?tab=readme-ov-file#train-your-models)準拠．

datasetは時系列全体を訓練するようにwrapperクラスを用意した．

### SARNN
image-based 前提．現在 (Feb. 3rd, 2025)はsorting_4_visionのみ：
```console
bash scripts/sorting_4_vision/sarnn_benchmark.sh
```

## TODO
T=Top, S=Second, L=Lowest priority

- [T] minGRU
    * training with sliced- and full-sequences
- [T] aligning
- [S] other visual tasks
- [L] migrate from conda to pypi
    * https://pytorch.org/blog/pytorch2-6/