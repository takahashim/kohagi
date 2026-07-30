# CoreML 開発治具

kohagi の CoreML バックエンドを検証、開発するための治具。
変換済みモデルの中身を読む、ANE 配置を確認する、バケットごとの実測を取る、
2つのモデルの出力を突き合わせる、といった作業を行う。

kohagi のワークスペースからは外してある（ルートの `Cargo.toml` が `tools` を `exclude`）。
依存が kohagi の `Cargo.lock` と公開物に入らないようにするためで、ビルドは明示的に行う。

```console
cargo build --release --manifest-path tools/coreml-jigs/Cargo.toml
```

macOS 専用。
`coreml-inspect`、`computeplan`、`bucket-latency` は Core ML の API を叩くため、Apple Silicon の実機が必要になる。
`parity` は kohagi のバイナリを2回動かして出力を比べるだけなので、比較する側のバックエンドが動く環境であればよい。
`milblob` と `mil-inventory` はファイルを読むだけで、Core ML には触らない。
blob 形式の単体テストだけは環境に依存しない。

## `coreml-inspect`

変換済みディレクトリの中身を、モデル自身の記述から読んで報告する。

```console
coreml-inspect <dir|bundle> [--json]
```

ファイル名や `config.json` から推測せず、実際の入出力仕様を読む。
そのうえで、実行時に無言で壊れる三つの食い違いを検出する。

* `seq-<N>` と名乗るバンドルの実際の長さが `N` でない
* `config.json` の `hidden_size` がモデルの出力幅と違う
* `input_ids` / `attention_mask` / `hidden` が無い

食い違いがあれば非ゼロ終了する。
どちらの形式もコンパイルせずに読むので、`.mlpackage` でも初回ロードの約20秒を払わない（3バケットのディレクトリで 0.5〜0.9 秒）。

- `.mlmodelc` は `MLModelAsset` 経由。multi-function バンドルの全 function を報告する
- `.mlpackage` は `model.mlmodel` の protobuf を直接読む。`MLModelAsset` はコンパイル済みモデルしか開けないためで、公開前に確認したいのはまさにこの形式である

`.mlpackage` からは変換器の来歴も出る（`userDefined` メタデータ）。
モデルカードに書くべき coremltools と torch のバージョン、変換日がここにある。

```console
$ coreml-inspect ruri-v3-130m-coreml
seq-128.mlpackage [package]  264.6 MB
  weights : 123 blobs, 264.4 MB
  main         seq 128, dim 512
      input_ids        [1, 128] int32
      attention_mask   [1, 128] int32
      hidden           [1, 128, 512] fp16
      source_dialect   TorchScript
      version          9.0
      conversion_date  2026-07-23
      source           torch==2.13.0
```

multi-function な `.mlpackage` では、protobuf の description が既定 function のものだけなので、他の function は名前を挙げるだけで検査しない。
同じバンドルの `.mlmodelc` を読めば全 function を検査でき、どちらの形式でも kohagi 本体はロード時に全 function を検証する。

同じ検証は kohagi 本体のロード時にも入っている（`src/coreml.rs` の `check_io`）。
こちらは公開前に配布物をまとめて確かめるためのもので、`--json` でチェックリストに組める。

## `milblob`

`weight.bin`（MIL blob storage 形式）の検査と編集。

```console
milblob dump      <weight.bin|bundle>            blob 一覧
milblob verify    <weight.bin|bundle>            構造検証、問題があれば exit 1
milblob roundtrip <weight.bin|bundle> [out.bin]  読んで書き戻し、バイト比較
milblob diff      <a> <b>                        2つの blob ファイルを比較
milblob negate    <weight.bin> <meta-offset>     fp16 blob を符号反転（in place）
```

`verify` は sentinel、64バイト境界、metadata の連鎖、dtype、`padding_size_in_bits`、末尾の余剰バイトを見る。
最初の問題で止めずに全部報告する。

`diff` は「重みが変わった」と「レイアウトが変わった」を区別する。
`cmp` ではこれが分からない。

`negate` は汎用エディタではない。
`layer_norm` の gamma を反転するとモデルの出力が厳密に予測できる形で変わるため、Rust が書いた blob を Core ML が受理するかの端から端までの確認に使える。

## `computeplan`

`MLComputePlan` による op ごとのデバイス割当。

```console
computeplan <model> [function] [--tsv out.tsv] [--baseline in.tsv]
```

`.mlpackage` を渡した場合は先にコンパイルする。`MLComputePlan` はコンパイル済みしか受け付けず、
`.mlpackage` を渡すとエラーを返すのではなく C++ 側でプロセスを落とすためである。

`--baseline` を付けると記録済みの TSV と比較し、割当が変わっていれば非ゼロ終了する。
これが測定を判定に変える部分で、同じモデルが新しい macOS で別の配置になったことを検出できる。
比較するのは op 種別、preferred、supported の三列だけである。
`cost` は静的な推定値で、配置が変わらなくても動くため回帰とは見なさない。

割合だけでなく構造を報告する。

* **prologue**：先頭から続く ANE 以外の op 数。先頭に固まっているなら受け渡しは1回で、途中に混ざるならグラフが繰り返し分割されている（そのたびにテンソルの往復が入る）
* **stragglers**：prologue の外で CPU に置かれた ANE 可能な op
* **ceiling**：ANE を supported に持つ op の割合。ここに達していればグラフの書き換えで改善する余地はない

記録済みのベースラインは `baselines/` にある。

```console
computeplan <model> --baseline tools/coreml-jigs/baselines/plan-seq128-ruri-v3-130m.tsv
```

`ruri-v3-130m` の `seq-128` は ANE 723 / CPU 12、prologue 12、stragglers 0、ceiling 98.9% である。

## `mil-inventory`

MIL プログラムの op 構成と、2つのプログラムの差分。

```console
mil-inventory <bundle|model.mil|model.mlmodel> [--json]
mil-inventory <a> --diff <b>
```

MIL は2つの形で保存される。
`.mlmodelc` はテキスト（`model.mil`）、`.mlpackage` は protobuf（`Data/com.apple.CoreML/model.mlmodel`）である。
両方読む。emitter が作るのは後者で、前者だけを見ていると目標を取り違えかねないためである。

op 種別ごとの個数だけでなく**並び順**も比べる。
同じ op の集合でも別のプログラムに結線できるので、順序のほうが強い判定になる。

protobuf 側はスキーマなしで読む。
`Operation.type` へのフィールド番号パス（`502 → 2 → 2 → 3 → 2 → 3 → 1`）を実物から特定してあり、生成コードも `.proto` も要らない。
パスが見つからなければ「op が0個」ではなくエラーにする（別種のモデルを渡したときに空のグラフと読めてしまうため）。

`ruri-v3-130m` の `seq-128` では、`.mlpackage` と `.mlmodelc` が **1,539 op すべて同じ順序で一致**した。
つまり coremlc はこのモデルに対して融合も並べ替えもしていない。
op 列は `baselines/mil-seq128-ruri-v3-130m.json` に記録してある。

```console
$ mil-inventory seq-128.mlpackage --diff compiled/seq-128.mlmodelc
totals  : a 1539 ops (735 compute, 21 kinds), b 1539 ops (735 compute, 21 kinds)
sequence: identical, all 1539 operations in the same order
identical inventories (21 op kinds)
```

テキスト形式では `weight.bin` への参照数も報告する。

## `bucket-latency`

バケットごとの forward の実測。

```console
bucket-latency <dir|bundle> [--iters N] [--warmup N] [--rounds N]
```

既定は `--iters 20 --warmup 30 --rounds 5`。
ANE のレイテンシは系列長に対して単調でないため（bekko では 96 が 128 より遅い）、バケット構成はモデルごとに実測して決める。
ANE のレイテンシを測るときの作法を固定してある。

- ウォームアップを多めに取る（足りないと 50% 近い誤差が出る）
- 各ラウンドで全バケットを順に測る（途中でマシンが忙しくなったとき、後ろのバケットだけが損をしない）
- 最小値を採る。ばらつきは隣に出すので、ノイズの多い実行は平均に埋もれず見える

ラウンド間で 25% 以上ばらついたバケットは名前を挙げて警告する。
その状態でバケット間の差を読まないための歯止めである。

ロード時間は初回とそれ以降を分けて報告する。
プロセス内の最初のロードは CoreML と ANE サービスの起動を含むので、平均に混ぜるとバケットを増やす代償を読み間違える。

`ruri-v3-130m`（M2、キャッシュが温まった状態）の実測。

| バケット | ms/text | µs/token | ロード |
| --- | ---: | ---: | ---: |
| 128 | 4.14 | 32.3 | 0.11s |
| 256 | 8.69 | 34.0 | 0.08s |
| 512 | 23.61 | 46.1 | 0.10s |

## `parity`

同じテキストを2つのバックエンドに通して比較する。

```console
parity --kohagi ./target/release/kohagi --texts texts.txt \
       --common "--max-seq-length 512" \
       --a "--device cpu" \
       --b "--device coreml --coreml-dir <dir>"
```

`examples/parity_check.py` は PyTorch 参照との一致を見るもので、こちらは kohagi 同士をデバイス違いで比較する。
CoreML パスは CPU パスと fp16 の丸め分だけ違うはずで、`1 - cosine` が `1e-5` を大きく超えるなら精度以外の何かが変わっている。

比較を無意味にする設定は `--a` / `--b` ではなく **`--common`** に入れる。
`--prefix`（末尾の空白ひとつで足りる）や `--max-seq-length` が片側だけ違うと `1 - cosine` は桁で動き、測っているのはバックエンドではなく設定になる。
両側の実行コマンドは必ず出力する。

`1 - cosine` の最小、中央、最大、最悪行の最大絶対誤差と平均絶対誤差、非有限値の数を報告し、閾値（既定 `1e-4`）を超えたら非ゼロ終了する。
ID で突き合わせる（プロトコルが順序を保証しないため）。

**テキストは長いものを混ぜる。** local attention の窓は幅 128（前後 64）なので、24〜40 トークンの文だけでは
窓が一度も効かず、窓の誤りが見えない。194 トークンの入力を seq 512 に通すと窓の内外が両方現れる。

`ruri-v3-130m` の CPU 対 ANE、日本語6文での実測は `1 - cosine` が 3.6e-6 〜 4.6e-6 だった。

## `device-diff`

同じモデルを ANE と CPU で走らせて出力を比べる。

```console
device-diff <bundle> [function] [--tokens N]
```

演算の差（ANE は fp16、CPU は余裕がある）とグラフ・重みの差を分けるために書いた。

**書いた対象のモデルでは機能しない。** `-inf` でマスクした fp16 グラフは `CPUOnly` だと全出力が NaN になる
（`convert_coreml.py` の生成物でも同じ）。ANE を除外できる設定は `CPUOnly` だけで、`All` は ANE を含むので
一致しても情報がない。治具はその旨を出力して結論を出さない。

非有限値は `max` に畳まず数える。`f32::max(0.0, NaN)` は 0.0 を返すので、最大値の中に NaN が隠れてしまう。

## ライセンス

`src/blob.rs` の blob 形式は coremltools の `mlmodel/src/MILBlob/Blob/` を参照して実装した。
coremltools は BSD-3-Clause で、条文を `LICENSE-COREMLTOOLS-BSD` に置いている。
