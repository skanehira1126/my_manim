# my_manim

## 環境構築
Mac: https://docs.manim.community/en/stable/installation/macos.html

```
$ brew install py3cairo ffmpeg
$ brew install pango pkg-config scipy

### Tex
# full
$ brew install --cask mactex-no-gui
# small
$ brew install basictex --cask

# eval "$(/usr/libexec/path_helper)"

# brewのpythonのバージョンを確認し、そのバージョンに合わせる必要がある気がする
# setuptools 同梱のpkg_resourcesが必要 -> オプションで指定
$ rye install manim --extra-requirement setuptools
```


## メモ

### 基本的なCLIの使い方
`manim render`を利用してpythonファイルから動画を生成する。  
`manim`の後にsubcommandが設定されていない場合、`manim render`が呼び出される.

- `ql` / `qh` : `--quality`
    - `l`: 低画質
    - `h`: 高画質
- `-p`: 生成が終わった後、自動で動画を再生する
- `-s`: 生成が終わった後、Sceneの最後のプレビューを書き出す


### トランジションが想定通りの挙動をしないとき
manimの`Mobject.animate`はトランジションの最初の状態と最後の状態を補間するように遷移の描画を作る.   
そのため四角形の180度回転など開始と最後の状態が同じときは遷移がおかしくなる時がある.


