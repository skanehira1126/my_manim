# my_manim

## 環境構築
Mac: https://docs.manim.community/en/stable/installation/macos.html

```
$ brew install py3cairo ffmpeg
$ brew install pango pkg-config scipy

$ brew install --cask mactex-no-gui

# eval "$(/usr/libexec/path_helper)"

# brewのpythonのバージョンを確認し、そのバージョンに合わせる必要があった
$ rye pip 3.11
$ rye add manim
$ rye sync
```

