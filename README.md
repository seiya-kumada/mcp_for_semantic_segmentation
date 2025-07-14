# セマンティックセグメンテーション MCP サーバー

Claude Desktopと接続して画像のセマンティックセグメンテーションを実行するMCPサーバーです。

## 機能

- 画像のセマンティックセグメンテーション処理
- Claude Desktopとの統合
- 静的ファイル出力
- エラーハンドリング

## セットアップ

### 前提条件

- Python 3.13.5
- uv (仮想環境管理)
- Claude Desktop

### インストール

1. 仮想環境をアクティベート:
```bash
.venv\Scripts\activate
```

2. 依存関係をインストール:
```bash
uv pip install -e .
```


## Semantic Segmentationの訓練済みモデル
### 場所
```bash
C:\Users\seiyakumada\.cache\torch\hub\checkpoints
```
### モデルの詳細:
  - 名前: DeepLabV3 + ResNet50
  - 学習データ: COCO + PASCAL VOC
  - サイズ: 約160MB
  - クラス数: 21クラス（PASCAL VOC）

### キャッシュの仕組み:
  - 初回のみダウンロード
  - 2回目以降はキャッシュから読み込み
  - 手動削除も可能（再ダウンロードされる）

## 使用方法

### MCPサーバーの起動

```bash
python src/server.py
```

### Claude Desktopでの設定

Claude Desktopの設定ファイルに以下を追加:

```json
{
  "mcpServers": {
    "semantic-segmentation": {
      "command": "C:\\projects\\mcp_for_semantic_segmentation\\start_server.bat"
    }
  }
}
```

## プロジェクト構造

```
mcp_for_semantic_segmentation/
├── .venv/                          # 仮想環境
├── src/                            # ソースコード
│   ├── __init__.py
│   ├── server.py                   # MCPサーバー本体
│   ├── segmentation.py             # セグメンテーション処理
│   └── utils.py                    # ユーティリティ関数
├── static/                         # 静的ファイル
│   ├── input/                      # 入力画像保存ディレクトリ
│   └── output/                     # 出力画像保存ディレクトリ
├── models/                         # 学習済みモデル
├── requirements.txt                # 依存関係
├── pyproject.toml                 # プロジェクト設定
├── SPECIFICATION.md               # 仕様書
└── README.md                      # このファイル
```

## 開発

### テストの実行

```bash
pytest
```

### コードフォーマット

```bash
black src/
```

### リンティング

```bash
flake8 src/
```

## ライセンス

MIT License

## 貢献

プルリクエストやイシューの報告を歓迎します。
