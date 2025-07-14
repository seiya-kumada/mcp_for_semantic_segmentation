# MCP Semantic Segmentation Server - セットアップガイド

## 前提条件

- Python 3.13+
- uv (仮想環境管理ツール)
- Claude Desktop アプリケーション

## 1. 依存関係のインストール

```bash
# プロジェクトディレクトリに移動
cd C:\projects\mcp_for_semantic_segmentation

# 依存関係のインストール
uv sync
```

## 2. Claude Desktop設定

### 設定ファイルの場所
Claude Desktopの設定ファイルは以下の場所にあります：

**Windows:**
```
%APPDATA%\Claude\claude_desktop_config.json
```

**macOS:**
```
~/Library/Application Support/Claude/claude_desktop_config.json
```

### 設定内容
以下の内容を`claude_desktop_config.json`に追加してください：

**方法1: バッチファイルを使用（推奨）**
```json
{
  "mcpServers": {
    "semantic-segmentation": {
      "command": "start_server.bat",
      "cwd": "C:\\projects\\mcp_for_semantic_segmentation"
    }
  }
}
```

**方法2: 直接uvコマンドを使用**
```json
{
  "mcpServers": {
    "semantic-segmentation": {
      "command": "uv",
      "args": [
        "run",
        "python",
        "-m",
        "src"
      ],
      "cwd": "C:\\projects\\mcp_for_semantic_segmentation",
      "env": {
        "PYTHONPATH": "C:\\projects\\mcp_for_semantic_segmentation"
      }
    }
  }
}
```

**注意：** `cwd`のパスは実際のプロジェクトディレクトリに合わせて変更してください。

## 3. サーバーの動作確認

### 手動テスト
```bash
# プロジェクトディレクトリで実行
uv run python test_mvp.py
```

### サーバー起動テスト
```bash
# 手動でのテスト実行（推奨）
start_server_manual.bat

# Claude Desktop用のバッチファイル（標準入出力用）
start_server.bat

# PowerShellの場合
.\start_server.ps1

# 直接実行の場合
uv run python -m src
```

## 4. Claude Desktopとの接続

1. **Claude Desktopを再起動**
   - 設定ファイルを編集後、Claude Desktopを完全に終了し、再起動してください

2. **接続確認**
   - 新しいチャット画面で「🔌」アイコンをクリック
   - 「semantic-segmentation」サーバーが表示されることを確認

3. **動作テスト**
   - 画像ファイルを用意
   - 以下のようなプロンプトを試してください：

```
画像のセマンティックセグメンテーションを実行してください。
画像パス: C:\path\to\your\image.jpg
```

## 5. トラブルシューティング

### よくある問題と解決方法

#### 1. サーバーが起動しない
- **原因**: uvが見つからない
- **解決**: uvが正しくインストールされ、PATHに含まれているか確認

#### 2. Claude Desktopに表示されない
- **原因**: 設定ファイルの構文エラー
- **解決**: JSONの構文を確認し、カンマやクォートの位置を確認

#### 3. 画像処理でエラー
- **原因**: 画像ファイルの形式や権限
- **解決**: 対応形式（JPEG, PNG, BMP, TIFF）、10MB以下のファイルを使用

#### 4. 処理が遅い
- **原因**: CPUでの処理
- **解決**: 初回はモデルダウンロードのため時間がかかります

### ログの確認方法

```bash
# サーバーログを確認
uv run python -m src

# デバッグモードで実行
uv run python -c "
import asyncio
from src.server import MCPSemanticSegmentationServer
server = MCPSemanticSegmentationServer()
print('Server initialized successfully')
"
```

## 6. 使用例

### 基本的な使用方法

```
semantic_segmentation を使って画像を処理してください。
画像パス: C:\Users\username\Documents\sample.jpg
```

### 期待される出力形式

```json
{
  "result": {
    "output_url": "file:///C:/projects/mcp_for_semantic_segmentation/static/output/result_20241201_143022.png",
    "message": "セグメンテーションが完了しました",
    "processing_time": 2.5,
    "input_size": "1920x1080",
    "output_size": "1920x1080"
  }
}
```

## 7. 設定のカスタマイズ

### 処理パラメータの調整

`src/segmentation.py`で以下を調整可能：

- **デバイス設定**: CPU/GPU切り替え
- **画像サイズ**: 入力画像のリサイズサイズ
- **カラーマップ**: セグメンテーション結果の色設定

### ファイル管理

- **入力画像**: `static/input/`に自動保存
- **出力画像**: `static/output/`に結果保存
- **自動削除**: 7日以上古いファイルを自動削除

## 8. 開発者向け情報

### プロジェクト構造
```
mcp_for_semantic_segmentation/
├── src/
│   ├── server.py          # MCPサーバー本体
│   ├── segmentation.py    # セグメンテーション処理
│   └── utils.py           # ユーティリティ関数
├── static/
│   ├── input/             # 入力画像
│   └── output/            # 出力画像
├── test_mvp.py            # テストスクリプト
├── claude_desktop_config.json  # Claude Desktop設定
└── start_server.bat       # サーバー起動スクリプト
```

### 拡張のためのガイドライン

1. **新しいモデルの追加**: `src/segmentation.py`の`_load_model()`メソッドを修正
2. **新しいツールの追加**: `src/server.py`の`list_tools()`と`call_tool()`を拡張
3. **エラーハンドリングの改善**: `src/utils.py`にエラーコードを追加

## サポート

問題が発生した場合は、以下を確認してください：

1. `test_mvp.py`の実行結果
2. Claude Desktopのログ
3. 画像ファイルの形式・サイズ
4. 設定ファイルの構文