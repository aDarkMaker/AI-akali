# AI-Akali

华中科技大学风蓝动漫社自己的AI-Bot

## 项目架构：

## 如何部署：

## TODO-List：

## 开发工具：

### 激活虚拟环境

**macOS/Linux:**

```bash
# Create venv by uv
uv venv .bot

source .bot/bin/activate
```

**Windows:**

```cmd
# Create venv by uv
uv venv .bot

.bot/bin/activate
```

### 依赖安装

```bash
uv pip install -r requirements.txt
```

### Develop Tools

```bash
uv pip install pre-commit

pre-commit install

pre-commit autoupdate

uv pip install commitizen

bun init

bun install
```

### 修改提交

```bash
git add xxx

cz commit

git push -u origin main
```
