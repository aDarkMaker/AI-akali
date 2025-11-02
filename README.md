# AI-Akali

华中科技大学风蓝动漫社自己的 AI-Bot

基于 NoneBot2 和 OneBot 协议的 QQ 机器人框架，支持 WebSocket 和 HTTP 两种通信方式。

## 目录

- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [项目架构](#项目架构)
- [配置说明](#配置说明)
- [使用指南](#使用指南)
- [开发指南](#开发指南)
- [项目结构](#项目结构)
- [TODO](#todo)

## 环境要求

- Python 3.13+
- uv - Python 包管理器
- Bun - JavaScript 运行时（用于开发工具）

## 快速开始

### 1. 克隆项目

```bash
git clone <repository-url>
cd AIAkali
```

### 2. 创建虚拟环境

**macOS/Linux:**

```bash
uv venv .bot
source .bot/bin/activate
```

**Windows:**

```cmd
uv venv .bot
.bot\Scripts\activate
```

### 3. 安装依赖

```bash
uv pip install -r requirements.txt
```

### 4. 配置

编辑 `config/bot.yaml`，设置 QQ 服务连接信息：

```yaml
host: 127.0.0.1 # NoneBot 服务地址
port: 8765 # NoneBot 服务端口

qq_host: 127.0.0.1 # QQ 服务地址
qq_ws_port: 8080 # QQ WebSocket 端口
qq_ws_path: /event # QQ WebSocket 路径
qq_http_port: 5700 # QQ HTTP API 端口
access_token: "" # 访问令牌（如果 QQ 服务需要）
```

### 5. 启动服务

**启动 NoneBot 服务（包括 WebSocket 服务器）:**

```bash
python scripts/serve.py
```

**启动 QQ Bot 客户端:**

```bash
python scripts/qq_bot.py
```

## 项目架构

```
┌─────────────────┐
│   QQ 服务端      │
│  (WebSocket)    │
└────────┬────────┘
         │
         │ WebSocket
         │
┌────────▼────────┐
│  QQBot Client   │ (scripts/qq_bot.py)
│  - QQClient     │
│  - Handlers     │
└─────────────────┘
         │
         │ 内部通信
         │
┌────────▼────────┐
│  NoneBot Server │ (scripts/serve.py)
│  - FastAPI      │
│  - WebSocket    │
│    Server       │
└─────────────────┘
```

### 核心模块

**Bot 模块** (`src/bot/`)

- `qq_client.py`: WebSocket 客户端，用于连接 QQ 服务
- `http_client.py`: HTTP 客户端，提供 HTTP API 调用
- `handlers.py`: 消息和事件处理器，支持插件化扩展

**网络模块** (`src/network/`)

- `ws_server.py`: WebSocket 服务器，处理客户端连接
- `websocket.py`: WebSocket 连接包装类
- `client.py`: HTTP 客户端基类
- `protocol.py`: 协议相关工具函数

**工具模块** (`src/utils/`)

- `config.py`: 配置文件加载器

详细架构说明请查看 [CODE_STRUCTURE.md](docs/CODE_STRUCTURE.md)

## 配置说明

### `config/bot.yaml`

| 配置项         | 说明              | 默认值      |
| -------------- | ----------------- | ----------- |
| `host`         | NoneBot 服务地址  | `127.0.0.1` |
| `port`         | NoneBot 服务端口  | `8765`      |
| `qq_host`      | QQ 服务地址       | `127.0.0.1` |
| `qq_ws_port`   | QQ WebSocket 端口 | `8080`      |
| `qq_ws_path`   | QQ WebSocket 路径 | `/event`    |
| `qq_http_port` | QQ HTTP API 端口  | `5700`      |
| `access_token` | 访问令牌          | `""`        |

### 环境变量

可以在项目根目录创建 `.bot` 文件来设置环境变量，NoneBot 会自动加载。

## 使用指南

### 添加消息处理器

编辑 `scripts/qq_bot.py`，在 `_setup_handlers()` 方法中注册新的处理器：

```python
def _setup_handlers(self):
    """Setup message handlers"""

    async def hello_handler(message: dict, raw_message: str, sender: dict):
        if "hello" in raw_message.lower():
            group_id = message.get("group_id")
            nickname = sender.get("nickname", "Unknown")
            await self.client.send_group_message(
                group_id, f"Hello, {nickname}!"
            )

    self.message_handler.register_group_handler(hello_handler)
```

### 发送消息

```python
# 发送群消息
await self.client.send_group_message(group_id, "Hello!")

# 发送私聊消息
await self.client.send_private_message(user_id, "Hello!")
```

### 事件类型

处理器支持三种事件类型：

- `message`: 消息事件（群消息、私聊消息）
- `notice`: 通知事件
- `request`: 请求事件

## 开发指南

### 安装开发工具

```bash
# 安装 pre-commit
uv pip install pre-commit
pre-commit install
pre-commit autoupdate

# 安装 commitizen（用于规范化提交）
uv pip install commitizen

# 安装前端开发工具
bun install
```

### 代码规范

项目使用以下工具保证代码质量：

- **Black**: Python 代码格式化
- **isort**: 导入排序
- **flake8**: Python 代码检查
- **prettier**: Markdown/JSON/YAML 格式化
- **cspell**: 拼写检查

### 提交代码

```bash
# 1. 添加修改
git add <files>

# 2. 使用 commitizen 提交（会自动运行 pre-commit hooks）
cz commit

# 3. 推送
git push origin main
```

### 可用脚本

使用 Bun 运行的脚本（定义在 `package.json`）：

```bash
# 格式化代码
bun run format

# 检查格式化
bun run format:check

# 拼写检查
bun run spell-check

# 交互式拼写修复
bun run spell-check:interactive
```

## 项目结构

```
AIAkali/
├── config/                 # 配置文件
│   └── bot.yaml           # Bot 配置
├── data/                   # 数据目录
│   ├── cache/             # 缓存
│   ├── datasets/          # 数据集
│   ├── logs/              # 日志
│   └── models/            # 模型
├── docs/                   # 文档
│   └── CODE_STRUCTURE.md  # 代码结构文档
├── scripts/                # 脚本
│   ├── serve.py           # NoneBot 服务启动
│   └── qq_bot.py          # QQ Bot 启动
├── src/                    # 源代码
│   ├── bot/               # Bot 模块
│   │   ├── handlers.py    # 消息处理器
│   │   ├── http_client.py # HTTP 客户端
│   │   └── qq_client.py   # WebSocket 客户端
│   ├── network/           # 网络模块
│   │   ├── client.py      # HTTP 客户端基类
│   │   ├── protocol.py    # 协议工具
│   │   ├── server.py      # 服务器基类
│   │   ├── websocket.py   # WebSocket 包装
│   │   └── ws_server.py   # WebSocket 服务器
│   ├── plugins/           # 插件目录
│   ├── training/          # 训练相关
│   ├── utils/             # 工具模块
│   │   └── config.py      # 配置加载
│   └── web/               # Web 相关
├── tests/                  # 测试
├── .bot/                   # Python 虚拟环境（gitignore）
├── cspell.json            # 拼写检查配置
├── pyproject.toml         # Python 项目配置
├── requirements.txt       # Python 依赖
└── README.md              # 本文件
```

## 贡献

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`cz commit`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 开发规范

- 遵循 PEP 8 Python 代码规范
- 使用 commitizen 进行规范化提交
- 确保所有 pre-commit hooks 通过
- 添加适当的注释和文档

## 许可证

待添加

## 致谢

- [NoneBot2](https://github.com/nonebot/nonebot2) - 异步 Python QQ 机器人框架
- [OneBot](https://onebot.dev/) - 聊天机器人应用接口标准

---

**注意**: 本项目仅供学习和研究使用。
