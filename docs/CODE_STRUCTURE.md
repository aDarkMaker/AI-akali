# Code Structure Documentation

## Project Overview

This document describes the structure and functions of the AIAkali project codebase.

## Directory Structure

```
src/
├── bot/                    # Bot client modules
│   ├── __init__.py
│   ├── qq_client.py        # WebSocket client for QQ
│   ├── http_client.py      # HTTP client for QQ API
│   └── handlers.py         # Message/event handlers
├── network/                # Network layer modules
│   ├── __init__.py
│   ├── client.py           # HTTP client base class
│   ├── websocket.py        # WebSocket connection wrapper
│   ├── ws_server.py        # WebSocket server
│   ├── protocol.py         # Protocol utilities
│   └── server.py           # Server base class
└── utils/                  # Utility modules
    ├── __init__.py
    └── config.py           # Configuration loader

scripts/
├── serve.py                # NoneBot service startup
└── qq_bot.py               # QQ Bot startup script
```

## Modules

### `src/bot/qq_client.py`

**Class: `QQClient`**

WebSocket client for connecting to QQ service.

#### Methods:

- `__init__(host: str, port: int, access_token: str)` - Initialize client with connection parameters
- `_get_ws_url(path: str) -> str` - Build WebSocket URL
- `_get_ws_headers() -> Dict[str, str]` - Get WebSocket headers with auth
- `connect(path: str)` - Connect to WebSocket server
- `disconnect()` - Close connection
- `send(action: str, params: Dict, echo: str) -> Dict` - Send API request and wait for response
- `_handle_api_response(message: Dict) -> bool` - Handle API response by echo matching
- `on(event_type: str, handler: Callable)` - Register event handler
- `_recv_loop()` - Main receive loop for processing messages
- `run(path: str)` - Start client connection and receive loop
- `send_group_message(group_id: int, message: str)` - Send message to group
- `send_private_message(user_id: int, message: str)` - Send message to user

### `src/bot/http_client.py`

**Class: `QQHttpClient`**

HTTP client for QQ API.

#### Methods:

- `__init__(host: str, port: int, access_token: str)` - Initialize HTTP client
- `_get_headers() -> Dict[str, str]` - Get HTTP headers with auth
- `request(action: str, params: Dict) -> Dict` - Send HTTP API request
- `send_group_message(group_id: int, message: str) -> Dict` - Send group message via HTTP
- `send_private_message(user_id: int, message: str) -> Dict` - Send private message via HTTP
- `close()` - Close HTTP client connection

### `src/bot/handlers.py`

**Class: `MessageHandler`**

Handler for message events with pluggable processors.

#### Methods:

- `__init__()` - Initialize with empty handler lists
- `register_group_handler(handler: Callable)` - Register group message handler
- `register_private_handler(handler: Callable)` - Register private message handler
- `handle_message(message: Dict) -> None` - Route message to appropriate handler
- `handle_group_message(message: Dict, raw_message: str, sender: Dict) -> None` - Process group message
- `handle_private_message(message: Dict, raw_message: str) -> None` - Process private message

**Class: `NoticeHandler`**

Handler for notice events.

#### Methods:

- `handle_notice(message: Dict) -> None` - Process notice event

**Class: `RequestHandler`**

Handler for request events.

#### Methods:

- `handle_request(message: Dict) -> None` - Process request event

### `src/network/ws_server.py`

**Class: `WebSocketServer`**

WebSocket server for handling client connections.

#### Methods:

- `__init__(host: str, port: int)` - Initialize server with host and port
- `on_action(action: str, handler: Callable)` - Register action handler
- `_handle_message(websocket, message: str) -> dict` - Process incoming message
- `_create_response(data: dict, success: bool) -> dict` - Create response message
- `_create_error_response(msg: str, retcode: int) -> dict` - Create error response
- `_client_handler(websocket, path: str)` - Handle client connection lifecycle
- `start()` - Start WebSocket server
- `broadcast(message: dict)` - Broadcast message to all connected clients

### `src/network/websocket.py`

**Class: `WSConnection`**

WebSocket connection wrapper.

#### Methods:

- `__init__(ws: WebSocketClientProtocol)` - Initialize with websocket
- `send(data: Dict)` - Send JSON data
- `recv() -> Dict` - Receive and parse JSON message
- `on(event_type: str, handler: Callable)` - Register event handler
- `handle(message: Dict)` - Dispatch message to registered handler

### `src/network/client.py`

**Class: `HttpClient`**

HTTP client base class using httpx.

#### Methods:

- `__init__(base_url: str, headers: Dict)` - Initialize with base URL and headers
- `post(endpoint: str, data: dict, headers: Dict)` - POST request
- `close()` - Close HTTP client

### `src/network/protocol.py`

Protocol utilities for message handling.

#### Functions:

- `parse_message(message: Dict) -> Dict` - Parse message (passthrough)
- `build_response(echo: str, data: Any) -> Dict` - Build response message
- `build_message(event_type: str, **kwargs) -> Dict` - Build event message

### `src/network/server.py`

**Class: `Server`**

Server base class using NoneBot driver.

#### Methods:

- `__init__()` - Initialize with NoneBot driver
- `run(host: str, port: int)` - Run server

### `src/utils/config.py`

Configuration loading utilities.

#### Functions:

- `load_config(config_path: str, base_dir: Path) -> Dict[str, Any]` - Load YAML configuration file

### `scripts/serve.py`

NoneBot service startup script.

#### Functions:

- `register_routes()` - Register HTTP routes and start WebSocket server
- `root()` - Root endpoint handler

#### Entry Point:

Runs NoneBot server on `127.0.0.1:8765` and WebSocket server on `127.0.0.1:8080`

### `scripts/qq_bot.py`

QQ Bot startup script.

#### Class: `QQBot`

Main bot class.

#### Methods:

- `__init__(config_path: str)` - Initialize bot with configuration
- `_setup_handlers()` - Register message handlers
- `start()` - Start bot connection and event loop

#### Functions:

- `setup_signal_handlers()` - Setup SIGINT/SIGTERM handlers
- `main()` - Main entry point

## Data Flow

1. **Bot Startup** (`scripts/qq_bot.py`)
    - Loads configuration
    - Creates `QQClient` instance
    - Registers handlers
    - Connects to WebSocket server
    - Enters receive loop

2. **Message Processing** (`src/bot/handlers.py`)
    - Receives message from `QQClient`
    - Routes to `MessageHandler.handle_message()`
    - Calls registered handlers
    - Handlers can send responses via `QQClient`

3. **WebSocket Server** (`src/network/ws_server.py`)
    - Accepts client connections
    - Processes incoming messages
    - Responds to API requests
    - Broadcasts events

4. **Network Layer** (`src/network/`)
    - `WSConnection`: Low-level WebSocket wrapper
    - `HttpClient`: HTTP request handling
    - `protocol`: Message format utilities

## Key Design Patterns

1. **Handler Registration**: Event handlers can be registered before or after connection
2. **Async/Await**: All network operations are asynchronous
3. **Separation of Concerns**: Clear separation between network, bot logic, and utilities
4. **Pluggable Handlers**: Message handlers can be easily extended

## Configuration

Configuration is loaded from `config/bot.yaml` with the following structure:

```yaml
qq_host: 127.0.0.1
qq_ws_port: 8080
qq_ws_path: /event
access_token: ""
```
