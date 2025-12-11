from starlette.websockets import WebSocketDisconnect
from fastapi import FastAPI, WebSocket
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        # 进入通信循环
        while True:
            # 尝试接收数据
            data = await websocket.receive_text()

            # 接收成功，处理数据
            print(f"Received data: {data}")
            await websocket.send_text(f"You sent: {data}")

    # 捕获 WebSocketDisconnect 异常
    except WebSocketDisconnect as e:
        # 在这里执行清理操作，如从活动连接列表中移除客户端
        print(f"Client disconnected. Code: {e.code}, Reason: {e.reason}")
    except Exception as e:
        # 捕获其他非预期的异常
        print(f"An unexpected error occurred: {e}")
