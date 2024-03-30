import os
import time
import gc
from multiprocessing import Process

from http.server import socketserver, SimpleHTTPRequestHandler 
import ssl

from flask import Flask, render_template
from flask_socketio import SocketIO, emit

from llama_cpp import Llama

SERVER_IP = '0.0.0.0'
HTTPS_PORT = 8443
LLM_PORT = 5000

SSL_CERT_PATH = "/home/bd4sur/bd4sur.crt"
SSL_PRIVATE_KEY_PATH = "/home/bd4sur/key_unencrypted.pem"



MODEL_PATH = [
    "/home/bd4sur/ai/Qwen15/Qwen15-14B-Chat-q4_k_m.gguf",
    "/home/bd4sur/ai/Qwen15/Qwen15-72B-Chat-q4_k_m.gguf"
]

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(12).hex()
socketio = SocketIO(app, cors_allowed_origins="*")

LLM = None
SYSTEM_PROMPT = ""
CHAT_HISTORY = []
IS_RUNNING = False


def start_https_server():
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(SSL_CERT_PATH, SSL_PRIVATE_KEY_PATH)
    httpd = socketserver.TCPServer((SERVER_IP, HTTPS_PORT), SimpleHTTPRequestHandler)
    httpd.socket = ssl_context.wrap_socket(
        httpd.socket,
        server_side=True)
    print(f"Started HTTPS Server {SERVER_IP}:{HTTPS_PORT}")
    httpd.serve_forever()





def load_gguf_model(model_path):
    global LLM
    del LLM
    gc.collect()
    LLM = Llama(
        model_path=model_path,
        chat_format="chatml",
        n_ctx=16384,
        n_threads=36,
        n_gpu_layers=81
    )

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('reload_llm', namespace='/chat')
def reset_chat_history(req):
    global CHAT_HISTORY
    print("请求：切换LLM")
    if IS_RUNNING == True:
        print("生成中，无法切换LLM")
        emit("reload_llm_response", {"is_success": False})
        return
    model_index = req["model_index"]
    load_gguf_model(MODEL_PATH[model_index])
    CHAT_HISTORY = []
    print("切换LLM成功")
    emit("reload_llm_response", {"is_success": True, "model_index": model_index})

@socketio.on('reset', namespace='/chat')
def reset_chat_history(req):
    global CHAT_HISTORY
    print("请求：清除对话历史")
    if IS_RUNNING == True:
        print("生成中，无法清除对话历史")
        emit("reset_response", {"is_success": False})
        return
    print("清除成功")
    CHAT_HISTORY = []
    emit("reset_response", {"is_success": True})

@socketio.on('interrupt', namespace='/chat')
def interrupt(req):
    global IS_RUNNING
    IS_RUNNING = False
    print("请求：中断生成")

@socketio.on('submit', namespace='/chat')
def predict(req):
    global IS_RUNNING
    if IS_RUNNING == True:
        print("Pass")
        return
    IS_RUNNING = True

    req_content = req["content"]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_message, assistant_message in CHAT_HISTORY:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": assistant_message})
    messages.append({"role": "user", "content": req_content})

    emit("chat_response", {"role": "user", "content": req_content, "timestamp": time.ctime(), "status": "start"})

    output = LLM.create_chat_completion(
        messages=messages,
        stream=True,
        temperature=0.5,
        top_p=0.9,
        top_k=2
    )
    response = ""
    for chunk in output:
        if IS_RUNNING == False:
            print("已中断")
            break
        IS_RUNNING = True
        delta = chunk['choices'][0]['delta']
        if 'content' in delta:
            response += delta['content']
            emit("chat_response", {"role": "assistant", "content": response, "timestamp": time.ctime(), "status": "generating"})
    emit("chat_response", {"role": "assistant", "content": "", "timestamp": time.ctime(), "status": "end"})
    IS_RUNNING = False
    CHAT_HISTORY.append((req_content, response))


@socketio.on('connect', namespace='/chat')
def test_connect():
    emit('my response', {'data': 'Connected'})

@socketio.on('disconnect', namespace='/chat')
def test_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    # HTTPS Server
    https_server_process = Process(target=start_https_server)
    https_server_process.daemon = True
    https_server_process.start()
    # https_server_process.join()

    # LLM Server (flask app)
    load_gguf_model(MODEL_PATH[1])
    socketio.run(app, host=SERVER_IP, port=LLM_PORT, ssl_context=(SSL_CERT_PATH, SSL_PRIVATE_KEY_PATH))
