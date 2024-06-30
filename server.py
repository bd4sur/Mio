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

CURRENT_LLM_CONFIG_KEY = "qwen2-72b-16k"

LLM_CONFIG = {
    "qwen15-1b8-32k": {
        "model_path": "/home/bd4sur/ai/Qwen15/Qwen15-1B8-Chat-q8_0.gguf",
        "context_length": 32768
    },
    "qwen15-7b-32k": {
        "model_path": "/home/bd4sur/ai/Qwen15/Qwen15-7B-Chat-q4_k_m.gguf",
        "context_length": 32768
    },
    "qwen15-14b-32k": {
        "model_path": "/home/bd4sur/ai/Qwen15/Qwen15-14B-Chat-q4_k_m.gguf",
        "context_length": 32768
    },
    "qwen15-72b-16k": {
        "model_path": "/home/bd4sur/ai/Qwen15/Qwen15-72B-Chat-q4_k_m.gguf",
        "context_length": 16384
    },
    "qwen15-110b-16k": {
        "model_path": "/home/bd4sur/ai/Qwen15/Qwen15-110B-Chat-q4_k_m.gguf",
        "context_length": 16384
    },
    "qwen2-1b5-128k": {
        "model_path": "/home/bd4sur/ai/Qwen2/Qwen2-1B5-Instruct-q8_0.gguf",
        "context_length": 131072
    },
    "qwen2-7b-128k": {
        "model_path": "/home/bd4sur/ai/Qwen2/Qwen2-7B-Instruct-q6_k.gguf",
        "context_length": 131072
    },
    "qwen2-72b-16k": {
        "model_path": "/home/bd4sur/ai/Qwen2/Qwen2-72B-Instruct-q4_k_m.gguf",
        "context_length": 16384
    }
}

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(12).hex()
socketio = SocketIO(app, cors_allowed_origins="*")

LLM = None
IS_LLM_GENERATING = False


def start_https_server():
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(SSL_CERT_PATH, SSL_PRIVATE_KEY_PATH)
    httpd = socketserver.TCPServer((SERVER_IP, HTTPS_PORT), SimpleHTTPRequestHandler)
    httpd.socket = ssl_context.wrap_socket(
        httpd.socket,
        server_side=True)
    print(f"Started HTTPS Server {SERVER_IP}:{HTTPS_PORT}")
    httpd.serve_forever()





def load_gguf_model(model_path, context_length=16384):
    global LLM
    print(f"Loading {model_path} ...")
    del LLM
    gc.collect()
    LLM = Llama(
        model_path=model_path,
        chat_format="chatml",
        n_ctx=context_length,
        n_threads=36,
        n_gpu_layers=-1,
        verbose=False
    )
    print(f"Loaded {model_path}")


@socketio.on('change_llm', namespace='/chat')
def change_llm(msg):
    global CURRENT_LLM_CONFIG_KEY
    if IS_LLM_GENERATING == True:
        emit("change_llm_response", {"is_success": False, "message": "生成中，无法切换LLM。"})
        return
    llm_config_key = msg["llm_config_key"]
    if llm_config_key not in LLM_CONFIG:
        emit("change_llm_response", {"is_success": False, "message": "LLM设置不正确。"})
        return
    if llm_config_key != CURRENT_LLM_CONFIG_KEY:
        llm_config = LLM_CONFIG[llm_config_key]
        load_gguf_model(llm_config["model_path"], llm_config["context_length"])
        CURRENT_LLM_CONFIG_KEY = llm_config_key
        emit("change_llm_response", {"is_success": True, "message": f"LLM已切换为{llm_config_key}。"})
    else:
        emit("change_llm_response", {"is_success": True, "message": f"LLM未切换，仍为{llm_config_key}。"})



@socketio.on('interrupt', namespace='/chat')
def interrupt(msg):
    global IS_LLM_GENERATING
    IS_LLM_GENERATING = False
    print("请求：中断生成")


@socketio.on('submit', namespace='/chat')
def predict(msg):
    global IS_LLM_GENERATING
    if IS_LLM_GENERATING == True:
        print("Pass")
        return
    IS_LLM_GENERATING = True

    emit("chat_response", {"timestamp": time.ctime(), "status": "start", "llm_output": None})

    output = LLM.create_chat_completion(
        messages=msg["chatml"],
        stream=True,
        temperature=msg["config"]["temperature"],
        top_p=msg["config"]["temperature"],
        top_k=msg["config"]["top_k"]
    )

    response = ""
    for chunk in output:
        if IS_LLM_GENERATING == False:
            print("已中断")
            break
        IS_LLM_GENERATING = True
        delta = chunk['choices'][0]['delta']
        if 'content' in delta:
            response += delta['content']
            emit("chat_response", {
                "timestamp": time.ctime(),
                "status": "generating",
                "llm_output": {"role": "assistant", "content": response}
            })
    print(f"LLM Response: {response}")
    emit("chat_response", {
        "timestamp": time.ctime(),
        "status": "end",
        "llm_output": {"role": "assistant", "content": response}
    })
    IS_LLM_GENERATING = False


if __name__ == '__main__':
    # HTTPS Server
    https_server_process = Process(target=start_https_server)
    https_server_process.daemon = True
    https_server_process.start()
    # https_server_process.join()

    # LLM Server (flask app)
    llm_config = LLM_CONFIG[CURRENT_LLM_CONFIG_KEY]
    load_gguf_model(llm_config["model_path"], llm_config["context_length"])
    socketio.run(app, host=SERVER_IP, port=LLM_PORT, ssl_context=(SSL_CERT_PATH, SSL_PRIVATE_KEY_PATH))
