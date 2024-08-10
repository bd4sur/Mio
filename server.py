import os
import time
import gc
from multiprocessing import Process
from threading import Thread

from http.server import socketserver, SimpleHTTPRequestHandler 
import ssl

from flask import Flask, render_template
from flask_socketio import SocketIO, emit

from llama_cpp import Llama
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers import TextIteratorStreamer

USE_SSL = True

SERVER_IP = '0.0.0.0'
HTTPS_PORT = 8443 if USE_SSL else 8088
LLM_PORT = 5000

SSL_CERT_PATH = "/home/bd4sur/bd4sur.crt"
SSL_PRIVATE_KEY_PATH = "/home/bd4sur/key_unencrypted.pem"

CURRENT_LLM_CONFIG_KEY = "Qwen2-7B-Q80-128K"

LLM_CONFIG = {
    "Qwen2-1.5B-Q80-128K": {
        "model_path": "/home/bd4sur/ai/Qwen2/Qwen2-1B5-Instruct-q8_0.gguf",
        "context_length": 131072
    },
    "Qwen2-7B-Q80-128K": {
        "model_path": "/home/bd4sur/ai/Qwen2/Qwen2-7B-Instruct-q8_0.gguf",
        "context_length": 131072
    },
    "Qwen2-57B-A14B-Q4KM-128K": {
        "model_path": "/home/bd4sur/ai/Qwen2/Qwen2-57B-A14B-Instruct-q4_k_m.gguf",
        "context_length": 131072
    },
    "Qwen2-72B-Q4KM-16K": {
        "model_path": "/home/bd4sur/ai/Qwen2/Qwen2-72B-Instruct-q4_k_m.gguf",
        "context_length": 16384
    },
    "Qwen2-72B-GPTQ-Int4": {
        "model_path": "/home/bd4sur/ai/Qwen2/Qwen2-72B-Instruct-GPTQ-Int4",
        "context_length": 16384
    }
}

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(12).hex()
socketio = SocketIO(app, cors_allowed_origins="*")

LLM = None
IS_LLM_GENERATING = False


def start_https_server():
    httpd = socketserver.TCPServer((SERVER_IP, HTTPS_PORT), SimpleHTTPRequestHandler)
    if USE_SSL:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(SSL_CERT_PATH, SSL_PRIVATE_KEY_PATH)
        httpd.socket = ssl_context.wrap_socket(httpd.socket, server_side=True)
    print(f"Started HTTPS Server {SERVER_IP}:{HTTPS_PORT}")
    httpd.serve_forever()


def llm_gc():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_gguf_model(model_path, context_length=16384):
    print(f"Loading GGUF Model {model_path} ...")
    model = Llama(
        model_path=model_path,
        chat_format="chatml",
        n_ctx=context_length,
        n_threads=36,
        n_gpu_layers=-1,
        verbose=False
    )
    print(f"Loaded GGUF Model {model_path}")
    return ("gguf", model, None, None)



def load_torch_model_and_tokenizer(model_path):
    print(f"Loading torch(Transformers) Model {model_path} ...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, resume_download=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        resume_download=True,
        torch_dtype=torch.float16
    ).eval()

    config = GenerationConfig.from_pretrained(
        model_path, trust_remote_code=True, resume_download=True,
    )

    print(f"Loaded torch(Transformers) Model {model_path} ...")

    return ("torch", model, tokenizer, config)




def load_model(model_path, context_length=16384):
    global LLM
    del LLM
    llm_gc()
    model_type = "gguf" if model_path.split(".")[-1] == "gguf" else "torch"
    if model_type == "gguf":
        LLM = load_gguf_model(model_path, context_length)
    elif model_type == "torch":
        LLM = load_torch_model_and_tokenizer(model_path)


@socketio.on('get_current_llm_key', namespace='/chat')
def get_current_llm_key():
    emit("get_current_llm_key_callback", {"current_llm_key": CURRENT_LLM_CONFIG_KEY})


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
        load_model(llm_config["model_path"], llm_config["context_length"])
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

    model_type = LLM[0]
    model = LLM[1]
    tokenizer = LLM[2]
    # llm_config = LLM[3]

    response = ""

    if model_type == "gguf":
        output = model.create_chat_completion(
            messages=msg["chatml"],
            stream=True,
            temperature=msg["config"]["temperature"],
            top_p=msg["config"]["temperature"],
            top_k=msg["config"]["top_k"]
        )
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

    elif model_type == "torch":
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        text = tokenizer.apply_chat_template(
            msg["chatml"],
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
        generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=512)
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        for new_text in streamer:
            if IS_LLM_GENERATING == False:
                print("已中断")
                break
            IS_LLM_GENERATING = True
            response += new_text
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
    load_model(llm_config["model_path"], llm_config["context_length"])

    if USE_SSL:
        socketio.run(app, host=SERVER_IP, port=LLM_PORT, ssl_context=(SSL_CERT_PATH, SSL_PRIVATE_KEY_PATH))
    else:
        socketio.run(app, host=SERVER_IP, port=LLM_PORT)
