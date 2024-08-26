import os
import time
import gc
import base64
from multiprocessing import Process
from threading import Thread

import ssl
from http.server import socketserver, SimpleHTTPRequestHandler 
from flask import Flask
from flask_socketio import SocketIO, emit

# LLM
from llama_cpp import Llama
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers import TextIteratorStreamer

# TTS (ChatTTS)
import ctts
from ctts.tools.seeder import TorchSeedContext
from ctts.tools.audio import pcm_arr_to_mp3_view
from ctts.tools.logger import get_logger
from ctts.tools.audio import load_audio

USE_SSL = True

SERVER_IP = '0.0.0.0'
HTTPS_PORT = 8443 if USE_SSL else 8088
API_PORT = 5000

SSL_CERT_PATH = "/home/bd4sur/bd4sur.crt"
SSL_PRIVATE_KEY_PATH = "/home/bd4sur/key_unencrypted.pem"

TTS_MODEL_PATH = "/home/bd4sur/ai/_model/ChatTTS"

CURRENT_LLM_CONFIG_KEY = "Qwen2-7B-Q80-128K"

LLM_CONFIG = {
    "Qwen2-1.5B-Q80-128K": {
        "model_path": "/home/bd4sur/ai/_model/Qwen2/Qwen2-1B5-Instruct-q8_0.gguf",
        "context_length": 131072
    },
    "Qwen2-7B-Q80-128K": {
        "model_path": "/home/bd4sur/ai/_model/Qwen2/Qwen2-7B-Instruct-q8_0.gguf",
        "context_length": 131072
    },
    "Qwen2-57B-A14B-Q4KM-128K": {
        "model_path": "/home/bd4sur/ai/_model/Qwen2/Qwen2-57B-A14B-Instruct-q4_k_m.gguf",
        "context_length": 131072
    },
    "Qwen2-72B-Q4KM-16K": {
        "model_path": "/home/bd4sur/ai/_model/Qwen2/Qwen2-72B-Instruct-q4_k_m.gguf",
        "context_length": 16384
    },
    "Qwen2-72B-GPTQ-Int4": {
        "model_path": "/home/bd4sur/ai/_model/Qwen2/Qwen2-72B-Instruct-GPTQ-Int4",
        "context_length": 16384
    },
    "Qwen1.5-110B-Q4KM-16K": {
        "model_path": "/home/bd4sur/ai/_model/Qwen15/Qwen15-110B-Chat-q4_k_m.gguf",
        "context_length": 16384
    }
}

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(12).hex()
socketio = SocketIO(app, cors_allowed_origins="*")

logger = get_logger("Mio")

LLM = None
IS_LLM_GENERATING = False

TTS = ctts.CTTS(logger)


def start_https_server():
    httpd = socketserver.TCPServer((SERVER_IP, HTTPS_PORT), SimpleHTTPRequestHandler)
    if USE_SSL:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(SSL_CERT_PATH, SSL_PRIVATE_KEY_PATH)
        httpd.socket = ssl_context.wrap_socket(httpd.socket, server_side=True)
    print(f"Started HTTPS Server {SERVER_IP}:{HTTPS_PORT}")
    httpd.serve_forever()

def save_mp3_file(wav, index):
    data = pcm_arr_to_mp3_view(wav)
    mp3_filename = f"speech_{index}.mp3"
    with open(mp3_filename, "wb") as f:
        f.write(data)
    logger.info(f"Audio file saved to {mp3_filename}")

def load_tts_model():
    logger.info("Initializing ChatTTS...")
    if TTS.load(model_path=TTS_MODEL_PATH, compile=False):
        logger.info("TTS Models loaded successfully.")
    else:
        logger.error("TTS Models load failed.")

@socketio.on('generate', namespace='/tts')
def tts_generate(msg):
    stream = False
    session_id = msg["session_id"]
    speaker = msg["speaker"]
    refine_enabled = msg["refine_enabled"]
    refine_prompt = msg["refine_prompt"]
    generation_prompt = msg["generation_prompt"]
    texts = msg["texts"]

    emit("generate_start", {"is_success": True, "session_id": session_id})

    with TorchSeedContext(42):
        if refine_enabled:
            texts = TTS.infer(
                texts,
                skip_refine_text=False,
                refine_text_only=True,
                params_refine_text=ctts.CTTS.RefineTextParams(prompt=refine_prompt),
            )
            logger.info(texts)

        emit("generate_refine", {"is_success": True, "session_id": session_id, "refined_text": texts})

        if speaker == "sample":
            speaker_sample_audio = TTS.sample_audio_speaker(load_audio("sample.mp3", 24000))
            speaker_sample_text = "与样例音频完全对应的[uv_break]文本转写"
            speaker_embedding = None
        else:
            speaker_sample_audio = None
            speaker_sample_text = None
            speaker_embedding = "蘁淰敕欀摃誌緘義囡胹讵祪萀梂晳癧亇婇嚕儝揇偩贻咆煴淀蠀欐萵貺箚弃胦菍弁夞皥焆喦卢狧乩夏淔莨臃赽奛溕筡誑緶貿捨讖卢瑫嬅哙硚惣蚵刻玏炉跸徱澾登嬖絢烇嫷媓蔢产虜椪眕俟徊吞詸愣備恍珳湉璑訷珽菹訴痙濽圴謗瘾皡憖啤囊偐惏嶩役磅惃碝贬貇行楝薇磉数綊蟊弤夋荄壪攫撧杶岈硯葳赛悫宸岩稼琜串汏僎灡峂蝇筋茹聈柵焵皿綏缊橥爝澺縬樢訣潙许壚朔仑螽穨糼稰礌漖噍脠庭穪栽嚽袿蟢朁睬筸獸蜍荃俜椉狴掠歾泓葁潚蚗刣悬縶執萏淪肬涼覎培煟苇攁蕘瞥覹緌玽忖熒苼偶巴氶壡卝僕聥栘袴瞗匥弯剫堎搒烅芡渢蒺仉濃猿焳觔吼嚾簬伋諿圀晑牣缄澜枡溒甆欌槙螶璭惝賙扣氒嘕質僜乧畭徉蟖裔既流橊卺奪襾耨嬖脡甆槡巢誸倦訐忂匼俵宰凥覡穰捠斋孖瀤謹讗揲害祩歊蠯旸忎継亍憭徿礯蜷絕凵腂凾疼渴痳旑賧槢浃圕畧晖庞捻翺岊澛縃婳哵喳唗趢咊綼倅佹艅丽趔攪懦蟜牢庨蒘薪蜩煐揈羄获话涴婔傊庪蚫曃氻肙瞥响丹粫璯蕷舺捆搞爳瞻僱潜袄恛懝嗀碥嶎椓一奥濇嵊卂燡懼礅護懭爋蚿檠蟔氖謻淫曇乯槙孓僷疶笺慛誏籜扰固嚲幦吲朸罺眅晝噱簭椼嘎坷嬢粆师恢埨伮跭侂庒瞭幕擛裌藩屙径皎蕾猨徲徎俬渰畣瓂嵭璌砟勗睃沭吾嗅端匈椃棒瓁刉觤伎虗貉柨燜緷奦曛綡拷撮箓縳蠺綢臑栳愆蛴聱嫼亞人翢疋貼横査艼妽菪梷薓棆焉彘撙蝳籯嬎谡毮牥狊垦岩刡趄虾葤纵爩媳泟惏撙剗瓕濂届竨跘匊殱幓你侜羯籕匐璾凡樃俋臺虘蝄懇罶悥孆击捪蛖畋屁蠐蟦埙夬俟抗籵惉柌箼瞀庻勨串捅窮氶賰燧捵蕓汐藈噱臷児汱留翷枾昅想慱羆蚅聢珹礦諅坔嚇缤冫窙蟓壡洦啓茖汬嶉賭汯紡屒揁熀蛾数篧哞撌塔妥蓗懘犌富圃胃莧絗喘葔改脧焛摆儭庥挖謪擾緖蓐卼褟萎磗侻恏嫒愗欮樞羻喻厚欫参姿剝堬絊挒暘擋緷貧妖欷牶诬囌揋膝湷觸柗灚烚誵暡讟卒縉乍跊疥褧皏菈吓穭脓呲挿燐藒澬珹嗧茪芝灲吋崩请瀓蜋棦掙沝刴彸褕缥誐喘胤櫂愄娇肥吥匚佯揔舔瑪燣孲珬谱炆夤梑狕祠痸浾薐萂暟葯俴涊怰蕲眞煍嘷趌褖弹硒囑琋焧截嵨蘈卥呬畸痾厾橓槔赒熰毪稵囨瀺綰穧楳囹籽窷俆坵萵澳瘏穉焬睳洲蓴懬膄揳妦悰尯堇翩葾弉忲昦蟝慎摏衃榶硟兡啥焛堵汼殗搩枌狎斳蒞貼敱叏刳梋莯椥刣吿埓仹熖悲嫿嫤哆怔祸嵢狴斻肎唤樵糪禾瓺摏璂跨卶欢刖薁嬼蚨壳栮余育熪跭讘勖亾擕硬悦痕屺櫞袁椤穟帀㴃"

        params_infer_code = ctts.CTTS.InferCodeParams(
            prompt=generation_prompt,
            spk_smp=speaker_sample_audio,
            txt_smp=speaker_sample_text,
            spk_emb=speaker_embedding,
            temperature=0.3,
            top_P=0.7,
            top_K=20,
            stream_batch=24,
            stream_speed=24000
        )

        wavs = TTS.infer(
            texts,
            stream,
            skip_refine_text=True,
            params_infer_code = params_infer_code
        )
    logger.info("TTS Inference finished.")

    for channel_index, wav in enumerate(wavs):
        if stream:
            for _, w in enumerate(wav):
                mp3file = pcm_arr_to_mp3_view(w)
                mp3base64 = base64.b64encode(mp3file).decode("ascii")
                emit("generate_response", {"is_success": True, "session_id": session_id, "channel_index": channel_index, "mp3data": mp3base64})
                logger.info(f"Sent {_} mp3 chuck")
        else:
            mp3file = pcm_arr_to_mp3_view(wav)
            mp3base64 = base64.b64encode(mp3file).decode("ascii")
            emit("generate_response", {"is_success": True, "session_id": session_id, "channel_index": channel_index, "mp3data": mp3base64})
            # save_mp3_file(wav, time.monotonic_ns())
    logger.info("TTS Audio generation finished.")


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

    # TTS Server
    load_tts_model()

    # LLM Server (flask app)
    llm_config = LLM_CONFIG[CURRENT_LLM_CONFIG_KEY]
    load_model(llm_config["model_path"], llm_config["context_length"])

    if USE_SSL:
        socketio.run(app, host=SERVER_IP, port=API_PORT, debug=False, log_output=False, ssl_context=(SSL_CERT_PATH, SSL_PRIVATE_KEY_PATH))
    else:
        socketio.run(app, host=SERVER_IP, port=API_PORT, debug=False, log_output=False)
