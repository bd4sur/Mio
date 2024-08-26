import os, sys

now_dir = os.getcwd()
sys.path.append(now_dir)

import torch

import argparse

import gradio as gr

import random
from typing import Optional
from time import sleep

from tools.audio import float_to_int16, has_ffmpeg_installed, load_audio
from tools.logger import get_logger
from tools.seeder import TorchSeedContext
from tools.normalizer import normalizer_en_nemo_text, normalizer_zh_tn

logger = get_logger(" WebUI ")

import ctts

chat = ctts.Chat(get_logger("ChatTTS"))

custom_path: Optional[str] = None

has_interrupted = False
is_in_generate = False

seed_min = 1
seed_max = 4294967295

use_mp3 = has_ffmpeg_installed()
if not use_mp3:
    logger.warning("no ffmpeg installed, use wav file output")

# 音色选项：用于预置合适的音色
voices = {
    "Default": {"seed": 2},
    "Timbre1": {"seed": 1111},
    "Timbre2": {"seed": 2222},
    "Timbre3": {"seed": 3333},
    "Timbre4": {"seed": 4444},
    "Timbre5": {"seed": 5555},
    "Timbre6": {"seed": 6666},
    "Timbre7": {"seed": 7777},
    "Timbre8": {"seed": 8888},
    "Timbre9": {"seed": 9999},
}


def generate_seed():
    return gr.update(value=random.randint(seed_min, seed_max))


# 返回选择音色对应的seed
def on_voice_change(vocie_selection):
    return voices.get(vocie_selection)["seed"]


def on_audio_seed_change(audio_seed_input):
    with TorchSeedContext(audio_seed_input):
        rand_spk = chat.sample_random_speaker()
    return rand_spk


def load_chat(cust_path: Optional[str], coef: Optional[str]) -> bool:
    if cust_path == None:
        ret = chat.load(coef=coef, compile=False)
    else:
        logger.info("local model path: %s", cust_path)
        ret = chat.load(
            "custom", custom_path=cust_path, coef=coef, compile=False
        )
        global custom_path
        custom_path = cust_path
    if ret:
        try:
            chat.normalizer.register("en", normalizer_en_nemo_text())
        except ValueError as e:
            logger.error(e)
        except:
            logger.warning("Package nemo_text_processing not found!")
            logger.warning(
                "Run: conda install -c conda-forge pynini=2.1.5 && pip install nemo_text_processing",
            )
        try:
            chat.normalizer.register("zh", normalizer_zh_tn())
        except ValueError as e:
            logger.error(e)
        except:
            logger.warning("Package WeTextProcessing not found!")
            logger.warning(
                "Run: conda install -c conda-forge pynini=2.1.5 && pip install WeTextProcessing",
            )
    return ret


def reload_chat(coef: Optional[str]) -> str:
    global is_in_generate

    if is_in_generate:
        gr.Warning("Cannot reload when generating!")
        return coef

    chat.unload()
    gr.Info("Model unloaded.")
    if len(coef) != 230:
        gr.Warning("Ingore invalid DVAE coefficient.")
        coef = None
    try:
        global custom_path
        ret = load_chat(custom_path, coef)
    except Exception as e:
        raise gr.Error(str(e))
    if not ret:
        raise gr.Error("Unable to load model.")
    gr.Info("Reload succeess.")
    return chat.coef


def on_upload_sample_audio(sample_audio_input: Optional[str]) -> str:
    if sample_audio_input is None:
        return ""
    sample_audio = load_audio(sample_audio_input, 24000)
    spk_smp = chat.sample_audio_speaker(sample_audio)
    del sample_audio
    return spk_smp


def _set_generate_buttons(generate_button, interrupt_button, is_reset=False):
    return gr.update(
        value=generate_button, visible=is_reset, interactive=is_reset
    ), gr.update(value=interrupt_button, visible=not is_reset, interactive=not is_reset)


def refine_text(
    text,
    text_seed_input,
    refine_text_flag,
):
    global chat

    if not refine_text_flag:
        sleep(1)  # to skip fast answer of loading mark
        return text

    with TorchSeedContext(text_seed_input):
        text = chat.infer(
            text,
            skip_refine_text=False,
            refine_text_only=True,
        )

    return text[0] if isinstance(text, list) else text


def generate_audio(
    text,
    temperature,
    top_P,
    top_K,
    spk_emb_text: str,
    stream,
    audio_seed_input,
    sample_text_input,
    sample_audio_code_input,
):
    global chat, has_interrupted

    if not text or has_interrupted or not spk_emb_text.startswith("蘁淰"):
        return None

    params_infer_code = ctts.Chat.InferCodeParams(
        spk_emb=spk_emb_text,
        temperature=temperature,
        top_P=top_P,
        top_K=top_K,
    )

    if sample_text_input and sample_audio_code_input:
        params_infer_code.txt_smp = sample_text_input
        params_infer_code.spk_smp = sample_audio_code_input
        params_infer_code.spk_emb = None

    with TorchSeedContext(audio_seed_input):
        wav = chat.infer(
            text,
            skip_refine_text=True,
            params_infer_code=params_infer_code,
            stream=stream,
        )
        if stream:
            for gen in wav:
                audio = gen[0]
                if audio is not None and len(audio) > 0:
                    yield 24000, float_to_int16(audio).T
                del audio
        else:
            yield 24000, float_to_int16(wav[0]).T


def interrupt_generate():
    global chat, has_interrupted

    has_interrupted = True
    chat.interrupt()


def set_buttons_before_generate(generate_button, interrupt_button):
    global has_interrupted, is_in_generate

    has_interrupted = False
    is_in_generate = True

    return _set_generate_buttons(
        generate_button,
        interrupt_button,
    )


def set_buttons_after_generate(generate_button, interrupt_button, audio_output):
    global has_interrupted, is_in_generate

    is_in_generate = False

    return _set_generate_buttons(
        generate_button,
        interrupt_button,
        audio_output is not None or has_interrupted,
    )



ex = [
    [
        "四川美食确实以辣闻名，但也有不辣的选择。比如甜水面、赖汤圆、蛋烘糕、叶儿粑等，这些小吃口味温和，甜而不腻，也很受欢迎。",
        0.3,
        0.7,
        20,
        2,
        42,
        True,
    ],
    [
        "What is your favorite english food?",
        0.5,
        0.5,
        10,
        245,
        531,
        True,
    ],
    [
        "chat T T S is a text to speech model designed for dialogue applications. [uv_break]it supports mixed language input [uv_break]and offers multi speaker capabilities with precise control over prosodic elements like [uv_break]laughter[uv_break][laugh], [uv_break]pauses, [uv_break]and intonation. [uv_break]it delivers natural and expressive speech,[uv_break]so please[uv_break] use the project responsibly at your own risk.[uv_break]",
        0.8,
        0.4,
        7,
        70,
        165,
        False,
    ],
]



def main():

    with gr.Blocks() as demo:
        gr.Markdown("# ChatTTS WebUI")
        gr.Markdown("- **GitHub Repo**: https://github.com/2noise/ChatTTS")
        gr.Markdown("- **HuggingFace Repo**: https://huggingface.co/2Noise/ChatTTS")

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Input Text",
                    lines=4,
                    max_lines=4,
                    placeholder="Please Input Text...",
                    value=ex[0][0],
                    interactive=True,
                )
                sample_text_input = gr.Textbox(
                    label="Sample Text",
                    lines=4,
                    max_lines=4,
                    placeholder="If Sample Audio and Sample Text are available, the Speaker Embedding will be disabled.",
                    interactive=True,
                )
            with gr.Column():
                with gr.Tab(label="Sample Audio"):
                    sample_audio_input = gr.Audio(
                        value=None,
                        type="filepath",
                        interactive=True,
                        show_label=False,
                        waveform_options=gr.WaveformOptions(
                            sample_rate=24000,
                        ),
                        scale=1,
                    )
                with gr.Tab(label="Sample Audio Code"):
                    sample_audio_code_input = gr.Textbox(
                        lines=12,
                        max_lines=12,
                        show_label=False,
                        placeholder="Paste the Code copied before after uploading Sample Audio.",
                        interactive=True,
                    )

        with gr.Row():
            refine_text_checkbox = gr.Checkbox(
                label="Refine text", value=ex[0][6], interactive=True
            )
            temperature_slider = gr.Slider(
                minimum=0.00001,
                maximum=1.0,
                step=0.00001,
                value=ex[0][1],
                label="Audio Temperature",
                interactive=True,
            )
            top_p_slider = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                step=0.05,
                value=ex[0][2],
                label="top_P",
                interactive=True,
            )
            top_k_slider = gr.Slider(
                minimum=1,
                maximum=20,
                step=1,
                value=ex[0][3],
                label="top_K",
                interactive=True,
            )

        with gr.Row():
            voice_selection = gr.Dropdown(
                label="Timbre",
                choices=voices.keys(),
                value="Default",
                interactive=True,
            )
            audio_seed_input = gr.Number(
                value=ex[0][4],
                label="Audio Seed",
                interactive=True,
                minimum=seed_min,
                maximum=seed_max,
            )
            generate_audio_seed = gr.Button("\U0001F3B2", interactive=True)
            text_seed_input = gr.Number(
                value=ex[0][5],
                label="Text Seed",
                interactive=True,
                minimum=seed_min,
                maximum=seed_max,
            )
            generate_text_seed = gr.Button("\U0001F3B2", interactive=True)

        with gr.Row():
            spk_emb_text = gr.Textbox(
                label="Speaker Embedding",
                max_lines=3,
                show_copy_button=True,
                interactive=True,
                scale=2,
            )
            dvae_coef_text = gr.Textbox(
                label="DVAE Coefficient",
                max_lines=3,
                show_copy_button=True,
                interactive=True,
                scale=2,
            )
            reload_chat_button = gr.Button("Reload", scale=1, interactive=True)

        with gr.Row():
            auto_play_checkbox = gr.Checkbox(
                label="Auto Play", value=False, scale=1, interactive=True
            )
            stream_mode_checkbox = gr.Checkbox(
                label="Stream Mode",
                value=False,
                scale=1,
                interactive=True,
            )
            generate_button = gr.Button(
                "Generate", scale=2, variant="primary", interactive=True
            )
            interrupt_button = gr.Button(
                "Interrupt",
                scale=2,
                variant="stop",
                visible=False,
                interactive=False,
            )

        text_output = gr.Textbox(
            label="Output Text",
            interactive=False,
            show_copy_button=True,
        )

        sample_audio_input.change(
            fn=on_upload_sample_audio,
            inputs=sample_audio_input,
            outputs=sample_audio_code_input,
        ).then(fn=lambda: gr.Info("Sampled Audio Code generated at another Tab."))

        # 使用Gradio的回调功能来更新数值输入框
        voice_selection.change(
            fn=on_voice_change, inputs=voice_selection, outputs=audio_seed_input
        )

        generate_audio_seed.click(generate_seed, outputs=audio_seed_input)

        generate_text_seed.click(generate_seed, outputs=text_seed_input)

        audio_seed_input.change(
            on_audio_seed_change, inputs=audio_seed_input, outputs=spk_emb_text
        )

        reload_chat_button.click(
            reload_chat, inputs=dvae_coef_text, outputs=dvae_coef_text
        )

        interrupt_button.click(interrupt_generate)

        @gr.render(inputs=[auto_play_checkbox, stream_mode_checkbox])
        def make_audio(autoplay, stream):
            audio_output = gr.Audio(
                label="Output Audio",
                value=None,
                format="mp3" if use_mp3 and not stream else "wav",
                autoplay=autoplay,
                streaming=stream,
                interactive=False,
                show_label=True,
                waveform_options=gr.WaveformOptions(
                    sample_rate=24000,
                ),
            )
            generate_button.click(
                fn=set_buttons_before_generate,
                inputs=[generate_button, interrupt_button],
                outputs=[generate_button, interrupt_button],
            ).then(
                refine_text,
                inputs=[
                    text_input,
                    text_seed_input,
                    refine_text_checkbox,
                ],
                outputs=text_output,
            ).then(
                generate_audio,
                inputs=[
                    text_output,
                    temperature_slider,
                    top_p_slider,
                    top_k_slider,
                    spk_emb_text,
                    stream_mode_checkbox,
                    audio_seed_input,
                    sample_text_input,
                    sample_audio_code_input,
                ],
                outputs=audio_output,
            ).then(
                fn=set_buttons_after_generate,
                inputs=[generate_button, interrupt_button, audio_output],
                outputs=[generate_button, interrupt_button],
            )

        gr.Examples(
            examples=ex,
            inputs=[
                text_input,
                temperature_slider,
                top_p_slider,
                top_k_slider,
                audio_seed_input,
                text_seed_input,
                refine_text_checkbox,
            ],
        )

    parser = argparse.ArgumentParser(description="ChatTTS demo Launch")
    parser.add_argument(
        "--server_name", type=str, default="0.0.0.0", help="server name"
    )
    parser.add_argument("--server_port", type=int, default=8080, help="server port")
    parser.add_argument("--root_path", type=str, default=None, help="root path")
    parser.add_argument(
        "--custom_path", type=str, default=None, help="custom model path"
    )
    parser.add_argument(
        "--coef", type=str, default=None, help="custom dvae coefficient"
    )
    args = parser.parse_args()

    logger.info("loading ChatTTS model...")

    if load_chat(args.custom_path, args.coef):
        logger.info("Models loaded successfully.")
    else:
        logger.error("Models load failed.")
        sys.exit(1)

    spk_emb_text.value = on_audio_seed_change(audio_seed_input.value)
    dvae_coef_text.value = chat.coef

    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        root_path=args.root_path,
        inbrowser=True,
        show_api=False,
    )


if __name__ == "__main__":
    main()
