docker run -p 10096:10095 -it \
  --rm \
  --privileged=true \
  --name funasr \
  --volume /home/bd4sur/ai/funasr/models:/workspace/models \
  --workdir /workspace/FunASR/runtime \
  registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.10 \
    /bin/bash run_server_2pass.sh \
      --download-model-dir /workspace/models \
      --vad-dir damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
      --model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx \
      --online-model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx  \
      --punc-dir damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx \
      --itn-dir thuduj12/fst_itn_zh \
      --hotword /workspace/models/hotwords.txt \
      --certfile /workspace/models/bd4sur.crt \
      --keyfile /workspace/models/key_unencrypted.pem
