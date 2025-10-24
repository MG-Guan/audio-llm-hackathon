from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent

import torch
import sys
#import torchaudio
import time
import click

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"

system_prompt = (
    "Generate audio following instruction.\n\n<|scene_desc_start|>\nAudio is recorded from a quiet room.\n<|scene_desc_end|>"
)

fin = open(sys.argv[1], "r")
user_content = fin.read()
fin.close()

messages = [
    Message(
        role="system",
        content=system_prompt,
    ),
    Message(
        role="user",
        content=user_content.strip(),
    ),
]
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

torch.cuda.empty_cache()

serve_engine = HiggsAudioServeEngine(MODEL_PATH, AUDIO_TOKENIZER_PATH,
                                     torch_dtype=torch.float16,
                                     kv_cache_lengths=[256, 512], device=device)

output: HiggsAudioResponse = serve_engine.generate(
    chat_ml_sample=ChatMLSample(messages=messages),
    max_new_tokens=1024,
    temperature=0.3,
    top_p=0.95,
    top_k=50,
    stop_strings=["<|end_of_text|>", "<|eot_id|>"],
)
#torchaudio.save(f"output.wav", torch.from_numpy(output.audio)[None, :], output.sampling_rate)

import numpy as np
import soundfile as sf

# output.audio 是 numpy 数组（[-1, 1] 浮点）；若是 torch.Tensor 先 .cpu().numpy()
audio = output.audio.astype(np.float32)
sf.write("output.wav", audio, output.sampling_rate)  # 直接写 WAV (float32)