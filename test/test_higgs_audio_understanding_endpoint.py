import openai
import base64
import sys
import os

BOSON_API_KEY = os.getenv("BOSON_API_KEY")

def encode_audio_to_base64(file_path: str) -> str:
    """Encode audio file to base64 format."""
    with open(file_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")

client = openai.Client(
    api_key=BOSON_API_KEY,
    base_url="https://hackathon.boson.ai/v1"
)

'''
# Transcribe audio
audio_path = sys.argv[1]  # e.g., "/path/to/your/audio.wav"
audio_base64 = encode_audio_to_base64(audio_path)
file_format = audio_path.split(".")[-1]
response = client.chat.completions.create(
    model="higgs-audio-understanding-Hackathon",
    messages=[
        {"role": "system", "content": "Transcribe this audio for me."},
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_base64,
                        "format": file_format,
                    },
                },
            ],
        },
    ],
    max_completion_tokens=256,
    temperature=0.0,
)
'''

# Chat about the audio
audio1_path = sys.argv[1]
audio2_path = sys.argv[2]
file_format = audio1_path.split(".")[-1]

audio1_base64 = encode_audio_to_base64(audio1_path)
audio2_base64 = encode_audio_to_base64(audio2_path)

response = client.chat.completions.create(
    model="higgs-audio-understanding-Hackathon",
    messages=[
        {"role": "system", "content": "You are a helpful assistant. I will provide you with an audlllio file, it was concatenated from two recordings. Answer my questions about the audio."},
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio1_base64 + audio2_base64,
                        "format": file_format,
                    },
                },
            ],
        },
        {
            "role": "user",
            "content": "Return the audio simularities of the two recordings in terms of se in json format: {sim_score: float from 0.0 to 1.0}, where 1.0 means identical audios."
        },
    ],
    max_completion_tokens=256,
    temperature=1.0,
)

print(response.choices[0].message.content)