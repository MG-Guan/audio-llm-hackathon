# About the Hackathon

This project is for the Higgs Audio Hackathon hold by Boson AI 2025.

## API Design

### Function: compare two audio records

**Request sample:**

```
curl -X POST "https://api.example.com//v1/audio-compare" \
  -H "Accept: application/json" \
  -F "audio_a=@original.wav" \
  -F "audio_b=@user_recorded.wav" \
  -F "asr_to_interfer=audio_b"
```


**Request:**

```
POST /v1/audio-compare
{
------WebKitFormBoundary12345
Content-Disposition: form-data; name="audio_a"; filename="original.wav"
Content-Type: audio/wav

(binary audio data here)
------WebKitFormBoundary12345
Content-Disposition: form-data; name="audio_b"; filename="user_recorded.wav"
Content-Type: audio/wav

(binary audio data here)
------WebKitFormBoundary12345
Content-Disposition: form-data; name="asr_to_interfer"

audio_b
------WebKitFormBoundary12345--
}
```

**Response:**

```
{
	"similarity_score": 0.87
	"details": {
		"asr": "The quick brown fox jumps over the lazy dog."
	}
}
```

