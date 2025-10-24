# About the Hackathon

This project is for the Higgs Audio Hackathon hold by Boson AI 2025.

## API Design

### Function: score audio record

**Request sample:**

```
curl -X POST "https://api.example.com/v1/score" \
  -H "Accept: application/json" \
  -F "audio=@user_recorded.wav" \
  -F "uid=PeppaPig-s06e02" \
  -F "cid=part02
```


**Request:**


**Response:**

```
{
  "sim_score": 0.8,
  "audio_sim_score": 0.7,
  "text_sim_score": 0.9,
}
```

### Server Side Design

**File Strucutre**

**


