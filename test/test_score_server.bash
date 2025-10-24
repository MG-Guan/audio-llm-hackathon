#!/bin/bash

model="whisper-small"

uid="$1"
cid="$2"
file_path="$3"

if [ -z "$uid" ] || [ -z "$cid" ] || [ -z "$file_path" ]; then
  echo "Usage: $0 <uid> <cid> <file_path>"
  exit 1
fi

printf "Sending request to score server with uid=%s, cid=%s, file_path=%s\n" "$uid" "$cid" "$file_path"

curl -X POST http://127.0.0.1:9000/score \
  -F "uid=${uid}" \
  -F "cid=${cid}" \
  -F "model=${model}" \
  -F "audio=@${file_path}"