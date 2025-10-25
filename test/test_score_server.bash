#!/bin/bash

model_type="$1"
uid="$2"
cid="$3"
file_path="$4"

if [ -z "$model_type" ] || [ -z "$uid" ] || [ -z "$cid" ] || [ -z "$file_path" ]; then
  echo "Usage: $0 <model_type> <uid> <cid> <file_path>"
  exit 1
fi

printf "Sending request to score server with moddel_type=%s uid=%s, cid=%s, file_path=%s\n" "$model_type" "$uid" "$cid" "$file_path"

curl -X POST http://127.0.0.1:9000/score \
  -F "uid=${uid}" \
  -F "cid=${cid}" \
  -F "model=${model_type}" \
  -F "audio=@${file_path}"