# Audio Score Server Deployment Guide

This service loads all configured runner types (Whisper style embeddings and the Higgs remote scorer) inside a single process. Each host can expose the API through a Unix domain socket and place Nginx in front of it, or rely on the bundled `run_server_with_gate.py` launcher which manages both the workers and Nginx.

## 1. Environment preparation
- Create or reuse a Python environment with the project dependencies installed.
- Ensure `ffmpeg` is available so `pydub` can decode `.mov` files.
- Ensure the reference audio cache (`config.json`) is populated on the host.
- Optional: export `AUDIO_SCORE_DISABLE_CACHE_FLUSH=1` if multiple instances will rebuild the cache concurrently.
- If the machine hosts multiple GPUs, set `CUDA_VISIBLE_DEVICES` per worker or through the service unit to avoid device contention.

## 2. Launching the worker
Run one systemd service per worker. The process binds to a Unix socket (removing any stale file first), enables dynamic batching (defaults to max batch size 8, 10 ms wait window), and handles every configured model.

Example systemd unit (save as `/etc/systemd/system/audio-score.service`):

```ini
[Unit]
Description=Audio Score Server (%i)
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/ec2-user/higgs-audio/hackathon/audio_score_server
Environment="PYTHONPATH=/home/ec2-user/higgs-audio"
Environment="AUDIO_SCORE_SERVER_SOCKET=/run/audio-score/audio-score.sock"
Environment="AUDIO_SCORE_DISABLE_CACHE_FLUSH=1"
ExecStart=/usr/bin/python3 main.py
User=ec2-user
Group=ec2-user
RuntimeDirectory=audio-score
RuntimeDirectoryMode=0755
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

`run_server.py` respects `AUDIO_SCORE_MAX_BATCH_SIZE`, `AUDIO_SCORE_MAX_BATCH_WAIT_MS`, `AUDIO_SCORE_MODEL_KEYS`, and the socket/host/port overrides. Tune those via additional `Environment=` lines if needed.

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now audio-score.service
```

## 3. Gateway / Nginx frontend
For hosts running more than one worker, the repository ships a helper that supervises both the workers and an Nginx instance:

```bash
cd /home/ec2-user/higgs-audio/hackathon/audio_score_server
python run_server_with_gate.py --workers 2 --listen-port 9000
```

The script:
- spawns the requested number of `run_server.py` workers bound to Unix sockets under `run/`,
- generates a temporary Nginx configuration with round-robin (`least_conn`) upstreaming,
- writes logs to `logs/` and request bodies to `logs/client_body_temp/` (avoids `/var/lib/nginx` permission issues),
- shuts everything down cleanly on `Ctrl+C`.

You can wrap this command with a process manager (systemd, supervisord, tmux) for unattended operation. Use `--model-keys`, `--worker-log-level`, `--client-max-body`, etc., to customise behaviour.

If you prefer to manage Nginx separately, expose the API over HTTP while talking to the Unix socket internally:

```nginx
upstream audio_score {
    server unix:/run/audio-score/audio-score.sock;
}

server {
    listen 80;
    server_name _;

    location /score/ {
        proxy_pass http://audio_score/score/;
        proxy_set_header Host $host;
    }
}
```

Reload Nginx after adding the configuration:

```bash
sudo nginx -s reload
```

## 4. Optional tuning
- Adjust `AUDIO_SCORE_MAX_BATCH_WAIT_MS` to change the latency/throughput trade-off (default 10 ms).
- Set `AUDIO_SCORE_MAX_BATCH_SIZE` if you need a value other than the default 8.
- Use `AUDIO_SCORE_MODEL_KEYS`/`AUDIO_SCORE_MODEL_KEY` if you need to temporarily restrict which models load in a specific instance.
- Set `AUDIO_SCORE_SERVER_HOST`/`AUDIO_SCORE_SERVER_PORT` to expose over TCP instead of Unix sockets.
- Spread workers across GPUs when scaling horizontally; multiple workers sharing a single GPU will still serialize model inference.
