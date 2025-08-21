# Canto Transcript Backend

A websocket server that receives audio data and transcribes it using SenseVoice small from Alibaba. (pseudo realtime)

A fork of [streaming-sensevoice](https://github.com/pengzhendong/streaming-sensevoice)

Python version: 3.12.1 (Confirmed working)

## Setup

- python3.12 -m venv venv

- source .venv/bin/activate

- pip install -r requirements.txt

## Run

- python main.py

  (It should begin to download the model and then start up the server `Downloading Model from https://www.modelscope.cn`)

  You'll receive an error `Loading remote code failed: model, No module named 'model'` but it should continue to run.

  `Uvicorn running on http://127.0.0.1:8000`

- Open up `http://127.0.0.1:8000` in your browser to test the websocket server
