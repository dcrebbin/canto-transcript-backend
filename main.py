"""
Auther: ISJDOG

## Cli

```bash
python realtime_ws_server_demo.py --help
```

## Debug with vscode:

```
{
  // Use IntelliSense to learn about possible attributes.
  // Hover to view descriptions of existing attributes.
  // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python Debugger: Current File",
      "type": "debugpy",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "env": {
        "SENSEVOICE_MODEL_PATH": "iic/SenseVoiceSmall",
        "DEVICE": "cuda",
      }
    }
  ]
}
```
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from urllib.parse import parse_qs

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from pysilero import VADIterator

from loguru import logger

import numpy as np

import sys, uuid

import soundfile as sf
import io


class Config(BaseSettings, cli_parse_args=True, cli_use_class_docs_for_groups=True):
    HOST: str = Field("127.0.0.1", description="Host")
    PORT: int = Field(8000, description="Port")
    DEBUG: bool = Field(False, description="Debug mode")
    SENSEVOICE_MODEL_PATH: str = Field(
        "iic/SenseVoiceSmall", description="SenseVoice model path"
    )
    DEVICE: str = Field("cpu", description="Device")
    SILEROVAD_VERSION: str = Field("v5", description="SileroVAD version, v4 or v5")
    SAMPLERATE: int = Field(16000, description="Sample rate")
    CHUNK_DURATION: float = Field(0.1, description="Chunk duration (s)")
    VAD_MIN_SILENCE_DURATION_MS: int = Field(
        150, description="VAD min slience duration (ms)"
    )
    VAD_THRESHOLD: float = Field(0.5, description="VAD threshold")


config = Config()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from streaming_sensevoice import StreamingSenseVoice

# load model on startup
StreamingSenseVoice.load_model(model=config.SENSEVOICE_MODEL_PATH, device=config.DEVICE)


class TranscriptionChunk(BaseModel):
    timestamps: list[int]
    raw_text: str
    final_text: str | None = None
    spk_id: int | None = None


class TranscriptionResponse(BaseModel):
    type: str = "TranscriptionResponse"
    id: int
    begin_at: float
    end_at: float | None
    data: TranscriptionChunk
    is_final: bool
    session_id: str | None = None


class VADEvent(BaseModel):
    type: str = "VADEvent"
    is_active: bool


class ErrorMessage(BaseModel):
    type: str = "Error"
    message: str
    code: int | None = None
    session_id: str | None = None


@app.get("/")
async def clientHost():
    return FileResponse("realtime_ws_client.html", media_type="text/html")


@app.websocket("/api/realtime/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()

        session_id = str(uuid.uuid4())
        logger.info(f"Session {session_id} opened")

        query_params = parse_qs(websocket.scope["query_string"].decode())
        try:
            chunk_duration = float(
                (query_params.get("chunk_duration", [config.CHUNK_DURATION]))[0]
            )
            vad_threshold = float(
                (query_params.get("vad_threshold", [config.VAD_THRESHOLD]))[0]
            )
            vad_min_silence_duration_ms = int(
                (
                    query_params.get(
                        "vad_min_silence_duration_ms",
                        [config.VAD_MIN_SILENCE_DURATION_MS],
                    )
                )[0]
            )
        except Exception as e:
            logger.exception("Invalid query parameters")
            await websocket.send_json(
                ErrorMessage(
                    message=f"Invalid query parameters: {e}", code=400, session_id=session_id
                ).model_dump()
            )
            await websocket.close(code=1002, reason="Invalid query parameters")
            return

        try:
            sensevoice_model = StreamingSenseVoice(
                model=config.SENSEVOICE_MODEL_PATH, device=config.DEVICE
            )
            vad_iterator = VADIterator(
                version=config.SILEROVAD_VERSION,
                threshold=vad_threshold,
                min_silence_duration_ms=vad_min_silence_duration_ms,
            )
        except Exception as e:
            logger.exception("Failed to initialize model or VAD")
            await websocket.send_json(
                ErrorMessage(
                    message=f"Initialization error: {e}", code=500, session_id=session_id
                ).model_dump()
            )
            await websocket.close(code=1011, reason="Initialization error")
            return

        audio_buffer = np.array([], dtype=np.float32)
        chunk_size = int(chunk_duration * config.SAMPLERATE)

        speech_count = 0
        currentAudioBeginTime = 0.0

        asrDetected = False

        transcription_response: TranscriptionResponse = None
        while True:
            data = await websocket.receive_bytes()

            # mp3 decode
            buffer = io.BytesIO(data)
            try:
                buffer.name = "a.mp3"

                samples, sr = sf.read(buffer, dtype="float32")
                audio_buffer = np.concatenate((audio_buffer, samples))
            except sf.LibsndfileError as e:
                # Fallback: interpret payload as raw PCM16LE mono at configured sample rate
                try:
                    raw_bytes = data
                    if len(raw_bytes) % 2 != 0:
                        logger.warning(
                            "Odd-length PCM16 payload; truncating last byte"
                        )
                        raw_bytes = raw_bytes[:-1]
                    pcm_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
                    if pcm_int16.size == 0:
                        logger.warning("Empty PCM16 payload; skipping chunk")
                        continue
                    samples = (pcm_int16.astype(np.float32)) / 32768.0
                    sr = config.SAMPLERATE
                    audio_buffer = np.concatenate((audio_buffer, samples))
                except Exception as pe:
                    logger.warning(
                        f"Failed to interpret as PCM16: {pe}; original decode error: {e}"
                    )
                    continue
            finally:
                buffer.close()

            if sr != config.SAMPLERATE:
                msg = f"Sample rate mismatch: expected {config.SAMPLERATE}, got {sr}"
                logger.error(msg)
                await websocket.send_json(
                    ErrorMessage(message=msg, code=4001, session_id=session_id).model_dump()
                )
                await websocket.close(code=1003, reason="Sample rate mismatch")
                break

            while len(audio_buffer) >= chunk_size:
                chunk = audio_buffer[:chunk_size]
                audio_buffer = audio_buffer[chunk_size:]

                for speech_dict, speech_samples in vad_iterator(chunk):
                    if "start" in speech_dict:
                        sensevoice_model.reset()

                        currentAudioBeginTime: float = (
                            speech_dict["start"] / config.SAMPLERATE
                        )

                        if asrDetected:
                            logger.debug(
                                f"{speech_count}: VAD *NOT* end: \n{transcription_response.data.raw_text}\n{str(transcription_response.data.timestamps)}"
                            )
                            speech_count += 1
                        asrDetected = False

                        logger.debug(
                            f"{speech_count}: VAD start: {currentAudioBeginTime}"
                        )
                        await websocket.send_json(VADEvent(is_active=True).model_dump())

                    is_last = "end" in speech_dict

                    for res in sensevoice_model.streaming_inference(
                        speech_samples, is_last
                    ):

                        if len(res["text"]) > 0:
                            asrDetected = True

                        if asrDetected:
                            transcription_response = TranscriptionResponse(
                                id=speech_count,
                                begin_at=currentAudioBeginTime,
                                end_at=None,
                                data=TranscriptionChunk(
                                    timestamps=res["timestamps"], raw_text=res["text"]
                                ),
                                is_final=False,
                                session_id=session_id,
                            )
                            await websocket.send_json(
                                transcription_response.model_dump()
                            )

                    if is_last:
                        if asrDetected:
                            speech_count += 1
                            asrDetected = False

                            transcription_response.is_final = True
                            transcription_response.end_at = (
                                speech_dict["end"] / config.SAMPLERATE
                            )

                            await websocket.send_json(
                                transcription_response.model_dump()
                            )
                            logger.debug(
                                f"{speech_count}: VAD end: {speech_dict['end'] / config.SAMPLERATE}\n{transcription_response.data.raw_text}\n{str(transcription_response.data.timestamps)}"
                            )
                        else:
                            logger.debug(
                                f"{speech_count}: VAD end: {speech_dict['end'] / config.SAMPLERATE}\nNo Speech"
                            )
                        await websocket.send_json(
                            VADEvent(is_active=False).model_dump()
                        )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        # Catch-all to ensure the client receives an error message
        logger.exception("Unexpected error in websocket handler")
        try:
            await websocket.send_json(
                ErrorMessage(
                    message=f"Unexpected server error: {e}", code=1011, session_id=locals().get("session_id")
                ).model_dump()
            )
        except Exception:
            pass
    finally:
        try:
            if "sensevoice_model" in locals() and sensevoice_model is not None:
                try:
                    sensevoice_model.reset()
                except Exception:
                    pass
                del sensevoice_model
        except Exception:
            pass
        try:
            if "vad_iterator" in locals():
                del vad_iterator
        except Exception:
            pass
        try:
            if "audio_buffer" in locals():
                del audio_buffer
        except Exception:
            pass
        try:
            logger.info(f"Session {locals().get('session_id', 'unknown')} closed")
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT)
