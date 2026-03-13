#!/usr/bin/env python3

import argparse
import base64
import json
import os
import socket
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download

BEAM_SIZE = 3
BEST_OF = 3
CONDITION_ON_PREVIOUS_TEXT_MIN_SECONDS = 6.0


def cmd_download(repo_id: str, model_dir: str) -> int:
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=model_dir,
        local_dir_use_symlinks=False,
    )
    return 0


def make_model(model_dir: str, device: str, compute_type: str) -> WhisperModel:
    return WhisperModel(model_dir, device=device, compute_type=compute_type)


def transcribe(model: WhisperModel, audio_f32_b64: str, sample_rate: int, language: str) -> dict:
    if sample_rate != 16000:
        raise ValueError(f"unsupported sample rate {sample_rate}; expected 16000")

    audio_bytes = base64.b64decode(audio_f32_b64.encode("ascii"))
    audio = np.frombuffer(audio_bytes, dtype=np.float32)
    duration_seconds = float(audio.size) / float(sample_rate)
    segments_iter, info = model.transcribe(
        audio,
        language=None if language == "auto" else language,
        task="transcribe",
        beam_size=BEAM_SIZE,
        best_of=BEST_OF,
        condition_on_previous_text=(
            duration_seconds >= CONDITION_ON_PREVIOUS_TEXT_MIN_SECONDS
        ),
        vad_filter=False,
        word_timestamps=False,
    )

    raw_parts = []
    segments = []
    for segment in segments_iter:
        text = (segment.text or "").strip()
        if not text:
            continue
        raw_parts.append(text)
        segments.append(
            {
                "text": text,
                "start_ms": int(segment.start * 1000),
                "end_ms": int(segment.end * 1000),
            }
        )

    transcript = {
        "raw_text": " ".join(raw_parts).strip(),
        "detected_language": getattr(info, "language", None),
        "segments": segments,
    }
    return {"type": "transcript", "transcript": transcript}


def handle_connection(conn: socket.socket, model: WhisperModel, language: str) -> None:
    request_line = b""
    while not request_line.endswith(b"\n"):
        chunk = conn.recv(65536)
        if not chunk:
            break
        request_line += chunk
    if not request_line:
        return

    request = json.loads(request_line.decode("utf-8"))
    if request.get("type") != "transcribe":
        response = {"type": "error", "message": "unsupported request"}
    else:
        try:
            response = transcribe(
                model,
                request["audio_f32_b64"],
                int(request["sample_rate"]),
                language,
            )
        except Exception as exc:  # noqa: BLE001
            response = {"type": "error", "message": str(exc)}

    conn.sendall(json.dumps(response).encode("utf-8") + b"\n")


def cmd_serve(
    socket_path: str,
    model_dir: str,
    language: str,
    device: str,
    compute_type: str,
    idle_timeout_ms: int,
) -> int:
    model = make_model(model_dir, device, compute_type)

    sock_path = Path(socket_path)
    sock_path.parent.mkdir(parents=True, exist_ok=True)
    if sock_path.exists():
        sock_path.unlink()

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(sock_path))
    server.listen(1)
    server.settimeout(None if idle_timeout_ms == 0 else idle_timeout_ms / 1000.0)

    try:
        while True:
            try:
                conn, _ = server.accept()
            except TimeoutError:
                break
            with conn:
                handle_connection(conn, model, language)
    finally:
        server.close()
        if sock_path.exists():
            sock_path.unlink()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download")
    download.add_argument("--repo-id", required=True)
    download.add_argument("--model-dir", required=True)

    serve = subparsers.add_parser("serve")
    serve.add_argument("--socket-path", required=True)
    serve.add_argument("--model-dir", required=True)
    serve.add_argument("--language", required=True)
    serve.add_argument("--device", required=True)
    serve.add_argument("--compute-type", required=True)
    serve.add_argument("--idle-timeout-ms", required=True, type=int)

    args = parser.parse_args()
    if args.command == "download":
        return cmd_download(args.repo_id, args.model_dir)
    return cmd_serve(
        args.socket_path,
        args.model_dir,
        args.language,
        args.device,
        args.compute_type,
        args.idle_timeout_ms,
    )


if __name__ == "__main__":
    raise SystemExit(main())
