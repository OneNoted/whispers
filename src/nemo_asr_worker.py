#!/usr/bin/env python3

import argparse
import base64
import json
import os
import socket
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

MODEL_READY_METADATA = ".model-ready.json"


def model_metadata(model_ref: str, family: str, local_model_path: str | None = None) -> dict:
    return {
        "repo_id": model_ref,
        "family": family,
        "local_model_path": local_model_path,
    }


def write_ready_metadata(
    model_dir: str, model_ref: str, family: str, local_model_path: str | None = None
) -> None:
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir, MODEL_READY_METADATA).write_text(
        json.dumps(model_metadata(model_ref, family, local_model_path))
    )


def make_parakeet_model(model_ref: str, device: str):
    import nemo.collections.asr as nemo_asr
    import torch
    from omegaconf import open_dict

    model_path = Path(model_ref)
    if model_path.exists():
        map_location = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        model = nemo_asr.models.ASRModel.restore_from(str(model_path), map_location=map_location)
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_ref)
    model = model.eval()
    if device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
        # Parakeet's CUDA-graphs decoder path is unstable on this stack; keep GPU inference,
        # but force the standard greedy decoder for correctness.
        decoding_cfg = model.cfg.decoding
        with open_dict(decoding_cfg):
            decoding_cfg.greedy.use_cuda_graph_decoder = False
        model.change_decoding_strategy(decoding_cfg, decoder_type=model.cur_decoder, verbose=False)
    return model


def make_canary_qwen_model(model_ref: str, device: str):
    import torch
    from nemo.collections.speechlm2.models import SALM

    model = SALM.from_pretrained(model_ref)
    model = model.eval()
    if device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
    return model


def make_model(model_ref: str, family: str, device: str):
    if family == "parakeet":
        return make_parakeet_model(model_ref, device)
    if family == "canary_qwen":
        return make_canary_qwen_model(model_ref, device)
    raise ValueError(f"unsupported NeMo model family: {family}")


def cmd_download(model_ref: str, family: str, model_dir: str, cache_dir: str) -> int:
    os.environ.setdefault("HF_HOME", cache_dir)
    from huggingface_hub import snapshot_download

    snapshot_path = Path(snapshot_download(repo_id=model_ref, cache_dir=cache_dir))
    local_model_path = None
    if family == "parakeet":
        candidate = snapshot_path / "parakeet-tdt_ctc-1.1b.nemo"
        if candidate.exists():
            local_model_path = str(candidate)
    write_ready_metadata(model_dir, model_ref, family, local_model_path)
    return 0


def write_temp_wav(audio: np.ndarray, sample_rate: int) -> str:
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="whispers-nemo-")
    os.close(fd)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = np.round(pcm * np.iinfo(np.int16).max).astype(np.int16)
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return path


def parakeet_transcribe(model, audio: np.ndarray) -> str:
    output = model.transcribe(
        [audio.astype(np.float32, copy=False)],
        batch_size=1,
        num_workers=0,
        verbose=False,
    )
    if not output:
        return ""
    first = output[0]
    if hasattr(first, "text"):
        return (first.text or "").strip()
    return str(first).strip()


def canary_qwen_transcribe(model, wav_path: str) -> str:
    answer_ids = model.generate(
        prompts=[
            [
                {
                    "role": "user",
                    "content": f"Transcribe the following: {model.audio_locator_tag}",
                    "audio": [wav_path],
                }
            ]
        ],
        max_new_tokens=128,
    )
    text = model.tokenizer.ids_to_text(answer_ids[0].cpu())
    return (text or "").strip()


def transcribe(model, family: str, audio_f32_b64: str, sample_rate: int, language: str) -> dict:
    if sample_rate != 16000:
        raise ValueError(f"unsupported sample rate {sample_rate}; expected 16000")
    if language not in ("", "auto", "en"):
        raise ValueError(
            f"NeMo experimental backend currently supports English only; got language={language}"
        )

    audio_bytes = base64.b64decode(audio_f32_b64.encode("ascii"))
    audio = np.frombuffer(audio_bytes, dtype=np.float32)
    wav_path = None
    try:
        if family == "parakeet":
            text = parakeet_transcribe(model, audio)
        elif family == "canary_qwen":
            wav_path = write_temp_wav(audio, sample_rate)
            text = canary_qwen_transcribe(model, wav_path)
        else:
            raise ValueError(f"unsupported NeMo model family: {family}")
    finally:
        if wav_path is not None:
            try:
                os.remove(wav_path)
            except FileNotFoundError:
                pass

    transcript = {
        "raw_text": text,
        "detected_language": "en" if text else None,
        "segments": (
            [{"text": text, "start_ms": 0, "end_ms": int(len(audio) / sample_rate * 1000)}]
            if text
            else []
        ),
    }
    return {"type": "transcript", "transcript": transcript}


def handle_connection(conn: socket.socket, model, family: str, language: str) -> None:
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
                family,
                request["audio_f32_b64"],
                int(request["sample_rate"]),
                language,
            )
        except Exception as exc:  # noqa: BLE001
            response = {"type": "error", "message": str(exc)}

    conn.sendall(json.dumps(response).encode("utf-8") + b"\n")


def cmd_serve(
    socket_path: str,
    model_ref: str,
    family: str,
    language: str,
    device: str,
    idle_timeout_ms: int,
    cache_dir: str,
    startup_lock_path: str | None,
) -> int:
    os.environ.setdefault("HF_HOME", cache_dir)
    try:
        load_started = time.perf_counter()
        model = make_model(model_ref, family, device)
        print(
            f"[whispers:nemo] model loaded in {(time.perf_counter() - load_started):.2f}s",
            file=os.sys.stderr,
            flush=True,
        )
    except Exception:
        if startup_lock_path:
            try:
                Path(startup_lock_path).unlink(missing_ok=True)
            except TypeError:
                if Path(startup_lock_path).exists():
                    Path(startup_lock_path).unlink()
        raise

    sock_path = Path(socket_path)
    sock_path.parent.mkdir(parents=True, exist_ok=True)
    if sock_path.exists():
        sock_path.unlink()

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(sock_path))
    server.listen(1)
    server.settimeout(None if idle_timeout_ms == 0 else idle_timeout_ms / 1000.0)
    if startup_lock_path:
        try:
            Path(startup_lock_path).unlink(missing_ok=True)
        except TypeError:
            if Path(startup_lock_path).exists():
                Path(startup_lock_path).unlink()

    try:
        while True:
            try:
                conn, _ = server.accept()
            except TimeoutError:
                break
            with conn:
                handle_connection(conn, model, family, language)
    finally:
        server.close()
        if sock_path.exists():
            sock_path.unlink()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download")
    download.add_argument("--model-ref", required=True)
    download.add_argument("--family", required=True)
    download.add_argument("--model-dir", required=True)
    download.add_argument("--cache-dir", required=True)

    serve = subparsers.add_parser("serve")
    serve.add_argument("--socket-path", required=True)
    serve.add_argument("--model-ref", required=True)
    serve.add_argument("--family", required=True)
    serve.add_argument("--language", required=True)
    serve.add_argument("--device", required=True)
    serve.add_argument("--idle-timeout-ms", required=True, type=int)
    serve.add_argument("--cache-dir", required=True)
    serve.add_argument("--startup-lock-path")

    args = parser.parse_args()
    if args.command == "download":
        return cmd_download(args.model_ref, args.family, args.model_dir, args.cache_dir)
    return cmd_serve(
        args.socket_path,
        args.model_ref,
        args.family,
        args.language,
        args.device,
        args.idle_timeout_ms,
        args.cache_dir,
        args.startup_lock_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
