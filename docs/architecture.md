# Architecture

Whispers is one intended-text pipeline behind one toggle:

```text
PipeWire capture
  → streaming sherpa-onnx / Nemotron preview
  → full-buffer sherpa-onnx / Parakeet final ASR
  → vocabulary aliases
  → deterministic number, symbol, and line formatting
  → conservative local llama.cpp / Qwen enhancement
  → focus-safe Wayland delivery
```

## Boundaries

The `whispers` process owns control, state, safety policy, normalization, and
desktop delivery. It launches two independently managed inference runtimes:

- An internal framed subprocess hosts sherpa-onnx. Nemotron provides responsive
  previews while recording; Parakeet reprocesses the complete in-memory audio
  after release and is the source of delivered text. Control frames are JSON;
  audio frames are raw little-endian `f32` samples. The versioned protocol keeps
  native ABI details outside the service state machine.
- A loopback-only llama.cpp server hosts enhancement. Its managed Vulkan backend
  offloads to a compatible GPU and falls back to CPU without changing the Rust
  build. Requests have a bounded timeout and output validator. Failure always
  returns to deterministic text.

Both runtimes remain warm in the user service. The enhancer also primes its
prompt cache during service startup. Runtimes are downloaded during explicit
setup, verified with pinned SHA-256 digests, and loaded dynamically. They are
not Cargo dependencies.

## State and delivery

The public service states are `idle`, `listening`, `finalizing`, and
`delivering`. Cancel is valid while listening or finalizing. Once desktop
delivery begins, it is reported as in progress instead of falsely promising
that already-dispatched input was cancelled.

Whispers records the focused window at start. It types with `wtype` only when
that focus identity was captured and is still current; otherwise it copies the
result and tells the user. Clipboard paste is a fallback, with Ctrl+Shift+V
limited to terminal surfaces.

## Privacy and failure policy

Audio is held in memory for the active dictation and is not retained. Temporary
fallback WAV files are deleted immediately after `whisper-cli` exits.
Transcripts are not logged. Setup downloads public artifacts; the dictation
pipeline makes no network requests.

Enhancement output is rejected if it is empty, invalid, excessively expanded,
or removes protected symbols or numbers. Final ASR falls back first to the
streaming result, then to an existing local whisper.cpp model; enhancement falls
back to vocabulary plus deterministic normalization.

## Build budget

`scripts/check-build-budget.sh` enforces at most 10 direct and 40 resolved Rust
packages, rejects retired heavyweight crates, and fails if a `vendor/` source
tree returns. The current target is 7 direct and 28 resolved packages.
