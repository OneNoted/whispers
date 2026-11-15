# Troubleshooting

Start with the built-in report:

```sh
whispers doctor
```

`FAIL` means core dictation cannot work; `WARN` describes an optional fallback
or desktop integration.

## Setup was interrupted

Rerun the same command. Downloads resume into the data directory, checksums are
verified before install, and completed targets are skipped.

## Audio does not start

Confirm PipeWire sees a default source and that raw capture works:

```sh
wpctl status
pw-cat --record --raw --format f32 --rate 16000 --channels 1 >/dev/null
```

Set `[audio].device` only when the default PipeWire source is not the microphone
you want.

## Whisper.cpp shows no partial text

Expected when `[recognition].runtime = "whisper-cli"`. The external CLI runs
after the recording stops, so the OSD only shows listening and transcribing.
Switch to `sherpa-onnx` to get the live preview.

## Whisper.cpp is slow or the GPU stays idle

The backend package, not whispers, selects the device:

```sh
pacman -Q whisper-cpp ggml-vulkan
whisper-cli -m /path/to/ggml-large-v3-turbo.bin -f sample.wav -nt
```

The log must print `using Vulkan0 backend`. With only `ggml` installed the CLI
falls back to CPU and a large model takes seconds per sentence. `whispers
doctor` reports the configured CLI and model path.

## Text was copied instead of typed

Whispers refuses to type into a different window than the one where dictation
began. When focus changes, the result is placed on the clipboard and a
notification explains why. Return to the intended window and paste manually.

If focus did not change, verify `wtype` and `wl-copy` are installed. Discord
clients use clipboard paste because their editors can drop rapid virtual-keyboard
input. Terminal fallback uses Ctrl+Shift+V; other applications use Ctrl+V.

For another application that drops characters during direct typing, set a small
delay without changing Discord's clipboard-first behavior:

```toml
[delivery]
typing_delay_ms = 5
```

## Enhancement falls back

Deterministic symbol handling and vocabulary still run if local enhancement is
unavailable. Check:

```sh
whispers doctor
systemctl --user status whispers.service
```

The enhancer rejects output that drops protected symbols or numbers, expands the
transcript excessively, or returns invalid text. That fallback is intentional.

## Existing version 1 configuration

Run `whispers setup`. Version 1 is not interpreted as version 2. Setup backs up
the original, imports compatible audio/language/model choices and vocabulary,
then writes the smaller schema. Legacy files and models are left in place for
manual cleanup after you are satisfied with the rewrite.
