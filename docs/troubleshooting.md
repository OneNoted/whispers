# Troubleshooting

## Start with setup

If local dictation is not behaving correctly, rerun the guided setup first:

```sh
whispers setup
```

That flow validates the current config, offers model downloads, and can fix `/dev/uinput` access for paste injection.

## `wl-copy` is missing

`whispers` uses `wl-copy` for clipboard injection. Install `wl-clipboard` (or otherwise make `wl-copy` available on `PATH`) and try again.

## `/dev/uinput` is missing or not writable

Paste injection depends on `/dev/uinput`.

Recommended path:

```sh
whispers setup
```

If you handle it manually instead, the fallback guidance is:

- Load the kernel module: `sudo modprobe uinput`
- Persist it across reboots if needed: create `/etc/modules-load.d/whispers-uinput.conf` with `uinput`
- Create a dedicated `uinput` group and a `udev` rule for `/dev/uinput`
- Log out and back in after group membership changes

## Cloud checks

If cloud ASR or rewrite is configured but not working, validate the current provider, credentials, and reachability:

```sh
whispers cloud check
```

## Inspect rewrite resource paths

These helpers are useful when you want to confirm which runtime files the current config points at:

```sh
whispers app-rule path
whispers glossary path
whispers rewrite-instructions-path
```

## Local `whisper_cpp` hang diagnostics

If the main `whispers` process ever gets stuck after playback when using local `whisper_cpp`, enable the built-in hang diagnostics for the next repro:

```sh
WHISPERS_HANG_DEBUG=1 whispers
```

When that mode is enabled, `whispers` writes runtime status and hang bundles under `${XDG_RUNTIME_DIR:-/tmp}/whispers/`:

- `main-status.json` shows the current dictation stage and recent stage metadata.
- `hang-<pid>-<stage>-<timestamp>.log` is emitted if `whisper_cpp` spends too long in model load or transcription.

Those bundles include the current status snapshot plus best-effort stack and open-file diagnostics. If the hang reproduces, capture the newest `hang-*.log` file along with `main-status.json`.
