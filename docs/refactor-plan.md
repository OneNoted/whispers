# Whispers Refactor Plan

Status: active
Workspace: `refactor-plan` at `/home/notes/Projects/whispers-refactor-plan`
Planning goal: reduce module sprawl and dependency tangles without mixing in feature work or behavior changes.

## Working Rules

- Keep refactor work in this workspace, not the shared feature workspace.
- Prefer behavior-preserving extractions first. Delay semantic changes until the new boundaries are in place.
- Keep each checkpoint to one logical concern and one Conventional Commit description.
- Run targeted tests after each checkpoint, then broaden to `cargo test` when the phase is stable.
- Do not start with OSD polish or naming cleanup. Fix structure first.

## Current Diagnosis

The main mess is not the top-level flow. The main mess is that a few large modules own too many responsibilities at once:

- `src/main.rs` is the de facto crate root for almost everything.
- `src/bin/whispers-rewrite-worker.rs` and `src/bin/whispers-osd.rs` pull shared code in via `#[path = ...]` instead of a shared library crate.
- `src/postprocess.rs` mixes planning, backend routing, fallback, and finalization.
- `src/agentic_rewrite.rs` mixes runtime policy logic with file-backed CLI admin.
- `src/asr.rs` duplicates backend lifecycle logic across batch and live paths.
- `src/app.rs` mixes orchestration, runtime state, injection policy, and session persistence.
- `src/personalization.rs`, `src/session.rs`, `src/config.rs`, and `src/setup.rs` each bundle multiple separate concerns.

## Recommended Order

1. Establish crate boundaries.
2. Fix dependency direction in the runtime path.
3. Split the largest domain modules by responsibility.
4. Split config/setup/model/completion orchestration.
5. Finish with platform adapters and status reporting cleanup.

## Phase 1: Crate Boundaries

Goal: stop sharing code between binaries through `#[path = ...]` includes and give the project a real library surface.

### Checkpoint 1.1

- Commit: `refactor: add library crate and thin binary entrypoints`
- Deliverables:
  - Add `src/lib.rs`.
  - Move module declarations out of `src/main.rs`.
  - Make `src/main.rs` a thin CLI entrypoint.
  - Make `src/bin/whispers-rewrite-worker.rs` and `src/bin/whispers-osd.rs` use library modules instead of `#[path = ...]`.
- Validation:
  - `cargo test`
  - `cargo test --bin whispers`
  - `cargo test --bin whispers-rewrite-worker`

### Checkpoint 1.2

- Commit: `refactor: isolate binary-only startup code`
- Deliverables:
  - Move PID lock and process signaling helpers into a small runtime support module.
  - Keep binary-specific CLI/bootstrap logic out of domain modules.
- Validation:
  - `cargo test main::tests`
  - `cargo test`

## Phase 2: Runtime Path

Goal: make the dictation path read as orchestration over smaller components instead of one large cross-module knot.

### Checkpoint 2.1

- Commit: `refactor: extract agentic rewrite runtime policy engine`
- Deliverables:
  - Split `src/agentic_rewrite.rs` into runtime policy code and file-backed admin/store code.
  - Runtime modules should not print to stdout or mutate files.
  - CLI-facing add/list/remove/path helpers should depend on the store layer, not the runtime layer.
- Validation:
  - `cargo test agentic_rewrite`
  - `cargo test postprocess`

### Checkpoint 2.2

- Commit: `refactor: split postprocess planning and execution`
- Deliverables:
  - Extract a planning layer from `src/postprocess.rs` for transcript preparation and session intent.
  - Extract an execution layer for local/cloud rewrite calls.
  - Keep final acceptance and fallback rules in a smaller decision layer.
- Validation:
  - `cargo test postprocess`
  - `cargo test session`
  - `cargo test personalization`

### Checkpoint 2.3

- Commit: `refactor: unify asr backend lifecycle`
- Deliverables:
  - Remove duplicated backend switching across `prepare_transcriber`, `prepare_live_transcriber`, `transcribe_audio`, and `transcribe_live_audio`.
  - Centralize fallback policy in one place.
- Validation:
  - `cargo test asr`
  - `cargo test faster_whisper`
  - `cargo test nemo_asr`

### Checkpoint 2.4

- Commit: `refactor: split rewrite routing from prompt rendering`
- Status:
  - completed sub-checkpoints: routing split, prompt rendering split, local rewrite engine extraction, output cleanup plus thin facade
  - phase status: complete
- Deliverables:
  - Separate route selection from prompt/template rendering in `src/rewrite.rs`.
  - Keep giant prompt contracts out of routing logic.
- Validation:
  - `cargo test rewrite`
  - `cargo test rewrite_profile`

### Checkpoint 2.5

- Commit: `refactor: split app controller from dictation runtime state`
- Status:
  - completed sub-checkpoints: extracted runtime state transitions, isolated OSD helpers, kept `run()` as controller orchestration
  - phase status: complete
- Deliverables:
  - Keep `src/app.rs` as orchestration.
  - Extract dictation runtime state, preview pacing, session updates, and injection decisions into smaller modules.
  - Minimize direct side effects inside the main dictation loop.
- Validation:
  - `cargo test app`
  - `cargo test session`
  - targeted manual smoke test for `whispers voice`

## Phase 3: Domain Modules

Goal: split large pure-ish logic files by domain instead of by size.

### Checkpoint 3.1

- Commit: `refactor: split personalization store and rewrite candidates`
- Status:
  - completed sub-checkpoints: extracted file-backed dictionary/snippet store, moved rewrite transcript and candidate generation out of the facade, kept `crate::personalization::*` call sites stable via re-exports
  - phase status: complete
- Deliverables:
  - Split `src/personalization.rs` into:
    - store and CLI mutation helpers
    - text transformation rules
    - rewrite candidate building and ranking
- Validation:
  - `cargo test personalization`

### Checkpoint 3.2

- Commit: `refactor: split session persistence from backtrack planning`
- Deliverables:
  - Move JSON load/save/prune logic away from backtrack heuristics.
  - Make backtrack planning operate on in-memory data structures.
- Validation:
  - `cargo test session`
  - `cargo test postprocess`

### Checkpoint 3.3

- Commit: `refactor: split cleanup lexicon analysis and rendering`
- Deliverables:
  - Split `src/cleanup.rs` into lexical rules, analysis, and rendering pieces.
  - Keep the public cleanup API stable until follow-up cleanup is done.
- Validation:
  - `cargo test cleanup`

## Phase 4: Config and Command Surface

Goal: remove duplicated sources of truth and reduce direct file mutation from high-level commands.

### Checkpoint 4.1

- Commit: `refactor: split config schema defaults and editing`
- Deliverables:
  - Split `src/config.rs` into schema, defaults/template, load/migrate, and edit/update modules.
  - Put TOML mutation behind a small config editor API.
- Validation:
  - `cargo test config`
  - `cargo test cli`

### Checkpoint 4.2

- Commit: `refactor: extract setup flow phases`
- Deliverables:
  - Break `src/setup.rs` into prompt/selection, config apply, side effects, and summary/reporting phases.
  - Keep interactive behavior unchanged.
- Validation:
  - `cargo test setup`

### Checkpoint 4.3

- Commit: `refactor: unify model management workflows`
- Deliverables:
  - Reduce duplication across `src/model.rs`, `src/asr_model.rs`, and `src/rewrite_model.rs`.
  - Share download/select/status plumbing where behavior is actually the same.
- Validation:
  - `cargo test model`
  - `cargo test asr_model`
  - `cargo test rewrite_model`

### Checkpoint 4.4

- Commit: `refactor: isolate shell completion installers`
- Deliverables:
  - Separate shell detection, script generation, install-path policy, and shell rc mutation in `src/completions.rs`.
- Validation:
  - `cargo test completions`

### Checkpoint 4.5

- Commit: `docs: derive config docs from canonical source`
- Deliverables:
  - Stop maintaining defaults separately in code, `config.example.toml`, and the README snippet.
  - Pick one canonical source and generate or reuse it everywhere else.
- Validation:
  - `cargo test config`
  - manual check of `README.md` and `config.example.toml`

## Phase 5: Platform Adapters and Reporting

Goal: separate policy from OS effects in smaller but high-value modules.

### Checkpoint 5.1

- Commit: `refactor: extract injection adapter layer`
- Deliverables:
  - Separate injection policy from evdev and clipboard execution in `src/inject.rs`.
- Validation:
  - `cargo test inject`

### Checkpoint 5.2

- Commit: `refactor: split audio recorder and dsp helpers`
- Deliverables:
  - Separate recorder lifecycle and device interaction from reusable audio transforms in `src/audio.rs`.
- Validation:
  - `cargo test audio`

### Checkpoint 5.3

- Commit: `refactor: build status snapshots before rendering`
- Deliverables:
  - Make `src/status.rs` build a pure status snapshot first, then render it.
  - Reduce direct backend probing inside rendering code.
- Validation:
  - `cargo test status`

## Not Now

- Rewriting the user-facing CLI.
- Replacing `tokio` structure or async strategy.
- Changing OSD visuals.
- Large naming-only passes.
- Folding unrelated feature work into refactor commits.

## Per-Checkpoint Template

Use this each time work starts on a new item:

1. Confirm the checkpoint and write the exact Conventional Commit description with `jj desc -m`.
2. Restate the non-goals for that checkpoint.
3. Move code without changing behavior.
4. Run targeted tests for touched modules.
5. If the checkpoint is complete, create the next working-copy change with `jj new`.
6. Update this file with status notes before moving on.

## Progress Log

- [x] Phase 1.1 complete
- [x] Phase 1.2 complete
- [x] Phase 2.1 complete
- [x] Phase 2.2 complete
- [x] Phase 2.3 complete
- [x] Phase 2.4 complete
- [x] Phase 2.5 complete
- [x] Phase 3.1 complete
- [ ] Phase 3.2 complete
- [ ] Phase 3.3 complete
- [ ] Phase 4.1 complete
- [ ] Phase 4.2 complete
- [ ] Phase 4.3 complete
- [ ] Phase 4.4 complete
- [ ] Phase 4.5 complete
- [ ] Phase 5.1 complete
- [ ] Phase 5.2 complete
- [ ] Phase 5.3 complete
