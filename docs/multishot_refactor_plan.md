# Refactor Plan: Introducing MultiShot (Holocine Shot Attention) on `reimplement/upstream-2026-01-21`

> Goal: On `reimplement/upstream-2026-01-21`, reintroduce the MultiShot features from the main branch in the **style used by upstream/Kijai for new features**; follow the **minimal interference** principle so existing upstream nodes keep working; new nodes must be **structurally compatible** with the old implementation (current main).

## 0. Baseline Info (must read)
- Current working branch: `reimplement/upstream-2026-01-21`
- Merge base: `393fe78ec2d5a515ce4ff794067ea4d72c830643`
- Diff range: `393fe78..main`
- MultiShot files (from main-side diff):
  - `__init__.py`
  - `custom_linear.py`
  - `nodes.py`
  - `nodes_model_loading.py`
  - `nodes_sampler.py`
  - `wanvideo/modules/attention.py`
  - `wanvideo/modules/model.py`
  - `wanvideo/modules/shot_utils.py` (new)
  - `wanvideo/modules/tokenizers.py`
  - `wanvideo/modules/t5.py`

> Reference diff (per file):
> - `git diff 393fe78..main -- <file>`

---

## 1. Design Principles (match upstream/Kijai style)
- **Minimal interference**: Prefer new nodes/functions; avoid modifying existing upstream logic unless required.
- **Explicit enablement**: Shot attention/shot mask/shot embedding must be explicitly enabled and must not affect default behavior.
- **Backward-compatible structure**: Node class names, I/O types, return order, CATEGORY/display names must match main.
- **Explicit validation and errors**: Provide clear errors for missing params or incompatible configs.

### 1.1 Additional requirements to match Kijai style (relative to current main)
**Goal:** Without changing node structure/behavior, align implementation with upstream/Kijai style: optional injection, low intrusion, clear failure.

- **No impact on default path**
  - All MultiShot features are enabled only through explicit nodes (`WanVideoHolocineSetShotAttention`).
  - Do not add implicit switches or auto-inference in the default sampling path.

- **Centralized config injection**
  - All shot-attention config is stored under `transformer_options["shot_attention"]` and parsed in one place.
  - Avoid writing the same params in multiple sampler/model locations to prevent hidden state.

- **Clear failure/fallback paths**
  - Missing `[global caption]`, `holocine_args`, or `text_cut_positions` must error with a clear reason.
  - If sparse backend is unavailable, fall back explicitly with a log (no silent fallback).

- **Clear module boundaries**
  - Core logic for structured prompt, shot indices, and cross-attn mask lives in `shot_utils.py`.
  - Sampler/model only wire inputs; do not spread logic.

- **Consistent logging and validation**
  - Use log.info/warning/error; avoid print.
  - Strictly validate LoRA types, token ratio, and mask channels with readable messages.

---

## 2. MultiShot Feature Breakdown (overview)

### 2.1 Node Layer (Node API)
**New nodes (as defined in main):**
1) `WanVideoHolocineShotArgs`
2) `WanVideoHolocineShotBuilder`
3) `WanVideoHolocinePromptEncode`
4) `WanVideoHolocineSetShotAttention`

**Compatibility checklist (must match main):**
- [ ] Class names and `NODE_CLASS_MAPPINGS` keys match main
- [ ] `CATEGORY` and `NODE_DISPLAY_NAME_MAPPINGS` match main
- [ ] Input/output type names match (e.g., `HOLOCINE_SHOTARGS`, `WANVID_HOLOCINE_SHOT_LIST`)
- [ ] Parameter defaults, tooltips, and option ranges match
- [ ] Output order matches main

**Node I/O quick reference (based on main):**
- `WanVideoHolocineShotArgs`
  - Inputs: `image_embeds(WANVIDIMAGE_EMBEDS)`, `shot_cut_frames(STRING)`
  - Outputs: `holocine_args(HOLOCINE_SHOTARGS)`
  - Behavior: Infer total frames + parse/normalize shot cut frames (4t+1)

- `WanVideoHolocineShotBuilder`
  - Inputs: `shot_caption(STRING)` + optional `shot_list(WANVID_HOLOCINE_SHOT_LIST)`, `shot_lora(WANVIDLORA)`, `smooth_window(INT)`
  - Outputs: `shot_list(WANVID_HOLOCINE_SHOT_LIST)`
  - Behavior: Chain-build a shot list; each shot can attach LoRA and smooth_window

- `WanVideoHolocinePromptEncode`
  - Inputs: `global_caption(STRING)`, `shot_list(WANVID_HOLOCINE_SHOT_LIST)`, `negative_prompt(STRING)`, `t5(WANTEXTENCODER)`, `image_embeds(WANVIDIMAGE_EMBEDS)`
  - Optional: `custom_shot_cut_frames(STRING)`, `append_shot_summary(BOOLEAN)`, `force_offload(BOOLEAN)`, `model_to_offload(WANVIDEOMODEL)`, `use_disk_cache(BOOLEAN)`, `device(gpu/cpu)`
  - Outputs: `text_embeds(WANVIDEOTEXTEMBEDS)`, `holocine_args(HOLOCINE_SHOTARGS)`, `positive_prompt(STRING)`
  - Behavior: Build structured prompt + compute shot_cuts + output holocine_args (includes shot_loras & smooth_windows)

- `WanVideoHolocineSetShotAttention`
  - Inputs: `model(WANVIDEOMODEL)`, `enable(BOOLEAN)`, `global_token_ratio_or_number(FLOAT)`
  - Optional: `mask_type(none/id/normalized/alternating)`, `backend(full/sparse_fallback/sparse_flash_attn)`, `i2v_mode(BOOLEAN)`
  - Outputs: `model(WANVIDEOMODEL)`
  - Behavior: Write `transformer_options["shot_attention"]`

> Reference diff: `git diff 393fe78..main -- nodes.py`

---

### 2.2 Structured Prompt (MultiShot entry)
- Prompt tags: `[global caption]`, `[per shot caption]`, `[shot cut]`
- Text parsing: `parse_structured_prompt()` uses offset_mapping to derive token spans
- Output: `text_embeds["text_cut_positions"]`

> Reference diff:
> - `git diff 393fe78..main -- wanvideo/modules/shot_utils.py`
> - `git diff 393fe78..main -- nodes.py`
> - `git diff 393fe78..main -- wanvideo/modules/tokenizers.py`
> - `git diff 393fe78..main -- wanvideo/modules/t5.py`

---

### 2.3 Shot Attention (self-attn + cross-attn)
- Self-attn: `sparse_shot_attention()` + fallback
- Cross-attn: `build_cross_attention_mask()` builds per-shot mask
- Backend: `full | sparse_fallback | sparse_flash_attn`
- Smooth windows: adjacent shots share tokens
- I2V mode: first frame as global prefix

> Reference diff:
> - `git diff 393fe78..main -- wanvideo/modules/attention.py`
> - `git diff 393fe78..main -- wanvideo/modules/model.py`
> - `git diff 393fe78..main -- nodes_sampler.py`
> - `git diff 393fe78..main -- wanvideo/modules/shot_utils.py`

---

### 2.4 Per-shot LoRA (attach LoRA per shot)
- `CustomLinear` adds per-shot LoRA injection points
- Sampler splits LoRA into shot payloads
- Only **linear LoRA** is supported; non-linear/DoRA/reshape are skipped with a warning

> Reference diff:
> - `git diff 393fe78..main -- custom_linear.py`
> - `git diff 393fe78..main -- nodes_sampler.py`
> - `git diff 393fe78..main -- wanvideo/modules/model.py`

---

### 2.5 Shot Embedding + Shot Mask Channel (model input augmentation)
- Detect `shot_embedding.weight` during load
- Shot mask types: `none | id | normalized | alternating`
- If checkpoint requires extra channels but mask_type=none, raise an error

> Reference diff:
> - `git diff 393fe78..main -- nodes_model_loading.py`
> - `git diff 393fe78..main -- wanvideo/modules/model.py`

---

## 3. Refactor Checklist (step-by-step validation)

### A. Add shot_utils foundation
- [x] Add `wanvideo/modules/shot_utils.py` (keep main content)
- [x] Add `sys.modules.setdefault` compatibility mapping in `__init__.py`
- [x] `parse_structured_prompt()` handles fast/slow tokenizer offsets

**Diff reference:**
- `git diff 393fe78..main -- wanvideo/modules/shot_utils.py`
- `git diff 393fe78..main -- __init__.py`

---

### B. Nodes: Holocine Shot / Prompt / SetShotAttention
- [x] Add `WanVideoHolocineShotArgs` with matching I/O
- [x] `WanVideoHolocineShotBuilder` supports chained shot_list + LoRA + smooth_window
- [x] `WanVideoHolocinePromptEncode`: structured prompt + holocine_args output
- [x] `WanVideoHolocineSetShotAttention`: write shot_attention config
- [x] NODE mappings, display name, CATEGORY fully aligned
- [x] Prompt tags match main (`[global caption]`, `[per shot caption]`, `[shot cut]`)
- [x] Write `text_cut_positions` into `text_embeds` dict

**Diff reference:**
- `git diff 393fe78..main -- nodes.py`

---

### C. Sampler injection (shot_indices + text_cut_positions)
- [x] `nodes_sampler.py` receives and validates `holocine_args`
- [x] `build_shot_indices()` builds shot_indices
- [x] Inject `shot_indices/shot_attention_cfg/shot_mask_type/text_cut_positions/smooth_windows` into model
- [x] Disable shot attention during CFG negative pass

**Diff reference:**
- `git diff 393fe78..main -- nodes_sampler.py`

---

### D. Model layer: shot attention + cross-attn mask
- [x] `wanvideo/modules/attention.py` adds sparse_shot_attention + fallback
- [x] `wanvideo/modules/model.py` parses `shot_attention_cfg`
- [x] Cross-attn mask wires into `build_cross_attention_mask()`
- [x] Shot embedding + shot mask concatenation is complete
- [x] `nodes_model_loading.py` detects `shot_embedding.weight` and writes model config

**Diff reference:**
- `git diff 393fe78..main -- wanvideo/modules/attention.py`
- `git diff 393fe78..main -- wanvideo/modules/model.py`

---

### E. Per-shot LoRA support
- [x] `custom_linear.py` adds shot_lora cache and injection
- [x] `nodes_sampler.py` aggregates LoRA payload per shot and injects into transformer
- [x] `wanvideo/modules/model.py` sets `CustomLinear.runtime_context`

**Diff reference:**
- `git diff 393fe78..main -- custom_linear.py`
- `git diff 393fe78..main -- nodes_sampler.py`
- `git diff 393fe78..main -- wanvideo/modules/model.py`

---

### F. Tokenizer compatibility and logs
- [x] `tokenizers.py` fallback fast -> slow
- [x] `t5.py` log style aligned with upstream

**Diff reference:**
- `git diff 393fe78..main -- wanvideo/modules/tokenizers.py`
- `git diff 393fe78..main -- wanvideo/modules/t5.py`

---

### Z. Docs and process
- [x] Create and maintain MultiShot refactor plan (checklist + implementation log)

## 4. Key risk points (validate carefully)
- **Do not break default path**: shot attention must be explicitly enabled; default None should not affect upstream behavior.
- **Missing prompt tags**: missing `[global caption]` must raise a clear error to avoid silent failure.
- **Mask channel**: if model requires extra channels, mask_type=none must error (avoid shape mismatch).
- **LoRA compatibility**: non-linear LoRA must be skipped with a warning to avoid silent wrong output.

---

## 5. Suggested validation steps (run after each step)
1. Add only `shot_utils.py` + `__init__.py` mapping, run a simple import test:
   - `python -c "import wanvideo.modules.shot_utils"`
2. Add and register nodes; start ComfyUI and verify node presence and I/O.
3. Use a main example prompt (with `[global caption]` tags) to run once:
   - Text encode only (no shot_attention)
4. Enable `WanVideoHolocineSetShotAttention` and run a short video, verify:
   - shot_indices built, cross-attn mask constructed without errors
5. Add per-shot LoRA and check logs for correct inject/skip messages.

---

## 6. Notes
- This document only covers **MultiShot (Holocine)** features; other main-branch additions are out of scope.
- Extend with additional breakdowns if needed.

---

## 7. Implementation log
- 2026-01-21 Commit 1: Add `docs/multishot_refactor_plan.md`, remove old doc `docs/reimplement_main_since_upstream.md` (baseline established). **Checklist update:** `Z. Docs and process` done.
- 2026-01-21 Commit 2: Add `wanvideo/modules/shot_utils.py` and set `wanvideo.modules.shot_utils` mapping in `__init__.py`. **Checklist update:** `A. shot_utils foundation` done.
- 2026-01-21 Commit 3: Add fast->slow fallback in `tokenizers.py`, unify logs in `t5.py`. **Checklist update:** `F. Tokenizer compatibility and logs` done.
- 2026-01-21 Commit 4: Add Holocine nodes and structured prompt flow; `WanVideoTextEncode` outputs `text_cut_positions`. **Checklist update:** `B. Nodes` done.
- 2026-01-21 Commit 5: Add per-shot LoRA injection and runtime_context support in `custom_linear.py`. **Checklist update:** `E. custom_linear.py` done.
- 2026-01-21 Commit 6: Add sparse shot attention + fallback in `wanvideo/modules/attention.py`. **Checklist update:** `D. attention.py` done.
- 2026-01-21 Commit 7: Wire shot attention, cross-attn mask, shot embedding/mask, and runtime_context in `wanvideo/modules/model.py`. **Checklist update:** `D. model.py` and `E. runtime_context` done.
- 2026-01-21 Commit 8: Detect `shot_embedding.weight` in `nodes_model_loading.py` and pass config into model. **Checklist update:** `D. nodes_model_loading.py` done.
- 2026-01-21 Commit 9: Wire holocine_args, shot attention params, shot_indices/smooth_windows, and per-shot LoRA in `nodes_sampler.py`; disable shot attention during CFG negative pass. **Checklist update:** `C. Sampler injection` and `E. nodes_sampler.py` done.
