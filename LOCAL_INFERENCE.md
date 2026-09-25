# Local inference with layaForWeb (Python)

**Status:** experimental — model files assembled, runner patched, verification pending.

## What we have locally

| File | Path | Size |
|------|------|------|
| ONNX model | `/tmp/laya_rebuild/laya_q4e8.onnx` | 3.6 MB |
| External data | `/tmp/laya_rebuild/laya_q4e8.onnx.data` | 291 MB (assembled from 12 parts) |
| Tokenizer | `/tmp/laya_rebuild/tokenizer.json` | 3.6 MB |
| Tokenizer config | `/tmp/laya_rebuild/tokenizer_config.json` | 308 B |
| Fixed tokenizer config | `/tmp/laya_rebuild/tokenizer_config_fixed.json` | 276 B |

Source: [VishalMysore/layaForWeb](https://huggingface.co/VishalMysore/layaForWeb)  
Demo: [vishalmysore.github.io/layaForWeb](https://vishalmysore.github.io/layaForWeb)  
Upstream repo: [github.com/vishalmysore/layaForWeb](https://github.com/vishalmysore/layaForWeb)

## Runner script

`scripts/run_local.py` is a standalone Python entry point that mirrors the browser demo's `systemOne` call:

```bash
python3 scripts/run_local.py \
  --model-dir /tmp/laya_rebuild \
  --variant q4e8 \
  --state "App crashes on launch. I have a demo in one hour!" \
  --questions \
    "team:choice:Which team should handle this? bug:Something is broken,how_to:A usage question,sales:Pricing or plans" \
    "urgency:score:How urgent is this? Can wait,This week,Today,Right now" \
    "angry:noul:The customer sounds angry"
```

It supports two model sources:

- `--model-dir` — a folder with the assembled model files (used for local testing)
- `--hf-model` — a Hugging Face repo name (downloads on first run into `--cache-dir`)

## Fixes applied to the runner

The runner was originally written for the upstream repo's manifest-based layout. We adapted it for the assembled flat layout:

1. **`download_from_dir` no longer requires `manifest.json`** — it falls back to a flat layout (`laya_<variant>.onnx` + `laya_<variant>.onnx.data` + tokenizer files). This is what our locally assembled model uses.
2. **`build_sequence` handles tokenizer `Encoding` objects** — the upstream tokenizer returns `Encoding` objects from `opt_ids`, not plain lists of ints. The marker/extension loop now calls `.ids` on each encoding.
3. **`tokenizer_config.json` is patched at load time** — `PreTrainedTokenizerFast` in this transformers version rejects non-dict values for `pad_token`, `unk_token`, etc. The runner reads `tokenizer_config_fixed.json` (manually cleaned) instead of the raw config, or falls back to a patched in-memory config.

## Known issues / next steps

1. **Post-processing mismatch** — the first full run hit an `IndexError` in the probability dict comprehension (`post_process_inf`). The ONNX model exports two outputs (`logits`, `act_logits`), and the runner's output shaping may not match the upstream checkpoint's expected layout. Needs a side-by-side against the upstream `laya` Python package's `systemOne` on the same input.
2. **No manifest-based verification yet** — `scripts/verify_model.py` expects a full `manifest.json`-driven build and a working `laya` Python package import chain, which currently fails on this environment's `transformers` version. Not a blocker for the runner itself, but means we haven't reproduced the upstream fidelity numbers locally.
3. **Tokenizer config provenance** — we have both `tokenizer_config.json` (raw, from HF) and `tokenizer_config_fixed.json` (hand-edited). The runner prefers the fixed one; the build scripts should produce the fixed config automatically instead of relying on a manual edit.

## Environment

- Python 3.11
- onnxruntime 1.28.0
- numpy 2.4.6
- transformers 4.49.0 (installed; import chain has known issues unrelated to the runner)
- tokenizers (bundled with transformers)

## References

- Upstream layaForWeb: https://github.com/vishalmysore/layaForWeb
- Hugging Face model: https://huggingface.co/VishalMysore/layaForWeb
- Live demo: https://vishalmysore.github.io/layaForWeb/
- Original Laya checkpoint: https://huggingface.co/convaiinnovations/laya
