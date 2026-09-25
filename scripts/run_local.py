#!/usr/bin/env python3
"""Run a pre-built layaForWeb ONNX model locally with Python + ONNX Runtime.

This is the Python equivalent of the browser demo: load a quantized ONNX model
from disk or Hugging Face, build token sequences with question markers and
question-type tokens, and return choice / score / yes-no probabilities with
confidence.

The model files can be built with ``scripts/build_model.py`` or downloaded
pre-built from ``VishalMysore/layaForWeb`` on Hugging Face.

Requirements
------------
    pip install tokenizers onnxruntime numpy huggingface_hub

Example
-------
    python scripts/run_local.py \
        --hf-model VishalMysore/layaForWeb \
        --state "App crashes on launch. I have a demo in one hour!" \
        --questions \
            team:choice:Which team should handle this? bug:"Something broken",how_to:"Usage question" \
            urgency:score:How urgent is this? "Can wait","This week","Today","Right now" \
            angry:noul:The customer sounds angry

    # Or point at a local model directory:
    python scripts/run_local.py \
        --model-dir /path/to/layaForWeb/build/model \
        --state "..." \
        --questions ...
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = ["choice", "score", "noul"]


def parse_questions(raw):
    """Parse ``--questions`` specs.

    Each spec is ``name:type:rest``. For ``noul``, ``rest`` is the instruction.
    For ``choice`` / ``score``, ``rest`` is ``instruction,label1,label2,...``
    where labels may be ``label:desc`` pairs (for choice) or plain words.
    """
    questions = {}
    for token in raw:
        parts = token.split(":", 2)
        if len(parts) < 3:
            raise ValueError(f"bad question spec {token!r}: expected name:type:rest")
        name, qtype, rest = parts
        qtype = qtype.lower()
        if qtype not in ("choice", "score", "noul"):
            raise ValueError(f"unknown type {qtype!r}")

        # Strip surrounding quotes if the whole rest is quoted
        if rest.startswith('"') and rest.endswith('"') and rest.count('"') == 2:
            rest = rest[1:-1]

        if qtype == "noul":
            questions[name] = {"type": "noul", "instructions": rest, "criteria": None}
            continue

        # Split instructions from criteria: the instruction is everything before
        # the first label:description pair (word followed by colon).
        # After the instruction, criteria are label:desc1,label2:desc2,...
        instr = rest
        criteria_raw = None
        depth = 0
        split_at = -1
        for i, ch in enumerate(rest):
            if ch == '"':
                depth ^= 1
            elif ch == ":" and depth == 0:
                # This could be a label:desc in the criteria. Only split if
                # there's no colon before any comma (meaning the criteria
                # hasn't started yet), or the colon is before the first comma.
                rest_before = rest[:i]
                # Find the last comma before this colon
                last_comma = rest_before.rfind(",")
                first_colon_in_after = rest.find(":", i + 1)
                first_comma = rest.find(",")
                if first_comma < 0 or i < first_comma:
                    # Colon appears before any comma in the entire string —
                    # this is the first label:desc pair, so split before it.
                    split_at = i
                    # Restore the colon in the instruction part
                    instr = rest[:i + 1].strip()
                    criteria_raw = rest[i + 1:].strip()
                    break
        if split_at < 0:
            instr = rest.strip()

        if qtype == "choice":
            criteria = {}
            if criteria_raw:
                for p in criteria_raw.split(","):
                    p = p.strip()
                    if not p:
                        continue
                    if ":" in p:
                        k, v = p.split(":", 1)
                        criteria[k.strip()] = v.strip().strip('"')
                    else:
                        criteria[p] = ""
            questions[name] = {"type": "choice", "instructions": instr, "criteria": criteria}
        else:  # score
            criteria = [p.strip().strip('"') for p in criteria_raw.split(",") if p.strip()] if criteria_raw else []
            questions[name] = {"type": "score", "instructions": instr, "criteria": criteria}
    return questions


def download_from_hf(repo, variant, out_dir):
    """Download model files from Hugging Face into *out_dir*."""
    from huggingface_hub import hf_hub_download
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tj = hf_hub_download(repo, "tokenizer.json", local_dir=out_dir)
    tc = hf_hub_download(repo, "tokenizer_config.json", local_dir=out_dir)
    rc = hf_hub_download(repo, "rl_agent_config.json", local_dir=out_dir)
    manifest = json.loads(hf_hub_download(repo, "manifest.json", local_dir=out_dir))
    v = manifest["variants"][variant]
    onnx = hf_hub_download(repo, v["onnx"], local_dir=out_dir)

    data_parts = []
    data_dir = out_dir / v["data"]["name"].replace(".data", "")
    data_dir.mkdir(exist_ok=True)
    for part in v["data"]["parts"]:
        data_parts.append(hf_hub_download(repo, part, local_dir=data_dir))

    # Assemble data file from parts
    assembled_data = out_dir / v["data"]["name"]
    if not assembled_data.exists():
        with open(assembled_data, "wb") as f:
            for p in sorted(data_parts):
                f.write(open(p, "rb").read())

    return {
        "tokenizer_json": tj,
        "tokenizer_config": tc,
        "rl_config": rc,
        "onnx": onnx,
        "data": assembled_data,
        "manifest": manifest,
    }


def download_from_dir(model_dir, variant):
    """Read model files from a local built folder.
    Works with both full manifest-based builds and minimal flat layouts
    (onnx + .onnx.data + tokenizer.json + tokenizer_config.json).
    """
    model_dir = Path(model_dir)
    manifest_path = model_dir / "manifest.json"
    rl_config_path = model_dir / "rl_agent_config.json"
    tok_json_path = model_dir / "tokenizer.json"
    tok_cfg_path = model_dir / "tokenizer_config.json"

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        v = manifest["variants"][variant]
        return {
            "tokenizer_json": str(tok_json_path) if tok_json_path.exists() else str(model_dir / "tokenizer.json"),
            "tokenizer_config": str(tok_cfg_path) if tok_cfg_path.exists() else str(model_dir / "tokenizer_config.json"),
            "rl_config": str(rl_config_path) if rl_config_path.exists() else None,
            "onnx": str(model_dir / v["onnx"]),
            "data": str(model_dir / v["data"]["name"]),
            "manifest": manifest,
        }
    # Minimal flat layout: assume variant name matches onnx file base
    onnx_name = f"laya_{variant}.onnx"
    onnx_path = model_dir / onnx_name
    if not onnx_path.exists():
        # search for any .onnx file
        onnx_files = list(model_dir.glob("*.onnx"))
        if onnx_files:
            onnx_path = onnx_files[0]
            onnx_name = onnx_path.name
        else:
            raise FileNotFoundError(f"No .onnx file in {model_dir}")
    data_path = model_dir / onnx_name.replace(".onnx", ".onnx.data")
    return {
        "tokenizer_json": str(tok_json_path) if tok_json_path.exists() else str(model_dir / "tokenizer.json"),
        "tokenizer_config": str(tok_cfg_path) if tok_cfg_path.exists() else str(model_dir / "tokenizer_config.json"),
        "rl_config": str(rl_config_path) if rl_config_path.exists() else None,
        "onnx": str(onnx_path),
        "data": str(data_path),
        "manifest": None,
    }


def assemble_external_data(onnx_path, data_path):
    """Assemble split external data files into a single .onnx.data file."""
    data_path = Path(data_path)
    if data_path.exists() and data_path.stat().st_size > 0:
        return  # Already assembled

    # Find parts in the same directory
    parts_dir = data_path.parent
    pattern = data_path.name.replace(".data", "").replace(".onnx", "")
    pattern += "part*"

    import glob
    part_files = sorted(glob.glob(str(parts_dir / pattern)))

    if not part_files:
        return

    with open(data_path, "wb") as f:
        for p in part_files:
            f.write(open(p, "rb").read())
    print(f"assembled {data_path} ({data_path.stat().st_size} bytes)", file=sys.stderr)


def build_sequence(tok, state, questions, max_len=512):
    """Build token sequence with question markers and qtype tokens.

    Ported from laya/core.py (build_sequence) and laya/agent.py (system_one).
    """
    from tokenizers import Tokenizer

    MASK = "[MASK]"

    # Serialize state
    if isinstance(state, str):
        state_text = state
    elif isinstance(state, dict):
        state_text = json.dumps(state, ensure_ascii=False)
    elif isinstance(state, list):
        state_text = json.dumps(state, ensure_ascii=False)
    else:
        state_text = str(state)

    items = []
    for qid, q in questions.items():
        qtype = q["type"].lower()
        instruction = q["instructions"]

        if qtype == "noul":
            crit = q.get("criteria") or {"false": "", "true": ""}
            opt_ids = [
                tok.encode(f"A: {crit.get('false', 'No')}", add_special_tokens=False),
                tok.encode(f"B: {crit.get('true', 'Yes')}", add_special_tokens=False),
            ]
            question_text = f"Answer the following question about the text: {instruction}"
        elif qtype == "choice":
            opt_texts = []
            for label, desc in q["criteria"].items():
                if desc:
                    opt_texts.append(f"{label}: {desc}")
                else:
                    opt_texts.append(label)
            opt_ids = [tok.encode(t, add_special_tokens=False) for t in opt_texts]
            question_text = f"{instruction}: {', '.join(opt_texts)}"
        else:  # score
            if not q.get("criteria"):
                opt_texts = [str(i) for i in range(5)]
            else:
                opt_texts = list(q["criteria"])
            opt_ids = [tok.encode(t, add_special_tokens=False) for t in opt_texts]
            question_text = f"{instruction}: {' > '.join(opt_texts)}"

        # Add special tokens: state_text + MASK + [SEP] + question
        full_text = state_text + " " + MASK + " " + question_text + " [SEP]"
        enc = tok.encode(full_text, add_special_tokens=False)
        ids = list(enc.ids)
        markers = []

        for opt in opt_ids:
            opt_ids_list = opt.ids if hasattr(opt, 'ids') else list(opt)
            for _ in opt_ids_list:
                markers.append(len(ids))
            ids.extend(opt_ids_list)

        items.append({"ids": ids[:max_len], "markers": markers, "qtype": QTYPES[qtype]})

    # Pad items to same length and prepare tensors
    n = len(items)
    if n == 0:
        return None

    kmax = max((len(it["markers"]) for it in items), default=0)
    L = max((len(it["ids"]) for it in items), default=0)

    all_ids = np.zeros((n, L), dtype=np.int64)
    att_mask = np.zeros((n, L), dtype=np.int64)
    mpos = np.zeros((n, kmax), dtype=np.int64)
    mmask = np.zeros((n, kmax), dtype=bool)
    qtype_arr = np.zeros((n,), dtype=np.int64)

    for i, it in enumerate(items):
        ids = it["ids"]
        markers = it["markers"]
        all_ids[i, :len(ids)] = ids
        att_mask[i, :len(ids)] = 1
        for j, m in enumerate(markers):
            if j < kmax:
                mpos[i, j] = m
                mmask[i, j] = True
        qtype_arr[i] = it["qtype"]

    return {
        "ids": all_ids,
        "att": att_mask,
        "mpos": mpos,
        "mmask": mmask,
        "qtype": qtype_arr,
        "n": n,
        "kmax": kmax,
        "L": L,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("model source")
    src.add_argument("--model-dir", help="path to a built model folder")
    src.add_argument("--hf-model", help="Hugging Face repo with pre-built model files")
    src.add_argument("--variant", default="q4e8", help="model variant (default: q4e8)")
    src.add_argument("--cache-dir", default="/tmp/laya_local_cache")

    inp = ap.add_argument_group("input")
    inp.add_argument("--state", default="App crashes on launch. I have a demo in one hour!")
    inp.add_argument("--questions", nargs="+",
                     default=[
                         "team:choice:Which team should handle this? bug:Something is broken,how_to:A usage question,sales:Pricing or plans",
                         "urgency:score:How urgent is this? Can wait,This week,Today,Right now",
                         "angry:noul:The customer sounds angry",
                     ])

    inf = ap.add_argument_group("inference")
    inf.add_argument("--providers", default="CPUExecutionProvider")
    inf.add_argument("--threads", type=int, default=1)

    a = ap.parse_args()

    # ---- resolve model ----------------------------------------------------------
    if a.hf_model:
        files = download_from_hf(a.hf_model, a.variant, a.cache_dir)
    elif a.model_dir:
        files = download_from_dir(a.model_dir, a.variant)
    else:
        ap.error("one of --model-dir or --hf-model is required")

    # ---- ONNX session -----------------------------------------------------------
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = a.threads
    session = ort.InferenceSession(
        files["onnx"],
        sess_options=sess_opts,
        providers=[p.strip() for p in a.providers.split(",")],
    )

    # ---- assemble external data if needed ----------------------------------------
    assemble_external_data(files["onnx"], files.get("data", ""))

    # ---- load tokenizer ---------------------------------------------------------
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(files["tokenizer_json"])

    # ---- parse questions --------------------------------------------------------
    try:
        questions = parse_questions(a.questions)
    except ValueError as e:
        print(f"bad --questions: {e}", file=sys.stderr)
        sys.exit(1)

    # ---- state -----------------------------------------------------------------
    state = a.state
    try:
        state = json.loads(state)
    except json.JSONDecodeError:
        pass

    # ---- build sequence --------------------------------------------------------
    batch = build_sequence(tok, state, questions)
    if batch is None:
        print("no questions to answer", file=sys.stderr)
        sys.exit(0)

    # ---- run inference ----------------------------------------------------------
    result = session.run(
        None,
        {
            "input_ids": batch["ids"],
            "attention_mask": batch["att"],
            "marker_pos": batch["mpos"],
            "marker_mask": batch["mmask"],
            "qtype": batch["qtype"],
        },
    )

    logits = result[0]
    act_logits = result[1]

    # ---- decode results --------------------------------------------------------
    answers = {}
    for i, (qid, q) in enumerate(questions.items()):
        qtype = q["type"]
        qlabels = list(q["criteria"].keys()) if qtype in ("choice", "score") and q.get("criteria") else None
        if qtype == "noul":
            # Binary yes/no
            prob = float(logits[i, 1])  # P("yes") = index 1
            answers[qid] = {
                "type": "noul",
                "instruction": q["instructions"],
                "p_yes": round(prob, 4),
                "confidence": round(1 - abs(2 * prob - 1), 4),
            }
        elif qtype == "choice":
            if not qlabels:
                qlabels = list(range(logits.shape[1]))
            probs = {str(qlabels[j]): round(float(logits[i, j]), 4) for j in range(logits.shape[1])}
            top = max(probs, key=probs.get)
            answers[qid] = {
                "type": "choice",
                "instruction": q["instructions"],
                "options": probs,
                "top": top,
                "confidence": round(1 - abs(2 * probs[top] - 1), 4),
            }
        else:  # score
            if not qlabels:
                qlabels = list(range(logits.shape[1]))
            probs = {str(qlabels[j]): round(float(logits[i, j]), 4) for j in range(logits.shape[1])}
            score = sum(j * probs[k] for k, j in zip(qlabels, range(logits.shape[1])))
            answers[qid] = {
                "type": "score",
                "instruction": q["instructions"],
                "probs": probs,
                "score": round(score, 4),
                "confidence": round(1 - abs(2 * probs[qlabels[int(score)]] - 1), 4),
            }

    # Add act_logits for escalation routing
    answers["_raw"] = {
        "logits": [[round(float(v), 4) for v in row] for row in act_logits],
    }

    print(json.dumps(answers, indent=2))


if __name__ == "__main__":
    main()
