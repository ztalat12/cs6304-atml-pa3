"""Task 2 pre-flight on the real models (one process): prompt layout, frozen parameter sets, connector gradients,
calibration, LoRA size. Prints one JSON line that the notebook turns into PASS / FAIL lines."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.lora import inject_lora  # noqa: E402
from common.utils import gpu_peak, seed_everything  # noqa: E402
from task2.t2_features import FeatureStore  # noqa: E402
from task2.t2_model import LM_ID, SeqBuilder  # noqa: E402
from task2.t2_train import SHARED, Data, build_model, make_batch  # noqa: E402


def main():
    sel_path, feat_dir = sys.argv[1], sys.argv[2]
    from transformers import AutoTokenizer
    seed_everything()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sel = json.load(open(sel_path))
    tok = AutoTokenizer.from_pretrained(LM_ID)
    b = SeqBuilder(tok)
    ids, lab, vis = b.build("qa", 196, question="Is the man wearing a hat?", target="yes")
    ids0, _, _ = b.build("qa", 0, question="Is the man wearing a hat?", target="yes")
    feats = FeatureStore(feat_dir)
    data = Data(sel, b, 196, SHARED["grounding_sampling"])
    vlm, cal = build_model(dev, sel, data, feats, log=lambda *a: None)
    r = {"rendered_template": b.rendered_example, "placeholder_token": tok.convert_ids_to_tokens(b.placeholder),
         "n_visual_positions": sum(vis), "first_visual_index": vis.index(1), "n_prefix_tokens": len(b.pre_ids),
         "prefix_text": tok.decode(b.pre_ids), "supervised_text": tok.decode([t for t, l in zip(ids, lab) if l != -100]),
         "decoded_text_part": tok.decode([t for t, v in zip(ids, vis) if not v]),
         "no_image_equals_text_part": ids0 == [t for t, v in zip(ids, vis) if not v],
         "qwen_all_frozen": all(not p.requires_grad for p in vlm.lm.parameters()),
         "connector_trainable": all(p.requires_grad for p in vlm.connector.parameters()),
         "connector_params": sum(p.numel() for p in vlm.connector.parameters()), "calibration": cal}
    vlm.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    vlm.train()
    bt, f = make_batch(data, feats, data.captions[:8], dev)
    with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
        nll = vlm.example_nll(bt, f)
    nll.mean().backward()
    r["connector_grad_norms"] = [float(p.grad.norm()) for p in vlm.connector.parameters()]
    r["qwen_grads_none"] = all(p.grad is None for p in vlm.lm.parameters())
    r["initial_caption_nll"] = float(nll.mean())
    info = inject_lora(vlm.lm, SHARED["lora_targets"], SHARED["lora_r"], SHARED["lora_alpha"], SHARED["lora_dropout"], seed=0,
                       log=lambda *a: None)
    r["lora_params"] = info["n_adapter_params"]
    r["lora_modules"] = info["n_modules"]
    r.update(gpu_peak())
    print(json.dumps(r))


if __name__ == "__main__":
    main()
