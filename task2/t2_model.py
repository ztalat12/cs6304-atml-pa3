"""Task 2 model: frozen SigLIP features -> trainable MLP connector -> frozen Qwen3-0.6B (+ LoRA in Stages 2-3).

Sequence layout (Qwen's own chat template, visual tokens first inside the user message):
    <|im_start|>user\n [Z_1 ... Z_196] \n {instruction}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n {target}<|im_end|>
Z_i are continuous vectors (connector outputs) placed where placeholder tokens sit; the placeholders' own embeddings
are never used. Loss: caption / answer tokens + <|im_end|>; visual tokens, prompt and padding are masked.
For "no image" inputs the visual positions are removed entirely (shorter prefix; mask and positions rebuilt).
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.utils import load_causal_lm

LM_ID = os.environ.get("PA3_T2_LM", "Qwen/Qwen3-0.6B")    # env override: offline tests only
IMG_MARK = "IMGMARK"
CAPTION_INSTRUCTION = "Write a short caption for this image."            # D-T2-1
ANSWER_SUFFIX = "\nAnswer the question using a single word or phrase."   # D-T2-1
CAPS = {"question": 80, "caption": 64, "answer": 16}                     # content-token caps (manual)


class Connector(nn.Module):
    """Linear 768 -> 1024, GELU, Linear 1024 -> 1024, then a FIXED output scale s (calibrated once, D-T2-5)."""

    def __init__(self, d_in=768, d_out=1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_out), nn.GELU(), nn.Linear(d_out, d_out))
        self.register_buffer("scale", torch.ones(()))

    def forward(self, f):
        return self.net(f) * self.scale


class SeqBuilder:
    """Builds token ids, labels and the visual-position mask for one example. Pieces are tokenised separately and
    concatenated, so the content caps are exact and every example uses the same boundaries."""

    def __init__(self, tok):
        self.tok = tok
        self.im_end = tok.convert_tokens_to_ids("<|im_end|>")
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        ph = tok.convert_tokens_to_ids("<|image_pad|>")
        self.placeholder = ph if isinstance(ph, int) and ph != tok.unk_token_id else self.pad
        rendered = tok.apply_chat_template([{"role": "user", "content": IMG_MARK + "\nQQQ"}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
        before, after = rendered.split(IMG_MARK)
        mid, tail = after.split("QQQ")
        enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]  # noqa: E731
        self.pre_ids, self.mid_ids, self.tail_ids = enc(before), enc(mid), enc(tail)
        self.caption_ids = enc(CAPTION_INSTRUCTION)
        self.suffix_ids = enc(ANSWER_SUFFIX)
        self.enc = enc
        self.rendered_example = rendered

    def build(self, kind, n_vis, question=None, target=None):
        """Return (ids, labels, vis). kind = 'caption' or 'qa'; vis marks the visual placeholder positions."""
        if kind == "caption":
            instr = self.caption_ids
        else:
            instr = self.enc(question)[:CAPS["question"]] + self.suffix_ids
        body = self.mid_ids + instr + self.tail_ids
        ids = list(self.pre_ids) + [self.placeholder] * n_vis + body
        vis = [0] * len(self.pre_ids) + [1] * n_vis + [0] * len(body)
        labels = [-100] * len(ids)
        if target is not None:
            cap = CAPS["caption"] if kind == "caption" else CAPS["answer"]
            t_ids = self.enc(target)[:cap] + [self.im_end]
            ids += t_ids
            vis += [0] * len(t_ids)
            labels += t_ids
        return ids, labels, vis


def collate(items, pad_id, left_pad=False):
    """items: list of (ids, labels, vis). Position ids count only real tokens (so padding never shifts positions)."""
    L = max(len(i[0]) for i in items)
    B = len(items)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    lab = torch.full((B, L), -100, dtype=torch.long)
    vis = torch.zeros((B, L), dtype=torch.bool)
    att = torch.zeros((B, L), dtype=torch.long)
    for b, (i, l, v) in enumerate(items):
        n = len(i)
        sl = slice(L - n, L) if left_pad else slice(0, n)
        ids[b, sl] = torch.tensor(i)
        lab[b, sl] = torch.tensor(l)
        vis[b, sl] = torch.tensor(v, dtype=torch.bool)
        att[b, sl] = 1
    pos = (att.cumsum(1) - 1).clamp(min=0)
    return {"input_ids": ids, "labels": lab, "vis_mask": vis, "attention_mask": att, "position_ids": pos}


class VLM(nn.Module):
    def __init__(self, lm, connector):
        super().__init__()
        self.lm = lm
        self.connector = connector

    def embed(self, batch, feats):
        """Text embeddings everywhere; connector outputs at the visual positions (gradients flow to the connector)."""
        emb = self.lm.get_input_embeddings()(batch["input_ids"])
        if feats is not None and batch["vis_mask"].any():
            z = self.connector(feats.to(emb.device, torch.float32))     # FP16 cache -> FP32 is exact
            z = z.to(emb.dtype)
            emb = emb.clone()
            emb[batch["vis_mask"]] = z.reshape(-1, z.shape[-1])
        return emb

    def forward(self, batch, feats):
        emb = self.embed(batch, feats)
        out = self.lm.model(inputs_embeds=emb, attention_mask=batch["attention_mask"], position_ids=batch["position_ids"])
        return out.last_hidden_state

    def example_nll(self, batch, feats, exclude_last=False):
        """Mean token NLL WITHIN each example (supervised tokens only). exclude_last drops the final <|im_end|>
        (reference-answer NLL excludes termination). Only supervised positions go through the output layer."""
        h = self.forward(batch, feats)[:, :-1]
        tgt = batch["labels"][:, 1:].clone()
        if exclude_last:
            sup = tgt != -100
            last = sup & (sup.cumsum(1) == sup.sum(1, keepdim=True))
            tgt[last] = -100
        sup = tgt != -100
        tok_nll = F.cross_entropy(self.lm.lm_head(h[sup]).float(), tgt[sup], reduction="none")
        ex = sup.nonzero()[:, 0]
        sums = torch.zeros(sup.shape[0], device=h.device, dtype=torch.float32).index_add_(0, ex, tok_nll)
        return sums / sup.sum(1).clamp(min=1)

    @torch.no_grad()
    def greedy(self, batch, feats, max_new=16, eos_ids=()):
        """Batched greedy decoding with a KV cache (left-padded prompts)."""
        emb = self.embed(batch, feats)
        att = batch["attention_mask"].clone()
        pos = batch["position_ids"].clone()
        eos = torch.tensor(list(eos_ids), device=emb.device)
        try:
            out = self.lm(inputs_embeds=emb, attention_mask=att, position_ids=pos, use_cache=True, logits_to_keep=1)
        except TypeError:
            out = self.lm(inputs_embeds=emb, attention_mask=att, position_ids=pos, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1)
        B = nxt.shape[0]
        gen = [nxt]
        done = torch.isin(nxt, eos)
        cur_pos = pos[:, -1]
        for _ in range(max_new - 1):
            if done.all():
                break
            att = torch.cat([att, torch.ones((B, 1), dtype=att.dtype, device=att.device)], 1)
            cur_pos = cur_pos + 1
            e = self.lm.get_input_embeddings()(nxt[:, None])
            out = self.lm(inputs_embeds=e, attention_mask=att, position_ids=cur_pos[:, None], past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, int(eos[0])), nxt)
            gen.append(nxt)
            done |= torch.isin(nxt, eos)
        return torch.stack(gen, 1)


def load_lm(dev, dtype=None):
    """Qwen3-0.6B with EVERY original parameter frozen (gradients still flow through it to the connector)."""
    dtype = dtype or (torch.float16 if dev.type == "cuda" else torch.float32)
    lm = load_causal_lm(LM_ID, dtype)
    for p in lm.parameters():
        p.requires_grad_(False)
    return lm.to(dev)


@torch.no_grad()
def calibrate_scale(vlm, builder, texts, feats, dev, log=print):
    """Rule (D-T2-5): s = mean L2 norm of Qwen input embeddings over the tokens of training texts
                          / mean L2 norm of the freshly initialised connector's outputs over training visual tokens.
    Training data only. Afterwards visual and text tokens enter Qwen at the same average size."""
    E = vlm.lm.get_input_embeddings()
    tnorm = torch.cat([E(torch.tensor(builder.enc(t), device=dev)).float().norm(dim=-1) for t in texts])
    vlm.connector.scale.fill_(1.0)
    vn = torch.cat([vlm.connector(feats[i:i + 64].to(dev).float()).norm(dim=-1).flatten() for i in range(0, feats.shape[0], 64)])
    s = float(tnorm.mean() / vn.mean())
    vlm.connector.scale.fill_(s)
    stats = {"rule": "s = mean ||text embedding|| / mean ||connector(F)|| at initialisation, training data only",
             "text_norm_mean": float(tnorm.mean()), "text_norm_std": float(tnorm.std()), "n_text_tokens": int(tnorm.numel()),
             "visual_norm_mean_before": float(vn.mean()), "n_visual_tokens": int(vn.numel()), "scale": s}
    log(f"connector scale s = {s:.5f} (text {stats['text_norm_mean']:.4f} / visual {stats['visual_norm_mean_before']:.4f})")
    return stats
