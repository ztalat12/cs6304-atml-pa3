"""Build a tiny random Qwen3 model + character-level tokenizer with Qwen's special tokens and a
Qwen3-style chat template (offline tests only; the real model is used on Kaggle)."""
import sys
from pathlib import Path

import torch

TEMPLATE = (
    "{%- if messages[0]['role'] == 'system' %}{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}{%- endif %}"
    "{%- for m in messages %}"
    "{%- if m['role'] == 'user' %}{{- '<|im_start|>user\\n' + m['content'] + '<|im_end|>\\n' }}"
    "{%- elif m['role'] == 'assistant' %}{{- '<|im_start|>assistant\\n' }}"
    "{%- if loop.last %}{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}{{- m['content'] + '<|im_end|>\\n' }}"
    "{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}{%- endif %}"
)


def make(out, n_layers=28, hidden=32, inter=48, heads=4, kv=2, head_dim=8):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    words = [chr(c) for c in range(32, 127)] + ["\n"]
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
    vocab = {w: i for i, w in enumerate(specials + words)}
    vocab["[UNK]"] = len(vocab)
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Split(pattern="", behavior="isolated")
    tk.decoder = decoders.Fuse()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="<|endoftext|>", eos_token="<|im_end|>",
                                  additional_special_tokens=specials[1:])
    tok.chat_template = TEMPLATE
    tok.save_pretrained(out)
    cfg = Qwen3Config(vocab_size=len(vocab), hidden_size=hidden, intermediate_size=inter, num_hidden_layers=n_layers,
                      num_attention_heads=heads, num_key_value_heads=kv, head_dim=head_dim, max_position_embeddings=4096,
                      tie_word_embeddings=True, pad_token_id=vocab["<|endoftext|>"], eos_token_id=vocab["<|im_end|>"])
    torch.manual_seed(0)
    m = Qwen3ForCausalLM(cfg).to(torch.bfloat16)        # the real checkpoint is stored in BF16 too
    m.save_pretrained(out)
    return out


def make_siglip(out):
    """Tiny random SigLIP vision encoder with the REAL output size (196 patches x 768) and a standard processor."""
    from transformers import SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = SiglipVisionConfig(image_size=224, patch_size=16, hidden_size=768, intermediate_size=64, num_hidden_layers=1,
                             num_attention_heads=2)
    torch.manual_seed(0)
    SiglipVisionModel(cfg).save_pretrained(out)
    SiglipImageProcessor().save_pretrained(out)
    return out


if __name__ == "__main__":
    # python tests/make_tiny.py /tmp/tinyqwen            (Task 1 and Task 2)
    # python tests/make_tiny.py /tmp/tinysiglip --siglip  (Task 2)
    if "--siglip" in sys.argv:
        make_siglip(sys.argv[1])
    else:
        make(sys.argv[1] if len(sys.argv) > 1 else "/tmp/tinyqwen")
