"""Task 2 data: COCO 2014 captions + instance annotations, VQAv2 complementary pairs.

Produces ONE selection file (t2_selection.json) holding every image / question / pair ID used anywhere, so the
whole experiment is reproducible from it. Every random choice uses seed 6304.

Splits: train2014 images -> training (captions, grounding pools, VQA training pairs)
        val2014 images   -> development and final test. Each val image is used by AT MOST ONE evaluation item,
                            so all evaluation sets are image-disjoint from each other (and from training).
"""
from __future__ import annotations

import json
import re
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common.utils import SEED, Checks, write_json

COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
VQA_BASE = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/"
VQA_ZIPS = ["v2_Questions_Train_mscoco.zip", "v2_Questions_Val_mscoco.zip", "v2_Annotations_Train_mscoco.zip",
            "v2_Annotations_Val_mscoco.zip", "v2_Complementary_Pairs_Train_mscoco.zip", "v2_Complementary_Pairs_Val_mscoco.zip"]
# every needed JSON, matched by file name pattern (so slightly different names inside a zip or a Kaggle dataset still work)
NEEDED = {
    "captions_train": r"^captions_train2014\.json$", "captions_val": r"^captions_val2014\.json$",
    "instances_train": r"^instances_train2014\.json$", "instances_val": r"^instances_val2014\.json$",
    "q_train": r"OpenEnded_mscoco_train2014_questions\.json$", "q_val": r"OpenEnded_mscoco_val2014_questions\.json$",
    "a_train": r"mscoco_train2014_annotations\.json$", "a_val": r"mscoco_val2014_annotations\.json$",
    "pairs_train": r"train2014_complementary_pairs\.json$", "pairs_val": r"val2014_complementary_pairs\.json$",
}
SOURCE_OF = {"captions_train": "coco", "captions_val": "coco", "instances_train": "coco", "instances_val": "coco",
             "q_train": "v2_Questions_Train_mscoco.zip", "q_val": "v2_Questions_Val_mscoco.zip",
             "a_train": "v2_Annotations_Train_mscoco.zip", "a_val": "v2_Annotations_Val_mscoco.zip",
             "pairs_train": "v2_Complementary_Pairs_Train_mscoco.zip", "pairs_val": "v2_Complementary_Pairs_Val_mscoco.zip"}

# ----------------------------------------------------------------------------- decisions (README: D-T2-*)
CFG = dict(
    n_caption_train_images=6000, captions_per_train_image=3, n_caption_dev_images=96,
    caption_choice="lowest annotation id",              # D-T2-4: which captions of an image are used
    grounding_pool_images="the 6,000 caption-training images",   # D-T2-3
    min_area_frac_presence=0.01,                         # D-T2-2: the 'yes' image needs one instance >= 1% of the image area
    min_area_frac_count=0.01,                            # D-T2-2: every counted instance >= 1% of the image area, no crowd labels
    count_range=(1, 4),
    grounding_dev_pairs={"presence": 12, "counting": 12},     # D-T2-6: 24 pairs / 48 questions
    grounding_test_pairs={"presence": 32, "counting": 32},    # D-T2-6: 64 pairs / 128 questions
    vqa_train_pairs={"yes/no": 640, "number": 256, "other": 640},
    vqa_dev_pairs={"yes/no": 12, "number": 8, "other": 12},
    vqa_test_pairs={"yes/no": 36, "number": 24, "other": 36},
    vqa_min_agreement=8,
)

PLURAL = {"person": "people", "mouse": "mice", "knife": "knives", "sheep": "sheep", "skis": "skis", "scissors": "scissors",
          "bus": "buses", "couch": "couches", "sandwich": "sandwiches", "wine glass": "wine glasses",
          "toothbrush": "toothbrushes", "bench": "benches", "hair drier": "hair driers", "glass": "glasses"}


def plural(name):
    if name in PLURAL:
        return PLURAL[name]
    if name.endswith(("s", "x", "ch", "sh")):
        return name + "es"
    if name.endswith("y") and name[-2:-1] not in "aeiou":
        return name[:-1] + "ies"
    return name + "s"


def article(name):
    return ("an " if name[0] in "aeiou" else "a ") + name


def presence_question(name):
    return f"Does this image contain {article(name)}?"


def counting_question(name):
    return f"How many {plural(name)} can you see?"


# ----------------------------------------------------------------------------- official VQA answer normalisation
_CONTRACTIONS = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
                 "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
                 "hadnt": "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't",
                 "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll",
                 "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd",
                 "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
                 "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", "mustnt": "mustn't",
                 "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
                 "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've",
                 "she'dve": "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
                 "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd",
                 "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
                 "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've",
                 "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've",
                 "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd",
                 "thered've": "there'd've", "there'dve": "there'd've", "therere": "there're", "theres": "there's",
                 "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're",
                 "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've",
                 "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", "whats": "what's",
                 "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've",
                 "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's",
                 "whove": "who've", "whyll": "why'll", "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've",
                 "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've", "yall": "y'all",
                 "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", "y'alld've": "y'all'd've",
                 "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll",
                 "youre": "you're", "youve": "you've"}
_MANUAL = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
           "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
_ARTICLES = {"a", "an", "the"}
_PERIOD = re.compile(r"(?!<=\d)(\.)(?!\d)")       # copied as-is from the official code
_COMMA = re.compile(r"(\d)(\,)(\d)")
_PUNCT = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]


def vqa_normalize(ans: str) -> str:
    """Port of the official VQA evaluation normalisation (processPunctuation + processDigitArticle) from
    GT-Vision-Lab/VQA PythonEvaluationTools/vqaEvaluation/vqaEval.py. Applied to predictions AND references."""
    t = (ans or "").replace("\n", " ").replace("\t", " ").strip()
    out = t
    for p in _PUNCT:
        if (p + " " in t or " " + p in t) or (re.search(_COMMA, t) is not None):
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    out = _PERIOD.sub("", out, re.UNICODE)            # the official code passes re.UNICODE as the count; kept for fidelity
    words = []
    for w in out.lower().split():
        w = _MANUAL.get(w, w)
        if w not in _ARTICLES:
            words.append(w)
    return " ".join(_CONTRACTIONS.get(w, w) for w in words)


def vqa_soft_accuracy(pred_norm: str, refs_norm: list) -> float:
    """s(a) = 1/10 * sum_i min(1, (1/3) * #{j != i : a_j == a})   (leave-one-annotator-out average)."""
    accs = []
    for i in range(len(refs_norm)):
        others = refs_norm[:i] + refs_norm[i + 1:]
        accs.append(min(1.0, sum(o == pred_norm for o in others) / 3.0))
    return float(np.mean(accs))


# ----------------------------------------------------------------------------- annotation files
def _download(url, dest, log=print):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    tmp.rename(dest)
    log(f"  downloaded {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def _match(key, name):
    return re.search(NEEDED[key], name) is not None


def _extract_needed(zpath, cache, keys):
    with zipfile.ZipFile(zpath) as zf:
        for m in zf.namelist():
            base = Path(m).name
            for k in keys:
                if _match(k, base) and not (cache / base).exists():
                    (cache / base).write_bytes(zf.read(m))


def _locate(key, roots):
    for r in roots:
        r = Path(r)
        if r.exists():
            for p in r.rglob("*.json"):
                if _match(key, p.name):
                    return p
    return None


def fetch_annotations(cache_dir, search_roots=("/kaggle/input",), log=print):
    """Return {key: path} for every needed JSON: attached files first (any Kaggle dataset that contains them),
    otherwise download the official zips. Raises with a clear list of what is missing."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    found = {k: _locate(k, list(search_roots) + [cache]) for k in NEEDED}
    missing = [k for k, v in found.items() if v is None]
    errors = {}
    if any(SOURCE_OF[k] == "coco" for k in missing):
        try:
            log("downloading COCO 2014 annotations (241 MB) ...")
            _extract_needed(_download(COCO_ANN_URL, cache / "annotations_trainval2014.zip", log), cache,
                            [k for k in missing if SOURCE_OF[k] == "coco"])
        except Exception as e:      # noqa: BLE001
            errors[COCO_ANN_URL] = repr(e)
    for zname in sorted({SOURCE_OF[k] for k in missing if SOURCE_OF[k] != "coco"}):
        try:
            log(f"downloading {zname} ...")
            _extract_needed(_download(VQA_BASE + zname, cache / zname, log), cache,
                            [k for k in missing if SOURCE_OF[k] == zname])
        except Exception as e:      # noqa: BLE001
            errors[VQA_BASE + zname] = repr(e)
    found = {k: _locate(k, list(search_roots) + [cache]) for k in NEEDED}
    still = [k for k, v in found.items() if v is None]
    if still:
        raise FileNotFoundError(f"could not obtain {still}; download errors: {errors}. Attach a Kaggle dataset that "
                                f"contains these JSON files (any folder layout) and run again.")
    return found


# ----------------------------------------------------------------------------- selection helpers
def _instances_index(inst):
    cats = {c["id"]: c["name"] for c in inst["categories"]}
    area = {im["id"]: im["width"] * im["height"] for im in inst["images"]}
    per = defaultdict(lambda: defaultdict(list))   # image -> category -> [(area, iscrowd)]
    for a in inst["annotations"]:
        per[a["image_id"]][a["category_id"]].append((a["area"], a["iscrowd"]))
    return cats, area, per


def _presence_ok(img, c, per, area, frac):
    return any(a >= frac * area[img] and not cr for a, cr in per[img].get(c, []))


def _count_of(img, c, per, area, frac, lo, hi):
    anns = per[img].get(c, [])
    if not anns or any(cr for _, cr in anns) or any(a < frac * area[img] for a, _ in anns):
        return None
    n = len(anns)
    return n if lo <= n <= hi else None


def build_grounding_pool(image_ids, cats, area, per, cfg, rng):
    """Reusable training pools. Every eligible positive (image, category) is paired with one random image that has no
    label of that category (presence); every countable (image, category) with one random image of a DIFFERENT count."""
    ids = sorted(image_ids)
    presence, counting = [], []
    for c in sorted(cats):
        pos = [i for i in ids if _presence_ok(i, c, per, area, cfg["min_area_frac_presence"])]
        neg = [i for i in ids if c not in per[i]]
        if pos and neg:
            negs = rng.choice(len(neg), size=len(pos), replace=True)
            presence += [{"cat": c, "name": cats[c], "pos": p, "neg": neg[j]} for p, j in zip(pos, negs)]
        byc = defaultdict(list)
        for i in ids:
            if c in per[i]:
                k = _count_of(i, c, per, area, cfg["min_area_frac_count"], *cfg["count_range"])
                if k is not None:
                    byc[k].append(i)
        if len(byc) >= 2:
            for k, imgs in sorted(byc.items()):
                others = [(kk, j) for kk, js in sorted(byc.items()) if kk != k for j in js]
                picks = rng.choice(len(others), size=len(imgs), replace=True)
                counting += [{"cat": c, "name": cats[c], "a": a, "count_a": k, "b": others[p][1], "count_b": others[p][0]}
                             for a, p in zip(imgs, picks)]
    return presence, counting


def _pair_to_questions(kind, p, split):
    if kind == "presence":
        q = presence_question(p["name"])
        return [{"image": f"{split}/{p['pos']}", "question": q, "answer": "yes", "type": "presence", "cat": p["name"]},
                {"image": f"{split}/{p['neg']}", "question": q, "answer": "no", "type": "presence", "cat": p["name"]}]
    q = counting_question(p["name"])
    return [{"image": f"{split}/{p['a']}", "question": q, "answer": str(p["count_a"]), "type": "counting", "cat": p["name"]},
            {"image": f"{split}/{p['b']}", "question": q, "answer": str(p["count_b"]), "type": "counting", "cat": p["name"]}]


def _grounding_candidates(kind, cats, area, per, ids, cfg):
    """Per-category candidate images for evaluation pairs (computed once)."""
    if kind == "presence":
        return {c: [i for i in ids if _presence_ok(i, c, per, area, cfg["min_area_frac_presence"])] for c in cats}
    out = {}
    for c in cats:
        byc = defaultdict(list)
        for i in ids:
            if c in per[i]:
                k = _count_of(i, c, per, area, cfg["min_area_frac_count"], *cfg["count_range"])
                if k is not None:
                    byc[k].append(i)
        out[c] = dict(byc)
    return out


def _eval_grounding(kind, n, cats, per, ids, cand, used, rng):
    """n evaluation pairs of one kind from val2014. Categories are visited round-robin in a seeded order (so the
    pairs cover many categories); images are never reused (`used` is shared by all evaluation sets)."""
    order = [int(c) for c in rng.permutation(sorted(cats))]
    pairs, tries = [], 0
    while len(pairs) < n and tries < 40 * n + len(order):
        c = order[tries % len(order)]
        tries += 1
        if kind == "presence":
            pos = [i for i in cand[c] if i not in used]
            if not pos:
                continue
            p = int(pos[rng.integers(len(pos))])
            for _ in range(2000):                        # rejection-sample a negative image
                q = int(ids[rng.integers(len(ids))])
                if q not in used and q != p and c not in per[q]:
                    break
            else:
                continue
            pairs.append({"cat": c, "name": cats[c], "pos": p, "neg": q})
            used.update([p, q])
        else:
            byc = {k: [i for i in v if i not in used] for k, v in cand[c].items()}
            byc = {k: v for k, v in byc.items() if v}
            if len(byc) < 2:
                continue
            ks = sorted(byc)
            k1, k2 = rng.choice(len(ks), 2, replace=False)
            a = int(byc[ks[k1]][rng.integers(len(byc[ks[k1]]))])
            b = int(byc[ks[k2]][rng.integers(len(byc[ks[k2]]))])
            pairs.append({"cat": c, "name": cats[c], "a": a, "count_a": ks[k1], "b": b, "count_b": ks[k2]})
            used.update([a, b])
    return pairs


def _vqa_index(qfile, afile):
    qs = {q["question_id"]: q for q in json.load(open(qfile))["questions"]}
    anns = {a["question_id"]: a for a in json.load(open(afile))["annotations"]}
    return qs, anns


def _vqa_item(qid, qs, anns):
    """Consensus = most common NORMALISED answer (ties: first seen); agreement = its count out of 10.
    Training target = the most common ORIGINAL string inside the consensus group (ties: first seen)."""
    a = anns[qid]
    raw = [x["answer"] for x in a["answers"]]
    norm = [vqa_normalize(x) for x in raw]
    cons, agree = Counter(norm).most_common(1)[0]
    group_raw = [r.strip() for r, nn in zip(raw, norm) if nn == cons]
    target = Counter(group_raw).most_common(1)[0][0]
    return {"qid": qid, "image_id": qs[qid]["image_id"], "question": qs[qid]["question"], "answer_type": a["answer_type"],
            "consensus": cons, "agreement": agree, "target": target, "answers": raw}


def _vqa_pairs(pairs_file, qs, anns, min_agree):
    out = defaultdict(list)
    for q1, q2 in json.load(open(pairs_file)):
        if q1 not in anns or q2 not in anns or q1 not in qs or q2 not in qs:
            continue
        a, b = _vqa_item(q1, qs, anns), _vqa_item(q2, qs, anns)
        if a["agreement"] < min_agree or b["agreement"] < min_agree:
            continue
        if a["consensus"] == b["consensus"] or a["answer_type"] != b["answer_type"]:
            continue
        if a["question"].strip().lower() != b["question"].strip().lower() or a["image_id"] == b["image_id"]:
            continue
        out[a["answer_type"]].append((a, b))
    return out


def build_selection(paths, out_path, log=print, cfg=None):
    """Select everything with seed 6304 and save it. Returns (selection, checks)."""
    C = Checks()
    cfg = cfg or CFG
    rng = np.random.default_rng(SEED)
    sel = {"seed": SEED, "config": cfg}

    # ---- captions (train): 6,000 random train2014 images with >= 3 captions; their 3 lowest-id captions
    ct = json.load(open(paths["captions_train"]))
    cv = json.load(open(paths["captions_val"]))
    caps_t, caps_v = defaultdict(list), defaultdict(list)
    for a in ct["annotations"]:
        caps_t[a["image_id"]].append((a["id"], a["caption"].strip()))
    for a in cv["annotations"]:
        caps_v[a["image_id"]].append((a["id"], a["caption"].strip()))
    del ct, cv
    train_imgs = sorted(i for i in caps_t if len(caps_t[i]) >= cfg["captions_per_train_image"])
    cap_train_ids = sorted(int(x) for x in rng.choice(train_imgs, cfg["n_caption_train_images"], replace=False))
    sel["caption_train"] = [{"image": f"train2014/{i}", "ann_id": aid, "caption": c}
                            for i in cap_train_ids for aid, c in sorted(caps_t[i])[:cfg["captions_per_train_image"]]]
    log(f"captions: {len(sel['caption_train'])} training examples from {len(cap_train_ids)} images")

    # ---- grounding training pools (from the caption-training images)
    it = json.load(open(paths["instances_train"]))
    cats, area_t, per_t = _instances_index(it)
    del it
    pres, cnt = build_grounding_pool(cap_train_ids, cats, area_t, per_t, cfg, rng)
    sel["grounding_train_pool"] = {"presence": [_pair_to_questions("presence", p, "train2014") for p in pres],
                                   "counting": [_pair_to_questions("counting", p, "train2014") for p in cnt]}
    log(f"grounding pools: {len(pres)} presence pairs, {len(cnt)} counting pairs")
    iv = json.load(open(paths["instances_val"]))
    cats_v, area_v, per_v = _instances_index(iv)
    del iv
    val_imgs = sorted(set(caps_v) & set(area_v))
    used = set()

    # ---- VQA evaluation first (the scarcer natural pairs get first pick of val images)
    qs_v, an_v = _vqa_index(paths["q_val"], paths["a_val"])
    vp = _vqa_pairs(paths["pairs_val"], qs_v, an_v, cfg["vqa_min_agreement"])
    C.observe("eligible validation complementary pairs by type", {k: len(v) for k, v in vp.items()})
    for split_name, counts in [("vqa_test", cfg["vqa_test_pairs"]), ("vqa_dev", cfg["vqa_dev_pairs"])]:
        chosen = []
        for t, n in counts.items():
            cand = vp[t]
            k = 0
            for j in rng.permutation(len(cand)):
                a, b = cand[j]
                if a["image_id"] in used or b["image_id"] in used:
                    continue
                used.update([a["image_id"], b["image_id"]])
                chosen.append([{**a, "image": f"val2014/{a['image_id']}", "type": t},
                               {**b, "image": f"val2014/{b['image_id']}", "type": t}])
                k += 1
                if k == n:
                    break
            C.check(k == n, f"{split_name}: {n} '{t}' pairs selected", n, k)
        sel[split_name] = chosen
    del qs_v, an_v, vp
    qs_t, an_t = _vqa_index(paths["q_train"], paths["a_train"])
    tp = _vqa_pairs(paths["pairs_train"], qs_t, an_t, cfg["vqa_min_agreement"])
    C.observe("eligible training complementary pairs by type", {k: len(v) for k, v in tp.items()})
    chosen = []
    for t, n in cfg["vqa_train_pairs"].items():
        for j in rng.choice(len(tp[t]), n, replace=False):
            a, b = tp[t][j]
            chosen.append([{**a, "image": f"train2014/{a['image_id']}", "type": t},
                           {**b, "image": f"train2014/{b['image_id']}", "type": t}])
    sel["vqa_train"] = chosen
    del qs_t, an_t, tp

    # ---- grounding evaluation
    cand = {kind: _grounding_candidates(kind, cats_v, area_v, per_v, val_imgs, cfg) for kind in ["presence", "counting"]}
    for split_name, counts in [("grounding_test", cfg["grounding_test_pairs"]), ("grounding_dev", cfg["grounding_dev_pairs"])]:
        pairs = []
        for kind, n in counts.items():
            ps = _eval_grounding(kind, n, cats_v, per_v, val_imgs, cand[kind], used, rng)
            C.check(len(ps) == n, f"{split_name}: {n} {kind} pairs", n, len(ps))
            pairs += [_pair_to_questions(kind, p, "val2014") for p in ps]
        sel[split_name] = pairs

    # ---- caption development (any val image still unused); its lowest-id caption
    free = [i for i in val_imgs if i not in used]
    dev_caps = sorted(int(x) for x in rng.choice(free, cfg["n_caption_dev_images"], replace=False))
    used.update(dev_caps)
    sel["caption_dev"] = [{"image": f"val2014/{i}", "ann_id": sorted(caps_v[i])[0][0], "caption": sorted(caps_v[i])[0][1]}
                          for i in dev_caps]

    # ---- fixed mismatched-image mapping for the final-test suites (decided before any generation)
    sel["mismatch_rule"] = ("pairs permuted with the seeded rng; the pair at position p takes its mismatched images from the "
                            "pair at position (p + 1) mod n, same slot (first / second question)")
    for suite in ["grounding_test", "vqa_test"]:
        n = len(sel[suite])
        perm = [int(x) for x in rng.permutation(n)]
        nxt = {perm[p]: perm[(p + 1) % n] for p in range(n)}
        sel[f"{suite}_mismatch"] = [[sel[suite][nxt[i]][0]["image"], sel[suite][nxt[i]][1]["image"]] for i in range(n)]

    # ---- isolation checks
    def imgs(name):
        if name in ("caption_dev", "caption_train"):
            return {e["image"] for e in sel[name]}
        return {q["image"] for pair in sel[name] for q in pair}
    train_set = imgs("caption_train") | imgs("vqa_train") | {q["image"] for k in ["presence", "counting"]
                                                             for pair in sel["grounding_train_pool"][k] for q in pair}
    evals = ["caption_dev", "grounding_dev", "grounding_test", "vqa_dev", "vqa_test"]
    C.check(all(not (imgs(e) & train_set) for e in evals), "no evaluation image appears in training or replay", "0 overlaps",
            {e: len(imgs(e) & train_set) for e in evals})
    ov = {f"{a}&{b}": len(imgs(a) & imgs(b)) for i, a in enumerate(evals) for b in evals[i + 1:]}
    C.check(sum(ov.values()) == 0, "all development / test sets are image-disjoint from each other", "all 0", ov)
    for suite in ["grounding_test", "vqa_test"]:
        bad = sum(m in {sel[suite][i][0]["image"], sel[suite][i][1]["image"]}
                  for i, ms in enumerate(sel[f"{suite}_mismatch"]) for m in ms)
        C.check(bad == 0, f"{suite}: mismatched images never come from the question's own pair", 0, bad)
    exp_caps = cfg["n_caption_train_images"] * cfg["captions_per_train_image"]
    C.check(len(sel["caption_train"]) == exp_caps, "caption training examples", exp_caps, len(sel["caption_train"]))
    exp_vqa = sum(cfg["vqa_train_pairs"].values())
    C.check(len(sel["vqa_train"]) == exp_vqa, "VQAv2 training pairs", exp_vqa, len(sel["vqa_train"]))
    C.check(all(p[0]["consensus"] != p[1]["consensus"] and min(p[0]["agreement"], p[1]["agreement"]) >= cfg["vqa_min_agreement"]
                for s in ["vqa_train", "vqa_dev", "vqa_test"] for p in sel[s]),
            "every VQAv2 pair: different consensus answers, >= 8/10 agreement on both images")
    C.observe("grounding training pools (pairs)", {k: len(v) for k, v in sel["grounding_train_pool"].items()})
    C.observe("example generated questions", [sel["grounding_dev"][0][0]["question"], sel["grounding_dev"][-1][0]["question"]])
    sel["all_images"] = sorted(train_set | set().union(*[imgs(e) for e in evals]))
    C.observe("images needing features", len(sel["all_images"]))
    write_json(out_path, sel)
    return sel, C
