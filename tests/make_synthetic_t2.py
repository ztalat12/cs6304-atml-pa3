"""Synthetic COCO-2014 + VQAv2 files and small JPEG images with the real file names (offline Task 2 tests only)."""
import json
import random
import shutil
from pathlib import Path

import numpy as np

CATS = ["person", "dog", "cat", "car", "bus", "apple", "chair", "bench", "knife", "sheep"]
SMALL_CFG = dict(n_caption_train_images=200, n_caption_dev_images=10,
                 grounding_dev_pairs={"presence": 2, "counting": 2}, grounding_test_pairs={"presence": 4, "counting": 4},
                 vqa_train_pairs={"yes/no": 10, "number": 4, "other": 10}, vqa_dev_pairs={"yes/no": 2, "number": 1, "other": 2},
                 vqa_test_pairs={"yes/no": 3, "number": 2, "other": 3})


def make(root, coco2017_for=()):
    """Write annotation JSONs into root/ann and images into root/coco/{train2014,val2014}. Image ids listed in
    coco2017_for are written with COCO-2017 names in root/coco/val2017 instead (to test that fallback)."""
    rng = random.Random(0)
    root = Path(root)
    shutil.rmtree(root, ignore_errors=True)
    ann = root / "ann"
    ann.mkdir(parents=True)
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(CATS)]

    def coco(split, ids):
        imgs = [{"id": i, "width": 640, "height": 480} for i in ids]
        caps = [{"id": i * 10 + k, "image_id": i, "caption": f"a photo number {i} variant {k}"} for i in ids for k in range(5)]
        anns, aid = [], 0
        for i in ids:
            for c in rng.sample(cats, rng.randint(0, 3)):
                for _ in range(rng.randint(1, 5)):
                    aid += 1
                    anns.append({"id": aid, "image_id": i, "category_id": c["id"], "area": rng.uniform(0.002, 0.1) * 640 * 480,
                                 "iscrowd": int(rng.random() < 0.05)})
        json.dump({"images": imgs, "annotations": caps}, open(ann / f"captions_{split}.json", "w"))
        json.dump({"images": imgs, "annotations": anns, "categories": cats}, open(ann / f"instances_{split}.json", "w"))

    def vqa(split, ids):
        qs, ans, pairs, qid = [], [], [], 0
        for k in range(0, len(ids) - 1, 2):
            for t, a1, a2, q in [("yes/no", "yes", "no", "Is it sunny?"), ("number", "2", "3", "How many cups?"),
                                 ("other", "red", "blue", "What color is the car?")]:
                pq = []
                for img, a in [(ids[k], a1), (ids[k + 1], a2)]:
                    qid += 1
                    qs.append({"question_id": qid, "image_id": img, "question": q})
                    ans.append({"question_id": qid, "answer_type": t, "question_type": "x",
                                "answers": [{"answer": x} for x in [a.capitalize()] + [a] * 8 + ["maybe"]]})
                    pq.append(qid)
                pairs.append(pq)
        json.dump({"questions": qs}, open(ann / f"v2_OpenEnded_mscoco_{split}_questions.json", "w"))
        json.dump({"annotations": ans}, open(ann / f"v2_mscoco_{split}_annotations.json", "w"))
        json.dump(pairs, open(ann / f"v2_mscoco_{split}_complementary_pairs.json", "w"))

    train_ids, val_ids = list(range(1, 401)), list(range(10001, 10601))
    coco("train2014", train_ids)
    coco("val2014", val_ids)
    vqa("train2014", train_ids)
    vqa("val2014", val_ids)
    from PIL import Image
    g = np.random.default_rng(0)
    for split, ids in [("train2014", train_ids), ("val2014", val_ids)]:
        for i in ids:
            if i in coco2017_for:
                d, name = root / "coco" / "val2017", f"{i:012d}.jpg"
            else:
                d, name = root / "coco" / split, f"COCO_{split}_{i:012d}.jpg"
            d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(g.integers(0, 255, (48, 64, 3), dtype=np.uint8)).save(d / name, quality=80)
    return root
