"""Quick checks of the split, the models, one small training run and the detector."""
import numpy as np
import torch
from torch import nn

from deep_learning import data as dataset
from deep_learning.evaluate import summarize
from deep_learning.models import CLASSES, CLASSES_WITH_LIGHTS, build_model

passed = 0


def check(name, condition, detail=""):
    global passed
    if not condition:
        raise SystemExit(f"FAILED: {name} {detail}")
    passed += 1
    print(f"ok  {name} {detail}")


device = torch.device("cuda")

for split in ("real_train", "unreal_train"):
    _, meta = dataset.load_split(split)
    validation, gap = dataset.validation_rows(meta)
    training = ~validation & ~gap
    train_keys = set(zip(meta["sign"][training], meta["index"][training]))
    validation_keys = set(zip(meta["sign"][validation], meta["index"][validation]))
    check(f"{split}: no base image on both sides", not train_keys & validation_keys)
    check(f"{split}: every sign has validation images", set(np.unique(meta["sign"][validation])) == set(range(CLASSES)))
    check(f"{split}: gap separates the sets", not (validation & gap).any() and gap.sum() > 0, f"{validation.mean():.1%} validation, {gap.mean():.1%} gap")
    for sign in range(CLASSES):
        train_top = meta["index"][training & (meta["sign"] == sign)].max()
        validation_bottom = meta["index"][validation & (meta["sign"] == sign)].min()
        check(f"{split}: sign {sign} block is after training", train_top < validation_bottom, f"{train_top} < {validation_bottom}")

clips = {
    split: set(dataset.load_split(f"lisa_{split}")[1]["clip"].tolist()) for split in ("train", "val", "test")
}
check("no LISA clip is in two splits", not (clips["train"] & clips["val"] or clips["train"] & clips["test"] or clips["val"] & clips["test"]))
check("LISA test uses only the official sequences", all("Sequence" in clip for clip in clips["test"]))
for split in ("train", "val", "test"):
    _, meta = dataset.load_split(f"lisa_{split}")
    check(f"lisa_{split} holds all three light colours", set(np.unique(meta["sign"])) == {14, 15, 16})
    check(f"lisa_{split} holds day and night", set(np.unique(meta["night"])) == {False, True})

lit_train, lit_validation = dataset.build_sets("clean", lights=True)
check("lights join training", set(np.unique(lit_train.labels)) == set(range(CLASSES_WITH_LIGHTS)))
check("lights join validation", set(np.unique(lit_validation.labels)) == set(range(CLASSES_WITH_LIGHTS)))
check("clean regime has clean lights only", (lit_train.challenge == 0).all())
check("17-class CNN outputs 17 classes", build_model("cnn", CLASSES_WITH_LIGHTS)(torch.rand(2, 3, 32, 32)).shape == (2, CLASSES_WITH_LIGHTS))

train_set, validation_set = dataset.build_sets("clean")
check("clean regime holds only clean images", (train_set.challenge == 0).all())
check("validation holds real damaged images", len(np.unique(validation_set.challenge)) == 13)
print(f"    clean training {len(train_set.labels):,}, validation {len(validation_set.labels):,}")

weights = dataset.class_weights(train_set.labels)
counts = np.bincount(train_set.labels, minlength=CLASSES)
check("rarest sign gets the largest weight", int(weights.argmax()) == int(counts.argmin()))

images = dataset.to_tensor(train_set.inputs[:32], device)
check("images are scaled around zero", images.min() >= -0.5 and images.max() <= 0.5)
check("images are channels first", images.shape == (32, 3, 32, 32))
check("augmentation keeps the shape", dataset.augment(images).shape == images.shape)
check("augmentation changes images", (dataset.augment(images) != images).any())

for name in ("mlp", "cnn"):
    model = build_model(name).to(device)
    check(f"{name} outputs 14 classes", model(images).shape == (32, CLASSES))
    print(f"    {name}: {sum(p.numel() for p in model.parameters()):,} weights")

features = torch.rand(32, 348, device=device)
check("feature MLP outputs 14 classes", build_model("feature_mlp").to(device)(features).shape == (32, CLASSES))

# A model that can't memorize 256 images has a broken training step.
torch.manual_seed(0)
model = build_model("cnn").to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_function = nn.CrossEntropyLoss()
rows = np.random.default_rng(0).choice(len(train_set.labels), 256, replace=False)
inputs = dataset.to_tensor(train_set.inputs[rows], device)
labels = torch.from_numpy(train_set.labels[rows].astype(np.int64)).to(device)
model.train()
for _ in range(150):
    optimizer.zero_grad()
    loss = loss_function(model(inputs), labels)
    loss.backward()
    optimizer.step()
model.eval()
memorized = (model(inputs).argmax(dim=1) == labels).float().mean().item()
check("CNN can memorize 256 images", memorized > 0.98, f"{memorized:.3f}")

metrics = summarize(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
check("accuracy formula", metrics["accuracy"] == 0.75)
check("macro F1 formula", abs(metrics["f1"] - (2 / 3 + 0.8) / 2) < 1e-9)

# Wide classes: CURE-TSR, LISA, GTSRB and the Indian set
from deep_learning.wide_classes import GTSRB_TO_WIDE, INDIAN_SKIPPED, INDIAN_TO_WIDE, RENAMED, WIDE_CLASSES, WIDE_NAMES

check("wide names are unique", len(set(WIDE_NAMES)) == WIDE_CLASSES, str(WIDE_CLASSES))
check("first 17 wide classes keep their places", all(WIDE_NAMES[i] == RENAMED.get(n, n) for i, n in enumerate(dataset.CLASS_NAMES)))
check("every GTSRB class has a wide label", sorted(GTSRB_TO_WIDE) == list(range(43)))
check("all 85 Indian folders are mapped or skipped", len(INDIAN_TO_WIDE) + len(INDIAN_SKIPPED) == 85 and not set(INDIAN_TO_WIDE) & set(INDIAN_SKIPPED))
used = set(range(CLASSES_WITH_LIGHTS)) | set(GTSRB_TO_WIDE.values()) | set(INDIAN_TO_WIDE.values())
check("every wide class has a source", used == set(range(WIDE_CLASSES)))
check("GTSRB stop, yield and no entry merge with CURE-TSR", [GTSRB_TO_WIDE[c] for c in (14, 13, 17)] == [5, 12, 11])
check("Indian give way and stop merge with CURE-TSR", INDIAN_TO_WIDE["GIVE_WAY"] == 12 and INDIAN_TO_WIDE["STOP"] == 5)
for source in ("gtsrb", "indian"):
    labels = {split: set(np.unique(dataset.load_split(f"{source}_{split}")[1]["sign"]).tolist()) for split in ("train", "val", "test")}
    check(f"{source} test classes were all trained on", labels["test"] <= labels["train"])
    check(f"{source} labels are wide labels", max(max(v) for v in labels.values()) < WIDE_CLASSES)
_, gtsrb_val = dataset.load_split("gtsrb_val")
_, gtsrb_train = dataset.load_split("gtsrb_train")
check("GTSRB validation keeps all 20 damage types", len({(c, l) for c, l in zip(gtsrb_val["challenge"], gtsrb_val["level"]) if c}) == 20)
check("GTSRB training has 4 damaged copies per image", np.bincount(gtsrb_train["index"]).max() == 5)
wide_train, wide_validation = dataset.build_sets("clean", lights=True, wide=True)
check(f"wide training covers all {WIDE_CLASSES} classes", set(np.unique(wide_train.labels)) == set(range(WIDE_CLASSES)))
check("wide clean regime holds only clean images", (wide_train.challenge == 0).all())
check(f"wide CNN outputs {WIDE_CLASSES} classes", build_model("cnn", WIDE_CLASSES)(torch.rand(2, 3, 32, 32)).shape == (2, WIDE_CLASSES))

# Global classes, with Mapillary
from deep_learning.global_classes import GLOBAL_CLASSES, GLOBAL_NAMES, MAPILLARY_MEANINGS, OTHER_SIGN, mapillary_class

check("global names are unique", len(set(GLOBAL_NAMES)) == GLOBAL_CLASSES, str(GLOBAL_CLASSES))
check("first 84 global classes keep their places", all(GLOBAL_NAMES[i] == n or n == "end of speed limit 80" for i, n in enumerate(WIDE_NAMES)))
check("every Mapillary meaning names a global class", set(MAPILLARY_MEANINGS.values()) <= set(GLOBAL_NAMES))
check("Mapillary variants share a class", mapillary_class("regulatory--keep-right--g1") == mapillary_class("regulatory--keep-right--g4") == GLOBAL_NAMES.index("keep right"))
check("Mapillary speed limits merge into one class", mapillary_class("regulatory--maximum-speed-limit-30--g1") == mapillary_class("regulatory--maximum-speed-limit-led-80--g1") == 0)
check("blue square pedestrian crossings join the class", mapillary_class("information--pedestrians-crossing--g1") == GLOBAL_NAMES.index("pedestrian crossing"))
check("plates under signs are left out", mapillary_class("complementary--distance--g1") is None)
check("rare named signs become other sign", mapillary_class("warning--texts--g1") == OTHER_SIGN)
labels = {split: set(np.unique(dataset.load_split(f"mapillary_{split}")[1]["sign"]).tolist()) for split in ("train", "val", "test")}
check("Mapillary test classes were all trained on", labels["test"] <= labels["train"])
check("Mapillary labels are global labels", max(max(v) for v in labels.values()) < GLOBAL_CLASSES)
global_train, _ = dataset.build_sets("clean", lights=True, wide=True, mapillary=True)
check(f"global training covers all {GLOBAL_CLASSES} classes", set(np.unique(global_train.labels)) == set(range(GLOBAL_CLASSES)))
check(f"global CNN outputs {GLOBAL_CLASSES} classes", build_model("cnn", GLOBAL_CLASSES)(torch.rand(2, 3, 32, 32)).shape == (2, GLOBAL_CLASSES))

# YOLO detector, on 30 unseen test frames of each dataset it learned from, spread across its sequences
from pathlib import Path

from deep_learning import detector
from image_processing import photos
from scripts.prepare_detection import CLASSES as DETECTED, OUTPUT as DETECTION, SOURCES


def labelled_boxes(frame, kind):
    path = DETECTION / "labels" / "test" / f"{Path(frame).stem}.txt"
    boxes = [[float(v) for v in line.split()] for line in path.read_text().splitlines()] if path.exists() else []
    return [box for box in boxes if int(box[0]) == kind]


yolo = detector.load_detector()
for source, kind in SOURCES.items():
    frames = (DETECTION / "lists" / f"{source}_test.txt").read_text().split()
    frames = [f for f in frames[:: max(1, len(frames) // 60)] if labelled_boxes(f, kind)][:30]
    hits, total, inside = 0, 0, True
    for frame in frames:
        photo = photos.load_photo(open(frame, "rb"))
        height, width = photo.shape[:2]
        found = detector.find_objects(yolo, photo)
        inside &= all(0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height for x0, y0, x1, y1, *_ in found)
        # A labelled object counts as found when a box of the right kind covers its centre
        for _, x, y, _, _ in labelled_boxes(frame, kind):
            hits += any(k == DETECTED[kind] and x0 <= x * width <= x1 and y0 <= y * height <= y1 for x0, y0, x1, y1, k, _ in found)
            total += 1
    check(f"YOLO finds at least half the {DETECTED[kind]}s in {len(frames)} unseen {source} frames", hits >= total / 2, f"{hits}/{total}")
    check(f"YOLO boxes stay inside the {source} frames", inside)

# Once YOLO has said sign or light, the classifier chooses only among that kind
scores = np.full(GLOBAL_CLASSES, 0.5 / (GLOBAL_CLASSES - 1))
scores[GLOBAL_NAMES.index("parking")] = 0.5
as_light = detector.keep_kind(scores, GLOBAL_NAMES, "light")
check("a light can only be named as a light", GLOBAL_NAMES[int(as_light.argmax())].endswith(" light") and abs(as_light.sum() - 1) < 1e-9)
check("a sign keeps its best sign", GLOBAL_NAMES[int(detector.keep_kind(scores, GLOBAL_NAMES, "sign").argmax())] == "parking")
check("a sign can't be named as a light", detector.keep_kind(np.eye(GLOBAL_CLASSES)[GLOBAL_NAMES.index("green light")], GLOBAL_NAMES, "sign")[GLOBAL_NAMES.index("green light")] == 0)
check("a sign scored 0 everywhere still gets a sign", not GLOBAL_NAMES[int(detector.keep_kind(np.eye(GLOBAL_CLASSES)[GLOBAL_NAMES.index("green light")], GLOBAL_NAMES, "sign").argmax())].endswith(" light"))

# Worn paint, added after a phone test, on a red disc with black lettering on a green background
from image_processing.restoration import WORN_PAINT, add_wear

painted = np.full((32, 32, 3), (40, 90, 40), np.uint8)
rows, columns = np.mgrid[:32, :32]
disc = np.hypot(rows - 15.5, columns - 15.5) <= 12
painted[disc] = (200, 30, 30)
painted[13:19, 9:23] = 10
worn = [add_wear(painted, level, np.random.default_rng(level)) for level in range(1, 6)]
check("worn copies keep their size and type", all(w.shape == painted.shape and w.dtype == np.uint8 for w in worn))
redness = [float((w[disc][:, 0].astype(int) - w[disc][:, 1]).mean()) for w in worn]
check("more wear takes more red off the sign", all(a > b for a, b in zip(redness, redness[1:])), str([round(r) for r in redness]))
check("black lettering stays dark", all(w[13:19, 9:23].mean() < 100 for w in worn))
check("the corners around the sign stay as they were", all((w[:3, :3] == painted[:3, :3]).all() for w in worn))
_, worn_meta = dataset.load_split("gtsrb_val_wear")
_, clean_meta = dataset.load_split("gtsrb_val")
check("every clean validation sign has all 5 worn levels", len(worn_meta["sign"]) == 5 * int((clean_meta["challenge"] == 0).sum()) and set(worn_meta["level"]) == {1, 2, 3, 4, 5})
check("worn copies are marked as worn paint", (worn_meta["challenge"] == WORN_PAINT).all())
_, worn_meta = dataset.load_split("real_train_wear")
_, clean_meta = dataset.load_split("real_train")
held = dataset.validation_rows(worn_meta)[0]
check("worn copies keep CURE-TSR's validation block", held.any() and set(worn_meta["index"][held]) <= set(clean_meta["index"][dataset.validation_rows(clean_meta)[0]]))

# Real damaged-sign photos from Kaggle: only stop, give way and no entry name one of our classes
known = {GLOBAL_NAMES.index(name) for name in ("stop", "yield", "no entry")}
damaged = {split: dataset.load_split(f"damaged_{split}")[1] for split in ("train", "val", "test")}
check("damaged photos only name stop, give way and no entry", all(set(np.unique(m["sign"]).tolist()) <= known for m in damaged.values()))
check("every damaged split has all three signs", all(set(np.unique(m["sign"]).tolist()) == known for m in damaged.values()))

print(f"\nAll {passed} checks passed.")
