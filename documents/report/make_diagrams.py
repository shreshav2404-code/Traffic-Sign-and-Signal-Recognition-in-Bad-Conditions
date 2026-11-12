"""Draw the report's and slides' diagrams: the pipeline, the architecture, the CNN, the fuzzy controller and the protocol.

Run with the project's .venv: python documents/report/make_diagrams.py
Colours follow the course each block belongs to, as in the charts. Sizes are in inches.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures"
INK = "#1F2A33"
MUTED = "#4F5A64"
# Course colours: border and soft fill
COURSE = {
    "image": ("#2A78D6", "#E6F0FB"),
    "fuzzy": ("#E07B16", "#FDF0E2"),
    "neural": ("#1B9E77", "#E4F4EE"),
    "app": ("#6B4FBB", "#EFEAF9"),
    "data": ("#5B6670", "#EEF0F2"),
}
LABELS = {"data": "Data", "image": "Image processing", "fuzzy": "Fuzzy logic", "neural": "Neural networks and deep learning", "app": "Detection and app"}
plt.rcParams.update({"font.family": "Arial"})
POINT = 1 / 72


def box(axis, x, y, w, h, title, lines=(), kind="data", title_size=9, line_size=7.5):
    """A rounded block: bold title near the top, then short grey lines, spaced by font size."""
    edge, fill = COURSE[kind]
    axis.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", linewidth=1.3, edgecolor=edge, facecolor=fill))
    title_lines = title.count("\n") + 1
    text_height = title_lines * title_size * POINT * 1.25 + len(lines) * line_size * POINT * 1.4
    top = y + h / 2 + text_height / 2
    axis.text(x + w / 2, top, title, ha="center", va="top", fontsize=title_size, fontweight="bold", color=INK, linespacing=1.15)
    cursor = top - title_lines * title_size * POINT * 1.25 - 0.03
    for line in lines:
        axis.text(x + w / 2, cursor, line, ha="center", va="top", fontsize=line_size, color=MUTED)
        cursor -= line_size * POINT * 1.4


def arrow(axis, start, end, colour=INK, width=1.3):
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, linewidth=width, color=colour, shrinkA=0, shrinkB=0))


def canvas(width, height):
    figure = plt.figure(figsize=(width, height))
    axis = figure.add_axes((0, 0, 1, 1))
    axis.set_xlim(0, width)
    axis.set_ylim(0, height)
    axis.axis("off")
    return figure, axis


def save(figure, name):
    figure.savefig(OUT / name, dpi=220, facecolor="white")
    plt.close(figure)
    print(f"Saved {name}")


def legend(axis, x, y, kinds, size=7.5):
    for kind in kinds:
        edge, fill = COURSE[kind]
        axis.add_patch(FancyBboxPatch((x, y - 0.05), 0.16, 0.1, boxstyle="round,pad=0,rounding_size=0.02", linewidth=1.1, edgecolor=edge, facecolor=fill))
        axis.text(x + 0.22, y, LABELS[kind], va="center", fontsize=size, color=INK)
        x += 0.34 + len(LABELS[kind]) * size * POINT * 0.52


def pipeline():
    """What happens to one photo, from upload to the two answers."""
    width = 7.2
    figure, axis = canvas(width, 3.1)
    w, h, gap = 1.62, 1.0, (width - 0.1 - 4 * 1.62) / 3
    xs = [0.05 + i * (w + gap) for i in range(4)]
    top, bottom = 1.85, 0.42
    upper = [
        ("1. Input", ["phone or street photo,", "or a damaged sample"], "data"),
        ("2. YOLO26n detector", ["finds signs and lights", "confidence ≥ 0.30", "or the user drags a box"], "app"),
        ("3. Crop and resize", ["square crop, margin 1.15", "resized to 32 × 32 × 3"], "image"),
        ("4. Measure quality", ["brightness (mean)", "contrast (RMS)", "noise (colour roughness)"], "image"),
    ]
    lower = [
        ("8. Show the answer", ["two result cards, top 3", "confidence and verdict", "rules fired, PSNR, SSIM"], "app"),
        ("7. CNN classifier", ["original and enhanced", "crop, 106 classes", "sign or light from YOLO"], "neural"),
        ("6. Enhance", ["Gaussian denoise", "→ RGB levels stretch", "→ gamma darken"], "image"),
        ("5. Fuzzy controller", ["14 Mamdani rules", "centroid defuzzification", "→ 3 strengths, 0 to 1"], "fuzzy"),
    ]
    for x, (title, lines, kind) in zip(xs, upper):
        box(axis, x, top, w, h, title, lines, kind)
    for x, (title, lines, kind) in zip(xs, lower):
        box(axis, x, bottom, w, h, title, lines, kind)
    for a in range(3):
        arrow(axis, (xs[a] + w + 0.03, top + h / 2), (xs[a + 1] - 0.03, top + h / 2))
        arrow(axis, (xs[a + 1] - 0.03, bottom + h / 2), (xs[a] + w + 0.03, bottom + h / 2))
    arrow(axis, (xs[3] + w / 2, top - 0.03), (xs[3] + w / 2, bottom + h + 0.03))
    # The original crop also goes straight to the classifier, for the side-by-side comparison
    arrow(axis, (xs[2] + w / 2, top - 0.03), (xs[1] + w / 2 + 0.25, bottom + h + 0.03), colour="#8C969F", width=1.0)
    axis.text(xs[2] + w / 2 + 0.08, top - 0.2, "original crop", fontsize=7, color=MUTED, ha="left", style="italic")
    legend(axis, 0.1, 0.17, ("data", "image", "fuzzy", "neural", "app"))
    save(figure, "fig_pipeline.png")


def architecture():
    """The four layers of the system, from datasets to the app."""
    width = 7.2
    figure, axis = canvas(width, 4.35)
    label_w, x0, gap = 0.82, 0.95, 0.07

    def layer(y, h, name):
        axis.add_patch(FancyBboxPatch((0.05, y), label_w, h, boxstyle="round,pad=0,rounding_size=0.06", linewidth=0, facecolor="#263238"))
        axis.text(0.05 + label_w / 2, y + h / 2, name, ha="center", va="center", color="white", fontsize=8.5, fontweight="bold", linespacing=1.25)

    rows = [
        (3.35, 0.92, "Data\nsources", [
            ("CURE-TSR", ["2,206,100 sign images", "14 classes, 12 damages"], "data"),
            ("LISA", ["41,923 light crops", "day and night"], "data"),
            ("GTSRB, Indian", ["real sign photos", "43 + 85 classes"], "data"),
            ("Mapillary", ["40,968 street photos", "six continents"], "data"),
            ("Kaggle, Roboflow", ["3,194 + 1,996 photos", "of damaged signs"], "data"),
        ]),
        (2.27, 0.92, "Prepar-\nation", [
            ("Pack arrays", ["32 × 32 × 3 NumPy", "merged class labels"], "data"),
            ("Add damage", ["dark, blur, noise, haze", "worn paint, 5 levels"], "image"),
            ("Leak-free splits", ["by clip, track, block", "gap before validation"], "data"),
            ("Enhanced copies", ["fuzzy controller", "on every image"], "fuzzy"),
            ("YOLO frames", ["1,280 px, 2 classes", "39,301 for training"], "app"),
        ]),
        (1.12, 1.0, "Core\nmodules", [
            ("image_processing", ["transforms, filters", "convolution, metrics"], "image"),
            ("fuzzy_logic", ["membership sets", "14 rules, controller"], "fuzzy"),
            ("deep_learning", ["MLP, feature MLP, CNN", "NumPy MLP by hand"], "neural"),
            ("YOLO26n", ["COCO-pretrained", "fine-tuned twice"], "app"),
        ]),
        (0.05, 0.92, "Evaluation\nand app", [
            ("Experiments", ["3 seeds per model", "choose on validation"], "neural"),
            ("Scored once", ["test sets untouched", "until the final run"], "neural"),
            ("262 checks", ["data, models, fuzzy", "image tools, app"], "data"),
            ("Streamlit app", ["4 tabs, YOLO or box", "phone via a tunnel"], "app"),
        ]),
    ]
    for y, h, name, blocks in rows:
        layer(y, h, name)
        w = (width - 0.05 - x0 - gap * (len(blocks) - 1)) / len(blocks)
        for i, (title, lines, kind) in enumerate(blocks):
            box(axis, x0 + i * (w + gap), y, w, h, title, lines, kind, 8, 6.8)
    for y in (3.35, 2.27, 1.12):
        arrow(axis, (0.46, y - 0.005), (0.46, y - 0.13), colour=MUTED, width=1.5)
    save(figure, "fig_architecture.png")


def cnn():
    """The small CNN, block by block, with the shape after each stage."""
    width = 7.2
    figure, axis = canvas(width, 2.15)
    stages = [
        ("Input", ["RGB crop"], "3 × 32 × 32", 1.5, "data"),
        ("Block 1", ["2 × conv 3×3, 32", "batch norm, ReLU", "max pool 2×2"], "32 × 16 × 16", 1.42, "neural"),
        ("Block 2", ["2 × conv 3×3, 64", "batch norm, ReLU", "max pool 2×2"], "64 × 8 × 8", 1.28, "neural"),
        ("Block 3", ["2 × conv 3×3, 128", "batch norm, ReLU", "max pool 2×2"], "128 × 4 × 4", 1.14, "neural"),
        ("Classifier", ["flatten 2,048", "dropout 0.3, FC 256", "ReLU, dropout 0.3"], "256", 1.14, "neural"),
        ("Output", ["FC + softmax"], "17 or 106", 1.0, "app"),
    ]
    w, gap = 1.0, 0.208
    mid = 1.25
    x = 0.05
    for i, (title, lines, shape, h, kind) in enumerate(stages):
        box(axis, x, mid - h / 2, w, h, title, lines, kind, 8.5, 6.8)
        axis.text(x + w / 2, mid - h / 2 - 0.06, shape, ha="center", va="top", fontsize=7.4, color=COURSE[kind][0], fontweight="bold")
        if i:
            arrow(axis, (x - gap + 0.02, mid), (x - 0.02, mid))
        x += w + gap
    axis.text(width / 2, 0.08, "816,369 weights for 17 classes and 839,242 for 106. AdamW, learning rate 0.001, cosine schedule, batch 256, class-weighted loss.", ha="center", fontsize=7.3, color=MUTED)
    save(figure, "fig_cnn.png")


def fuzzy():
    """Mamdani inference, from three measurements to three enhancement strengths."""
    width = 7.2
    figure, axis = canvas(width, 1.9)
    steps = [
        ("Measure", ["brightness 0 to 1", "contrast 0 to 1", "noise 0 to 60"], "image"),
        ("Fuzzify", ["3 sets per input", "trapezoids and", "triangles"], "fuzzy"),
        ("14 rules", ["IF bright AND", "contrast medium", "THEN darken strong"], "fuzzy"),
        ("Defuzzify", ["clip output sets", "combine with max", "centroid, 0 to 1"], "fuzzy"),
        ("Enhance", ["denoise first", "then stretch", "then darken"], "image"),
    ]
    w, gap, y, h = 1.22, 0.255, 0.38, 1.35
    x = 0.05
    for i, (title, lines, kind) in enumerate(steps):
        box(axis, x, y, w, h, title, lines, kind, 9, 7.2)
        if i:
            arrow(axis, (x - gap + 0.03, y + h / 2), (x - 0.03, y + h / 2))
        x += w + gap
    axis.text(width / 2, 0.1, "Written in NumPy for whole batches; within 0.0001 of scikit-fuzzy's engine on 100 random inputs.", ha="center", fontsize=7.3, color=MUTED)
    save(figure, "fig_fuzzy.png")


def protocol():
    """How the models grew, and the rule that keeps test scores honest."""
    width = 7.2
    figure, axis = canvas(width, 2.55)
    stages = [
        ("17-class models", ["CURE-TSR + LISA", "MLP, feature MLP, CNN", "36 runs"], "neural"),
        ("84-class wide CNN", ["+ GTSRB", "+ Indian set", "6 runs"], "neural"),
        ("106-class global CNN", ["+ Mapillary", "+ 'other sign'", "6 runs, 30 epochs"], "neural"),
        ("Global CNN + wear", ["+ worn-paint copies", "+ Kaggle photos", "3 runs, app default"], "neural"),
    ]
    w, gap, y, h = 1.6, 0.2, 1.3, 1.0
    x = 0.05
    for i, (title, lines, kind) in enumerate(stages):
        box(axis, x, y, w, h, title, lines, kind, 8.5, 7)
        if i:
            arrow(axis, (x - gap + 0.03, y + h / 2), (x - 0.03, y + h / 2))
        x += w + gap
    yolo = [("YOLO26n", ["COCO-pretrained"]), ("Detector 1", ["13 epochs, LISA + Mapillary"]), ("Detector 2", ["+10 epochs, damaged signs"])]
    w2, gap2 = 1.95, 0.32
    x = (width - 3 * w2 - 2 * gap2) / 2
    for i, (title, lines) in enumerate(yolo):
        box(axis, x, 0.48, w2, 0.62, title, lines, "app", 8.5, 7)
        if i:
            arrow(axis, (x - gap2 + 0.03, 0.79), (x - 0.03, 0.79))
        x += w2 + gap2
    axis.text(width / 2, 0.14, "Each model: train → choose on validation (3 seeds, mean ± spread) → score the test sets once → never change it after", ha="center", fontsize=7.6, color=INK, fontweight="bold")
    save(figure, "fig_protocol.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pipeline()
    architecture()
    cnn()
    fuzzy()
    protocol()
