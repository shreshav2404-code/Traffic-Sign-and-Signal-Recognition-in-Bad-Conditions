"""Check that the report PDF and the slides quote the numbers in results/, rounded as written.

Run with a Python that has pandas, pymupdf and python-pptx. Prints each check and fails on any miss.
"""
import csv
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pymupdf
from pptx import Presentation

HERE = Path(__file__).resolve().parent
FINAL = HERE.parents[1] / "results" / "final"
report = " ".join(page.get_text() for page in pymupdf.open(HERE / "Traffic_Sign_Recognition_Report.pdf"))
report = " ".join(report.split())
slides = " ".join(shape.text_frame.text for slide in Presentation(HERE / "Traffic_Sign_Recognition_Presentation.pptx").slides for shape in slide.shapes if shape.has_text_frame)
for slide in Presentation(HERE / "Traffic_Sign_Recognition_Presentation.pptx").slides:
    for shape in slide.shapes:
        if shape.has_table:
            slides += " " + " ".join(cell.text for row in shape.table.rows for cell in row.cells)
slides = " ".join(slides.split())


def rows(name):
    with open(FINAL / name, newline="") as file:
        return list(csv.DictReader(file))


failed = 0


def expect(text, where, why):
    global failed
    found = text in (report if where == "report" else slides)
    failed += not found
    print(f"{'ok ' if found else 'MISSING'}  {where:<7} {text:<10} {why}")


# 17-class models on real signs, simulated signs and lights (original input unless stated)
summary = rows("summary_test.csv")
def f1(model, trained, source, group):
    row = next(r for r in summary if r["model"] == model and r["trained_on"] == trained and r["input"] == source and r["group"] == group and r["subset"] == "overall")
    return float(row["f1_mean"]), float(row["accuracy_mean"])

for model, trained, source in (("cnn", "all", "original"), ("cnn", "all enhanced", "fuzzy"), ("cnn", "clean", "fuzzy"), ("cnn", "clean", "original"), ("mlp", "all", "original"), ("feature_mlp", "all", "original")):
    for group in ("signs_real", "signs_simulated", "lights"):
        value, _ = f1(model, trained, source, group)
        expect(f"{value:.3f}", "report", f"{model} {trained} {source} {group}")
value, accuracy = f1("cnn", "all", "original", "signs_real")
expect(f"{accuracy:.1%}", "report", "best CNN accuracy, real signs")
expect(f"{f1('cnn', 'all', 'original', 'lights')[1]:.1%}", "report", "best CNN accuracy, lights")
expect(f"{f1('cnn', 'clean', 'original', 'lights')[1]:.1%}", "report", "clean CNN light accuracy")
expect(f"{f1('cnn', 'clean', 'fuzzy', 'lights')[1]:.1%}", "report", "clean CNN + fuzzy light accuracy")

# Global CNNs with and without wear, test sets
for row in rows("global_wear_summary_test.csv"):
    if row["input"] == "original" and row["trained_on"] in ("all", "all + wear"):
        expect(f"{float(row['f1_mean']):.3f}", "report", f"global {row['trained_on']} {row['group']}")
for group, trained in (("signs_real", "all + wear"), ("worn_signs_real", "all + wear"), ("worn_signs_real", "all"), ("lights", "all + wear")):
    row = next(r for r in rows("global_wear_summary_test.csv") if r["input"] == "original" and r["trained_on"] == trained and r["group"] == group)
    expect(f"{float(row['f1_mean']):.3f}", "slides", f"global {trained} {group}")

# Wide CNN test
for row in rows("wide_summary_test.csv"):
    if row["input"] == "original" and row["trained_on"] == "all":
        expect(f"{float(row['f1_mean']):.3f}", "report", f"wide all {row['group']}")

# Detectors
for name in ("detector_test.csv", "detector_wear_test.csv"):
    for row in rows(name):
        for key in ("precision", "recall", "map50"):
            expect(f"{float(row[key]):.3f}", "report", f"{name} {row['dataset']} {key}")
for row in rows("detector_wear_test.csv"):
    expect(f"{float(row['map50']):.3f}", "slides", f"detector 2 {row['dataset']} mAP50")

# Image quality and the fixed setting
best = json.loads((FINAL / "fixed_setting.json").read_text())["best"]
expect(f"{best['f1']:.3f}", "report", "best fixed setting on validation")
for row in rows("study_restoration.csv"):
    if row["filter"] in ("Median",) and row["damage"].startswith("Salt"):
        expect(f"{float(row['psnr']):.2f}", "report", f"median filter, {row['damage']}")
for row in rows("mlp_study.csv"):
    # The file keeps 4 decimals, so 0.5255 is rounded half up, as the report does
    value = Decimal(row["best_val_f1"]).quantize(Decimal("0.001"), ROUND_HALF_UP)
    expect(str(value), "report", f"MLP study {row['study']} {row['hidden']} {row['optimizer']} {row['learning_rate']} {row['activation']}")

print(f"\n{'All numbers found' if not failed else f'{failed} numbers missing'}")
raise SystemExit(1 if failed else 0)
