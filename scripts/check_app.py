"""Run the Streamlit app headlessly through its main paths and check the photo helpers."""
import io
from pathlib import Path

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

from image_processing import photos

ROOT = Path(__file__).resolve().parents[1]
passed = 0


def check(name, condition, detail=""):
    global passed
    if not condition:
        raise SystemExit(f"FAILED: {name} {detail}")
    passed += 1
    print(f"ok  {name} {detail}")


def run_clean(app, name):
    app.run(timeout=120)
    messages = [e.value for e in app.exception]
    check(name, not messages, str(messages)[:400])


# Photo helpers
photo = np.zeros((400, 600, 3), np.uint8)
photo[100:300, 200:400] = (200, 30, 30)
buffer = io.BytesIO()
Image.fromarray(photo).save(buffer, "PNG")
buffer.seek(0)
check("uploaded photo loads as RGB", photos.load_photo(buffer).shape == (400, 600, 3))
box = photos.crop_box(photo.shape, {"left": 150, "top": 100, "width": 300, "height": 200})
check("crop box is in pixels", box == (150, 100, 450, 300), str(box))
outside = photos.crop_box(photo.shape, {"left": 500, "top": 300, "width": 300, "height": 300})
check("crop box stays inside the photo", outside == (500, 300, 600, 400), str(outside))
cropped = photos.crop_and_resize(photo, photos.crop_box(photo.shape, {"left": 200, "top": 100, "width": 200, "height": 200}))
check("crop resizes to 32x32", cropped.shape == (32, 32, 3) and cropped.dtype == np.uint8)
check("crop keeps the red sign", cropped[..., 0].mean() > 150 and cropped[..., 1].mean() < 60)
wide = np.zeros((200, 800, 3), np.uint8)
wide[50:150, 350:450] = 255
shrunk = photos.crop_and_resize(wide, (0, 0, 800, 200))
rows, columns = np.nonzero(shrunk[..., 0] > 128)
check("a wide photo keeps its shape", abs((rows.max() - rows.min()) - (columns.max() - columns.min())) <= 1)
try:
    photos.crop_box(photo.shape, {"left": 50, "top": 50, "width": 3, "height": 3})
    check("tiny crop is refused", False)
except ValueError:
    check("tiny crop is refused", True)

# App paths
app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
run_clean(app, "app starts on the default sample")
check("all four tabs exist", len(app.tabs) == 4)
check("two result cards are drawn", sum('class="card-sign"' in m.value for m in app.markdown) == 2)
check("the verdict is drawn", any('class="verdict"' in m.value for m in app.markdown))
check("three measurements are drawn", any(m.label == "Brightness" for m in app.metric))

classifier = next(s for s in app.selectbox if s.label == "Classifier")
for name in classifier.options:
    next(s for s in app.selectbox if s.label == "Classifier").set_value(name)
    run_clean(app, f"classifier works: {name}")

condition = next(s for s in app.selectbox if s.label == "Condition")
for name in condition.options:
    next(s for s in app.selectbox if s.label == "Condition").set_value(name)
    run_clean(app, f"condition works: {name}")

sign = next(s for s in app.selectbox if s.label == "Sign or light")
for name in sign.options:
    next(s for s in app.selectbox if s.label == "Sign or light").set_value(name)
    run_clean(app, f"class works: {name}")

next(s for s in app.selectbox if s.label == "Sign or light").set_value("Red light")
run_clean(app, "red light opens")
check("all four demo classifiers are offered", len(next(s for s in app.selectbox if s.label == "Classifier").options) == 4)
check("lights offer only their damage types", len(next(s for s in app.selectbox if s.label == "Condition").options) == 5)
next(r for r in app.radio if r.label == "Scene").set_value("Night")
run_clean(app, "night scene works")
next(s for s in app.selectbox if s.label == "Sign or light").set_value("Stop")
run_clean(app, "back to a sign")
next(s for s in app.selectbox if s.label == "Condition").set_value("Darkening")
run_clean(app, "back to darkening")
for value in (1, 5):
    next(s for s in app.slider if s.label == "Damage level").set_value(value)
    run_clean(app, f"level {value} works")
next(s for s in app.slider if s.label == "Gamma").set_value(2.0)
run_clean(app, "gamma slider works")

check("the final results table is drawn", any(len(d.value) == 24 for d in app.dataframe))
check("the wide CNN results table is drawn", any(len(d.value) == 4 and "GTSRB" in d.value.columns for d in app.dataframe))
# Clean, all conditions, and all conditions with worn signs, each on original and fuzzy input
check("the global CNN results table is drawn", any(len(d.value) == 6 and "Mapillary" in d.value.columns for d in app.dataframe))
check("the worn-sign results table is drawn", any(len(d.value) == 6 and "Worn real signs" in d.value.columns for d in app.dataframe))
# Both detectors, each on LISA, Mapillary and the two damaged-sign sets
check("the YOLO results table is drawn", any(len(d.value) == 8 and "Detector" in d.value.columns for d in app.dataframe))
for name in next(s for s in app.selectbox if s.label == "Model").options:
    next(s for s in app.selectbox if s.label == "Model").set_value(name)
    run_clean(app, f"results work: {name}")

next(r for r in app.radio if r.label == "Image source").set_value("Upload photo")
run_clean(app, "upload mode opens without a file")
check("upload mode asks for a photo", any("Upload a photo" in i.value for i in app.info))
check("upload mode shows the known classes", any(e.label == "Signs and lights the model knows" for e in app.expander))

print(f"\nAll {passed} checks passed.")
