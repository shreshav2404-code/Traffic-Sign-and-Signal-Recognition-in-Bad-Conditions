"""Streamlit demo: read traffic signs from damaged images. Run: streamlit run app.py"""
import importlib.util
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from streamlit_cropper import st_cropper

from deep_learning import detector
from deep_learning.data import CLASS_NAMES, CONDITIONS
from deep_learning.evaluate import predict
from deep_learning.models import CLASSES, build_model, uses_features
from deep_learning.global_classes import GLOBAL_NAMES
from deep_learning.wide_classes import WIDE_NAMES
from fuzzy_logic import controller
from fuzzy_logic.membership import INPUT_SETS, INPUT_UNIVERSES, degree
from fuzzy_logic.rules import RULES, describe
from image_processing import convolution, enhancement, features, photos, quality, restoration, transformations

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "assets" / "samples.npz"
RESULTS = ROOT / "results" / "final" / "summary_test.csv"
CLASS_RESULTS = ROOT / "results" / "final" / "classes_test.csv"
BIGGER_RESULTS = {
    "Wide CNN, 84 classes": ROOT / "results" / "final" / "wide_summary_test.csv",
    "Global CNN, 106 classes": ROOT / "results" / "final" / "global_wear_summary_test.csv",
}
DETECTOR_RESULTS = {
    "First": ROOT / "results" / "final" / "detector_test.csv",
    "Fine-tuned on damaged signs": ROOT / "results" / "final" / "detector_wear_test.csv",
}
DEVICE = torch.device("cpu")
DISPLAY_SIZE = 256
TOOL_SIZE = 160
CONFIDENCE_WARNING = 0.6
DETECTOR_MARGIN = 1.15
FIND_AUTOMATICALLY = "Find them with YOLO"

# Example images for the bigger class lists, by number of classes
EXAMPLES = {len(WIDE_NAMES): ROOT / "assets" / "wide_examples.npz", len(GLOBAL_NAMES): ROOT / "assets" / "global_examples.npz"}

# Display name: (architecture, checkpoint file, class names, one-line description)
# Kept to four for the demo. Every other model's scores are in the results tab and the reports.
MODELS = {
    "Global CNN, 106 classes, worn signs too": ("cnn", "cnn_global_wear_all_seed0", GLOBAL_NAMES, "Knows 106 signs and lights from six continents, can answer 'other sign', and has learned from damaged images and from faded, peeled and rusted signs."),
    "Global CNN, 106 classes, clean images": ("cnn", "cnn_global_clean_seed0", GLOBAL_NAMES, "Knows 106 classes but has never seen damage, so it shows what enhancement can do."),
    "17-class CNN, all conditions": ("cnn", "cnn_lights_all_seed0", CLASS_NAMES, "The best model from the main 17-class experiments. It has learned from damaged images."),
    "17-class MLP, all conditions": ("mlp", "mlp_lights_all_seed0", CLASS_NAMES, "Raw pixels through two hidden layers, to compare with the CNN."),
}
MODEL_LABELS = {"mlp": "MLP", "feature_mlp": "Feature MLP", "cnn": "CNN"}
TRAINED_LABELS = {"clean": "Clean images", "all": "All conditions", "all + wear": "All conditions and worn signs", "clean enhanced": "Enhanced clean images", "all enhanced": "Enhanced images, all conditions"}
INPUT_LABELS = {"original": "Original", "fixed": "Fixed stretch", "fuzzy": "Fuzzy"}
DATASET_LABELS = {"signs_real": "Real signs", "signs_simulated": "Simulated signs", "lights": "Lights", "gtsrb": "GTSRB", "indian": "Indian set", "mapillary": "Mapillary"}
WORN_LABELS = {
    "worn_signs_real": "Worn real signs", "worn_signs_simulated": "Worn simulated signs", "worn_gtsrb": "Worn GTSRB",
    "worn_indian": "Worn Indian set", "worn_mapillary": "Worn Mapillary", "damaged_photos": "Real damaged photos",
}
LIGHT_COLUMNS = {"overall": "All lights", "day": "Day", "night": "Night", "red light": "Red", "yellow light": "Yellow", "green light": "Green"}
DETECTOR_COLUMNS = {"dataset": "Dataset", "class": "Finds", "frames": "Test frames", "precision": "Precision", "recall": "Recall", "map50": "mAP50", "map50_95": "mAP50-95"}

STYLE = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1240px;}
h1 {font-size: 2.1rem !important; letter-spacing: -0.02em;}
.lead {color: #A9B2BD; font-size: 1.02rem; margin: -0.5rem 0 1.2rem;}
.step {color: #F5B301; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; margin: 1.4rem 0 0.5rem;}
.card {background: #1A212B; border: 1px solid #2B3440; border-radius: 12px; padding: 1.1rem 1.2rem; height: 100%;}
.card.good {border-color: #2E8B57;}
.card.bad {border-color: #C0504D;}
.card-title {color: #A9B2BD; font-size: 0.85rem; margin-bottom: 0.3rem;}
.card-sign {font-size: 1.5rem; font-weight: 700; margin-bottom: 0.2rem;}
.card-note {font-size: 0.88rem; margin-bottom: 0.7rem;}
.card-note.good {color: #6FCF97;}
.card-note.bad {color: #EB8A86;}
.card-note.warn {color: #F5B301;}
.bar-label {display: flex; justify-content: space-between; font-size: 0.82rem; color: #C9D0D8; margin-top: 0.45rem;}
.bar-track {background: #0F1419; border-radius: 4px; height: 7px; overflow: hidden;}
.bar-fill {background: #F5B301; height: 100%;}
.verdict {background: #1A212B; border-left: 3px solid #F5B301; border-radius: 6px; padding: 0.7rem 1rem; margin-top: 0.9rem;}
</style>
"""


@st.cache_data
def load_samples():
    with np.load(SAMPLES) as data:
        return {name: data[name] for name in data.files}


@st.cache_data
def class_examples(count):
    """One clean sample image of every class a model knows."""
    if count in EXAMPLES:
        with np.load(EXAMPLES[count]) as data:
            return [enlarge(image, 96) for image in data["images"]]
    samples = load_samples()
    rows = [np.flatnonzero((samples["sign"] == sign) & (samples["challenge"] == 0))[0] for sign in range(count)]
    return [enlarge(samples["images"][row], 96) for row in rows]


@st.cache_resource
def load_model(architecture, checkpoint, classes):
    model = build_model(architecture, classes)
    state = torch.load(ROOT / "models" / f"{checkpoint}.pt", map_location=DEVICE, weights_only=True)["state"]
    model.load_state_dict(state)
    return model.eval()


@st.cache_data
def load_results(path):
    return pd.read_csv(path) if path.exists() else None


def available_models():
    needs = lambda spec: [ROOT / "models" / f"{spec[1]}.pt"] + ([EXAMPLES[len(spec[2])]] if len(spec[2]) in EXAMPLES else [])
    return {name: spec for name, spec in MODELS.items() if all(path.exists() for path in needs(spec))}


def classify(spec, image, kind=None):
    """Score every class, or with kind "sign" or "light" from YOLO, only the classes of that kind."""
    architecture, checkpoint, names, _ = spec
    inputs = features.extract_features(image)[np.newaxis] if uses_features(architecture) else image[np.newaxis]
    probabilities = predict(load_model(architecture, checkpoint, len(names)), inputs, DEVICE)[0]
    return probabilities if kind is None else detector.keep_kind(probabilities, names, kind)


def enlarge(image, size=DISPLAY_SIZE):
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_NEAREST)


def strongest_set(variable, value):
    """Name the fuzzy set a measurement belongs to most."""
    return max(INPUT_SETS[variable], key=lambda label: float(degree(variable, label, value)))


def choose_sample():
    samples = load_samples()
    sign = st.selectbox("Sign or light", range(len(CLASS_NAMES)), index=5, format_func=lambda s: CLASS_NAMES[s].capitalize())
    group = samples["sign"] == sign
    if sign >= CLASSES:
        scene = st.radio("Scene", ("Day", "Night"), horizontal=True)
        group &= samples["night"] == (scene == "Night")

    present = sorted(set(samples["challenge"][group].tolist()))
    condition = st.selectbox(
        "Condition",
        present,
        index=present.index(4),
        format_func=lambda c: CONDITIONS[c].capitalize(),
    )
    level = 0
    if condition != 0:
        level = st.slider("Damage level", 1, 5, 3)

    chosen = group & (samples["challenge"] == condition) & (samples["level"] == level)
    reference = group & (samples["challenge"] == 0)
    if not chosen.any():
        st.warning("That image isn't in the sample set, so this shows the clean original.")
        chosen = reference
    return samples["images"][chosen][0], samples["images"][reference][0], sign


def choose_upload():
    file = st.file_uploader("Upload a photo", type=["png", "jpg", "jpeg", "bmp"])
    if file is None:
        return None, None
    try:
        return photos.load_photo(file), file.file_id
    except (UnidentifiedImageError, OSError):
        st.error("That file couldn't be read as an image.")
        return None, None


def known_classes(expanded, names):
    with st.expander("Signs and lights the model knows", expanded=expanded):
        st.caption(f"This model learned only these {len(names)} classes. Any other sign is still named as one of them, sometimes confidently.")
        examples = class_examples(len(names))
        width = 6 if len(names) <= len(CLASS_NAMES) else 8
        for start in range(0, len(examples), width):
            for column, sign in zip(st.columns(width), range(start, min(start + width, len(examples)))):
                column.image(examples[sign], caption=names[sign].capitalize(), width="stretch")


def crop_photo(photo, photo_id):
    """Let the user drag a box around one sign, and return it at model size."""
    st.markdown('<div class="step">Select one sign or light</div>', unsafe_allow_html=True)
    st.caption("Drag the box and its corners, so it fills most of the box.")
    picker, preview = st.columns([3, 1])
    with picker:
        box = st_cropper(
            Image.fromarray(photo), box_color="#F5B301", aspect_ratio=(1, 1), return_type="box", key=f"crop-{photo_id}"
        )
    try:
        image = photos.crop_and_resize(photo, photos.crop_box(photo.shape, box))
    except ValueError as error:
        st.warning(str(error))
        return None
    with preview:
        st.image(enlarge(image, TOOL_SIZE), caption="What the model sees", width="stretch")
    return image


def detector_ready():
    return detector.DETECTOR.exists() and importlib.util.find_spec("ultralytics") is not None


@st.cache_resource
def get_detector():
    return detector.load_detector()


@st.cache_data
def detect(photo_id, _photo):
    return detector.find_objects(get_detector(), _photo)


def mark_objects(photo, found):
    """Draw each box with its number, sized to the photo."""
    marked = photo.copy()
    thickness = max(2, max(photo.shape[:2]) // 400)
    for number, (x0, y0, x1, y1, _, _) in enumerate(found, 1):
        cv2.rectangle(marked, (x0, y0), (x1, y1), (245, 179, 1), thickness)
        cv2.putText(marked, str(number), (x0, max(0, y0 - 3 * thickness)), cv2.FONT_HERSHEY_SIMPLEX, thickness / 3, (245, 179, 1), thickness)
    return marked


def pick_detection(photo, photo_id, spec):
    """Let YOLO find every sign and light, name each one, and return the one the user picks with its kind."""
    found = detect(photo_id, photo)
    if not found:
        st.warning("YOLO found no signs or lights here, so drag a box yourself.")
        return crop_photo(photo, photo_id), None

    st.markdown('<div class="step">Signs and lights found by YOLO</div>', unsafe_allow_html=True)
    crops = [photos.crop_and_resize(photo, photos.square_box(photo.shape, item[:4], DETECTOR_MARGIN)) for item in found]
    names = spec[2]
    rows = []
    for number, (item, crop) in enumerate(zip(found, crops), 1):
        settings = {name: float(value[0]) for name, value in controller.decide(**controller.measure(crop)).items()}
        before, after = classify(spec, crop, item[4]), classify(spec, enhancement.enhance(crop, **settings), item[4])
        rows.append({
            "Number": number, "YOLO found": item[4], "YOLO confidence": f"{item[5]:.0%}",
            "Original image": names[int(before.argmax())].capitalize(), "Enhanced image": names[int(after.argmax())].capitalize(),
        })
    picture, table = st.columns([3, 2])
    picture.image(mark_objects(photo, found), width="stretch")
    with table:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("The classifier chooses only among lights when YOLO found a light, and only among signs when it found a sign.")
        number = st.selectbox("Look closer at", range(len(found)), format_func=lambda n: f"{n + 1}. {rows[n]['Enhanced image']}")
        st.image(enlarge(crops[number], TOOL_SIZE), caption="What the model sees")
    return crops[number], found[number][4]


def result_card(title, probabilities, expected, names):
    order = np.argsort(probabilities)[::-1][:3]
    best = int(order[0])
    confidence = float(probabilities[best])

    status, note = "", ""
    if expected is not None:
        status = "good" if best == expected else "bad"
        note = "Correct" if best == expected else f"Wrong, this is {names[expected]}"
    if confidence < CONFIDENCE_WARNING:
        note = f"{note}. Low confidence" if note else "Low confidence"
    note_class = status or "warn"

    bars = "".join(
        f'<div class="bar-label"><span>{names[c].capitalize()}</span><span>{probabilities[c]:.1%}</span></div>'
        f'<div class="bar-track"><div class="bar-fill" style="width:{probabilities[c] * 100:.1f}%"></div></div>'
        for c in order
    )
    st.markdown(
        f'<div class="card {status}"><div class="card-title">{title}</div>'
        f'<div class="card-sign">{names[best].capitalize()}</div>'
        f'<div class="card-note {note_class}">{note}&nbsp;</div>{bars}</div>',
        unsafe_allow_html=True,
    )
    return best


def verdict(original, improved, expected, unsure, count):
    if unsure:
        return f"Neither prediction is confident, so this may not be one of the {count} known signs or lights."
    if expected is not None:
        if original != expected and improved == expected:
            return "Enhancement fixed this prediction."
        if original == expected and improved != expected:
            return "Enhancement broke this prediction, so it isn't always the right choice."
        if original == expected:
            return "Both predictions are correct."
        return "Both predictions are wrong, so enhancement didn't save this image."
    return "Both predictions agree." if original == improved else "Enhancement changed the prediction."


def show_quality(reference, damaged, improved):
    if np.array_equal(reference, damaged):
        st.caption("This is the clean original, so there's no damage to measure.")
        return
    columns = st.columns(2)
    columns[0].metric("PSNR", f"{quality.psnr(reference, improved):.1f} dB", f"{quality.psnr(reference, improved) - quality.psnr(reference, damaged):+.1f} dB")
    columns[1].metric("SSIM", f"{quality.ssim(reference, improved):.3f}", f"{quality.ssim(reference, improved) - quality.ssim(reference, damaged):+.3f}")


def demo_tab(image, reference, expected, spec, measured, settings, stages, kind=None):
    st.markdown('<div class="step">Step 1: Measure the image</div>', unsafe_allow_html=True)
    columns = st.columns(3)
    columns[0].metric("Brightness", f"{measured['brightness']:.2f}", strongest_set("brightness", measured["brightness"]).capitalize(), delta_color="off", delta_arrow="off")
    columns[1].metric("Contrast", f"{measured['contrast']:.2f}", strongest_set("contrast", measured["contrast"]).capitalize(), delta_color="off", delta_arrow="off")
    columns[2].metric("Noise", f"{measured['noise']:.1f}", strongest_set("noise", measured["noise"]).capitalize(), delta_color="off", delta_arrow="off")

    st.markdown('<div class="step">Step 2: Fuzzy logic chooses, then each stage runs</div>', unsafe_allow_html=True)
    captions = (
        "Original",
        f"Denoise {settings['denoise']:.0%}",
        f"Stretch {settings['stretch']:.0%}",
        f"Darken {settings['darken']:.0%}",
    )
    for column, stage, caption in zip(st.columns(4), stages, captions):
        column.image(enlarge(stage), caption=caption, width="stretch")

    st.markdown('<div class="step">Step 3: Classify both images</div>', unsafe_allow_html=True)
    names = spec[2]
    if kind is not None:
        st.caption(f"YOLO found a {kind} here, so the classifier chooses only among the {kind}s it knows.")
    before, after = classify(spec, image, kind), classify(spec, stages[-1], kind)
    unsure = expected is None and max(before.max(), after.max()) < CONFIDENCE_WARNING
    columns = st.columns(2)
    with columns[0]:
        original = result_card("Original image", before, expected, names)
    with columns[1]:
        improved = result_card("Enhanced image", after, expected, names)
    st.markdown(f'<div class="verdict">{verdict(original, improved, expected, unsure, len(names))}</div>', unsafe_allow_html=True)

    if reference is not None:
        st.markdown('<div class="step">Image quality against the clean original</div>', unsafe_allow_html=True)
        show_quality(reference, image, stages[-1])
    else:
        st.caption("Models learned from small dataset crops, so photos may be misread. Crop tightly around one sign or light, and check it is one of the known classes shown above.")


def fuzzy_tab(measured, settings):
    st.markdown("#### Where this image falls in each fuzzy set")
    figure, axes = plt.subplots(1, 3, figsize=(12, 2.8), facecolor="none")
    for axis, (variable, value) in zip(axes, measured.items()):
        axis.set_facecolor("none")
        for label, values in INPUT_SETS[variable].items():
            axis.plot(INPUT_UNIVERSES[variable], values, label=label)
        axis.axvline(value, color="#F5B301", linewidth=2, linestyle="--")
        axis.set_title(f"{variable.capitalize()}: {value:.2f}", color="#E8EAED", fontsize=11)
        axis.tick_params(colors="#A9B2BD", labelsize=8)
        for spine in axis.spines.values():
            spine.set_color("#2B3440")
        axis.legend(fontsize=8, frameon=False, labelcolor="#C9D0D8")
    figure.tight_layout()
    st.pyplot(figure, clear_figure=True, width="stretch")

    st.markdown("#### Rules that fired")
    firing = controller.firing_strengths(*(np.array([measured[name]]) for name in ("brightness", "contrast", "noise")))[0]
    fired = sorted(((float(s), describe(rule)) for s, rule in zip(firing, RULES) if s > 0), reverse=True)
    st.dataframe(
        pd.DataFrame(fired, columns=["Strength", "Rule"]),
        hide_index=True,
        width="stretch",
        column_config={"Strength": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1, width="small")},
    )

    st.markdown("#### Final strengths after defuzzification")
    columns = st.columns(3)
    for column, name in zip(columns, ("denoise", "stretch", "darken")):
        column.metric(name.capitalize(), f"{settings[name]:.2f}")


def tools_tab(image, reference):
    st.markdown("#### Operations on this image")
    gray = transformations.to_gray(image)
    gamma = st.slider("Gamma", 0.2, 3.0, 0.5, 0.1)
    operations = {
        "Negative": transformations.negative(image),
        f"Gamma {gamma:.1f}": transformations.gamma_correction(image, gamma),
        "Histogram equalized": transformations.equalize_histogram(image),
        "CLAHE": transformations.clahe(image),
        "Contrast stretch": transformations.stretch_levels(image),
        "Gaussian filter": restoration.gaussian_filter(image),
        "Median filter": restoration.median_filter(image),
        "Bilateral filter": restoration.bilateral_filter(image),
        "Non-local means": restoration.non_local_means(image),
        "Sharpen kernel": convolution.apply_kernel(image, "sharpen"),
        "Sobel edges": convolution.sobel_edges(gray),
        "Grayscale": gray,
    }
    names = list(operations)
    for start in range(0, len(names), 4):
        for column, name in zip(st.columns(4), names[start : start + 4]):
            result = operations[name]
            caption = name
            if reference is not None and result.ndim == 3 and not np.array_equal(reference, result):
                caption = f"{name}: {quality.psnr(reference, result):.1f} dB"
            column.image(enlarge(result, TOOL_SIZE), caption=caption, width="stretch")

    st.markdown("#### Brightness histogram")
    histogram = quality.intensity_histogram(image)
    chart = pd.DataFrame({"Pixels": np.add.reduce(histogram.reshape(32, 8), axis=1)}, index=np.arange(32) * 8)
    st.bar_chart(chart, x_label="Gray level", y_label="Pixels", color="#F5B301", height=220)


def overall_table(results):
    rows = []
    for (model, trained_on, source), group in results[results["subset"] == "overall"].groupby(["model", "trained_on", "input"], sort=False):
        scores = group.set_index("group")
        rows.append({
            "Model": MODEL_LABELS[model],
            "Trained on": TRAINED_LABELS[trained_on],
            "Input": INPUT_LABELS[source],
            "Real signs F1": f"{scores.loc['signs_real', 'f1_mean']:.3f} ± {scores.loc['signs_real', 'f1_std']:.3f}",
            "Simulated signs F1": f"{scores.loc['signs_simulated', 'f1_mean']:.3f} ± {scores.loc['signs_simulated', 'f1_std']:.3f}",
            "Lights F1": f"{scores.loc['lights', 'f1_mean']:.3f} ± {scores.loc['lights', 'f1_std']:.3f}",
            "order": scores.loc["signs_real", "f1_mean"],
        })
    return pd.DataFrame(rows).sort_values("order", ascending=False).drop(columns="order")


def lights_table(results, model, trained_on):
    rows = results[(results["model"] == model) & (results["trained_on"] == trained_on) & (results["group"] == "lights")]
    scenes = rows[rows["subset"].isin(LIGHT_COLUMNS)].pivot_table(index="input", columns="subset", values="accuracy_mean", sort=False)
    classes = load_results(CLASS_RESULTS)
    classes = classes[(classes["model"] == model) & (classes["trained_on"] == trained_on) & (classes["group"] == "lights")]
    colours = classes.pivot_table(index="input", columns="class", values="recall", sort=False)
    table = scenes.join(colours)[list(LIGHT_COLUMNS)].rename(columns=LIGHT_COLUMNS).reset_index()
    table["input"] = table["input"].map(INPUT_LABELS)
    return table.rename(columns={"input": "Input"})


def bigger_table(summary, labels=DATASET_LABELS):
    """Macro F1 on every dataset, for each training regime and input."""
    rows = []
    for (trained_on, source), group in summary.groupby(["trained_on", "input"], sort=False):
        scores = group.set_index("group")
        row = {"Trained on": TRAINED_LABELS[trained_on], "Input": INPUT_LABELS[source]}
        row.update({label: f"{scores.loc[g, 'f1_mean']:.3f} ± {scores.loc[g, 'f1_std']:.3f}" for g, label in labels.items() if g in scores.index})
        rows.append(row)
    return pd.DataFrame(rows)


def results_tab():
    results = load_results(RESULTS)
    if results is None:
        st.info("Run experiments/final_scores.py and experiments/summarize.py with --split test to fill this page.")
        return
    st.caption("Final test results. Each score is the mean ± spread over 3 training seeds, and the test sets were scored once.")
    st.markdown("#### Every model")
    st.dataframe(overall_table(results), hide_index=True, width="stretch")

    st.markdown("#### One model in detail")
    options = list(dict.fromkeys(zip(results["model"], results["trained_on"])))
    picked = st.selectbox(
        "Model",
        options,
        index=options.index(("cnn", "clean")),
        format_func=lambda o: f"{MODEL_LABELS[o[0]]}, {TRAINED_LABELS[o[1]].lower()}",
    )
    rows = results[(results["model"] == picked[0]) & (results["trained_on"] == picked[1]) & (results["group"] == "signs_real")]
    rows = rows[rows["subset"].str.startswith("condition: ")]
    chart = rows.pivot_table(index="subset", columns="input", values="f1_mean").rename(columns=INPUT_LABELS)
    chart.index = chart.index.str.removeprefix("condition: ").str.capitalize()
    st.markdown("**Macro F1 on real signs, by condition**")
    st.bar_chart(chart, horizontal=True, stack=False, height=560, x_label="", y_label="Macro F1", color=["#F5B301", "#6B7785", "#3D7DD8"][: len(chart.columns)])

    st.markdown("**Accuracy on traffic lights**")
    table = lights_table(results, *picked)
    columns = {name: st.column_config.NumberColumn(format="%.3f") for name in LIGHT_COLUMNS.values()}
    st.dataframe(table, hide_index=True, width="stretch", column_config=columns)

    for title, path in BIGGER_RESULTS.items():
        summary = load_results(path)
        if summary is None:
            continue
        st.markdown(f"#### {title}")
        st.caption("Macro F1 on each dataset's test images. With more classes to choose from, it scores lower on CURE-TSR than the 17-class models, which stay the main results.")
        st.dataframe(bigger_table(summary), hide_index=True, width="stretch")
        if summary["group"].isin(list(WORN_LABELS)).any():
            st.markdown("**Worn and damaged signs**")
            st.caption("Worn-paint copies of each test set at 5 levels, and real photos of stop, give way and no entry signs from Kaggle's Damaged Signs Dataset.")
            st.dataframe(bigger_table(summary, WORN_LABELS), hide_index=True, width="stretch")

    found = {name: load_results(path) for name, path in DETECTOR_RESULTS.items()}
    if any(table is not None for table in found.values()):
        detection = pd.concat([table.assign(detector=name) for name, table in found.items() if table is not None])
        detection = detection[["detector"] + [c for c in detection.columns if c != "detector"]]
        st.markdown("#### YOLO detector")
        st.caption("How well YOLO finds signs and lights in whole test frames. Each dataset is scored only on the class its people labelled, and a find counts when its box overlaps the labelled box by at least half (mAP50).")
        st.dataframe(detection.rename(columns={"detector": "Detector", **DETECTOR_COLUMNS}), hide_index=True, width="stretch")


def main():
    st.set_page_config(page_title="Sign and Signal Recognition", page_icon=":vertical_traffic_light:", layout="wide")
    st.markdown(STYLE, unsafe_allow_html=True)
    st.title("Traffic Sign and Signal Recognition in Bad Conditions")
    st.markdown('<p class="lead">Fuzzy logic measures each image, repairs it, and a neural network reads the sign or light.</p>', unsafe_allow_html=True)

    if not SAMPLES.exists():
        st.error("Sample images are missing. Run scripts/prepare_samples.py first.")
        st.stop()

    with st.sidebar:
        st.header("Setup")
        source = st.radio("Image source", ("Sample image", "Upload photo"), horizontal=True)
        image, reference, expected, photo, photo_id, finder, kind = None, None, None, None, None, None, None
        if source == "Sample image":
            image, reference, expected = choose_sample()
        else:
            photo, photo_id = choose_upload()
            if detector_ready():
                finder = st.radio("Signs in the photo", (FIND_AUTOMATICALLY, "Drag a box myself"))
        st.divider()
        models = available_models()
        if not models:
            st.error("No trained models were found.")
            st.stop()
        model_name = st.selectbox("Classifier", list(models))
        st.caption(models[model_name][3])

    if source == "Upload photo":
        known_classes(photo is None, models[model_name][2])
        if photo is None:
            st.info("Upload a photo in the sidebar to begin, or switch to a sample image.")
            st.stop()
        if finder == FIND_AUTOMATICALLY:
            image, kind = pick_detection(photo, photo_id, models[model_name])
        else:
            image = crop_photo(photo, photo_id)
        if image is None:
            st.stop()

    measured = controller.measure(image)
    settings = {name: float(value[0]) for name, value in controller.decide(**measured).items()}
    stages = (
        image,
        enhancement.enhance(image, denoise=settings["denoise"]),
        enhancement.enhance(image, denoise=settings["denoise"], stretch=settings["stretch"]),
        enhancement.enhance(image, **settings),
    )

    demo, fuzzy, tools, results = st.tabs(["Live demo", "Fuzzy decision", "Image tools", "Project results"])
    with demo:
        demo_tab(image, reference, expected, models[model_name], measured, settings, stages, kind)
    with fuzzy:
        fuzzy_tab(measured, settings)
    with tools:
        tools_tab(image, reference)
    with results:
        results_tab()


main()
