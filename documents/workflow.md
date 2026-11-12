# Project Workflow

How the project works, how to run it, and what's left.
Last updated: 4 November 2026.

## What the app does

```text
Upload a cropped sign or light, or a full photo where YOLO
     finds each sign and light, or the user drags a box
                         |
                         v
             Resize the image to 32x32
                         |
                         v
        Measure brightness, contrast and noise
                         |
                         v
        14 fuzzy rules choose three strengths
                         |
                         v
            Denoise -> Stretch -> Darken
                         |
                         v
          Classify the original and enhanced image
                         |
                         v
   Show before and after, predictions, confidence and rules fired
```

## Build progress

| Step | Status |
|---|---|
| 1. Faculty approval and report rules | Approval received, report rules **waiting on you** |
| 2. Project setup, GPU check and datasets | Done |
| 3. Image processing | Done, 43 checks pass |
| 4. Fuzzy controller | Done, 36 checks pass |
| 5. Classifiers, NumPy MLP and YOLO | Done, 108 checks pass |
| 5a. CNN on fuzzy-enhanced images | Done |
| 6. Streamlit app | Done, 75 checks pass |
| 7. Add LISA traffic lights | Done, 17 classes |
| 8. Final experiments on 3 seeds, using the test set once | Done, results in `results/final/` |
| 9. Try the app on real phone photos | 11 web street photos tested, phone photos in bad conditions **waiting on you** |
| 10. Wide 84-class CNN with GTSRB and an Indian set | Done, scored once on the test sets |
| 11. Global 106-class CNN with Mapillary | Done, scored once on the test sets, the app's default model |
| 12. YOLO detector for full photos | Done, scored once on the test sets, keeps boxes at 30% confidence or more |
| 13. Worn and damaged signs: worn-paint copies, Kaggle and Roboflow photos | Done, scored once on the test sets, the app's default CNN and detector |
| 14. Three reports and demo preparation | Not started |

## Folder structure

```text
mini_project/
├── app.py                    Streamlit app
├── assets/                   sample signs for the demo
├── .streamlit/               app theme
├── data/
│   ├── raw/                  CURE-TSR.zip, LISA.zip, GTSRB, the Indian set and Mapillary
│   ├── processed/            packed images, features and metadata
│   └── detection/            full frames and boxes for YOLO
├── deep_learning/
│   ├── data.py               loading, splits, class weights and augmentation
│   ├── models.py             MLP and small CNN
│   ├── train.py              training with early stopping
│   ├── evaluate.py           predictions and metrics
│   ├── numpy_mlp.py          hand-written backpropagation
│   ├── wide_classes.py       84 classes and the GTSRB and Indian mappings
│   ├── global_classes.py     106 classes and the Mapillary mapping
│   ├── detector.py           YOLO finds signs and lights in full photos
│   └── train_detector.py     YOLO training
├── documents/                decisions and workflow
├── experiments/              36 training runs, final scoring, studies and charts
├── fuzzy_logic/
│   ├── membership.py         fuzzy sets
│   ├── rules.py              14 rules
│   └── controller.py         Mamdani inference
├── image_processing/
│   ├── transformations.py    negative, gamma, stretching and equalization
│   ├── restoration.py        noise and filters
│   ├── convolution.py        hand-written convolution and kernels
│   ├── quality.py            PSNR, SSIM and image measurements
│   ├── features.py           HOG and colour features
│   ├── enhancement.py        the three fuzzy-controlled steps
│   └── photos.py             uploaded photo loading and cropping
├── models/                   trained checkpoints
├── results/                  charts, CSV files and training logs
├── scripts/                  setup, checks, demos and comparisons
└── requirements.txt
```

## How to run everything

Run these from `D:\mini_project`.

**Run the app**

```bash
.venv\Scripts\python.exe -m streamlit run app.py
```

**Set up the environment**

```bash
python -m venv .venv
```

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

```bash
.venv\Scripts\python.exe -m scripts.check_setup
```

**Prepare the data**

```bash
.venv\Scripts\python.exe -m scripts.prepare_cure_tsr
```

```bash
.venv\Scripts\python.exe -m scripts.prepare_lisa
```

```bash
.venv\Scripts\python.exe -m scripts.prepare_features
```

```bash
.venv\Scripts\python.exe -m scripts.prepare_enhanced
```

```bash
.venv\Scripts\python.exe -m scripts.prepare_samples
```

For the wide model, GTSRB goes in `data\raw\gtsrb\` and the Indian set in `data\raw\indian\`. This also writes the app's 84 example images.

```bash
.venv\Scripts\python.exe -m scripts.prepare_wide
```

```bash
.venv\Scripts\python.exe -m scripts.prepare_enhanced --splits gtsrb_val gtsrb_test indian_val indian_test
```

**Run the checks**

```bash
.venv\Scripts\python.exe -m scripts.check_image_processing
```

```bash
.venv\Scripts\python.exe -m scripts.check_fuzzy
```

```bash
.venv\Scripts\python.exe -m scripts.check_deep_learning
```

```bash
.venv\Scripts\python.exe -m scripts.check_app
```

**Make the demo charts**

```bash
.venv\Scripts\python.exe -m scripts.demo_image_processing
```

```bash
.venv\Scripts\python.exe -m scripts.demo_fuzzy
```

**Train one model**

Choose `mlp`, `feature_mlp` or `cnn`, and `clean` or `all`. Add `--lights` for all 17 classes, and `--enhanced` to train on fuzzy-enhanced images, which works for `mlp` and `cnn`.

```bash
.venv\Scripts\python.exe -m deep_learning.train --model cnn --regime all --seed 0 --lights
```

**Train and check the NumPy MLP**

```bash
.venv\Scripts\python.exe -m scripts.train_numpy_mlp
```

**Compare the models**

```bash
.venv\Scripts\python.exe -m scripts.compare_on_validation
```

```bash
.venv\Scripts\python.exe -m scripts.compare_lights
```

```bash
.venv\Scripts\python.exe -m scripts.evaluate_fuzzy_quality
```

**Run the final experiments**

Training is safe to restart, because finished runs are skipped.

```bash
.venv\Scripts\python.exe -m experiments.run_training
```

```bash
.venv\Scripts\python.exe -m experiments.fixed_baseline
```

```bash
.venv\Scripts\python.exe -m experiments.mlp_study
```

```bash
.venv\Scripts\python.exe -m experiments.final_scores --split val
```

```bash
.venv\Scripts\python.exe -m experiments.summarize --split val
```

**Score the test sets**

These have already been run once. `summarize` only redraws charts from the saved scores, so it can be rerun.

```bash
.venv\Scripts\python.exe -m experiments.final_scores --split test
```

```bash
.venv\Scripts\python.exe -m experiments.summarize --split test
```

```bash
.venv\Scripts\python.exe -m experiments.image_studies
```

**Train and score the wide model**

```bash
.venv\Scripts\python.exe -m experiments.run_training --wide
```

```bash
.venv\Scripts\python.exe -m experiments.wide_scores --split val
```

```bash
.venv\Scripts\python.exe -m experiments.wide_scores --split test
```

**Train and score the global model**

The fully annotated Mapillary zips go in `data\raw\mapillary\`. This also writes the YOLO frames and boxes.

```bash
.venv\Scripts\python.exe -m scripts.prepare_mapillary
```

```bash
.venv\Scripts\python.exe -m scripts.prepare_global
```

```bash
.venv\Scripts\python.exe -m scripts.prepare_enhanced --splits mapillary_val mapillary_test
```

```bash
.venv\Scripts\python.exe -m experiments.run_training --mapillary
```

```bash
.venv\Scripts\python.exe -m experiments.wide_scores --mapillary --split val
```

The test sets below have already been scored once. `wide_charts` only redraws the charts from the saved scores.

```bash
.venv\Scripts\python.exe -m experiments.wide_scores --mapillary --split test
```

```bash
.venv\Scripts\python.exe -m experiments.wide_charts --mapillary --split test
```

**Train and score the YOLO detector**

Run `prepare_mapillary` first, then these in order.

```bash
.venv\Scripts\python.exe -m scripts.prepare_detection
```

```bash
.venv\Scripts\python.exe -m scripts.label_lights
```

```bash
.venv\Scripts\python.exe -m deep_learning.train_detector
```

```bash
.venv\Scripts\python.exe -m experiments.detector_scores --split val --weights models\detector.pt
```

The test frames have already been scored once.

```bash
.venv\Scripts\python.exe -m experiments.detector_scores --split test --weights models\detector.pt
```

**Add worn and damaged signs**

The Kaggle Damaged Signs zip and the Roboflow Detection_of_damage_signs YOLOv8 zip go in `data\raw\damaged\`. These make worn-paint copies of every sign image, pack the real damaged-sign photos, and label the lights in their frames.

```bash
.venv\Scripts\python.exe -m scripts.prepare_wear
```

```bash
.venv\Scripts\python.exe -m scripts.prepare_damaged
```

```bash
.venv\Scripts\python.exe -m scripts.prepare_damaged --roboflow data\raw\damaged\Detection_of_damage_signs.v3i.yolov8.zip
```

```bash
.venv\Scripts\python.exe -m scripts.label_lights --sources damaged roboflow
```

```bash
.venv\Scripts\python.exe -m scripts.prepare_enhanced --splits real_train_wear unreal_train_wear real_test_wear unreal_test_wear gtsrb_val_wear gtsrb_test_wear indian_val_wear indian_test_wear mapillary_val_wear mapillary_test_wear damaged_val damaged_test
```

**Train and score the worn-sign models**

The new CNN and detector get new names, so the finished models stay as they are.

```bash
.venv\Scripts\python.exe -m experiments.run_training --wear
```

```bash
.venv\Scripts\python.exe -m deep_learning.train_detector --weights models\detector.pt --epochs 10 --name detector_wear --output models\detector_wear.pt
```

```bash
.venv\Scripts\python.exe -m experiments.wide_scores --wear --split val
```

```bash
.venv\Scripts\python.exe -m experiments.detector_scores --split val --weights models\detector_wear.pt
```

## Rules to follow

- Use the real test set only for final results.
- Tune on training images, using the validation block or slices.
- Keep frames from one clip on one side of every split.
- Never flip sign images.
- Apply denoising before stretching.
- Report real and simulated results separately, and signs and lights separately.
- Report each model's mean and spread over 3 seeds.
- Only write real measured results in the reports.
- Ask before installing anything new.
- Keep the laptop plugged in and awake during long training. When Windows sleeps, training pauses until it wakes.
- Windows Smart App Control is on and may block new, little-used library files. pandas 3.1.0 was blocked on 4 November, and pandas 2.2.3 works.

## Time left

These are rough estimates at a few hours a day.

| Work | Estimate |
|---|---|
| Real photo test | 1-2 hours |
| Demo preparation | A few hours |
| Three reports | 1-2 days |
