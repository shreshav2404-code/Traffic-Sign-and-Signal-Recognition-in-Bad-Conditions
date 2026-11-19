// Build the 8-slide project presentation, ordered by the three points the teacher asked for:
// 1 why this topic, 2 algorithms used, 3 implementation. Run with pptxgenjs on NODE_PATH:
//   node documents/report/build_deck.js
// Every number comes from the report (build_report.py), which takes them from results/.
const path = require("path");
const pptxgen = require("pptxgenjs");

const HERE = __dirname;
const FIG = path.join(HERE, "figures");
const RESULTS = path.join(HERE, "..", "..", "results");
const OUTPUT = path.join(HERE, "Traffic_Sign_Recognition_Presentation.pptx");
const SKILL = process.env.PPTX_SKILL;

const THEME = {
  name: "Road Signal",
  headFontFace: "Calibri",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1F2A33", lt1: "FFFFFF", dk2: "1F3A5F", lt2: "EEF2F6",
    accent1: "F5B301", accent2: "2A78D6", accent3: "E07B16", accent4: "1B9E77", accent5: "6B4FBB", accent6: "C0392B",
    hlink: "2A78D6", folHlink: "6B4FBB",
  },
};
const HEX = THEME.colors;
// Soft fills for the course colours, as in the report's diagrams
const TINT = { image: "E6F0FB", fuzzy: "FDF0E2", neural: "E4F4EE", app: "EFEAF9", data: "EEF0F2" };
const EDGE = { image: HEX.accent2, fuzzy: HEX.accent3, neural: HEX.accent4, app: HEX.accent5, data: "5B6670" };
const MUTED = "4F5A64";
const W = 13.333;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Traffic Sign and Signal Recognition in Bad Conditions";
pres.subject = "Mini project presentation";
pres.author = "Keshav Kumar";
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: HEX.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 1.05, w: 11.9, h: 1.6, fontSize: 42, bold: true, color: C.background1, valign: "top", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 2.75, w: 11.9, h: 0.6, fontSize: 20, color: "CADCFC", valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: HEX.lt1 },
  margin: [0.5, 0.6, 0.55, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.32, w: 12.1, h: 0.75, fontSize: 30, bold: true, color: C.text2, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "Traffic Sign and Signal Recognition in Bad Conditions", options: { x: 0.6, y: 7.05, w: 8, h: 0.3, fontSize: 10, color: "8A949D" } } },
  ],
  slideNumber: { x: 12.25, y: 7.05, w: 0.5, h: 0.3, fontSize: 10, color: "8A949D", align: "right" },
});
pres.defineSlideMaster({
  title: "CLOSING_DARK",
  background: { color: HEX.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.32, w: 12.1, h: 0.75, fontSize: 30, bold: true, color: C.background1, valign: "middle", align: "left", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 12.25, y: 7.05, w: 0.5, h: 0.3, fontSize: 10, color: "CADCFC", align: "right" },
});

// A rounded block with a bold title and short lines, coloured by course
function block(slide, x, y, w, h, kind, title, lines, opts = {}) {
  const runs = [{ text: title, options: { bold: true, fontSize: opts.titleSize || 15, color: HEX.dk1, breakLine: lines.length > 0 } }];
  lines.forEach((line, i) => runs.push({ text: line, options: { fontSize: opts.lineSize || 12, color: MUTED, breakLine: i < lines.length - 1 } }));
  slide.addText(runs, {
    shape: pres.shapes.ROUNDED_RECTANGLE, x, y, w, h, rectRadius: 0.12,
    fill: { color: TINT[kind] }, line: { color: EDGE[kind], width: 1.5 },
    align: "center", valign: "middle", margin: 4, isTextBox: true, objectName: opts.name || title,
  });
}

function arrow(slide, x1, y1, x2, y2, colour = HEX.dk1) {
  slide.addShape(pres.shapes.LINE, {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    flipH: x2 < x1, flipV: y2 < y1, line: { color: colour, width: 2, endArrowType: "triangle" },
  });
}

// A big number with a small label under it
function stat(slide, x, y, w, value, label, dark = false) {
  slide.addText([
    { text: value, options: { fontSize: 34, bold: true, color: dark ? HEX.accent1 : HEX.dk2, breakLine: true } },
    { text: label, options: { fontSize: 12, color: dark ? "CADCFC" : MUTED } },
  ], { x, y, w, h: 1.15, align: "left", valign: "top", margin: 0, isTextBox: true, objectName: `Stat ${label}` });
}

function tableRows(header, rows, opts = {}) {
  const head = header.map((text) => ({ text, options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 }, fontSize: opts.headSize || 12 } }));
  const body = rows.map((row, r) => row.map((text, c) => ({
    text: String(text),
    options: {
      fontSize: opts.size || 12, color: HEX.dk1,
      bold: (opts.boldFirst && c === 0) || (opts.boldRows || []).includes(r),
      fill: { color: (opts.highlight || []).includes(r) ? "FFF4D6" : r % 2 ? "F2F5F8" : "FFFFFF" },
    },
  })));
  return [head, ...body];
}

function legend(slide, x, y) {
  const items = [["data", "Data"], ["image", "Image processing"], ["fuzzy", "Fuzzy logic"], ["neural", "Neural networks and deep learning"], ["app", "Detection and app"]];
  items.forEach(([kind, label]) => {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: y + 0.06, w: 0.32, h: 0.2, rectRadius: 0.04, fill: { color: TINT[kind] }, line: { color: EDGE[kind], width: 1.2 } });
    const w = 0.085 * label.length + 0.1;
    slide.addText(label, { x: x + 0.4, y, w, h: 0.32, fontSize: 12, color: HEX.dk1, margin: 0, valign: "middle", isTextBox: true });
    x += 0.5 + w;
  });
}

// 1. Title
pres.addSection({ title: "Introduction" });
let slide = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
slide.addText("Traffic Sign and Signal Recognition in Bad Conditions", { placeholder: "title" });
slide.addText("Fuzzy-logic image enhancement, neural network classification and YOLO detection", { placeholder: "body" });
// Agenda: the three points the presentation must cover
[["1", "Why this topic"], ["2", "Algorithms used"], ["3", "Implementation"]].forEach(([n, label], i) => {
  slide.addText([
    { text: `${n}  `, options: { bold: true, color: HEX.accent1 } },
    { text: label, options: { color: HEX.lt1 } },
  ], { shape: pres.shapes.ROUNDED_RECTANGLE, x: 0.7 + i * 3.55, y: 3.4, w: 3.3, h: 0.55, rectRadius: 0.1, fill: { color: "2C4F7C" }, line: { color: "4A6A93", width: 1 }, fontSize: 17, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: `Agenda ${n}` });
});
stat(slide, 0.7, 4.45, 2.9, "0.819", "macro F1 on 203,374 real damaged test signs", true);
stat(slide, 3.75, 4.45, 2.9, "97.9%", "accuracy on 281,547 traffic-light test images", true);
stat(slide, 6.8, 4.45, 2.9, "106", "sign and light classes from 7 datasets", true);
stat(slide, 9.85, 4.45, 2.9, "0.786", "on worn real signs after wear training (0.597 before)", true);
slide.addText("Presented by: Keshav Kumar (20241CAI0072)     Department: PSAIAC", {
  x: 0.7, y: 6.55, w: 11.9, h: 0.4, fontSize: 14, color: "CADCFC", margin: 0, isTextBox: true,
});
slide.addNotes("One pipeline reads traffic signs and lights from damaged images. Fuzzy logic decides how to repair each image, and neural networks name the sign. Every score is the mean of 3 training runs, tested once.");

// 2. Point 1: why this topic, and the objectives
pres.addSection({ title: "1 Why this topic" });
slide = pres.addSlide({ masterName: "CONTENT", sectionTitle: "1 Why this topic" });
slide.addText("1 · Why this topic: reading signs where vision fails", { placeholder: "title" });
slide.addText("Why we chose it", { x: 0.6, y: 1.25, w: 5.8, h: 0.4, fontSize: 20, bold: true, color: C.text2, margin: 0, isTextBox: true });
slide.addText([
  { text: "Safety: ", options: { bold: true, bullet: true } },
  { text: "a driver-assistance camera must read signs at night, in haze, rain and snow, and on worn paint, exactly when vision is worst.", options: { breakLine: true } },
  { text: "A real gap: ", options: { bold: true, bullet: true } },
  { text: "a CNN trained on clean images reads only 0.683 macro F1 on real damaged signs, against 0.819 when trained on damage.", options: { breakLine: true } },
  { text: "Real roads: ", options: { bold: true, bullet: true } },
  { text: "our own phone test found a faded stop sign that no model could read.", options: { breakLine: true } },
  { text: "Three courses, one topic: ", options: { bold: true, bullet: true } },
  { text: "image processing repairs, fuzzy logic decides how much, neural networks read the sign.", options: {} },
], { x: 0.6, y: 1.7, w: 5.8, h: 2.75, fontSize: 14, color: HEX.dk1, paraSpaceAfter: 5, valign: "top", margin: 0, isTextBox: true });
slide.addImage({ path: path.join(FIG, "cure_samples.png"), x: 0.75, y: 4.55, w: 3.0, h: 3.0 / 1.284, altText: "CURE-TSR signs: clean, then darkening, noise and haze at levels 3 and 5" });
slide.addText("CURE-TSR signs: clean, then darkening, noise and haze at levels 3 and 5", { x: 3.95, y: 5.3, w: 2.45, h: 1.2, fontSize: 11, color: MUTED, italic: true, margin: 0, valign: "top", isTextBox: true });
slide.addText("Objectives", { x: 6.9, y: 1.25, w: 5.8, h: 0.4, fontSize: 20, bold: true, color: C.text2, margin: 0, isTextBox: true });
const objectives = [
  ["image", "1", "Image processing", "filters, histograms, convolution, measured with PSNR and SSIM"],
  ["fuzzy", "2", "Fuzzy logic", "a Mamdani controller that chooses denoise, stretch and darken strengths"],
  ["neural", "3", "Neural networks", "MLP, feature MLP and backpropagation written by hand"],
  ["neural", "4", "Deep learning", "CNNs on clean, damaged and enhanced images, 17 to 106 classes, worn signs"],
  ["app", "5", "Detection and app", "YOLO finds signs; a Streamlit app shows every step; honest scoring"],
];
objectives.forEach(([kind, n, head, text], i) => {
  const y = 1.75 + i * 1.02;
  slide.addText(n, { shape: pres.shapes.OVAL, x: 6.9, y: y + 0.08, w: 0.55, h: 0.55, fill: { color: EDGE[kind] }, color: HEX.lt1, bold: true, fontSize: 16, align: "center", valign: "middle", margin: 0, isTextBox: true });
  slide.addText([
    { text: head, options: { bold: true, fontSize: 15, color: HEX.dk1, breakLine: true } },
    { text, options: { fontSize: 13, color: MUTED } },
  ], { x: 7.6, y, w: 5.1, h: 0.85, valign: "middle", margin: 0, isTextBox: true });
});
slide.addNotes("Why this topic: it is a safety problem, there is a measured gap between clean-trained and damage-trained models, real roads have worn signs, and it needs all three courses. The objectives map onto the courses: image processing (1), fuzzy logic and neural networks (2, 3), and deep learning (4).");

// 3. Point 2: the algorithm used at each step
pres.addSection({ title: "2 Algorithms used" });
slide = pres.addSlide({ masterName: "CONTENT", sectionTitle: "2 Algorithms used" });
slide.addText("2 · Algorithms used, step by step", { placeholder: "title" });
const algorithms = [
  ["app", "Find", "YOLO26n detector, pretrained on COCO, fine-tuned twice on our frames", "Deep learning", "mAP50 0.701 on signs, 0.893 on damaged close-ups"],
  ["image", "Measure", "Mean brightness, RMS contrast, colour-roughness noise", "Image processing", "3 numbers describe the damage"],
  ["fuzzy", "Decide", "Mamdani fuzzy inference: 14 rules, min for AND, max to combine, centroid defuzzification", "Fuzzy logic", "matches scikit-fuzzy within 0.0001"],
  ["image", "Repair", "Gaussian filter → RGB levels stretch → gamma correction", "Image processing", "clean-trained CNN 0.683 → 0.715"],
  ["neural", "Classify", "CNN: 6 conv layers, batch norm, dropout; AdamW, weighted cross-entropy, early stopping", "Deep learning", "0.819 macro F1, 89.8% accuracy"],
  ["neural", "Compare", "MLP on pixels, MLP on HOG + colour features, NumPy MLP with backpropagation by hand", "Neural networks", "gradient error 5.2 × 10⁻⁷"],
];
const algoRows = [["Step", "Algorithm", "Course", "Result"].map((text) => ({ text, options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 }, fontSize: 14 } }))];
algorithms.forEach(([kind, step, algorithm, course, result]) => {
  algoRows.push([
    { text: step, options: { bold: true, fontSize: 15, color: HEX.dk1, fill: { color: TINT[kind] } } },
    { text: algorithm, options: { fontSize: 13.5, color: HEX.dk1, fill: { color: "FFFFFF" } } },
    { text: course, options: { fontSize: 13, color: EDGE[kind], bold: true, fill: { color: "FFFFFF" } } },
    { text: result, options: { fontSize: 13, color: MUTED, fill: { color: "FFFFFF" } } },
  ]);
});
slide.addTable(algoRows, { x: 0.6, y: 1.25, w: 9.1, colW: [1.1, 4.1, 1.6, 2.3], border: { type: "solid", pt: 0.75, color: "B7C1CA" }, valign: "middle", margin: 0.08 });
slide.addText([
  { text: "Also built and compared: ", options: { bold: true, color: HEX.dk1 } },
  { text: "histogram equalization, CLAHE, convolution by hand (equal to OpenCV), Sobel and Laplacian edges, and mean, median, bilateral and non-local means filters (median: 22.16 → 34.83 dB on 2% salt-and-pepper).", options: { color: MUTED } },
], { x: 0.6, y: 6.15, w: 9.1, h: 0.8, fontSize: 12.5, margin: 0, valign: "top", isTextBox: true });
slide.addText([
  { text: "IF ", options: { bold: true, color: EDGE.fuzzy } }, { text: "brightness is bright", options: { breakLine: true } },
  { text: "AND ", options: { bold: true, color: EDGE.fuzzy } }, { text: "contrast is medium", options: { breakLine: true } },
  { text: "THEN ", options: { bold: true, color: EDGE.fuzzy } }, { text: "darken is strong", options: {} },
], { shape: pres.shapes.ROUNDED_RECTANGLE, x: 9.95, y: 1.25, w: 2.8, h: 1.3, rectRadius: 0.1, fill: { color: TINT.fuzzy }, line: { color: EDGE.fuzzy, width: 1.25 }, fontSize: 14, color: HEX.dk1, margin: 8, valign: "middle", isTextBox: true, objectName: "Example rule" });
slide.addText("one of the 14 fuzzy rules", { x: 9.95, y: 2.6, w: 2.8, h: 0.3, fontSize: 11, italic: true, color: MUTED, margin: 0, isTextBox: true });
slide.addImage({ path: path.join(FIG, "fuzzy_before_after.png"), x: 10.25, y: 3.05, w: 2.6 * 0.839, h: 2.6, altText: "Darkened and hazy signs before and after fuzzy enhancement" });
slide.addText("Darkening and haze: damaged (top), after fuzzy repair (bottom)", { x: 9.95, y: 5.75, w: 2.8, h: 0.6, fontSize: 11, italic: true, color: MUTED, margin: 0, valign: "top", isTextBox: true });
slide.addNotes("Each step of the pipeline uses a named algorithm. YOLO finds signs, three image measures describe the damage, Mamdani fuzzy inference picks the repair strengths, a Gaussian filter, levels stretch and gamma correction repair the image, and the CNN names the sign. The MLPs are the neural-network baselines, including one with backpropagation written by hand.");

// 4. Point 3: implementation, the workflow drawn as native shapes
pres.addSection({ title: "3 Implementation" });
slide = pres.addSlide({ masterName: "CONTENT", sectionTitle: "3 Implementation" });
slide.addText("3 · Implementation: one photo, eight steps", { placeholder: "title" });
const bw = 2.75, bh = 1.5, gap = (W - 1.2 - 4 * bw) / 3;
const xs = [0, 1, 2, 3].map((i) => 0.6 + i * (bw + gap));
const top = 1.45, bottom = 3.85;
const upper = [
  ["data", "1. Input", ["phone or street photo,", "or a damaged sample"]],
  ["app", "2. YOLO26n detector", ["finds signs and lights", "confidence ≥ 0.30,", "or the user drags a box"]],
  ["image", "3. Crop and resize", ["square crop, margin 1.15", "resized to 32 × 32 × 3"]],
  ["image", "4. Measure quality", ["brightness (mean)", "contrast (RMS)", "noise (colour roughness)"]],
];
const lower = [
  ["app", "8. Show the answer", ["two result cards, top 3", "confidence and verdict", "rules fired, PSNR, SSIM"]],
  ["neural", "7. CNN classifier", ["original and enhanced crop", "106 classes, softmax", "sign or light from YOLO"]],
  ["image", "6. Enhance", ["Gaussian denoise", "→ RGB levels stretch", "→ gamma darken"]],
  ["fuzzy", "5. Fuzzy controller", ["14 Mamdani rules", "centroid defuzzification", "→ 3 strengths, 0 to 1"]],
];
upper.forEach(([kind, title, lines], i) => block(slide, xs[i], top, bw, bh, kind, title, lines));
lower.forEach(([kind, title, lines], i) => block(slide, xs[i], bottom, bw, bh, kind, title, lines));
for (let i = 0; i < 3; i++) {
  arrow(slide, xs[i] + bw + 0.04, top + bh / 2, xs[i + 1] - 0.04, top + bh / 2);
  arrow(slide, xs[i + 1] - 0.04, bottom + bh / 2, xs[i] + bw + 0.04, bottom + bh / 2);
}
arrow(slide, xs[3] + bw / 2, top + bh + 0.04, xs[3] + bw / 2, bottom - 0.04);
arrow(slide, xs[2] + bw / 2, top + bh + 0.04, xs[1] + bw / 2 + 0.3, bottom - 0.04, "8C969F");
slide.addText("the original crop is classified too", { x: xs[2] + bw / 2 + 0.1, y: top + bh + 0.22, w: 3, h: 0.3, fontSize: 11, italic: true, color: MUTED, margin: 0, isTextBox: true });
legend(slide, 0.6, 5.75);
slide.addText("Denoising goes first, because stretching would amplify leftover noise. Colours show which course each step belongs to.", {
  x: 0.6, y: 6.3, w: 12.1, h: 0.4, fontSize: 13, color: MUTED, margin: 0, isTextBox: true,
});
slide.addNotes("Walk through one photo: YOLO finds the sign, the crop is measured, the fuzzy controller sets three strengths, the image is enhanced in a fixed order, and the CNN classifies both the original and the enhanced crop so the app can show whether enhancement changed the answer.");

// 5. Implementation: architecture and technology stack
slide = pres.addSlide({ masterName: "CONTENT", sectionTitle: "3 Implementation" });
slide.addText("3 · Implementation: four-layer architecture and tools", { placeholder: "title" });
const layers = [
  ["Data sources", [["data", "CURE-TSR + LISA", ["2.2 M signs", "41,923 lights"]], ["data", "GTSRB + Indian", ["real sign photos", "43 + 85 classes"]], ["data", "Mapillary", ["40,968 photos", "six continents"]], ["data", "Kaggle + Roboflow", ["5,190 photos", "of damaged signs"]]]],
  ["Preparation", [["data", "Pack arrays", ["32 × 32 × 3", "merged labels"]], ["image", "Add damage", ["5 types × 5 levels", "incl. worn paint"]], ["data", "Leak-free splits", ["clip, track, block", "+ validation gap"]], ["fuzzy", "Enhanced copies", ["fuzzy controller", "on every image"]]]],
  ["Core modules", [["image", "image_processing", ["filters, convolution", "PSNR, SSIM"]], ["fuzzy", "fuzzy_logic", ["14 rules", "controller"]], ["neural", "deep_learning", ["MLP, CNN", "NumPy MLP"]], ["app", "YOLO26n", ["39,301 frames", "fine-tuned twice"]]]],
  ["Evaluation and app", [["neural", "Experiments", ["3 seeds", "choose on validation"]], ["neural", "Scored once", ["test sets", "untouched"]], ["data", "262 checks", ["data, models,", "fuzzy, app"]], ["app", "Streamlit app", ["4 tabs", "phone via tunnel"]]]],
];
layers.forEach(([name, blocks], r) => {
  const y = 1.3 + r * 1.27, h = 1.05, labelW = 1.3, x0 = 0.6 + labelW + 0.15, boxW = (8.55 - x0 - 3 * 0.12) / 4;
  slide.addText(name, { shape: pres.shapes.ROUNDED_RECTANGLE, x: 0.6, y, w: labelW, h, rectRadius: 0.1, fill: { color: "263238" }, color: HEX.lt1, bold: true, fontSize: 13, align: "center", valign: "middle", margin: 4, isTextBox: true, objectName: "Layer " + name });
  if (r < layers.length - 1) arrow(slide, 0.6 + labelW / 2, y + h + 0.02, 0.6 + labelW / 2, y + 1.25, "5B6670");
  blocks.forEach(([kind, title, lines], i) => block(slide, x0 + i * (boxW + 0.12), y, boxW, h, kind, title, lines, { titleSize: 12.5, lineSize: 11 }));
});
slide.addText("Technology stack", { x: 8.9, y: 1.25, w: 3.85, h: 0.4, fontSize: 20, bold: true, color: C.text2, margin: 0, isTextBox: true });
slide.addTable(tableRows(["Layer", "Tools"], [
  ["Language", "Python 3.11, virtual environment"],
  ["Deep learning", "PyTorch 2.15 to train, 2.5.1 for the app; RTX 4050 6 GB"],
  ["Detection", "Ultralytics 8.4, YOLO26 nano"],
  ["Image processing", "OpenCV, scikit-image, NumPy"],
  ["Fuzzy logic", "scikit-fuzzy sets, NumPy inference"],
  ["Evaluation", "scikit-learn, pandas, matplotlib"],
  ["App", "Streamlit, streamlit-cropper, Cloudflare tunnel"],
], { size: 11.5, headSize: 12.5, boldFirst: true }), { x: 8.9, y: 1.75, w: 3.85, colW: [1.4, 2.45], border: { type: "solid", pt: 0.75, color: "B7C1CA" }, valign: "middle", margin: 0.05 });
slide.addText("All 262 automated checks pass.", { x: 8.9, y: 5.75, w: 3.85, h: 0.45, fontSize: 14, bold: true, color: HEX.accent4, margin: 0, isTextBox: true });
slide.addNotes("image_processing, fuzzy_logic and deep_learning hold the methods; experiments runs training and scoring; app.py is the interface. Datasets are packed once into 32 x 32 arrays, which made training on 1.9 million images possible on a laptop GPU.");

// 6. Implementation: datasets and training
slide = pres.addSlide({ masterName: "CONTENT", sectionTitle: "3 Implementation" });
slide.addText("3 · Implementation: 7 datasets, 51 training runs", { placeholder: "title" });
slide.addTable(tableRows(["Dataset", "What it adds", "Images used"], [
  ["CURE-TSR", "14 signs, 12 damages × 5 levels", "2,206,100"],
  ["LISA", "red, yellow, green lights, day and night", "41,923 crops"],
  ["GTSRB", "real German sign photos", "51,839 crops"],
  ["Indian set", "real Indian sign photos", "4,359 crops"],
  ["Mapillary", "street signs, 6 continents, 'other sign'", "48,863 crops"],
  ["Kaggle damaged", "faded, rusted, vandalised signs", "3,194 photos"],
  ["Roboflow damaged", "close-ups by kind of damage", "1,996 photos"],
], { size: 12, headSize: 13, boldFirst: true }), { x: 0.6, y: 1.3, w: 7.3, colW: [1.8, 3.6, 1.9], border: { type: "solid", pt: 0.75, color: "B7C1CA" }, valign: "middle", margin: 0.06 });
slide.addText("Clean crops before synthetic damage; Kaggle and Roboflow frames also train YOLO.", { x: 0.6, y: 5.05, w: 7.3, h: 0.35, fontSize: 11, italic: true, color: MUTED, margin: 0, isTextBox: true });
stat(slide, 8.35, 1.3, 2.1, "51", "classifier training runs, 3 seeds each");
stat(slide, 10.6, 1.3, 2.1, "2", "YOLO detectors, 9.4 hours of training");
stat(slide, 8.35, 2.75, 2.1, "1.92 M", "training images for the final CNN");
stat(slide, 10.6, 2.75, 2.1, "106", "classes: 17 → 84 → 106");
slide.addText([
  { text: "No leakage: ", options: { bold: true } },
  { text: "splits by whole clip, track or block of frames; near-duplicates removed; test street photos kept out of training.", options: {} },
], { x: 8.35, y: 4.15, w: 4.4, h: 1.2, fontSize: 13, color: HEX.dk1, margin: 0, valign: "top", isTextBox: true });
const steps = [["neural", "Train", ["3 seeds per model"]], ["neural", "Choose on validation", ["mean ± spread"]], ["app", "Score test once", ["never change after"]]];
steps.forEach(([kind, title, lines], i) => {
  const x = 0.6 + i * 4.25;
  block(slide, x, 5.6, 3.6, 0.95, kind, title, lines, { titleSize: 15, lineSize: 12 });
  if (i) arrow(slide, x - 0.6, 6.075, x - 0.05, 6.075);
});
slide.addNotes("CURE-TSR is the core because every damaged image has a clean original, so image quality can be measured exactly. The other datasets widen the classes and add real photos and real damage. The protocol at the bottom applies to every model.");

// 7. Implementation: results, every headline score
slide = pres.addSlide({ masterName: "CONTENT", sectionTitle: "3 Implementation" });
slide.addText("3 · Results: training on damage beats repairing it", { placeholder: "title" });
slide.addTable(tableRows(["Model (test, macro F1)", "Classes", "Real signs", "Simulated", "Lights", "Worn real"], [
  ["Feature MLP, all conditions", "17", "0.701", "0.825", "0.898", "–"],
  ["MLP, all conditions", "17", "0.733", "0.878", "0.960", "–"],
  ["CNN, clean + fuzzy input", "17", "0.715", "0.791", "0.936", "–"],
  ["CNN, all conditions", "17", "0.819", "0.909", "0.980", "–"],
  ["Wide CNN, all conditions", "84", "0.757", "0.881", "0.946", "–"],
  ["Global CNN, all conditions", "106", "0.816", "0.888", "0.951", "0.597"],
  ["Global CNN + worn signs (app)", "106", "0.808", "0.881", "0.959", "0.786"],
], { size: 12, headSize: 12, boldFirst: true, highlight: [3, 6] }), { x: 0.6, y: 1.3, w: 7.6, colW: [2.9, 0.85, 1.0, 1.0, 0.85, 1.0], border: { type: "solid", pt: 0.75, color: "B7C1CA" }, valign: "middle", margin: 0.05 });
slide.addText([
  { text: "YOLO detector 2, test mAP50:  ", options: { bold: true, color: HEX.dk1 } },
  { text: "0.701 signs · 0.680 lights · 0.853 Kaggle damaged · 0.893 Roboflow close-ups (0.003 before fine-tuning)", options: { color: MUTED } },
], { x: 0.6, y: 4.0, w: 7.6, h: 0.75, fontSize: 14, margin: 0, valign: "top", isTextBox: true });
slide.addText([
  { text: "Best 17-class CNN: ", options: { bold: true, color: HEX.dk1 } },
  { text: "89.8% accuracy on 203,374 real signs, 97.9% on 281,547 lights. Fuzzy enhancement raised PSNR from 17.90 to 18.35 dB.", options: { color: MUTED } },
], { x: 0.6, y: 4.85, w: 7.6, h: 0.75, fontSize: 14, margin: 0, valign: "top", isTextBox: true });
slide.addText([
  { text: "Worn-sign training: ", options: { bold: true, color: HEX.dk1 } },
  { text: "worn real signs 0.597 → 0.786, worn GTSRB 0.629 → 0.867; the usual test sets moved by under 1 point.", options: { color: MUTED } },
], { x: 0.6, y: 5.7, w: 7.6, h: 0.75, fontSize: 14, margin: 0, valign: "top", isTextBox: true });
slide.addChart(pres.charts.BAR, [{
  name: "Real damaged signs, macro F1",
  labels: ["CNN, clean images", "CNN, clean + fuzzy", "CNN, enhanced images", "CNN, all conditions"],
  values: [0.683, 0.715, 0.814, 0.819],
}], {
  x: 8.5, y: 1.3, w: 4.25, h: 5.3, barDir: "bar",
  showTitle: true, title: "Real damaged signs (17 classes)", titleFontSize: 14, titleColor: HEX.dk1, titleFontFace: "+mn-lt",
  chartColors: [HEX.accent3, HEX.accent3, HEX.accent4, HEX.accent4], invertedColors: [HEX.accent3],
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", dataLabelFontSize: 13, dataLabelColor: HEX.dk1, dataLabelFontFace: "+mn-lt",
  valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.2, valAxisLabelFormatCode: "0.0", valAxisLabelFontSize: 11, valAxisLabelColor: MUTED, valAxisLabelFontFace: "+mn-lt",
  catAxisLabelFontSize: 12, catAxisLabelColor: HEX.dk1, catAxisLabelFontFace: "+mn-lt",
  valGridLine: { color: "E1E5E9", size: 0.75 }, catGridLine: { style: "none" }, showLegend: false,
});
slide.addNotes("All scores are means of 3 seeds, tested once. The chart shows the main lesson: training on damaged images (green) beats enhancing images for a model that never saw damage (orange). Wear training raised worn real signs from 0.597 to 0.786 while the usual sets moved by under 1 point.");

// 8. The app, conclusion and next steps
pres.addSection({ title: "Conclusion" });
slide = pres.addSlide({ masterName: "CLOSING_DARK", sectionTitle: "Conclusion" });
slide.addText("A working app, and what we learned", { placeholder: "title" });
const shotH = 3.55, shotW = shotH * 0.871;
slide.addImage({ path: path.join(FIG, "app_yolo_photo.jpg"), x: 0.6, y: 1.3, w: shotW, h: shotH, altText: "App: the worn-sign YOLO finds five objects in a street photo" });
slide.addImage({ path: path.join(FIG, "app_light_result.jpg"), x: 0.6 + shotW + 0.25, y: 1.3, w: shotW, h: shotH, altText: "App: fuzzy stages and both answers, green light" });
slide.addText("Left: the worn-sign YOLO finds five objects in a street photo. Right: fuzzy stages and both answers, green light 96%.", {
  x: 0.6, y: 4.95, w: 2 * shotW + 0.25, h: 0.6, fontSize: 12, italic: true, color: "CADCFC", margin: 0, valign: "top", isTextBox: true,
});
slide.addText("Conclusions", { x: 7.35, y: 1.25, w: 5.4, h: 0.4, fontSize: 20, bold: true, color: HEX.accent1, margin: 0, isTextBox: true });
slide.addText([
  { text: "Training on damaged images beats repairing them: 0.819 against 0.715 macro F1.", options: { bullet: true, breakLine: true } },
  { text: "Fuzzy enhancement still helps every model that never saw damage, and leaves clean images alone.", options: { bullet: true, breakLine: true } },
  { text: "Real photos exposed too few classes, close-up lights and worn paint; each fix was measured before it was kept.", options: { bullet: true } },
], { x: 7.35, y: 1.7, w: 5.4, h: 2.4, fontSize: 15, color: HEX.lt1, paraSpaceAfter: 6, margin: 0, valign: "top", isTextBox: true });
slide.addText("Next steps", { x: 7.35, y: 4.25, w: 5.4, h: 0.4, fontSize: 20, bold: true, color: HEX.accent1, margin: 0, isTextBox: true });
slide.addText([
  { text: "Larger input or a pretrained backbone, to read text and numbers", options: { bullet: true, breakLine: true } },
  { text: "Stronger sun-bleaching and real photos of old signs", options: { bullet: true, breakLine: true } },
  { text: "Night and rain photo tests; YOLO trained on damaged frames", options: { bullet: true } },
], { x: 7.35, y: 4.7, w: 5.4, h: 1.7, fontSize: 15, color: HEX.lt1, paraSpaceAfter: 4, margin: 0, valign: "top", isTextBox: true });
slide.addText("Thank you · Questions?", { x: 0.6, y: 6.5, w: 12.1, h: 0.45, fontSize: 18, bold: true, color: HEX.lt1, margin: 0, isTextBox: true });
slide.addNotes("The app runs every step live. The main lesson is measured, not assumed. Limitations: 32 x 32 input cannot read text, and extremely faded signs still fail.");

(async () => {
  await pres.writeFile({ fileName: OUTPUT });
  if (SKILL) {
    const { applyTheme } = require(path.join(SKILL, "scripts", "apply_theme.js"));
    await applyTheme(OUTPUT, THEME);
  }
  console.log(`Saved ${OUTPUT}`);
})();
