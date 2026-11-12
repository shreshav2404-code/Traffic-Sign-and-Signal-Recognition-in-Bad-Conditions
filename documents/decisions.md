# Project Decisions

Every major choice in this project, why it was made, and what was rejected.
Last updated: 2 November 2026.

## Open decisions

| Decision | Status |
|---|---|
| Does any course need the network coded from scratch? | **Waiting on faculty.** The small NumPy MLP covers this either way. |
| Report format and deadline for each course | **Waiting on faculty.** |

## Project scope

### 1. One project with three separate reports
- **Decision:** build one application, and write a separate report for each of Deep Learning, Image Processing and Analysis, and Neural Networks and Fuzzy Logic.
- **Why:** one pipeline can cover all three syllabuses, and each report can focus on its own objectives and experiments.
- **Note:** faculty approval for this was received by 31 October 2026.

### 2. Traffic sign and signal recognition in bad conditions
- **Decision:** the project reads traffic signs and signals from damaged or poor-quality images.
- **Why:** on clear general photos, enhancement has little to fix. Dark, hazy or noisy road images give enhancement and fuzzy logic a real job.
- **Rejected:** ChatGPT's NeuroVision AI idea on general photos, chest X-rays (needs more careful data handling), plant leaf disease (images are mostly clean), and object detection or segmentation as the main task.

### 3. Classification comes after enhancement
- **Decision:** the flow is measure, fuzzy decision, denoise, enhance, classify, with the original and enhanced results shown side by side.
- **Why:** classification directly gives the MLP, backpropagation, precision, recall, F1 and confusion matrix that the courses need.

### 4. Full photos use a drag box
- **Decision:** the app accepts a cropped photo, or a full photo where the user drags a box around one sign or light.
- **Why:** the classifier reads one object at a time, and a whole street scene would shrink the sign until it disappears.
- **Status:** built with the `streamlit-cropper` add-on. The box keeps a square shape, and the photo is cropped at full resolution.
- **Update:** YOLO now finds signs and lights automatically, and the drag box stays as a fallback. See decision 29.

### 5. MLP, feature MLP and CNN
- **Decision:** compare an MLP on raw pixels, an MLP on HOG and colour features, and a small CNN.
- **Why:** the MLP covers the syllabus, the feature MLP links image processing to neural networks, and the CNN covers convolution layers.

### 6. Future scope, in this order
1. Train on more datasets for both signs and signals, such as GTSRB, the Indian Traffic Sign Dataset and Bosch Small Traffic Lights.
2. Add YOLO so the app can find signs and signals in full photos by itself.

**Why not at first:** YOLO adds an estimated 3-5 days, adds a new way to fail, and detection isn't in the listed syllabus topics. Each extra dataset needs class mapping and cleaning.
**Update, 2 November 2026:** both steps were brought forward after the street photo test. See decisions 23, 28 and 29.

## Data

### 7. CURE-TSR is the main sign dataset
- **Decision:** use CURE-TSR, with GTSRB as the fallback.
- **Why:** it was built for this topic. It has 12 labelled conditions at 5 severity levels, and each damaged image has a matching clean original, so PSNR and SSIM can be measured.
- **Trade-offs:** 14 sign types against GTSRB's 43, one "speed limit" class, very small crops, and damage added by software.
- **Rejected:** CIFAR-10 and STL-10, which are general photos, and GTSRB as the main set, because it has no condition labels.

### 8. Keep all 14 signs
- **Decision:** keep all 14 signs, even though the real test set has no images of no overtaking, no left or priority.
- **How:** train on real and simulated images, then report real test results for the 11 signs and simulated test results for all 14, clearly labelled.
- **Why:** the user wants the full set, and the labelled results keep this honest.
- **Rejected:** moving real training images into testing, because near-identical frames from one clip could land on both sides.

### 9. LISA for signals, after the sign model works
- **Decision:** add red, yellow and green lights from LISA once the sign model works, for 17 classes in total.
- **Why:** it has about 112,000 labelled lights, and roughly half are at night, which suits the topic.
- **Fallback:** Bosch Small Traffic Lights, which has fewer lights and doesn't list night scenes.
- **Splits:** LISA's four official sequences are the test set, dayClip2 and nightClip4 are validation, and the other clips are training. The two `sample-` folders are exact copies, so they're left out.
- **Balance:** all yellow lights are kept, because they're only about 2.7% of labels. Red and green are limited to 12,000 each in training, 1,500 in validation and 6,000 in testing.
- **Damage:** LISA has no condition labels, so we add darkening, blur, noise and haze at 5 levels. Colour is never removed, because it's the label.
- **Reporting:** signs and lights are reported separately.

### 10. Pack images into arrays instead of extracting them
- **Decision:** resize all 2,206,100 CURE-TSR images to 32×32 and store them in NumPy arrays.
- **Why:** this avoids millions of tiny files and makes training much faster. It took under 2 minutes.

### 11. Protect the real test set
- **Decision:** use real test images only for the final results.
- **How:** tuning and validation use training images. Validation holds out the last 15% of each sign's frames, with a 5% gap before that block.
- **Why:** neighbouring frames are nearly identical, so a random split would inflate the scores.
- **Note:** early image processing checks read a sample of real test images, but nothing was tuned on them. Those checks now use training images.

## Fuzzy logic and enhancement

### 12. Three fuzzy outputs: denoise, stretch and darken
- **Decision:** use three separate outputs, not one enhancement strength.
- **Why:** dark, hazy, noisy and overexposed signs need different fixes.
- **Order:** denoise first, because stretching would amplify any noise left in the image.
- **Finding:** on validation images, fuzzy strongly helps clean-trained models but slightly hurts the CNN trained on all conditions. A CNN trained on enhanced images removes that loss. In the 17-class models it scores 0.907 macro F1 on signs, against 0.905 for the all-conditions CNN on original images.

### 13. Hand-written Mamdani inference
- **Decision:** write the inference with NumPy, using scikit-fuzzy's membership functions.
- **Why:** it runs on whole batches, can show which rules fired, and is easy to explain. It was checked against scikit-fuzzy's own engine.

### 14. New noise measure
- **Decision:** measure colour roughness at half resolution.
- **Why:** the standard Immerkaer estimate works on noise we add ourselves, but barely sees CURE-TSR noise, which is blocky and strongly coloured.
- **Rejected:** a median-based version, which didn't separate noise any better.

### 15. Stretch all RGB channels together
- **Decision:** stretch red, green and blue by the same amount.
- **Why:** it also restores faded colour. On level 3 darkened training signs it reached about 21 dB, against about 13 dB for gamma 0.5.
- **Note:** histogram equalization and CLAHE stay on brightness only, so colours don't shift.

## Training

### 16. Two training regimes
- **Decision:** train each model on clean images only, and on all conditions.
- **Why:** a clean-only model has never seen damage, so it shows what enhancement can do. An all-conditions model shows whether enhancement still adds anything.

### 17. Handle unbalanced classes
- **Decision:** weight the loss by inverse class frequency and report macro F1.
- **Why:** clean real training images range from about 2,480 for parking to 38 for no stopping.

### 18. Small rotations, shifts and zooms, but no flips
- **Decision:** augment images without flipping.
- **Why:** a flip can change a sign's meaning.

### 19. Small NumPy MLP with hand-written backpropagation
- **Decision:** keep this alongside the PyTorch models.
- **Why:** PyTorch hides backpropagation, and the viva will need it explained.

### 20. Tools
- **Decision:** Python 3.11, PyTorch 2.15.0 with CUDA 13.4, OpenCV, scikit-image, scikit-fuzzy, scikit-learn, matplotlib, pandas, Streamlit and streamlit-cropper, all inside a project `.venv`.
- **Why:** the laptop's main Python has a CPU-only PyTorch, which can't use the RTX 4050.

## Final choices

### 21. Fix the choices on validation, before the test sets
- **Decision:** the best model is the CNN trained on all conditions, on original images. The app opens on the CNN trained on clean images, labelled as such.
- **Why:** on validation over 3 seeds, the original and enhanced all-conditions CNNs tie at 0.902 macro F1 on real signs, so the simpler one leads and both are reported. The clean CNN is the app's default because it is where enhancement changes the answer: 0.794 on original images, 0.817 with fuzzy enhancement.
- **Note:** a fixed 50% stretch scores 0.818 on the same clean CNN, level with fuzzy at 0.817. Both are reported, and the fuzzy rules are not retuned after this point.
- **Test result:** the choice held. The all-conditions CNN scored 0.819 on real test signs, and the enhanced CNN 0.814. No model or fuzzy rule changed after the test scoring.

### 22. Report the weak no stopping sign instead of hiding it
- **Decision:** keep no stopping in every macro F1, and explain its effect in the report.
- **Why:** it scored 0.114 F1 on real test signs, after only 38 clean training images. Dropping it would raise the best score from 0.819 to 0.889, but that would be choosing what to count after seeing the test results.
- **Note:** to explain this, the best CNN read the real test set one extra time after scoring, per condition. Nothing was tuned or changed from it.

## Wide model

### 23. Add GTSRB and an Indian set for a second, 84-class model
- **Decision:** train a separate CNN on CURE-TSR, LISA, GTSRB and an Indian traffic sign set, and keep the 17-class results as the main experiments.
- **Why:** tests on downloaded street photos showed 17 classes is too few for a live demo, because any other sign gets a wrong name. The 17-class experiments were already finished and scored once, so they stay as they are.
- **Rejected:** the Kaggle Indian Traffic Sign Dataset (59 classes), because its 13,971 images come from one government image per class, resized and turned grey. The Indian set used is `kannanwisen/Indian-Traffic-Sign-Classification` on Hugging Face: 5,726 real photos in 85 classes. It states no licence.
- **Note:** GTSRB came from its Kaggle copy, because the official server didn't answer.

### 24. Merge the same sign across datasets
- **Decision:** one class per sign. Speed limits stay one class, because CURE-TSR doesn't record the number. Signs merge only when they look alike and mean the same, for example GTSRB stop and Indian stop with CURE-TSR stop.
- **Dropped:** eight Indian classes with fewer than 10 usable images, or none left for training: barrier ahead, crossroads, chevron, ferry, no handcarts, no tongas, turn right and unguarded level crossing.
- **Dropped:** 355 Indian phone photos of whole streets, because a sign in them is too small once the photo is shrunk to 32x32.

### 25. Keep near copies out of the Indian test set
- **Decision:** drop Indian test images whose 8x8 average hash matches a training image, and keep near copies together when splitting validation.
- **Why:** 768 of the set's 1,216 usable test images matched training images, so testing on them would inflate the scores. 448 test images in 64 classes remain.

### 26. Damage the new images the same way as the lights
- **Decision:** darkening, Gaussian blur, noise and haze at 5 levels. Training images get 4 random damaged copies each, and validation and test images get all 20.
- **Why:** GTSRB and the Indian set have no damage of their own. These four conditions are the ones already built for LISA, so the wide model sees the same kinds of damage across all four datasets.

### 27. Fix the wide model choices on validation, before its test sets
- **Decision:** the app opens on the wide CNN trained on all conditions, seed 0, reading original images. The wide CNN trained on clean images stays in the list, because it shows what enhancement can do.
- **Why:** on validation over 3 seeds, the all-conditions wide CNN scores best on real CURE-TSR signs (0.807 macro F1), GTSRB (0.898) and lights (0.994). Fuzzy enhancement doesn't help it (0.800 on real signs), but it lifts the clean wide CNN from 0.766 to 0.790 on real signs and from 0.778 to 0.864 on GTSRB.
- **Cost of more classes:** on the same real CURE-TSR validation signs, the wide CNN scores 0.807 against 0.902 for the 17-class CNN, because it can now mistake them for 67 more classes. The 17-class results stay the main experiments.
- **Weak spot:** the Indian set scores only 0.54 macro F1. Its median class has 20 clean training photos, and 42 of its 67 classes have fewer than 30. Its validation has a median of 4 photos per class, so that score is also noisy.
- **Test result:** scored once, after this choice. The all-conditions wide CNN reached 0.757 macro F1 on real CURE-TSR test signs (17-class CNN: 0.819), 0.881 on simulated signs (0.909), 0.946 on lights (0.980), 0.907 on GTSRB and 0.613 on the Indian set. Fuzzy enhancement again helped the clean wide CNN: 0.688 to 0.725 on real signs, 0.830 to 0.924 on lights and 0.796 to 0.878 on GTSRB. Nothing was changed after these scores.
- **Note:** 18 crops from downloaded street photos were also tried. The wide CNNs named the narrow road sign that the 17-class CNN couldn't, but missed some signs the 17-class CNN got right. Nothing was chosen from these photos.

### 28. Add Mapillary for a third, 106-class model
- **Decision:** train a global CNN on everything the wide CNN uses, plus 36,240 sign crops from the Mapillary Traffic Sign Dataset. It adds 21 classes and an "other sign" class to the 84.
- **Why:** the wide CNN still learned mostly from European and Indian signs of one style, and failed on street photos with other styles, such as the blue square pedestrian crossing. Mapillary has 40,968 ordinary street photos from six continents, labelled by people.
- **Data:** only the fully annotated part, checked against Mapillary's published checksums. Its official test images have no public labels, so its official validation photos are our test set, and 15% of its training photos are our validation set. Panoramas are left out.
- **Crops:** only clearly visible signs with a named type and a longer side of at least 20 pixels, cut out as squares like the other datasets, and damaged the same way.
- **Merges:** every variant of one Mapillary meaning becomes one class, because variants are the same sign drawn in different countries' styles. This goes further than decision 24, so a class can now hold more than one look, for example the warning triangle and the blue square pedestrian crossing.
- **New classes:** a new meaning becomes a class only with at least 100 clear training crops. Rarer named signs become "other sign", so the model can say a sign isn't one it knows. Plates under signs, unsure labels and look-alikes that mean something else are left out, 4,586 crops in all.
- **Licence:** Mapillary is CC BY-NC-SA, which allows non-commercial academic use with credit.
- **App:** at the user's request, the app's model list is kept to four: the global CNN trained on all conditions (opens first), the global CNN trained on clean images to show what enhancement fixes, and the best 17-class CNN and MLP for a live CNN against MLP comparison. Every other model's scores stay in the results tab and the reports.

## Detection

### 29. YOLO finds the signs, and the CNN still names them
- **Decision:** a small YOLO model, pretrained on COCO, learns two classes: sign and light. Each box it finds is cropped and goes through the same fuzzy enhancement and classifier as a hand-drawn box.
- **Why:** the fuzzy logic and the classifier stay at the centre of the project, and the detector only needs box positions, so any dataset with boxes can teach it.
- **Data:** LISA frames for lights, split by clip exactly as for the classifier, so its test sequences stay unseen. Only every third training frame is used, because neighbouring frames are nearly identical. Mapillary adds the signs, with its frames shrunk so the longer side is 1,280 pixels, and the same splits as decision 28.
- **Missing labels:** each dataset labels only one class, so the other class shows up unlabelled. To stop YOLO learning that Mapillary's traffic lights are background, COCO-pretrained YOLO marks the lights it finds with at least 50% confidence in Mapillary's training and validation frames. Test frames keep only human labels, and each dataset is scored only on the class its people labelled. LISA's unlabelled signs are left as they are, because LISA is a small share of the frames.
- **Rejected:** one YOLO model that names all 84 classes, because it would skip the fuzzy enhancement and the classifiers being compared.
- **Note:** flips are allowed when training the detector, because they don't change whether something is a sign or a light.

### 30. Fix the global model and detector choices on validation, before their test sets
- **Decision:** the app opens on the global CNN trained on all conditions, seed 0 as for the other models, reading original images. The global CNN trained on clean images stays second, because enhancement changes its answers the most. YOLO keeps boxes it is at least 30% sure of.
- **Longer training:** the all-conditions global CNN trains for 30 epochs with patience 10, not 12 epochs with patience 4. At 12 epochs, seed 0 scored only 0.692 macro F1 on Mapillary validation signs, and 30 epochs raised it to 0.787. Patience 4 then stopped seed 2 at epoch 14, at 0.737 on the whole validation set, while the learning rate was still high. With patience 10 it reached 0.806. The 17-class and wide models keep their old settings, so their scored results stay as they were.
- **Why this default:** on validation over 3 seeds, the all-conditions global CNN scores best on real CURE-TSR signs (0.833 macro F1), simulated signs (0.889), lights (0.996) and GTSRB (0.943). The clean global CNN with fuzzy enhancement is a little ahead on Mapillary (0.818 against 0.797) and the Indian set (0.718 against 0.704), but well behind on the rest, for example 0.787 on real signs.
- **Enhancement:** fuzzy enhancement lifts the clean global CNN on every group: real signs 0.757 to 0.787, lights 0.903 to 0.980, GTSRB 0.801 to 0.888 and Mapillary 0.755 to 0.818. It doesn't help the all-conditions global CNN (0.829 on real signs, 0.804 on Mapillary).
- **Cost of Mapillary:** none found. At the same 12 epochs as the wide CNN, the global CNN already scored the same or better on the shared validation groups: real signs 0.819 against 0.807, GTSRB 0.933 against 0.898, the Indian set 0.646 against 0.537, and lights 0.993 against 0.994.
- **Detector threshold:** on validation frames, F1 peaks near 0.30 confidence for Mapillary signs and 0.39 for LISA lights, and is nearly flat between them, so one threshold of 0.30 is used. At validation, YOLO scores 0.693 mAP50 on Mapillary signs (precision 0.823, recall 0.599) and 0.725 on LISA lights (precision 0.739, recall 0.755).
- **Test result:** scored once, after this choice, and the choice held. The all-conditions global CNN reached 0.816 macro F1 on real CURE-TSR test signs (wide CNN: 0.757, 17-class CNN: 0.819), 0.888 on simulated signs, 0.951 on lights, 0.940 on GTSRB, 0.719 on the Indian set and 0.797 on Mapillary. Fuzzy enhancement again lifted the clean global CNN on every group but the Indian set: real signs 0.713 to 0.745, lights 0.839 to 0.941, GTSRB 0.821 to 0.893 and Mapillary 0.750 to 0.815, which is the best Mapillary score of the four. YOLO scored 0.694 mAP50 on Mapillary test signs (precision 0.803, recall 0.618) and 0.663 on LISA test lights (precision 0.713, recall 0.615). Nothing was changed after these scores.

### 31. Let YOLO's sign or light answer limit the classifier in the app
- **Decision:** when YOLO finds a box in an uploaded photo, the classifier chooses only among the lights if YOLO says light, and only among the signs if YOLO says sign. Its scores within that kind are rescaled to sum to 1. Sample images and hand-drawn boxes are unchanged.
- **Why:** on 11 downloaded street photos, YOLO found all 18 hand-marked signs and lights, but both global CNNs named two close-up green lights as signs ("parking" and "turn right ahead"). They learned lights only from LISA's small, distant lights, which look very different. The 17-class CNN named both correctly, because it has only 14 signs to confuse them with.
- **Effect on the photos:** counting only the found signs and lights each model knows, the global CNNs named 12 of 16 correctly before the rule and 14 of 16 with it. The 17-class CNN named 11 of 12 either way. Fuzzy enhancement changed none of these answers, because the photos are clear daytime shots.
- **Note:** this rule was chosen from these street photos, after the test sets were scored. It changes only the app's photo mode, and no reported score. If YOLO calls the kind wrong, the classifier can't correct it.

### 32. Teach the global CNN and YOLO about worn signs, beside the finished models
- **Decision:** train a new global CNN, "all conditions + wear", and fine-tune YOLO on photos of damaged signs. The finished models and their scores stay as they are, and the new ones are reported beside them.
- **Why:** in a phone test, no model named an old stop sign with most of its red peeled off, and YOLO didn't find it. Painting the peeled areas red again brought stop back, so the models lean on colour. None of the datasets used so far has worn signs.
- **Worn paint:** a new damage type (`restoration.add_wear`, condition 13) fades the colour and peels patches away at 5 levels, inside a disc around the sign's centre. Peeled colour shows the pale backing sheet, white paint peels to darker metal, black lettering stays, and levels 3 to 5 add rust spots. Every clean sign training image gets 2 worn copies at random levels, and every validation and test image gets all 5 levels. Lights get none, because their colour is their label.
- **Kaggle Damaged Signs Dataset (CC0):** 3,194 photos of Portuguese signs, boxed in 9 sign groups. Only stop, give way and no entry match our classes, so only those crops join the classifier, with the usual 4 damaged copies for training. Every box teaches YOLO "sign". Photos are split 70/15/15 in blocks of 20 neighbouring file numbers, so repeat shots of one sign mostly stay on one side.
- **Roboflow Detection_of_damage_signs (CC BY 4.0, by Trafficresearch):** 1,996 photos boxed by kind of damage (faded, peeled, rusted, defaced and more), not by sign, so it only teaches YOLO. It keeps its own train, valid and test split.
- **Rule fixed before any result:** the new CNN becomes the app's default only if, on validation, it scores clearly better on worn signs and no worse than the spread of 3 seeds on the old groups. The new detector replaces the old one on the same rule. The test sets are scored once, after that choice.

### 33. The worn-sign CNN and detector become the app's defaults, chosen on validation
- **Detector:** the fine-tuned YOLO replaces the old one, as decision 32's rule asks, because it scores higher on every validation set: mAP50 0.777 against 0.725 on LISA lights, 0.701 against 0.693 on Mapillary signs, 0.884 against 0.790 on Kaggle's damaged signs, and 0.892 against 0.000 on Roboflow's damaged close-ups. Its F1 still peaks near 0.30 confidence for signs (0.29 to 0.36) and stays flat up to 0.49 for lights, so the threshold stays at 0.30.
- **CNN:** the global CNN trained on all conditions plus wear becomes the app's default, at the user's choice after seeing validation, although it misses decision 32's stricter rule. Over 3 seeds it scores far higher on worn signs (real CURE-TSR 0.855 against 0.633, simulated 0.911 against 0.734, GTSRB 0.875 against 0.644, Indian 0.540 against 0.405, Mapillary 0.642 against 0.464). On the old groups it is level on real signs (0.830 against 0.833) and lights (0.996), but lower by more than the seed spread on simulated signs (0.881 against 0.889), GTSRB (0.936 against 0.943) and Mapillary (0.783 against 0.797).
- **Why override the rule:** the project is about signs in bad conditions, and worn signs are common on real roads. The choice was made on validation only, and both CNNs are scored once on the test sets and reported side by side.
- **App:** the four models are now the worn-sign global CNN (opens first), the global CNN trained on clean images, and the 17-class CNN and MLP. The global CNN without wear leaves the app, but its scores stay in the results tab and reports.
- **Note:** Kaggle's 3 real classes are easy for every model on validation (0.96 macro F1), so the worn-paint copies carry most of the evidence about wear. Fuzzy enhancement leaves worn signs almost unchanged, because they are bright and sharp, and it didn't change their scores.
- **Test result:** scored once, after this choice, and it held. Over 3 seeds on original input, the worn-sign CNN scored far higher on worn test signs: real CURE-TSR 0.786 against 0.597, simulated 0.900 against 0.678, GTSRB 0.867 against 0.629, Indian 0.592 against 0.440 and Mapillary 0.649 against 0.469. On the usual test sets it moved by under 0.01: real signs 0.808 against 0.816, simulated 0.881 against 0.888, lights 0.959 against 0.951, GTSRB 0.932 against 0.940, Indian 0.724 against 0.719 and Mapillary 0.789 against 0.797. Both scored 0.968 on Kaggle's real damaged-sign crops. The fine-tuned detector scored mAP50 0.680 against 0.663 on LISA lights, 0.701 against 0.694 on Mapillary signs, 0.853 against 0.749 on Kaggle's damaged signs and 0.893 against 0.003 on Roboflow's. Nothing was changed after these scores.
- **The faded stop sign:** the fine-tuned detector now finds it, at 73% confidence, but no CNN names it: all 3 seeds give stop about 0%. About three quarters of its red is gone and its white letters have turned dark, which is past the strongest simulated wear. It stays a known limitation, rather than tuning the wear to one photo.

### 34. Move to PyTorch 2.5.1 after Windows blocked PyTorch 2.15
- **Problem:** on 11 November, Windows Smart App Control started blocking `torch.dll` and `torch_cpu.dll` from PyTorch 2.15.0, so training, the app and YOLO could not start. It blocks new, little-used library files, as it did with pandas 3.1.0 on 4 November. The setting stays on: turning it off is the user's choice, and Windows can't turn it back on without a reset.
- **Decision:** with the user's yes, install PyTorch 2.5.1 and torchvision 0.20.1 for CUDA 12.4, an older and widely used release that Windows allows. It was tried in a separate environment first, where it loaded and ran on the GPU.
- **Check:** all 59 saved CNN checkpoints and both YOLO detectors load unchanged, and all 262 checks pass (43 image processing, 36 fuzzy, 108 deep learning, 75 app). On the 12 validation groups, the app's model (worn-sign global CNN, seed 0, original input) scores within 0.0005 macro F1 of the scores stored under PyTorch 2.15, which is GPU rounding and changes no reported number. On a street photo, the first detector finds the same 6 objects at the same confidences as on 3 November. No test set was scored again.
