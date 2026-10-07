# IE 7615 – Object Recognition on a Class-Collected Dataset

Course project (**Discriminative Project**) for *IE 7615 Deep Learning for AI*, Northeastern University.
Author: **Sankalp Shinde** (Study Group 8) · contributed object: **OBJ059 · Comb**

Each student photographed one unique everyday object (~100 photos each). The merged dataset has **73 objects (OBJ001–OBJ073)**, listed in [`object_ids.csv`](object_ids.csv). The images themselves are shared through the course Google Drive and are not included in this repository.

| Milestone | Task | Notebook |
|---|---|---|
| 1 | Single-object recognition: compare 2 classifiers and 2 detectors | [`milestone1/Milestone1_All_4_Models.ipynb`](milestone1/Milestone1_All_4_Models.ipynb) |
| 2 | Multi-object detection with YOLOv8: report the ID and location of every object | [`milestone2/Milestone2_YOLOv8_MultiObject.ipynb`](milestone2/Milestone2_YOLOv8_MultiObject.ipynb) |

---

## Milestone 1 – Single-object recognition

Preprocessing: validation, exact-duplicate removal by MD5 hash (28 removed), resizing to 224×224 and a stratified 70/15/15 split. This leaves 7,315 images: 5,120 train, 1,097 validation and 1,098 test.

| Model | Type | Test accuracy | Macro-F1 | Params |
|---|---|---|---|---|
| Custom CNN | Classification, trained from scratch | 89.6% | 0.897 | 2.4 M |
| **EfficientNet-B0** | Classification, transfer learning | **95.4%** | **0.953** | 4.1 M |
| YOLO11s | One-stage detector | 93.6% | 0.936 | 9.5 M |
| Faster R-CNN (R50-FPN v2) | Two-stage detector | 88.9% | 0.888 | 43.6 M |

EfficientNet-B0 was selected as the final model. Full report: [`reports/Milestone1_Report.pdf`](reports/Milestone1_Report.pdf).

## Milestone 2 – Multi-object detection with YOLOv8

1. **Single-object boxes.** Each 224×224 photo gets a tight box around its object from OWLv2, an open-vocabulary detector prompted with the object's name. When OWLv2 isn't confident, the whole photo is used as the box.
2. **Multi-object dataset.** Single-object photos are concatenated into 640×640 composites on random grids (1×2 up to 3×3, so 2–9 objects per image). A class-balanced sampler ensures no Object ID appears twice in one composite. Each object is randomly scaled, positioned and flipped, with color jitter, on a random background. The boxes are exact because each object's position is known by construction.
   - 3,000 train, 500 validation and 500 test composites.
   - Each split is built only from the matching Milestone 1 split, so no photo appears in two splits (checked in code).
3. **Training.** COCO-pretrained **YOLOv8s**, fine-tuned at 640 px with early stopping. Class names are the Object IDs, so the model outputs IDs directly.
4. **Evaluation.**
   - mAP@0.5 and mAP@0.5:0.95.
   - Object-level precision, recall and F1, counting a detection as correct only with the right ID and IoU ≥ 0.5.
   - ID accuracy and exact-image accuracy.
   - Performance by number of objects, plus a stress test on an unseen 4×3 layout with 12 objects.

### Milestone 2 results (500 held-out test images, 2,309 objects)

| Metric | Value |
|---|---|
| mAP@0.5 | **0.821** |
| mAP@0.5:0.95 | **0.754** |
| Precision / Recall / F1 (correct ID and IoU ≥ 0.5, confidence ≥ 0.25) | 0.846 / 0.811 / 0.828 |
| ID accuracy of localized objects | **0.977** |
| Stress test: unseen 4×3 layout with 12 objects | 12 / 12 correct |
| Inference time | 17.4 ms / image on a T4 |

YOLOv8s trained for 71 epochs (best at epoch 51) in 62 minutes on a T4. Nearly all errors concern the box rather than the ID, and are mostly caused by the automatically generated reference boxes. Full report: [`reports/Milestone2_Report.pdf`](reports/Milestone2_Report.pdf).

## How to run

1. Open a notebook in Google Colab and select **Runtime → Change runtime type → T4 GPU**.
2. Add the shared dataset folder to *My Drive* as a shortcut. The notebook finds it automatically.
3. **Run all.** Outputs are saved to `MyDrive/IE_7615/Milestone_<n>_Results/`. If Colab disconnects, *Run all* again: finished steps reload from Drive, and YOLO training resumes from its last checkpoint.

The Milestone 2 notebook reuses the Milestone 1 preprocessing and boxes from Drive if they exist, and recreates them otherwise.

## Detect objects in your own image

Download the trained weights (`yolov8s_multiobject_best.pt`) from this repository's **Releases** page, then run:

```bash
pip install -r requirements.txt
cd milestone2
python predict.py --weights yolov8s_multiobject_best.pt --source my_image.jpg --save-dir predictions
```

Example output:

```
my_image.jpg: 3 objects detected
  OBJ001  Basketball               conf 0.97  box (12, 20, 198, 205)
  OBJ059  Comb                     conf 0.91  box (240, 35, 420, 190)
  OBJ013  Mobile phone             conf 0.88  box (455, 30, 610, 200)
```

In the notebook, `detect_objects(image)` returns the same information as a table.

## Repository structure

```
├── object_ids.csv                    # final Object IDs and names (73)
├── requirements.txt
├── milestone1/
│   └── Milestone1_All_4_Models.ipynb
├── milestone2/
│   ├── Milestone2_YOLOv8_MultiObject.ipynb
│   ├── predict.py                    # command-line detection
│   └── object_ids.py
└── reports/
    ├── Milestone1_Report.pdf
    └── Milestone2_Report.pdf
```
