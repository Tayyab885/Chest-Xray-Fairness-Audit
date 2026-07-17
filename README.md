# Fairness Audit of a Chest X-ray Classifier (NIH ChestX-ray14)

## Abstract

Deep learning models for chest radiograph interpretation now approach
radiologist-level performance on aggregate metrics. Aggregate accuracy can still
hide systematic underperformance on particular patient subgroups, which has direct
clinical consequences once a model is deployed for triage or diagnosis. This project
trains a DenseNet121 multi-label classifier for 14 thoracic pathologies on the NIH
ChestX-ray14 dataset using leakage-free, patient-level data splits, then audits it
for performance disparities across sex and age subgroups using per-subgroup AUROC
gaps and equalized-odds TPR/FPR gaps. We also attempt a group-balanced resampling
mitigation and report whether it narrows the observed gaps, including the case where
it does not. What the project offers is a reproducible protocol for measuring and
reporting subgroup fairness in medical image classifiers, together with the results
that protocol produces, whichever direction they point.

## Dataset

[NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) contains
112,120 frontal-view chest radiographs from 30,805 unique patients, each
labeled for the presence of up to 14 thoracic pathologies (Atelectasis,
Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia,
Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax).
Labels were extracted from radiology reports using NLP mining rather than
per-image expert adjudication, and are therefore noisy (see Limitations).
The release also ships an official patient-level train/validation and test
split (`train_val_list.txt`, `test_list.txt`), which this project uses as
the basis for all data partitioning.

### Download

```bash
kaggle datasets download -d nih-chest-xrays/data
```

Unzip the archive so that the following live somewhere under `data/`:

- the image folders (nested `images_001/images/` ... `images_012/images/` is
  fine as-is, since images are located automatically by filename via a
  recursive scan, with no manual flattening required)
- `Data_Entry_2017.csv` (per-image labels and patient/demographic metadata)
  directly in `data/`
- `train_val_list.txt` directly in `data/`
- `test_list.txt` directly in `data/`

## Method

### Leakage-free patient-level splits

A model can inflate its validation and test AUROC if images from the same patient
appear in more than one split, because different views or follow-up scans of one
patient are highly correlated. We use the official NIH `train_val_list.txt` and
`test_list.txt` partition, which is already defined by patient rather than by image,
and carve the validation set out of `train_val_list.txt` by patient ID, so no
patient's images are split across train and validation. A unit test asserts zero
patient-ID overlap between every pair of splits, and it gates the build.

### Model & training

The classifier is a DenseNet121 backbone pretrained on ImageNet with a 14-way sigmoid
output head, one unit per pathology, trained with `BCEWithLogitsLoss` for independent
multi-label prediction. Images are resized to 224×224 and trained with mixed precision
(AMP) at batch size 16 to fit the target GPU (NVIDIA RTX 3050, 6GB VRAM). Training
requires CUDA. A guard in the training entry point raises `SystemExit` when no CUDA
device is visible, which stops a run from quietly falling back to CPU and producing
timings and results that represent nothing.

### Fairness metrics

Subgroups are defined by sex (M/F) and age, binned into `<40`, `40-60`,
`60-80`, `80+`. For a fixed set of deep-dive labels (Cardiomegaly, Effusion,
Atelectasis, Pneumothorax), we report:

- Per-subgroup AUROC and its gap. AUROC is computed separately within each subgroup
  for a given label, and the gap is `max(AUROC) - min(AUROC)` across subgroups. A
  large gap means the model discriminates disease presence much better for some
  groups than others, independent of any decision threshold.
- Equalized-odds TPR and FPR gaps. At a single operating point (probability threshold
  chosen on the validation set via Youden's J statistic, the point maximizing
  TPR - FPR), we compute the true-positive-rate and false-positive-rate gap between
  subgroups. These capture disparities in the decisions the model would actually
  make, which its ranking ability alone can hide.

### Explainability

Grad-CAM runs on the last convolutional block of the DenseNet121 backbone
(`features.norm5`) to show which image regions drive each prediction. We generate
saliency maps per subgroup for the deep-dive labels, to check whether the model
attends to clinically plausible regions consistently across groups or picks up
different, possibly spurious cues for different groups.

### Mitigation

As a first mitigation attempt, we retrain the model with a `WeightedRandomSampler`
that upweights each example by `1 / group-size`, equalizing how often each sex
subgroup is sampled (`group_key="sex"` by default, `tag="mitigated"`). The mitigated
model goes through the same fairness audit as the baseline, and we compare the gaps
directly. Resampling may leave a gap unchanged or widen it, and those outcomes are
reported alongside the ones where it helps.

## Reproducing

```bash
pip install -r requirements.txt
```

Download the dataset (see above) into `data/`, then:

```bash
python -m src.train
python -m src.run_audit --checkpoint checkpoints/baseline.pt
python -m src.mitigate
python -m src.run_audit --checkpoint checkpoints/mitigated.pt
```

For fast development iterations without a full epoch over 112k images, set
`subsample_frac` in `configs/default.yaml` to a value below 1.0 to train on
a random patient-level subsample.

## Results

_Results pending the full training run. The tables below are empty on purpose. They
will be filled from the audit output, not from estimates._

**(a) Per-label and macro AUROC**

| Label | AUROC |
|---|---|
| Atelectasis | |
| Cardiomegaly | |
| Consolidation | |
| Edema | |
| Effusion | |
| Emphysema | |
| Fibrosis | |
| Hernia | |
| Infiltration | |
| Mass | |
| Nodule | |
| Pleural Thickening | |
| Pneumonia | |
| Pneumothorax | |
| **Macro** | |

**(b) Subgroup AUROC and gap by sex**

| Label | AUROC (M) | AUROC (F) | Gap |
|---|---|---|---|
| Cardiomegaly | | | |
| Effusion | | | |
| Atelectasis | | | |
| Pneumothorax | | | |

**(b2) Subgroup AUROC and gap by age**

| Label | <40 | 40-60 | 60-80 | 80+ | Gap |
|---|---|---|---|---|---|
| Cardiomegaly | | | | | |
| Effusion | | | | | |
| Atelectasis | | | | | |
| Pneumothorax | | | | | |

**(c) Equalized-odds gaps at Youden-J operating point**

| Label | TPR gap (sex) | FPR gap (sex) | TPR gap (age) | FPR gap (age) |
|---|---|---|---|---|
| Cardiomegaly | | | | |
| Effusion | | | | |
| Atelectasis | | | | |
| Pneumothorax | | | | |

**(d) Baseline vs. mitigated gap comparison**

| Label | Metric | Baseline gap | Mitigated gap | Δ |
|---|---|---|---|---|
| Cardiomegaly | AUROC gap (sex) | | | |
| Effusion | AUROC gap (sex) | | | |
| Atelectasis | AUROC gap (sex) | | | |
| Pneumothorax | AUROC gap (sex) | | | |

## Fairness Findings

_Pending the full training run._ This section will report which subgroups, by sex and
by age bin, show the largest AUROC and equalized-odds gaps for each deep-dive label,
whether those gaps agree across metrics, and what group-balanced resampling does to
them. If the mitigation fails to help, that is what will appear here.

## Grad-CAM

_Figures pending._ This section will present side-by-side Grad-CAM saliency
maps for representative examples from each subgroup, for each deep-dive
label, to visually assess whether the model attends to comparable anatomical
regions across sex and age groups.

## Limitations

- **Label noise.** ChestX-ray14 labels are NLP-mined from radiology reports,
  not adjudicated per-image by radiologists, and are known to contain a
  non-trivial error rate.
- **Compute budget.** Training runs on a single 6GB laptop GPU, which
  constrains batch size and may require subsampling or a shortened training
  schedule; absolute AUROC is not expected to match large-scale results such
  as CheXNet, which trained on more compute.
- **Demographic coverage.** NIH ChestX-ray14 provides only sex and age
  metadata, with no race or ethnicity, so this fairness audit is necessarily
  limited to those two axes and cannot speak to disparities along other
  clinically relevant demographic dimensions.
- **Single-dataset scope.** All results come from one institution's dataset
  with one label-extraction pipeline; no external validation on a separate
  hospital system or population is performed.

## References

Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M. (2017).
ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on
Weakly-Supervised Classification and Localization of Common Thorax Diseases.
*CVPR 2017*.

Rajpurkar, P., Irvin, J., Zhu, K., Yang, B., Mehta, H., Duan, T., Ding, D.,
Bagul, A., Langlotz, C., Shpanskaya, K., Lungren, M. P., & Ng, A. Y. (2017).
CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep
Learning. *arXiv:1711.05225*.
