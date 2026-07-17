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
(AMP). The reported results come from a full-data run on a Kaggle T4 (batch size 64,
`configs/kaggle.yaml`); the config also ships a batch-size-16 default that fits a 6GB
laptop GPU (NVIDIA RTX 3050) for local iteration. Training requires CUDA. A guard in the
training entry point raises `SystemExit` when no CUDA device is visible, which stops a
run from quietly falling back to CPU and producing timings and results that represent
nothing.

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

All numbers below are read directly from the audit CSVs in `results/baseline/`
and `results/mitigated/`. The reported model is the baseline unless a column
says otherwise. Training early-stopped on validation macro AUROC (best at epoch
5, patience 3); the test macro AUROC is 0.801, below the 0.840 validation peak,
as expected for held-out data.

**(a) Per-label and macro AUROC** (baseline, test set)

| Label | AUROC |
|---|---|
| Atelectasis | 0.764 |
| Cardiomegaly | 0.869 |
| Consolidation | 0.741 |
| Edema | 0.819 |
| Effusion | 0.827 |
| Emphysema | 0.900 |
| Fibrosis | 0.792 |
| Hernia | 0.917 |
| Infiltration | 0.684 |
| Mass | 0.807 |
| Nodule | 0.755 |
| Pleural Thickening | 0.768 |
| Pneumonia | 0.712 |
| Pneumothorax | 0.859 |
| **Macro** | **0.801** |

The per-label ordering matches what the ChestX-ray14 literature reports:
Hernia, Emphysema, and Cardiomegaly rank highest, while Infiltration and
Pneumonia sit near 0.68-0.71.

**(b) Subgroup AUROC and gap by sex** (baseline)

| Label | AUROC (M) | AUROC (F) | Gap |
|---|---|---|---|
| Cardiomegaly | 0.872 | 0.866 | 0.007 |
| Effusion | 0.834 | 0.816 | 0.018 |
| Atelectasis | 0.773 | 0.752 | 0.021 |
| Pneumothorax | 0.856 | 0.867 | 0.011 |

Sex AUROC gaps are small (all below 0.021). By ranking ability alone, the
baseline is roughly even across sex on these four labels.

**(b2) Subgroup AUROC and gap by age** (baseline)

| Label | <40 | 40-60 | 60-80 | 80+ | Gap |
|---|---|---|---|---|---|
| Cardiomegaly | 0.866 | 0.883 | 0.835 | 0.895 | 0.060 |
| Effusion | 0.825 | 0.822 | 0.834 | 0.798 | 0.036 |
| Atelectasis | 0.780 | 0.751 | 0.759 | 0.622 | 0.158 |
| Pneumothorax | 0.862 | 0.861 | 0.861 | 0.760 | 0.102 |

Age is where the model breaks down. Atelectasis AUROC on the 80+ group falls to
0.622 against 0.780 for under-40 (gap 0.158), and Pneumothorax drops to 0.760 on
80+ (gap 0.102). The oldest patients get the worst predictions on exactly the
labels where a miss matters clinically.

**(c) Equalized-odds gaps at Youden-J operating point** (baseline)

| Label | TPR gap (sex) | FPR gap (sex) |
|---|---|---|
| Cardiomegaly | 0.020 | 0.016 |
| Effusion | 0.001 | 0.023 |
| Atelectasis | 0.051 | 0.011 |
| Pneumothorax | 0.125 | 0.098 |

Equalized odds is computed for the sex subgroups only. The standout is
Pneumothorax: at a single decision threshold the model catches true positives
in women far more often than in men (TPR 0.86 vs 0.74, gap 0.125) and also fires
more false positives on women (FPR gap 0.098). A gap this size in a threshold-based
decision is the kind of disparity aggregate AUROC hides.

**(d) Baseline vs. mitigated gap comparison** (sex subgroups)

Mitigation is the `WeightedRandomSampler` retrain described in Method. It targets
the sex subgroups, so the comparison that matters is on the equalized-odds gaps it
optimizes. Negative Δ means the gap shrank (fairer).

| Label | Metric | Baseline | Mitigated | Δ |
|---|---|---|---|---|
| Cardiomegaly | TPR gap (sex) | 0.020 | 0.089 | +0.069 |
| Effusion | TPR gap (sex) | 0.001 | 0.004 | +0.003 |
| Atelectasis | TPR gap (sex) | 0.051 | 0.028 | −0.023 |
| Pneumothorax | TPR gap (sex) | 0.125 | 0.095 | −0.029 |
| Cardiomegaly | FPR gap (sex) | 0.016 | 0.027 | +0.011 |
| Effusion | FPR gap (sex) | 0.023 | 0.017 | −0.006 |
| Atelectasis | FPR gap (sex) | 0.011 | 0.001 | −0.010 |
| Pneumothorax | FPR gap (sex) | 0.098 | 0.064 | −0.034 |

Macro AUROC moves only 0.801 → 0.797 under mitigation, so accuracy is essentially
preserved. The fairness effect is mixed rather than a clean win: FPR gaps narrow on
three of four labels, and the largest disparity (Pneumothorax) improves on both TPR
and FPR. But the Cardiomegaly TPR gap blows up roughly fourfold (0.020 → 0.089),
which drags the mean TPR gap the wrong way (0.049 → 0.054). Group-balanced resampling
did not reliably close the equalized-odds gaps here, and it regressed one label badly.

## Fairness Findings

Three findings hold up across the metrics above.

**Age drives the largest disparities, not sex.** The baseline's sex AUROC gaps stay
under 0.021, but its age gaps reach 0.158 (Atelectasis) and 0.102 (Pneumothorax),
concentrated in the 80+ bin. The AUROC gaps and the equalized-odds gaps agree on
direction for the sex axis: Pneumothorax is the worst offender on both the ranking
metric and the threshold-based one.

**The worst subgroup harm is a threshold effect the aggregate score hides.** On
Pneumothorax, the baseline reaches 0.859 macro-label AUROC yet still splits TPR by
0.125 between sexes at its operating point. Ranking ability and decision fairness are
not the same property, and this project measures both because a model can look even on
one and be badly skewed on the other.

**A naive mitigation is not a reliable fix.** Group-balanced resampling on sex left
macro AUROC essentially unchanged (0.801 → 0.797) and narrowed most FPR gaps, but it
did not close the TPR gaps overall and quadrupled the Cardiomegaly TPR gap
(0.020 → 0.089). Reporting this negative result is the point: subgroup fairness has to
be measured per label and per metric after any intervention, because an intervention
that helps in aggregate can still regress a specific label, and only a direct
before/after audit surfaces it.

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
