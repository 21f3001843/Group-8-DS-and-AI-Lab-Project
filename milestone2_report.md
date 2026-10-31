**Chest X-ray Multi-Source Dataset (CheXpert and NIH ChestX-ray14)**

Milestone 2 Report: Data Preparation, Dataset Documentation, EDA and Data Splits

# Summary

This project combines two public chest radiograph datasets, CheXpert (Stanford) and NIH ChestX-ray14, to build a multi-label thoracic abnormality model. This report documents how the data were prepared before model experimentation in Milestone 3. Only frontal-view images are used. Six abnormality labels were selected using combined frequency and a requirement that each abnormality has a plausible anatomical localisation: Pleural Effusion, Cardiomegaly, Pneumothorax, Atelectasis, Consolidation and Lung Lesion. A seventh column, No Finding, is kept as a flag and is not a training target.

All images are converted to a common format (224 x 224, single channel, histogram equalised) by one deterministic function applied identically to both sources. Splits are made at patient level so that no patient appears in more than one split. Section 3.7 reports a limitation found during validation: the two sources remain distinguishable from simple image statistics, which affects how results should be reported in Milestone 3.

# 1. Data Preparation

## 1.1 Scope of the prepared dataset

- CheXpert: frontal views only (lateral views removed), 191,229 images.
- NIH ChestX-ray14: all 112,120 images (all are frontal views).
- Total: 303,349 images. Each row of the final manifest is one image with its patient ID, source, split and six label columns plus No Finding.

## 1.2 Label selection and harmonisation

The two datasets use different label vocabularies. Labels were ranked by total frequency across both sources (merging equivalent labels), then filtered by whether the abnormality has a clear anatomical location that can later be used for localisation. The mapping used is shown below.

| **Project label**      | **CheXpert column** | **NIH column(s)**              | **Note**                             |
| ---------------------- | ------------------- | ------------------------------ | ------------------------------------ |
| Pleural Effusion       | Pleural Effusion    | Effusion                       | Direct equivalent                    |
| ---                    | ---                 | ---                            | ---                                  |
| Cardiomegaly           | Cardiomegaly        | Cardiomegaly                   | Direct equivalent                    |
| ---                    | ---                 | ---                            | ---                                  |
| Pneumothorax           | Pneumothorax        | Pneumothorax                   | Direct equivalent                    |
| ---                    | ---                 | ---                            | ---                                  |
| Atelectasis            | Atelectasis         | Atelectasis                    | Direct equivalent                    |
| ---                    | ---                 | ---                            | ---                                  |
| Consolidation          | Consolidation       | Consolidation                  | Direct equivalent                    |
| ---                    | ---                 | ---                            | ---                                  |
| Lung Lesion            | Lung Lesion         | Mass OR Nodule                 | Approximate: union of two NIH labels |
| ---                    | ---                 | ---                            | ---                                  |
| No Finding (flag only) | No Finding          | Finding Labels == "No Finding" | Not a training target                |
| ---                    | ---                 | ---                            | ---                                  |

Labels not selected and the reason: Support Devices (not a pathology), Lung Opacity and Infiltration (non-specific umbrella terms), Edema (diffuse, no focal location; kept out of the shortlist), Pneumonia (clinical diagnosis rather than a radiographic sign; low count), Enlarged Cardiomediastinum (redundant with Cardiomegaly), Fracture, Pleural Other and Pleural Thickening (catch-all or subtle), Emphysema and Fibrosis (diffuse), and Hernia (227 positives only).

## 1.3 Missing and uncertain labels

- CheXpert blank (not mentioned in the report) values in the label columns are set to 0. The imputation is applied to the label columns only; Age and other metadata are left untouched so that missing values stay missing.
- CheXpert uncertain labels (-1) are kept as -1 in the manifest and are not resolved at preparation time. The policy is chosen when the dataset object is created: "ignore" (masked out of the loss, default), "zeros" or "ones". This allows the policy to be changed in Milestone 3 without rebuilding the data.
- NIH has binary labels only, so it has no uncertain class. A 0 in NIH means "not labelled positive", not "confirmed absent".

## 1.4 Image preprocessing pipeline

Every image goes through the same steps, in this order:

- Load and convert to single-channel grayscale.
- Resize to 224 x 224 using Lanczos interpolation. Aspect ratio is not preserved (stretch): NIH images are square, CheXpert images are wider than tall (median aspect ratio about 1.22), and letterbox padding would add black bars to CheXpert only.
- Global histogram equalisation (OpenCV equalizeHist) on the resized image.
- At load time: scale to \[0, 1\], optionally replicate to three channels, and normalise with ImageNet mean and standard deviation so that ImageNet-pretrained backbones can be used directly.
- Training only: small random affine augmentation (rotation up to 7 degrees, translation up to 5 percent, scale 0.93 to 1.07). Horizontal flips are deliberately not used because left and right carry anatomical meaning (heart position).

## 1.5 Final format and readiness for Milestone 3

- manifest.csv: one row per image with source, image path, patient ID, original split, final split (train, val, test, test_official), the six label columns (values 1, 0 or -1) and the No Finding flag.
- A cached array of preprocessed images (N x 224 x 224, uint8) indexed by a cache_idx column in the manifest, so training reads images from one file instead of decoding the original images each epoch.
- A dataset class that returns (image, labels, mask) per sample, where the mask removes uncertain labels from the loss. Class weights are computed from the training split only.
- The label set, input size, channel count and uncertain-label policy are parameters, so different architectures and input sizes can be tried without restructuring the data.

# 2. Dataset Documentation

## 2.1 Datasets selected

|                                   | **CheXpert**                                                                                        | **NIH ChestX-ray14**                                                                                                    |
| --------------------------------- | --------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Provider                          | Stanford ML Group, Stanford University                                                              | NIH Clinical Center                                                                                                     |
| ---                               | ---                                                                                                 | ---                                                                                                                     |
| Origin of images                  | Stanford Hospital, retrospective                                                                    | NIH Clinical Center PACS                                                                                                |
| ---                               | ---                                                                                                 | ---                                                                                                                     |
| Full release size                 | 224,316 radiographs, 65,240 patients                                                                | 112,120 frontal images, 30,805 patients                                                                                 |
| ---                               | ---                                                                                                 | ---                                                                                                                     |
| Used in this project              | 191,229 frontal images                                                                              | 112,120 images                                                                                                          |
| ---                               | ---                                                                                                 | ---                                                                                                                     |
| Image format and size (as loaded) | JPEG, downsampled copy; short side 320 px, width 320-444 and height 320-390 in a 1,000-image sample | PNG, 1024 x 1024                                                                                                        |
| ---                               | ---                                                                                                 | ---                                                                                                                     |
| Label source                      | Rule-based labeller on radiology reports; 14 observations as positive, negative or uncertain        | NLP text-mining of radiology reports; 14 findings, binary, multi-label; NIH reports labelling accuracy above 90 percent |
| ---                               | ---                                                                                                 | ---                                                                                                                     |
| Kaggle copy used                  | ashery/chexpert                                                                                     | nih-chest-xrays/data                                                                                                    |
| ---                               | ---                                                                                                 | ---                                                                                                                     |

## 2.2 Source, licence and usage permissions

**NIH ChestX-ray14.** Source: NIH Clinical Center (<https://nihcc.app.box.com/v/ChestXray-NIHCC>). Use of the images is unrestricted. The attribution requirements are: link to the NIH download site, cite Wang et al., "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases", CVPR 2017, and acknowledge the NIH Clinical Center as the data provider.

**CheXpert.** Source: Stanford ML Group (<https://stanfordmlgroup.github.io/competitions/chexpert/>). The dataset is released under a Stanford University Dataset Research Use Agreement; users must register and agree to it before download, and use is limited to research purposes. Reference: Irvin et al., "CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison", AAAI 2019. The agreement text should be consulted for the terms on redistribution of the images or derived data.

Both datasets were accessed through Kaggle copies in the notebook environment. Neither dataset is used for any clinical or commercial purpose in this project.

## 2.3 Metadata available

- CheXpert: Path (contains patient ID), Sex, Age, Frontal/Lateral, AP/PA, and 14 observation columns.
- NIH: Image Index, Finding Labels, Follow-up #, Patient ID, Patient Age, Patient Gender, View Position, Original Image Size, Original Image Pixel Spacing; official train_val and test split lists; and a bounding-box file (BBox_List_2017.csv) covering roughly 1,000 images.

## 2.4 Classes included

Counts below are image-level, over all frontal images of each source (CheXpert 191,229 images; NIH 112,120 images).

**CheXpert**

| **Label**        | **Positive** | **Negative (after blank to 0)** | **Uncertain (-1)** | **% positive** |
| ---------------- | ------------ | ------------------------------- | ------------------ | -------------- |
| Pleural Effusion | 76,963       | 104,688                         | 9,578              | 40.25          |
| ---              | ---          | ---                             | ---                | ---            |
| Cardiomegaly     | 23,451       | 161,071                         | 6,707              | 12.26          |
| ---              | ---          | ---                             | ---                | ---            |
| Pneumothorax     | 17,700       | 170,821                         | 2,708              | 9.26           |
| ---              | ---          | ---                             | ---                | ---            |
| Atelectasis      | 29,795       | 131,571                         | 29,863             | 15.58          |
| ---              | ---          | ---                             | ---                | ---            |
| Consolidation    | 13,015       | 153,833                         | 24,381             | 6.81           |
| ---              | ---          | ---                             | ---                | ---            |
| Lung Lesion      | 7,041        | 183,080                         | 1,108              | 3.68           |
| ---              | ---          | ---                             | ---                | ---            |

**NIH ChestX-ray14**

| **Label**                    | **Positive** | **Negative (not labelled positive)** | **Uncertain** | **% positive** |
| ---------------------------- | ------------ | ------------------------------------ | ------------- | -------------- |
| Pleural Effusion (Effusion)  | 13,317       | 98,803                               | n/a           | 11.88          |
| ---                          | ---          | ---                                  | ---           | ---            |
| Cardiomegaly                 | 2,776        | 109,344                              | n/a           | 2.48           |
| ---                          | ---          | ---                                  | ---           | ---            |
| Pneumothorax                 | 5,302        | 106,818                              | n/a           | 4.73           |
| ---                          | ---          | ---                                  | ---           | ---            |
| Atelectasis                  | 11,559       | 100,561                              | n/a           | 10.31          |
| ---                          | ---          | ---                                  | ---           | ---            |
| Consolidation                | 4,667        | 107,453                              | n/a           | 4.16           |
| ---                          | ---          | ---                                  | ---           | ---            |
| Lung Lesion (Mass or Nodule) | 11,207       | 100,913                              | n/a           | 10.00          |
| ---                          | ---          | ---                                  | ---           | ---            |

In CheXpert most "negative" labels are blanks rather than explicit negatives. The raw breakdown before imputation is shown in Section 3.3.

## 2.5 Sample images

_Figure 1. Sample CheXpert images at the native size of the Kaggle copy (390 x 320 pixels). Top row: random images. Bottom row: lowest-contrast images in the 1,000-image sample. The images are shown as they are provided in the dataset copy used (already histogram equalised, see Section 3.5)._

_\[Insert Figure 2: random sample grid of NIH images (one row), before preprocessing. Source: Phase 2 sampling cell.\]_

_\[Insert Figure 3: one example image per selected label (Pleural Effusion, Cardiomegaly, Pneumothorax, Atelectasis, Consolidation, Lung Lesion) and one No Finding example, for each source.\]_

_\[Insert Figure 4: 12-image grid after preprocessing (top row CheXpert, bottom row NIH). Source: Step 2 check cell.\]_

## 2.6 Localisation annotations

NIH provides bounding boxes (BBox_List_2017.csv, roughly 1,000 images) for a subset of findings; CheXpert provides no localisation annotations. NIH data is intended to support localisation of the selected abnormalities in Milestone 3. The bounding-box file has not yet been analysed or integrated into the manifest. Because NIH images are square, the stretch resize to 224 x 224 is a uniform scale of 224/1024 on both axes, so box coordinates can be rescaled directly once the file is inspected.

# 3. Exploratory Data Analysis and Preprocessing Documentation

## 3.1 Notebooks

The EDA and preprocessing notebook (dsai-preprocessing.ipynb together with the Phase 1, Phase 2 and pipeline cells) accompanies this report. The sections below summarise what each analysis showed and the decision it led to.

## 3.2 Metadata comparison

The two sources were compared on size, patients, images per patient, age, sex and view position (PA versus AP). Patient overlap between splits was checked in each source.

_\[Insert Figure 5: age distribution (density), sex split and view-position split by source. Source: Phase 1, Cell 2.\]_

_\[Insert Table: size by source and split (images, patients, images per patient). Source: Phase 1, Cell 1.\]_

## 3.3 Label analysis

CheXpert raw label breakdown before imputation (frontal images, 191,229 in total):

| **Label**        | **Positive**    | **Uncertain**   | **Negative (explicit 0)** | **Blank (not mentioned)** |
| ---------------- | --------------- | --------------- | ------------------------- | ------------------------- |
| Pleural Effusion | 76,963 (40.25%) | 9,578 (5.01%)   | 25,437 (13.30%)           | 79,251 (41.44%)           |
| ---              | ---             | ---             | ---                       | ---                       |
| Cardiomegaly     | 23,451 (12.26%) | 6,707 (3.51%)   | 8,005 (4.19%)             | 153,066 (80.04%)          |
| ---              | ---             | ---             | ---                       | ---                       |
| Pneumothorax     | 17,700 (9.26%)  | 2,708 (1.42%)   | 48,018 (25.11%)           | 122,803 (64.22%)          |
| ---              | ---             | ---             | ---                       | ---                       |
| Atelectasis      | 29,795 (15.58%) | 29,863 (15.62%) | 1,152 (0.60%)             | 130,419 (68.20%)          |
| ---              | ---             | ---             | ---                       | ---                       |
| Consolidation    | 13,015 (6.81%)  | 24,381 (12.75%) | 19,778 (10.34%)           | 134,055 (70.10%)          |
| ---              | ---             | ---             | ---                       | ---                       |
| Lung Lesion      | 7,041 (3.68%)   | 1,108 (0.58%)   | 977 (0.51%)               | 182,103 (95.23%)          |
| ---              | ---             | ---             | ---                       | ---                       |

Observations that shaped the decisions:

- Atelectasis and Consolidation have very large uncertain shares (15.6% and 12.8%), so the uncertain-label policy matters most for these two. This is why -1 is preserved and resolved at load time.
- Blank labels dominate the negative class (for example 68% blank against 0.6% explicit negative for Atelectasis). Treating blank as negative is the standard assumption but means the negative class is mostly "not mentioned".
- Prevalence differs strongly between sources: Cardiomegaly is 12.3% in CheXpert and 2.5% in NIH; Pleural Effusion is 40.3% and 11.9%; Lung Lesion is 3.7% and 10.0%. A model trained on pooled data could exploit source identity as a proxy for the label (see Section 3.7).
- No Finding was added as a flag. It is not an independent target (if it is 1, the six labels are 0), but all-zero for the six labels does not mean normal because other findings (for example Edema or Infiltration) may be present. The flag separates clean normals from other pathology.

_Figure 6. Share of images that are positive, uncertain or negative for each selected label, by source (all frontal images). NIH has no uncertain class._

_\[Insert Figure 7: shared-label prevalence comparison between sources and the full label inventory table. Source: Phase 1, Cell 3.\]_

_\[Insert Figure 8: Spearman train/test label-rank comparison for NIH (from the notebook).\]_

## 3.4 Image quality comparison

A random sample of 1,000 images per source was measured (resolution, intensity statistics, sharpness as variance of the Laplacian, noise estimate). Median values:

| **Metric**                               | **CheXpert** | **NIH** | **Comment**                          |
| ---------------------------------------- | ------------ | ------- | ------------------------------------ |
| File size (KB)                           | 51.7         | 395.6   | JPEG 320 px vs PNG 1024 px           |
| ---                                      | ---          | ---     | ---                                  |
| Mean intensity                           | 129.15       | 122.50  | CheXpert near 127.5 (flat histogram) |
| ---                                      | ---          | ---     | ---                                  |
| Std of intensity                         | 73.75        | 60.91   | Flat histogram gives about 73.9      |
| ---                                      | ---          | ---     | ---                                  |
| Dynamic range (p99 - p1)                 | 249          | 223     |                                      |
| ---                                      | ---          | ---     | ---                                  |
| Entropy (bits)                           | 7.99         | 7.48    | Maximum is 8.0                       |
| ---                                      | ---          | ---     | ---                                  |
| % pure white pixels                      | 2.32         | 0.01    |                                      |
| ---                                      | ---          | ---     | ---                                  |
| Sharpness at 320 px (Laplacian variance) | 905.8        | 124.5   | Affected by equalisation and JPEG    |
| ---                                      | ---          | ---     | ---                                  |
| Noise estimate at 320 px                 | 4.13         | 0.88    | Indicative only                      |
| ---                                      | ---          | ---     | ---                                  |

_Figure 9. Distributions of image quality metrics by source (1,000 images per source): mean, standard deviation, dynamic range, number of gray levels, percentage of black pixels, sharpness (log10 Laplacian variance at 320 px), noise estimate (log10) and file size (log10 KB). CheXpert values are tightly clustered for intensity statistics, consistent with uniform histogram equalisation._

## 3.5 Finding: the CheXpert images are already histogram equalised

The CheXpert intensity statistics (mean 129, std 73.8, entropy 7.99) are almost identical to those of a perfectly flat histogram (127.5, 73.9, 8.0) and vary very little between images (IQR of the standard deviation 73.58 to 73.99). To test this directly, the distance between each image's cumulative histogram and the equalised diagonal was computed (0 means perfectly flat):

| **Source** | **n** | **Mean distance** | **Median** | **75th percentile** | **Max** |
| ---------- | ----- | ----------------- | ---------- | ------------------- | ------- |
| CheXpert   | 1000  | 0.0061            | 0.0049     | 0.0062              | 0.0921  |
| ---        | ---   | ---               | ---        | ---                 | ---     |
| NIH        | 1000  | 0.0968            | 0.0926     | 0.1330              | 0.3409  |
| ---        | ---   | ---               | ---        | ---                 | ---     |

CheXpert images are therefore uniformly equalised in the copy used here (the stage at which this was applied is not documented in the Kaggle copy), whereas NIH images have natural intensity variation. Consequences: absolute original intensities cannot be recovered for CheXpert; CheXpert sharpness and noise values are inflated by the equalisation and are not a fair measure of scanner quality; and applying CLAHE to CheXpert would amplify noise further, so it is not used.

## 3.6 Preprocessing steps and justification

| **Step**                                 | **What it does**                                                          | **Why**                                                                                                                                |
| ---------------------------------------- | ------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Frontal-only filter (CheXpert)           | Removes lateral views                                                     | NIH contains frontal views only; mixing views would add a variation unrelated to the task                                              |
| ---                                      | ---                                                                       | ---                                                                                                                                    |
| Label harmonisation                      | Maps both vocabularies to six labels; Lung Lesion = Mass or Nodule in NIH | Allows the two sources to be trained together on a common label set                                                                    |
| ---                                      | ---                                                                       | ---                                                                                                                                    |
| Blank to 0 in label columns only         | Blank CheXpert labels become negative                                     | Blank means the finding was not mentioned; applying to Age would have produced false age 0 values                                      |
| ---                                      | ---                                                                       | ---                                                                                                                                    |
| Keep -1 uncertain                        | Preserved through to load time                                            | Policy (ignore, zeros, ones) is an experiment choice for Milestone 3; Atelectasis and Consolidation are affected most                  |
| ---                                      | ---                                                                       | ---                                                                                                                                    |
| Patient-level splitting                  | Splits grouped by patient ID                                              | Patients have several images; image-level splits would leak patient information across splits                                          |
| ---                                      | ---                                                                       | ---                                                                                                                                    |
| Grayscale conversion                     | Single channel                                                            | X-rays carry no colour; guarantees a consistent input                                                                                  |
| ---                                      | ---                                                                       | ---                                                                                                                                    |
| Resize to 224 x 224 (stretch)            | Lanczos resampling, no padding                                            | Standard input size for common pretrained backbones; padding was removed because it let a classifier identify the source (Section 3.7) |
| ---                                      | ---                                                                       | ---                                                                                                                                    |
| Global histogram equalisation            | Applied to every image after resize                                       | CheXpert is already equalised; applying the same operation to NIH makes the intensity distributions comparable                         |
| ---                                      | ---                                                                       | ---                                                                                                                                    |
| ImageNet normalisation, 3-channel option | Applied at load time                                                      | Allows ImageNet-pretrained models to be used without changes                                                                           |
| ---                                      | ---                                                                       | ---                                                                                                                                    |
| Affine augmentation, no flips            | Small rotation, shift, scale, training only                               | Adds robustness to positioning; flips would swap left and right anatomy                                                                |
| ---                                      | ---                                                                       | ---                                                                                                                                    |
| Class weights from the training split    | neg / pos per label                                                       | Compensates for class imbalance without using validation or test data                                                                  |
| ---                                      | ---                                                                       | ---                                                                                                                                    |

## 3.7 Validation of the preprocessing

After resize and equalisation, 200 images per source were compared (median values):

| **Source** | **Mean** | **Std** | **Distance to flat CDF** | **Laplacian variance** |
| ---------- | -------- | ------- | ------------------------ | ---------------------- |
| CheXpert   | 127.77   | 73.76   | 0.0014                   | 845.5                  |
| ---        | ---      | ---     | ---                      | ---                    |
| NIH        | 127.48   | 74.13   | 0.0039                   | 405.2                  |
| ---        | ---      | ---     | ---                      | ---                    |

Mean, standard deviation and flatness now match. A visual check of random preprocessed images from both sources showed no flipped, inverted, rotated or non-chest images and no obvious outliers. A residual sharpness difference remains (CheXpert about 2 times higher), which is expected because NIH is reduced by a factor of about 4.6 and CheXpert by about 1.4.

To test whether a model could still identify the source, a gradient-boosted classifier was trained to predict the source (CheXpert or NIH) from image features, using 600 images per source (one image per patient) with 5-fold cross-validation. Values are ROC AUC (0.5 means indistinguishable):

| **Features**               | **Letterbox padding (initial)** | **Stretch (chosen)** |
| -------------------------- | ------------------------------- | -------------------- |
| Intensity (mean, std)      | 0.968                           | 0.976                |
| ---                        | ---                             | ---                  |
| Intensity + sharpness      | 0.993                           | 0.990                |
| ---                        | ---                             | ---                  |
| Padding amount only        | 0.903                           | 0.557                |
| ---                        | ---                             | ---                  |
| Layout (16 x 16 thumbnail) | 0.974                           | 0.906                |
| ---                        | ---                             | ---                  |
| All features               | 0.997                           | 0.996                |
| ---                        | ---                             | ---                  |

Interpretation:

- Stretching removes the padding cue (0.903 to 0.557) and reduces the layout cue (0.974 to 0.906). This is why stretch was chosen over letterbox padding.
- The sources remain highly separable from intensity and sharpness statistics (AUC 0.97 to 0.99) even though median mean and standard deviation match. Matching medians does not imply matching distributions; the CheXpert per-image statistics are tightly clustered, which a classifier can exploit. The specific feature driving this has not yet been isolated.
- Because the source also predicts label prevalence (Section 3.3), pooled performance in Milestone 3 may partly reflect source recognition. Planned mitigations are to report all metrics per source as well as pooled, to compare a pooled model against per-source models, and to evaluate options for reducing the residual gap (for example matching sharpness).

## 3.8 Known limitations

- Labels in both sources are derived automatically from radiology reports and contain noise. NIH negatives mean "not labelled positive".
- Lung Lesion is an approximate match between CheXpert Lung Lesion and NIH Mass plus Nodule.
- Counts are image-level; CheXpert has about three images per patient on average, so patient-level counts are lower.
- Original CheXpert intensities are not recoverable in the copy used, because the images are equalised.
- The NIH bounding boxes have not yet been analysed.

# 4. Train / Validation / Test Split Details

## 4.1 Split strategy

- Splits are made at patient level (grouped by patient ID); a check confirms that zero patients appear in more than one split.
- NIH: the official test list is kept as the test split. The official train_val list is split 90/10 by patient into train and val.
- CheXpert: the official validation set (frontal images only) is kept as a separate test split. The official training set is split 90/10 by patient into train and val.
- The split uses a fixed random seed (42) so it can be reproduced.

## 4.2 Split sizes

| **Source** | **Split**                      | **Images** | **Patients** |
| ---------- | ------------------------------ | ---------- | ------------ |
| CheXpert   | train                          | 171,927    | 58,080       |
| ---        | ---                            | ---        | ---          |
| CheXpert   | val                            | 19,100     | 6,454        |
| ---        | ---                            | ---        | ---          |
| CheXpert   | test (official validation set) | 202        | 200          |
| ---        | ---                            | ---        | ---          |
| NIH        | train                          | 77,988     | 25,207       |
| ---        | ---                            | ---        | ---          |
| NIH        | val                            | 8,536      | 2,801        |
| ---        | ---                            | ---        | ---          |
| NIH        | test (official test set)       | 25,596     | 2,797        |
| ---        | ---                            | ---        | ---          |
| Total      |                                | 303,349    |              |
| ---        | ---                            | ---        | ---          |

## 4.3 Positive counts per split

| **Source / split** | **Pleural Effusion** | **Cardiomegaly** | **Pneumothorax** | **Atelectasis** | **Consolidation** | **Lung Lesion** | **No Finding** |
| ------------------ | -------------------- | ---------------- | ---------------- | --------------- | ----------------- | --------------- | -------------- |
| CheXpert train     | 69,277               | 20,994           | 15,849           | 26,804          | 11,688            | 6,356           | 15,219         |
| ---                | ---                  | ---              | ---              | ---             | ---               | ---             | ---            |
| CheXpert val       | 7,622                | 2,391            | 1,844            | 2,916           | 1,295             | 684             | 1,755          |
| ---                | ---                  | ---              | ---              | ---             | ---               | ---             | ---            |
| CheXpert test      | 64                   | 66               | 7                | 75              | 32                | 1               | 26             |
| ---                | ---                  | ---              | ---              | ---             | ---               | ---             | ---            |
| NIH train          | 7,852                | 1,549            | 2,394            | 7,405           | 2,584             | 7,362           | 45,485         |
| ---                | ---                  | ---              | ---              | ---             | ---               | ---             | ---            |
| NIH val            | 807                  | 158              | 243              | 875             | 268               | 793             | 5,015          |
| ---                | ---                  | ---              | ---              | ---             | ---               | ---             | ---            |
| NIH test           | 4,658                | 1,069            | 2,665            | 3,279           | 1,815             | 3,052           | 9,861          |
| ---                | ---                  | ---              | ---              | ---             | ---               | ---             | ---            |

Uncertain (-1) labels per CheXpert split (the CheXpert test split has none):

| **Split** | **Pleural Effusion** | **Cardiomegaly** | **Pneumothorax** | **Atelectasis** | **Consolidation** | **Lung Lesion** |
| --------- | -------------------- | ---------------- | ---------------- | --------------- | ----------------- | --------------- |
| train     | 8,635                | 6,058            | 2,463            | 26,836          | 21,876            | 1,007           |
| ---       | ---                  | ---              | ---              | ---             | ---               | ---             |
| val       | 943                  | 649              | 245              | 3,027           | 2,505             | 101             |
| ---       | ---                  | ---              | ---              | ---             | ---               | ---             |
| test      | 0                    | 0                | 0                | 0               | 0                 | 0               |
| ---       | ---                  | ---              | ---              | ---             | ---               | ---             |

## 4.4 Notes on the splits

- The CheXpert test split has only 202 images, so per-label evaluation on it is unreliable (for example 1 Lung Lesion and 7 Pneumothorax positives). Its labels come from the official radiologist-annotated validation set, so it is useful as a small reference set; a larger patient-level test set can be carved from the CheXpert training portion if needed.
- The NIH official test set has a much higher disease share than the NIH training set (for example Pneumothorax 10.4% vs 3.1%, Pleural Effusion 18.2% vs 10.1%, No Finding 38.5% vs 58.3%). Threshold-dependent metrics (accuracy, precision, F1) will therefore not be comparable between validation and test; AUROC or AUPRC with prevalence reported alongside is recommended.

## 4.5 Folder structure of the processed dataset

The processed dataset consists of the following files:

```
processed_dataset/
    manifest.csv                   (one row per image: source, patient_id, split, 6 labels, No Finding, cache_idx)
    cxr_224_stretch.npy            (N x 224 x 224, uint8, indexed by cache_idx)
    cxr_224_stretch_done.npy       (boolean flags marking successfully cached rows)
    failed_images.csv              (rows that could not be read, if any)
```

The split of each image is stored in the manifest (column split with values train, val, test and test_official), so a single array file serves all splits.

## 4.6 Hosted dataset

_\[Insert link to the hosted processed dataset (for example on Kaggle) after the caching step is run.\]_