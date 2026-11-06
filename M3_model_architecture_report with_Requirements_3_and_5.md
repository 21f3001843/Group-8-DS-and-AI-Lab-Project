Model Architecture: Selection, Justification and Visualisation

Sections 1, 2, 3, 5 and 6 of the Milestone 3 report

# Overview

The project builds a multi-label classifier for six chest X-ray findings (Pleural Effusion, Cardiomegaly, Pneumothorax, Atelectasis, Consolidation and Lung Lesion) and uses the model's uncertainty to decide whether a smartphone photo of an X-ray should be recaptured. CheXpert and NIH ChestX-ray14 are trained separately and never pooled, because after preprocessing a simple classifier can still tell the two sources apart from image statistics (source-prediction AUC about 0.99). CheXphoto, which contains photos and synthetic photo transformations of CheXpert X-rays, supplies the photo data. The NIH bounding boxes (880 images, 984 boxes) are used to check that the abnormality localisation is sensible.

Several training setups are used for different purposes. They are defined in Section 1.3 and shown in Figure 2.

# 1. Model Architecture Selection

## 1.1 Requirements that drive the choice

Multi-label output: six findings can be present at the same time, so each finding has its own sigmoid output.

Noisy and uncertain labels: CheXpert labels come from report text and include an uncertain (-1) class; NIH labels are binary.

Robustness to photo artifacts: the model is tested on photos of X-rays that it was not designed around.

Calibrated uncertainty: the recapture decision depends on knowing when the model is unsure.

A localisation check: it must be possible to produce a region map for each finding and score it against the NIH boxes.

Modest compute: training must fit on Kaggle-class GPUs, including several repeated runs for an ensemble.

## 1.2 Architectures selected

| Component | Choice | Details |

| --- | --- | --- |

| Primary backbone | DenseNet-121 (ImageNet-pretrained) | About 8M parameters, about 2.9 GFLOPs at 224 x 224; 1024 x 7 x 7 feature map |

| Comparison backbone | ResNet-50 (ImageNet-pretrained) | About 25.6M parameters, about 4.1 GFLOPs; 2048 x 7 x 7 feature map |

| Comparison backbone | EfficientNet-B0 (ImageNet-pretrained) | About 5.3M parameters, about 0.4 GFLOPs; 1280 x 7 x 7 feature map |

| Classification head | Average pool, dropout (about 0.2), linear layer, 6 sigmoid outputs | Used by every setup |

| Patch head (optional) | 1 x 1 convolution on the feature map giving 6 x P x P patch scores; image score is the maximum over patches | Used only in Setup E (box-supervised NIH model) |

| Uncertainty layer | Deep ensemble of 3 to 5 seeds, temperature scaling per finding, optional MC dropout | Applied to the best configuration |

| Localisation method | Grad-CAM ++ region maps from the trained classifier | Used by all setups; scored against NIH boxes |

How the selected components work. The classification head is the final part of the network that converts the backbone's feature map into six finding probabilities. Average pooling summarises the spatial feature map, dropout helps reduce overfitting, and a linear layer produces one score for each finding. A sigmoid converts each score to a value between 0 and 1. Because this is multi-label classification, more than one finding can be predicted for the same X-ray.

The optional patch head is an additional branch, not a replacement for the classification head. A 1 x 1 convolution produces a score for each of the six findings at each location in the feature map (a 6 x P x P grid). During Setup E, NIH bounding boxes provide spatial supervision: patches overlapping a box are treated as positive for that finding. The maximum patch score can be used as the image-level score. This tests whether box supervision teaches the network more useful spatial features.

The uncertainty layer is not a new neural-network layer. It is an inference and calibration procedure applied to trained models. A deep ensemble means training three to five copies of the same architecture with different random seeds. Their average prediction is the ensemble probability; disagreement (spread) between their predictions is an uncertainty signal. Temperature scaling, fitted separately for each finding on validation data, adjusts probabilities to improve calibration. MC dropout may be added as a second uncertainty estimate, but the ensemble is the primary choice in this report.

Overall approach. The three backbones are compared on CheXpert only, crossed with three uncertain-label policies (ignore, zeros, ones). The comparison runs in a funnel: short runs on a subset of the training patients, then full training of the top two combinations. The winner is chosen on CheXpert validation by mean AUROC with confidence intervals and by calibration error. That backbone is then used in every setup. Only the backbone changes between the compared models; the data, loss, optimiser and evaluation are identical, so the comparison is fair.

## 1.3 Experimental setups

| Setup | Trained on | Purpose | Evaluated on | Status |

| --- | --- | --- | --- | --- |

| A | CheXpert digital images (train split) | Baseline: how much performance is lost on photos and on another source | CheXpert test, NIH test (cross-source), CheXphoto photos | Core |

| B | CheXpert plus CheXphoto training photos (natural and synthetic) | Main classifier of the pipeline: a photo-aware model | CheXpert test, CheXphoto photos of held-out patients, NIH boxed images for localisation | Core |

| C | NIH (train split) | Independent baseline; reference for the cross-source gap and for localisation without a source shift | NIH test, NIH boxes, CheXpert test | Core (best backbone only) |

| E | NIH train images plus box supervision through the patch head | Tests whether training with boxes improves localisation compared with Setup C | NIH test and held-out NIH boxes | Conditional |

Setup E explained. Setup C is the NIH-trained image-level classification baseline. Setup E uses the same NIH training split and backbone, but adds the patch head and supervises it with the available bounding boxes. Comparing C with E on the same held-out NIH images tests whether box supervision improves localisation beyond the Grad-CAM ++ maps produced by an ordinary classifier. Because the annotations are limited, E is conditional: it is run only if the box-supervised approach is feasible and a patient-separated held-out evaluation can be maintained.

The uncertainty layer (ensemble and temperature scaling) is not a separate setup; it is added to the best of A and B (and E if it is run). Setup E is run only if the box supervision gives a clear improvement on NIH (decision rule: train C and E with identical settings, compare localisation and classification on the held-out NIH boxes and test set; if E is not clearly better, the boxes are kept as an evaluation set only).

Data separation: CheXphoto training photos come from CheXpert training patients. Patients used for photo evaluation are removed from every training set, and the split is checked by patient ID, so that no X-ray or patient is used for both training and photo testing.

## 1.4 How the setups fit together in the pipeline

Train the classifier. The main classifier is trained on CheXpert together with the CheXphoto training photos, which carry the CheXpert labels (Setup B). Setup A (CheXpert digital only) is trained for comparison.

Validate localisation on NIH boxed images. For each finding that has boxes, a Grad-CAM ++ map is computed on the held-out NIH box images and compared with the ground-truth box: pointing game (the map peak falls in the box), share of map energy inside the box, and hit rate at IoU thresholds of 0.1 and 0.3. The box images are split by patient into a tuning part (to choose the map method and threshold) and a test part (scored once). Five of the six findings have boxes (Consolidation does not). This is a cross-dataset check, so it also reflects the source difference between CheXpert-lineage models and NIH images.

Test on CheXphoto photos. The ensemble produces, for each photo, a calibrated probability per finding and an uncertainty estimate (spread across seeds).

Assess the recapture criteria. A photo is flagged for recapture if, for any of the six findings, the calibrated probability falls inside an uncertain band, or the spread across seeds exceeds a threshold. The band and threshold are set on validation data to reach a chosen flag rate, and results are reported across a range of rates (risk-coverage curve). The recapture rule is not trained; it is evaluated on held-out data.

Second photo and re-check. If a photo is flagged, a second version of the same film is generated for the simulation and classified by the same model. The predictions are combined (for example, by averaging calibrated probabilities; replacement and an uncertainty-weighted average can be compared on validation data). The combined result is then checked against the recapture criteria again. This loop can repeat until the criteria are no longer met or a pre-set maximum number of recaptures is reached; if the criteria are not met, the current prediction is final.

How recapture is evaluated. A real second photograph is not available in the dataset, so recapture benefit will be tested in a separate controlled simulation. Starting with held-out CheXphoto images, create artificially degraded or distorted versions using controlled severity levels (for example, blur, glare, reduced contrast, perspective distortion or cropping). Run each version through the model, then progressively reduce the applied degradation to create a sequence of improving image-quality levels for the same underlying X-ray. At each level, record the predicted probability/confidence, ensemble uncertainty, calibration and whether the classification is correct. The expected pattern is that predictions become more reliable as image quality improves; an increase in confidence alone is not sufficient, because a model can become confidently wrong. Report whether confidence and calibration improve alongside classification accuracy and whether uncertainty falls. This experiment estimates sensitivity to image quality; it does not demonstrate the benefit of a real user taking a second photograph.

Region-specific recapture. In the main line (Setups A and B), the model flags a photo using finding-level uncertainty. Grad-CAM ++ provides a region map that can guide which part of the film to inspect or retake; it is not a precise lesion boundary. Setup E is a conditional experiment that tests whether box supervision improves the region map.

# 2. Architecture Justification

## 2.1 Suitability for the dataset and problem

DenseNet-121 [1] is the backbone of CheXNet [5] and is the family used for the original CheXpert baselines [6], so published results exist for direct reference.

It is compact (about 8M parameters), so a 3 to 5 seed ensemble is affordable (about 40M parameters in total, against about 128M for five ResNet-50 models).

It is convolutional, so Grad-CAM ++ [13] region maps can be taken from the last feature map and scored against the NIH boxes.

A sigmoid output per finding fits the multi-label problem, and a masked loss handles the uncertain CheXpert labels without discarding images.

ImageNet-pretrained weights are used because CheXtransfer [8] found a statistically significant gain from pretraining on CheXpert, particularly for smaller architectures.

## 2.2 Evidence from the literature

| Source | Model | Reported result |

| --- | --- | --- |

| CheXpert paper [6], as tabulated in [7] | DenseNet-121 baselines | Mean AUC 0.893 (U-Ones), 0.895 (U-MultiClass), 0.894 (U-SelfTrained) on the CheXpert validation set, 5 competition findings |

| Pham et al. [7] | Single DenseNet-121 | 0.863 (U-Ignore), 0.860 (U-Ones), 0.894 (U-Ones with conditional training and label smoothing) |

| Pham et al. [7] | Ensemble of 6 CNNs | 0.940 mean AUC on the same validation set |

| CheXNet [5], ChestX-ray14 | DenseNet-121 | Atelectasis 0.809, Cardiomegaly 0.925, Effusion 0.864, Mass 0.868, Nodule 0.780, against 0.716, 0.807, 0.784, 0.706 and 0.671 for Wang et al. [17] |

| CheXtransfer [8] | 16 ImageNet CNNs on CheXpert | ImageNet accuracy does not predict CheXpert performance; pretraining gives a significant gain, larger for smaller models; size changes within a family move AUC by only 0.003 to 0.005 without pretraining |

| CheXphotogenic [11] | Eight CheXpert-challenge models | All dropped significantly on photos, by 0.036 MCC on average |

| CheXpedition [12] | CheXpert-challenge models | Average AUC 0.916 on photos against 0.924 on the original images |

These numbers come from different splits, label subsets and protocols, so they show how each architecture behaves in the literature and are not a head-to-head comparison with this project. The photo results show that the average loss on photos is small, which is why the photo evaluation here is reported per finding and per case, not only as an average.

## 2.3 Expected advantages over alternative approaches

| Alternative | Comparison and decision |

| --- | --- |

| ResNet-50 | Competitive on chest X-rays but about three times the parameters, which makes ensembles costly. Kept as a comparison backbone. |

| EfficientNet-B0 | Cheapest to train and ensemble, but fine-tuning is generally more sensitive to learning rate and batch size. Kept as a comparison backbone. |

| Vision transformer [4] | Typically needs large-scale pretraining and more compute, and attention maps are harder to compare with boxes. Not selected; this is a judgement and not a benchmark result. |

| Small custom CNN | Raghu et al. [9] found small networks can match ImageNet models on CheXpert, but pretraining speeds convergence and gave a significant gain in [8]. Not selected. |

| Detection network trained on boxes (for example Faster R-CNN) | Needs far more annotated boxes than the 880 available. Replaced by a classifier with Grad-CAM ++, and by the limited-supervision patch head of Li et al. [18] in Setup E. |

| Single model | Cheaper, but the ensemble of Pham et al. [7] improved on a single model (0.940 against 0.894) and the ensemble spread is the uncertainty signal needed for recapture decisions. |

| Pooled CheXpert and NIH training | Rejected: the sources are easily distinguished from image statistics and have very different label prevalence (for example Cardiomegaly 12.3 percent and 2.5 percent), so a pooled model could learn the source as a shortcut. |

## 2.4 Design decisions and modifications to the original architectures

The 1000-class ImageNet layer is replaced by a 6-output linear layer with sigmoid, with dropout before it so MC dropout [15] is available as a second uncertainty estimate.

Masked binary cross-entropy: uncertain (-1) labels are excluded (ignore) or mapped to 0 or 1; class weights come from the training split only.

Input: images are resized to 224 x 224, histogram equalised, and grayscale images are replicated into three channels to match the three-channel input expected by ImageNet-pretrained networks. The three channels are then normalised using the standard ImageNet channel means (0.485, 0.456, 0.406) and standard deviations (0.229, 0.224, 0.225). This normalisation rescales input values to the range/distribution expected by the pretrained weights; it is not an additional image-enhancement step. Images are stretched to a square (no padding), because padding bars allowed a classifier to identify the source.

Augmentation: small rotation, shift and scale on training images only. Horizontal flips are not used because left and right carry anatomical meaning.

Uncertainty: a deep ensemble of 3 to 5 independently trained seeds [14] is the primary uncertainty method. Temperature scaling [16], fitted on validation data separately for each finding, calibrates the predicted probabilities. MC dropout [15] is an optional additional estimate, rather than the main uncertainty signal.

Patch head (Setup E): a 1 x 1 convolution produces a grid of scores for each finding. Patches that overlap an NIH bounding box are supervised as positive for that finding; images without boxes retain image-level supervision only. The patch head is an optional box-supervised branch, while the ordinary classification head remains in place.

## 2.5 Challenges and trade-offs

Resolution against localisation: at 224 pixels the final feature map is only 7 x 7, so region maps are coarse for small findings. A larger input (for example 320 pixels, the native CheXphoto size) is tested as an ablation.

Small evaluation sets: the official CheXpert validation set has 202 frontal images and the NIH boxes cover 880 images, so differences of a point or two between architectures will not be statistically meaningful. Confidence intervals are reported.

Noisy labels: both label sets come from report text; NIH negatives mean "not labelled positive", and for some findings almost all CheXpert negatives are blanks (for example Atelectasis: 130,419 blank against 1,152 explicit negatives).

Source shift: models trained on CheXpert are scored on NIH boxes, and the NIH-trained reference (Setup C) is needed to separate the effect of the source shift from the weakness of the map method.

Simulated recapture: no real second photograph is available, so the recapture effect is estimated through controlled artificial degradation and progressive quality restoration of held-out CheXphoto images. This measures whether predictions become more accurate, better calibrated and less uncertain as quality improves; it is not a measured clinical or real-world recapture effect.

Compute: three backbones, three uncertain policies, ensembles and several setups are only feasible with the staged funnel.

# 3. Baseline Model Performance

## 3.1 Evaluation objective and protocol

The purpose of this section is to record pretrained-model performance before additional project-specific training or hyperparameter tuning, as required for Milestone 3. The primary baseline is a CheXpert-pretrained DenseNet-121 from TorchXRayVision (`densenet121-res224-chex`), evaluated without additional fine-tuning on the project's internal CheXpert validation split.

The evaluation uses the six project findings in the following order: Pleural Effusion, Cardiomegaly, Pneumothorax, Atelectasis, Consolidation and Lung Lesion. The validation set contains 19,100 frontal images from 6,454 patients. The split is patient-level and was reconstructed using `GroupShuffleSplit` with `test_size=0.10` and `random_state=42`. The same 19,100 image paths, in the same order, were verified against the validation split used in the earlier DenseNet-121 experiment. Raw labels and definite-label masks were also identical.

For each finding, only definite positive (1) and definite negative (0) labels contribute to the evaluation. Uncertain (-1) and missing labels are excluded using the `ignore` policy. AUROC and average precision (AUPRC) are reported. AUROC measures ranking discrimination across thresholds; AUPRC summarises the precision-recall trade-off and is particularly affected by the prevalence of positive labels. The XRV model outputs are used as ranking scores for these metrics; they are not treated as calibrated probabilities.

The XRV preprocessing was applied as required by the pretrained checkpoint: image loading and intensity normalisation to the XRV range, centre cropping, resizing to 224 x 224, and single-channel tensor input. This differs from the ImageNet-pretrained model's three-channel ImageNet-normalised preprocessing, so the two configurations should not be described as an isolated comparison of pretraining source alone.

## 3.2 Pretrained baseline results

The CheXpert-pretrained DenseNet-121 was evaluated on all 19,100 validation images. The number of evaluable images differs by finding because uncertain and missing labels are masked.

| Finding | Positive | Negative | Evaluated | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pleural Effusion | 7,622 | 2,586 | 10,208 | 0.9151 | 0.9602 |
| Cardiomegaly | 2,391 | 805 | 3,196 | 0.9001 | 0.9595 |
| Pneumothorax | 1,844 | 4,706 | 6,550 | 0.7133 | 0.4781 |
| Atelectasis | 2,916 | 105 | 3,021 | 0.6647 | 0.9756 |
| Consolidation | 1,295 | 2,004 | 3,299 | 0.9116 | 0.8703 |
| Lung Lesion | 684 | 84 | 768 | 0.6583 | 0.9333 |
| **Macro average** | — | — | — | **0.7939** | **0.8628** |

The model shows strong AUROC for Pleural Effusion, Cardiomegaly and Consolidation. Pneumothorax performance is lower. Atelectasis and Lung Lesion have particularly small definite-negative counts (105 and 84 respectively); their high AUPRC values must therefore be interpreted alongside the very high positive-label prevalence and limited negative sample size. These values do not establish strong or clinically reliable performance for those findings.

The ROC and precision-recall curves for this evaluation are saved in the Kaggle working directory at `m3_chexpert_pretrained/plots/roc_curves.png` and `m3_chexpert_pretrained/plots/precision_recall_curves.png`. The per-finding metrics and prediction arrays are saved as `m3_chexpert_pretrained/validation_metrics.csv` and `m3_chexpert_pretrained/validation_predictions.npz` respectively. The prediction archive contains predictions, targets, masks and finding names.

## 3.3 Research benchmark comparison

A relevant external reference is the TorchXRayVision benchmark, which reports per-finding AUROC for DenseNet-121 on 20% of each dataset. Its CheXpert table includes the multi-dataset-pretrained `densenet121-res224-all` model. The reported CheXpert AUROCs for our six findings are shown below.

| Finding | Project CheXpert-pretrained baseline AUROC | TorchXRayVision published AUROC |
| --- | ---: | ---: |
| Pleural Effusion (Effusion) | 0.9151 | 0.94 |
| Cardiomegaly | 0.9001 | 0.91 |
| Pneumothorax | 0.7133 | 0.85 |
| Atelectasis | 0.6647 | 0.91 |
| Consolidation | 0.9116 | 0.90 |
| Lung Lesion | 0.6583 | 0.84 |

The project baseline is relatively close to the published reference for Pleural Effusion, Cardiomegaly and Consolidation. Larger differences are present for Pneumothorax, Atelectasis and Lung Lesion. These published values are research benchmarks rather than universal industry acceptance thresholds. They are not a strict head-to-head comparison: the published row uses a multi-dataset-pretrained checkpoint and a different evaluation sample/protocol, whereas this project evaluates a CheXpert-pretrained checkpoint on its reconstructed patient-level internal validation split. Label masking, preprocessing and the number of definite negatives also affect comparability. No directly comparable per-finding AUPRC values were identified in this benchmark table, so AUPRC is reported for the project experiments without an external numerical comparison.

## 3.4 Supplementary fine-tuned experiment and comparison

A separate DenseNet-121 experiment used ImageNet-1K pretrained weights, replaced the original classifier with a six-output head, and fine-tuned the model on the full internal CheXpert training split for five epochs. The best checkpoint was from epoch 3. This is a trained project experiment, not the unadapted pretrained baseline described in Section 3.2. It is included here as a supplementary reference to show the performance observed after project-specific fine-tuning.

| Finding | ImageNet-pretrained, fine-tuned AUROC | ImageNet-pretrained, fine-tuned AUPRC | CheXpert-pretrained baseline AUROC | CheXpert-pretrained baseline AUPRC |
| --- | ---: | ---: | ---: | ---: |
| Pleural Effusion | 0.9559 | 0.9804 | 0.9151 | 0.9602 |
| Cardiomegaly | 0.9442 | 0.9790 | 0.9001 | 0.9595 |
| Pneumothorax | 0.8268 | 0.6526 | 0.7133 | 0.4781 |
| Atelectasis | 0.6749 | 0.9786 | 0.6647 | 0.9756 |
| Consolidation | 0.9427 | 0.9225 | 0.9116 | 0.8703 |
| Lung Lesion | 0.8091 | 0.9683 | 0.6583 | 0.9333 |
| **Macro average** | **0.8589** | **0.9136** | **0.7939** | **0.8628** |

The fine-tuned experiment has higher macro AUROC and macro AUPRC than the unadapted CheXpert-pretrained baseline on this validation split. The two runs nevertheless differ in more than whether weights were fine-tuned: they use different pretrained checkpoints and different input preprocessing pipelines. Therefore, the result is a comparison of two experimental configurations, not a controlled estimate of the effect of fine-tuning alone. Both were evaluated on the same validation images, in identical order, with identical raw labels and definite-label masks.

## 3.5 Candidate dataset for rapid experimentation

To reduce computational cost for rapid experiments, a candidate training subset was created from the internal training split. The cleaned CheXpert training metadata contains 191,027 frontal images from 64,534 patients. A patient-level 90:10 split using `GroupShuffleSplit` (`random_state=42`) produced 171,927 internal training images from 58,080 patients and 19,100 validation images from 6,454 patients.

The candidate training subset was sampled at patient level using `MultilabelStratifiedShuffleSplit` from the `iterative-stratification` package, with `test_size=0.90` and `random_state=42`. For each patient, each of the six finding indicators was set to positive if at least one image for that patient had a definite positive label for that finding. This produced a candidate training subset of 17,508 images from 5,808 patients, approximately 10% of the internal training patients. The complete 19,100-image internal validation set was retained unchanged as the candidate validation set.

A patient-overlap check found zero patients shared between candidate training and validation. The candidate subset's image-level label distribution was also compared with the complete internal training split. The largest absolute difference in positive-label proportion was 0.65 percentage points (Pleural Effusion); the largest absolute difference in missing-label proportion was 1.07 percentage points (Consolidation). Negative and uncertain label proportions were also close across the six findings.

| Finding | Positive difference (percentage points) | Negative difference (percentage points) | Uncertain difference (percentage points) | Missing difference (percentage points) |
| --- | ---: | ---: | ---: | ---: |
| Pleural Effusion | +0.65 | +0.18 | -0.14 | -0.69 |
| Cardiomegaly | +0.25 | -0.11 | -0.02 | -0.11 |
| Pneumothorax | -0.05 | -0.40 | +0.05 | +0.40 |
| Atelectasis | -0.23 | +0.06 | +0.15 | +0.02 |
| Consolidation | +0.37 | +0.42 | +0.28 | -1.07 |
| Lung Lesion | -0.14 | approximately 0.00 | -0.01 | +0.16 |

This supports the candidate subset's representativeness with respect to patient-level sampling and the observed distributions of the six target labels. It does not establish that all demographic, image-quality or acquisition characteristics are identically distributed. The two reported model evaluations in Sections 3.2 and 3.4 used the full internal validation split; the fine-tuned ImageNet experiment used the full internal training split. The candidate subset is documented as the reduced data resource for rapid experimentation and should not be incorrectly described as the training data used for those two reported runs.

## 3.6 Baseline conclusions and limitations

The unadapted CheXpert-pretrained DenseNet-121 provides the pretrained baseline required for this milestone. Its macro AUROC is 0.7939 and macro AUPRC is 0.8628 under the definite-label `ignore` evaluation policy. The per-finding results and published research references identify Pneumothorax, Atelectasis and Lung Lesion as findings requiring particular attention. For Atelectasis and Lung Lesion, the very small number of definite negatives limits the stability and interpretation of AUROC.

The external benchmark comparison is contextual rather than directly controlled. The candidate subset preserves the observed label distributions closely and maintains patient-level separation, making it suitable for faster exploratory runs. Further architecture comparisons and hyperparameter tuning should be reported separately from these pretrained baseline results.

# 5. End-to-End Setup

## 5.1 Implemented pipeline and scope

The implemented end-to-end model pipeline is documented in the Kaggle notebook `dsai-m3-end-to-end-pipeline(1).ipynb`. It covers data loading, preprocessing, patient-separated training and validation, DenseNet-121 training, checkpoint persistence and recovery, full validation inference, metric calculation and result generation. The pipeline was executed on a CUDA-enabled Kaggle GPU.

This section describes the parts that were implemented and evaluated. The photo-quality recapture decision loop, second-photo acquisition/combination and region-specific recapture workflow remain proposed project components; the notebook's image-level training and validation run does not by itself demonstrate that those components have been integrated or executed end to end.

## 5.2 Data preparation and loading

The notebook uses the cleaned frontal CheXpert data and a patient-level split. `GroupShuffleSplit` is configured with `test_size=0.10` and `random_state=42`, producing 171,927 training images from 58,080 patients and 19,100 validation images from 6,454 patients. Dataset and DataLoader checks verify the expected split sizes and that both loaders contain batches.

The six target findings are Pleural Effusion, Cardiomegaly, Pneumothorax, Atelectasis, Consolidation and Lung Lesion. Each sample is returned with an image tensor, target vector, observation mask and image path. The mask is used to exclude uncertain or missing labels from the loss and evaluation rather than treating them as definite negatives.

## 5.3 Image preprocessing

The implemented ImageNet-pretrained DenseNet-121 pipeline:
- reads each frontal X-ray as a grayscale image;
- resizes it to 224 × 224 and applies histogram equalisation;
- replicates the grayscale image into three channels;
- applies ImageNet channel normalisation;
- applies small random affine transformations to training images only (rotation up to 5 degrees, translation up to 2% and scale from 0.95 to 1.05);
- leaves validation images unaugmented.

The validation DataLoader does not shuffle the images. The notebook also checks the shape of a training batch and verifies that image tensors have three channels at 224 × 224, while targets and masks each contain six entries per image.

## 5.4 Model training and checkpoint management

The implemented training model is DenseNet-121 initialised with ImageNet-1K pretrained weights and modified with a six-output classification layer. The training configuration uses seed 42, five epochs, batch size 32, AdamW with learning rate 1 × 10⁻⁴ and weight decay 1 × 10⁻⁴, and gradient clipping at 1.0. Mixed-precision training is enabled on CUDA.

The loss is masked binary cross-entropy with logits. Element-wise losses are multiplied by the observation mask, and the sum is divided by the number of observed labels. This ensures that uncertain and missing labels do not contribute to the loss.

The notebook saves model and training state checkpoints and includes a checkpoint recovery verification procedure. The recovery test checks that model, optimiser, scheduler, gradient-scaler and random-number-generator states are present, and verifies that a saved checkpoint can be downloaded and restored. The best trained model is subsequently loaded for validation evaluation. The best checkpoint in the reported run was from epoch 3.

## 5.5 Validation inference and evaluation

After training, the best checkpoint is loaded and evaluated in inference mode on the complete 19,100-image validation set. For each batch, the pipeline collects the six target labels, observation masks, sigmoid prediction scores and image paths. Metrics are calculated separately for each finding using only entries with observed labels.

The evaluation produces AUROC and average precision (AUPRC), along with ROC and precision-recall curves. Additional notebook outputs include confusion-matrix visualisations, per-finding threshold analysis and a sample-image inference demonstration using a 0.5 threshold. The separate CheXpert-pretrained TorchXRayVision baseline evaluation is documented in Section 3.2; it is not the same training run as the ImageNet-pretrained fine-tuned model described in this section.

## 5.6 Saved outputs and reproducibility

The notebook writes the trained model and evaluation artifacts under `/kaggle/working/m3_baseline/`. The documented outputs include:
- `best_model.pth` and training metric/checkpoint files;
- `evaluation/full_validation_metrics.json`;
- `evaluation/validation_predictions.npz`, containing targets, observation masks, prediction scores and image paths;
- ROC and precision-recall curve images;
- confusion-matrix visualisations;
- threshold-analysis results and sample-inference outputs;
- an end-to-end pipeline diagram.

A final notebook step checks for the required deliverable files and packages them into `requirement_5_end_to_end_deliverables.zip`. These artifacts support reproducibility and inspection of the implemented training-and-evaluation pipeline.

## 5.7 Implementation boundary and remaining integration

The implemented workflow establishes an executable image-classification pipeline from prepared CheXpert data through model training, checkpoint recovery, validation inference and performance reporting. It does not yet establish a fully integrated recapture system. In particular, the following elements are not demonstrated as executed components of this notebook:
- inference on held-out CheXphoto photographs as part of the same end-to-end run;
- calibrated ensemble uncertainty and a validated recapture threshold;
- region-specific retake guidance driven by localisation maps;
- generation or capture of a second photograph, prediction combination and repeated recapture decisions.

These remain separate implementation and evaluation tasks. They should be marked complete only after the corresponding code, outputs and evaluation evidence are available.

# 6. Architecture Visualization

Figure 1 shows the model architecture. Figure 2 shows the complete pipeline, including the training setups, the localisation check on NIH boxes, the test on CheXphoto photos and the recapture loop.


<img width="652" height="527" alt="image" src="https://github.com/user-attachments/assets/067737bb-f3c0-4ff7-98d3-39de57a2075a" />

Figure 1. Model architecture. The input is resized, histogram equalised, converted to three channels and ImageNet-normalised. An ImageNet-pretrained backbone (initialised with weights learned on natural images and then fine-tuned on chest X-rays) feeds a classification head in all setups and an optional box-supervised patch head in Setup E. An ensemble with temperature scaling produces calibrated probabilities and an uncertainty estimate; Grad-CAM ++ provides the region map.

<img width="533" height="637" alt="image" src="https://github.com/user-attachments/assets/2c1de738-1d09-4efe-b2a6-77e82d659f80" />

Figure 2. Complete pipeline and setups. Setups A, B, C and E are trained as in step 1; localisation is validated on NIH boxed images (step 2); the model is tested on CheXphoto photos (step 3); and the recapture criteria are checked (step 4). If recapture is required, the second version is classified and combined with the first prediction (steps 5 to 7), after which the criteria are checked again. The loop continues until the criteria are no longer met or the pre-set recapture limit is reached.

## Components shown in the diagrams

| Component | What it does | Used in |

| --- | --- | --- |

| Preprocessed X-ray | Resized to 224 x 224, histogram equalised, grayscale replicated to 3 channels, then normalised using ImageNet channel means and standard deviations; same function for every source and photo | All setups |

| Backbone | ImageNet-pretrained convolutional network; its learned feature-extraction weights are fine-tuned on the X-ray training data and produce a 7 x 7 feature map | All setups |

| Classification head | Average-pools the feature map, applies dropout and a linear layer, then outputs six sigmoid probabilities (one per finding) | All setups |

| Patch head | Optional 1 x 1 convolution branch that scores each feature-map patch for each finding; box-supervised only in Setup E | Setup E only |

| Ensemble and temperature scaling | Ensemble of 3 to 5 independently trained seeds; averages probabilities, measures disagreement across seeds, and applies temperature scaling for calibration | Best configuration of A or B (and E if run) |

| Region map (Grad-CAM ++) | Shows which part of the image drove each prediction | All setups |

| Localisation check (NIH boxes) | Compares region maps with ground-truth boxes on held-out NIH box images | Step 2 |

| Photo test (CheXphoto) | Runs the model on photos of held-out patients and records probability and uncertainty | Step 3 |

| Recapture criteria | Flags a photo when the probability is in an uncertain band or the seed spread is high | Step 4 |

| Second photo, classify, combine | Classifies a progressively improved/degraded version of the same CheXphoto image; predictions are rechecked against the recapture criteria (controlled simulation) | Steps 5 to 7 |

# References

[1] Huang G., Liu Z., van der Maaten L., Weinberger K.Q. Densely Connected Convolutional Networks. CVPR 2017.

[2] He K., Zhang X., Ren S., Sun J. Deep Residual Learning for Image Recognition. CVPR 2016.

[3] Tan M., Le Q. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML 2019.

[4] Dosovitskiy A. et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR 2021.

[5] Rajpurkar P. et al. CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning. arXiv:1711.05225, 2017.

[6] Irvin J. et al. CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison. AAAI 2019.

[7] Pham H.H., Le T.T., Tran D.Q., Ngo D.T., Nguyen H.Q. Interpreting chest X-rays via CNNs that exploit hierarchical disease dependencies and uncertainty labels. arXiv:1911.06475.

[8] Ke A., Ellsworth W., Banerjee O., Ng A.Y., Rajpurkar P. CheXtransfer: Performance and Parameter Efficiency of ImageNet Models for Chest X-Ray Interpretation. CHIL 2021.

[9] Raghu M., Zhang C., Kleinberg J., Bengio S. Transfusion: Understanding Transfer Learning for Medical Imaging. NeurIPS 2019.

[10] Phillips N.A. et al. CheXphoto: 10,000+ Photos and Transformations of Chest X-rays for Benchmarking Deep Learning Robustness. arXiv:2007.06199, 2020.

[11] CheXphotogenic: Generalization of Deep Learning Models for Chest X-ray Interpretation to Photos of Chest X-rays. arXiv:2011.06129, 2020.

[12] Rajpurkar P. et al. CheXpedition: Investigating Generalization Challenges for Translation of Chest X-Ray Algorithms to the Clinical Setting. arXiv:2002.11379, 2020.

[13] Selvaraju R.R. et al. Grad-CAM ++: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV 2017.

[14] Lakshminarayanan B., Pritzel A., Blundell C. Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. NeurIPS 2017.

[15] Gal Y., Ghahramani Z. Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. ICML 2016.

[16] Guo C., Pleiss G., Sun Y., Weinberger K.Q. On Calibration of Modern Neural Networks. ICML 2017.

[17] Wang X. et al. ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. CVPR 2017.

[18] Li Z., Wang C., Han M., Xue Y., Wei W., Li L.-J., Fei-Fei L. Thoracic Disease Identification and Localization with Limited Supervision. CVPR 2018 (arXiv:1711.06373).

<img width="113" height="49" alt="Screenshot 2026-10-22 230005" src="https://github.com/user-attachments/assets/fa081b35-091e-4aec-bb30-667f4a1ee1c6" />
