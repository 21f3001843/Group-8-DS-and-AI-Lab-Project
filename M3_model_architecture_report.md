Model Architecture: Selection, Justification and Visualisation

Sections 1, 2 and 6 of the Milestone 3 report

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

# 6. Architecture Visualization

Figure 1 shows the model architecture. Figure 2 shows the complete pipeline, including the training setups, the localisation check on NIH boxes, the test on CheXphoto photos and the recapture loop.

Figure 1. Model architecture. The input is resized, histogram equalised, converted to three channels and ImageNet-normalised. An ImageNet-pretrained backbone (initialised with weights learned on natural images and then fine-tuned on chest X-rays) feeds a classification head in all setups and an optional box-supervised patch head in Setup E. An ensemble with temperature scaling produces calibrated probabilities and an uncertainty estimate; Grad-CAM ++ provides the region map.

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


## Figures


The source Word document contains 2 embedded figure image(s). They are not embedded in this Markdown file; retain the Word document for the original figures.
