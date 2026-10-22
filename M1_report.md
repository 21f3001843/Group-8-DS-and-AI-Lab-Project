**Milestone 1: Problem Definition and Literature Review**

**Problem Statement:**

**"**Can uncertainty-guided, region-specific image recapture improve the reliability of chest X-ray abnormality detection from smartphone photographs of physical X-ray films?"

**Objectives:**

The Deep learning community has utilized datasets such as "MIMIC-CXR-JPG", "CheXphoto", "VinDr-CXR" etc to improve model performance on classification and localization of several abnormalities visible in a chest scan. However, the aim of this project would be to integrate the ability of recapturing an image which the model could not classify with a certain confidence benchmark to improve the reliability of the abnormality detection when an X-ray is photographed by a smartphone.

The project acknowledges a degradation in the quality of the image processed when photographed, compared to digital files, hence rather than making a low confidence claim, the model would aim to obtain region specific recapture to boost confidence in the classification of the model.

Flow of questions to be addressed in the project:

1. Can a model remain reliable under real photographic distortions?
2. Can the system identify which specific area is inadequate?
3. Can uncertainty reliably identify cases needing further assessment?
4. Can a model choose a region where a better capture is likely to help?
5. Does combining these capabilities improve detection compared with existing approaches?

From initial research conducted, we believe points 1 to 3 have been addressed by existing research projects. This project aims to evaluate points 4 and 5.

**Literature Review**

We have broken down the literature review into addressing each question mentioned in the previous questions:

- [CheXphoto: 10,000+ Photos and Transformations of Chest X-rays for Benchmarking Deep Learning Robustness](https://proceedings.mlr.press/v136/phillips20a/phillips20a.pdf) : Models trained on CheXpert dataset experience a 10-14% drop in ROC AUC when tested on Smartphone captured CheXphoto images. CheXphoto ran a competition, now closed, whose number 1 on the leaderboard scored an AUC of 0.850 on film images and 0.890 on digital images.
- [Recalibration of deep learning models for abnormality detection in smartphone-captured chest radiograph](https://www.nature.com/articles/s41746-021-00393-9): This study quantified the performance loss and tested a method to recover it. They found that the original model's mean AUROC dropped from 0.75 on digital images to 0.67 on photographs during external validation. Their proposed recalibration method, which used fewer than 200 photographs for adaptation, recovered performance to an AUROC of 0.75, outperforming both transfer learning (0.71) and direct training on photographs (0.59)
- [RoMIA: a framework for creating Robust Medical Imaging AI models for chest radiographs](https://pmc.ncbi.nlm.nih.gov/articles/PMC10800823/): RoMIA adds three key steps to the model training and deployment flow: (i) Noise-added training, synthetic transformation, (ii) Fine-tuning with input mixing, and (iii) DCT-based denoising, which removes a fraction of high-frequency components of each image. Upon evaluating the CheXphoto dataset, models produced by RoMIA show 3%–5% improvement in robust accuracy, which corresponds to an average reduction of 22.6% in misclassifications
- [Localization and classification of abnormalities on chest X-ray images using a Mamba-YOLOvX model](https://www.sciencedirect.com/science/article/pii/S0957417425015519): Selective scanning mechanisms implemented in the YOLOvX backbone were used to focus on critical regions of the input data, and were shown effective in extracting features associated with both large and small abnormalities. The Mamba-YOLOvX model significantly outperformed the baseline YOLOvX model in localizing abnormalities with AP50 scores improving from a baseline of 0.16 (YOLOvX) to 0.174 (Mamba-YOLOvX +CSAD + Image Augmentation)
- [What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision](https://proceedings.neurips.cc/paper_files/paper/2017/file/2650d6089a6d640c5e85b2b88265dc2b-Paper.pdf):The essential theoretical foundation for this project's pipeline. It formalizes how to decompose pixel-wise and image-wise uncertainty into aleatoric uncertainty (which captures sensor noise, glare, and photographic occlusion) and epistemic uncertainty (unseen pathology), explaining why aleatoric uncertainty correlates with degraded input regions.
- [Look Closer to See Better: Recurrent Attention Convolutional Neural Network for Fine-grained Image Recognition](https://openaccess.thecvf.com/content_cvpr_2017/papers/Fu_Look_Closer_to_CVPR_2017_paper.pdf):The authors proposed a Recurrent Attention Convolutional Neural Network (RA-CNN) for fine-grained image recognition, demonstrating that discriminative region localization and region-based feature learning can reinforce one another. This work provides a methodological foundation for investigating region-focused analysis in chest X-ray classification. However, the proposed approach identifies regions that are informative for classification rather than regions whose image quality makes predictions unreliable. It also relies on digital region amplification rather than reacquisition of the image.

**Current Baseline metrics:**

Currently, state-of-the-art chest X-ray classification models perform exceptionally well on pristine digital DICOM images (routinely achieving AUROC scores of 0.85–0.95+ on datasets like CheXpert and MIMIC-CXR). However, when deployed in resource-limited settings via smartphone photography of physical films, these baselines degrade significantly.

According to benchmark studies (e.g., _CheXphoto_), real-world photographic distortions such as specular glare, perspective warp, and moiré patterns cause the AUROC to drop substantially, sometimes by 10-15%. Existing mitigation baselines rely on passive, post-hoc digital restoration (e.g., de-glaring GANs or homography correction). These passive baselines routinely fail because they algorithmically hallucinate over obscured regions rather than recovering true signal, keeping the **False Negative Rate (FNR)** high for subtle, focal pathologies like pneumothorax or small nodules.

Breaking down the baseline's planned for this project:

- Baseline A: A standard classification backbone (e.g., DenseNet-121 or ResNet-50) trained on clean DICOMs and tested directly on degraded photographs without any special pre-processing establishing an absolute performance floor.
- Baseline B: Using the same model but during training, heavy synthetic augmentations are applied to the clean X-rays. This includes geometric transformations and photometric transformations
- Baseline C: To represent the current state of the art in mobile medical digitization, the model attempts to "clean" the image algorithmically as part of the pre-processing pipeline before passing it to the classifier, rather than relying purely on model robustness. Cleaning the image would involve glare removal and homography rectification. This baseline would be the primary contributor as the layer we intend to add on top of this would prove our core hypothesis.

**Project Plan to improve baseline metrics:**

Rather than attempting to mathematically guess what lies beneath a glare artifact, our project shifts the paradigm from _passive image restoration_ to _active, uncertainty-guided re-acquisition_.

We plan on utilizing spatial aleatoric uncertainty intersected with diagnostic task saliency to automatically identify if an obscured region (e.g., a glare over the lung apex) is diagnostically critical. If so, the system prompts the user to take a targeted, close-up photograph of just that region.

To integrate captures, we plan on implementing a two-branch Convolutional Neural Network (CNN). The degraded full-frame photo will pass through a Global Branch for macro-anatomical context, while the clean close-up passes through a Local Branch for high-frequency fidelity. These mathematical feature maps will be dynamically fused using an attention mechanism.

We aim to achieve the following outcomes by replacing hallucinated data with ground-truth optical recaptures:

1. Recover AUROC: Close the performance gap between smartphone-photographed X-rays and pristine digital DICOM scans.
2. Reduce False Negative Rate: Significantly lower the False Negative Rate for focal/localized thoracic diseases that are most vulnerable to being masked by photographic artifacts.

In addition to the diagnostic metrics mentioned above, we also want to address certain uncertainty and quality metrics:

1. Expected Calibration Error (ECE): To measure if the model's confidence matches its actual accuracy.
2. Recapture Efficiency: The percentage of images where a recapture is triggered. If the system asks for a recapture on 99% of images, it is clinically impractical.

**Sample inputs**

<img width="430" height="286" alt="Screenshot 2026-10-22 224045" src="https://github.com/user-attachments/assets/8fc0209b-b1c0-4ab4-a5f8-652420534c94" />
<img width="352" height="302" alt="Screenshot 2026-10-22 224034" src="https://github.com/user-attachments/assets/ab58300b-a73c-4e43-98a1-5851d19dc954" />

**Expected outputs:**

1. Interactive Prototype:

The system will take a primary smartphone photograph as input. It will output a preliminary diagnostic score, an uncertainty assessment, and if the image is degraded, an actionable visual prompt requesting a secondary close-up photograph. After receiving the close-up, it outputs the final, highly confident diagnostic prediction.

1. Quantitative and Visual Benchmarking:

A comprehensive set of comparative metrics (AUROC, False Negative Rate, Expected Calibration Error) and side-by-side visual case studies evaluating your Global-Local fusion model against the standard baselines (Do Nothing, Data Augmentation, Post-Hoc Restoration)

**Acknowledgement:**

<img width="107" height="58" alt="Screenshot 2026-10-22 225932" src="https://github.com/user-attachments/assets/320ec95a-c2cd-4993-a90c-e27b299e08dc" />
<img width="113" height="49" alt="Screenshot 2026-10-22 230005" src="https://github.com/user-attachments/assets/0ce6fb73-133d-42c4-aca1-76438a03927a" />

