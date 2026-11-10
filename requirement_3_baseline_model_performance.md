# Requirement 3 – Baseline Model Performance

## 1. Objective

The objective of this requirement was to establish baseline performance for chest X-ray abnormality classification and compare two DenseNet-121 models:

1. **ImageNet-pretrained and CheXpert fine-tuned DenseNet-121:** A DenseNet-121 initialized with ImageNet weights and subsequently trained on the CheXpert dataset for six findings.
2. **CheXpert-pretrained DenseNet-121:** A DenseNet-121 from TorchXRayVision using the `densenet121-res224-chex` pretrained weights, evaluated without additional fine-tuning.

The comparison was conducted across the following six findings:

- Pleural Effusion
- Cardiomegaly
- Pneumothorax
- Atelectasis
- Consolidation
- Lung Lesion

## 2. Experimental Setup

### ImageNet-pretrained and fine-tuned model

The first model used the DenseNet-121 architecture initialized with ImageNet-pretrained weights. Its classification layer was replaced with a six-output layer corresponding to the target findings.

The model was trained using the CheXpert dataset with the following configuration:

| Parameter | Configuration |
|---|---|
| Architecture | DenseNet-121 |
| Initial weights | ImageNet1K V1 |
| Training epochs | 5 |
| Batch size | 32 |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Random seed | 42 |
| Best checkpoint | Epoch 3 |

### CheXpert-pretrained model

The second model used TorchXRayVision's pretrained DenseNet-121 with the `densenet121-res224-chex` weights.

The model was evaluated directly without additional fine-tuning on the candidate validation dataset.

Its inference pipeline used the TorchXRayVision image-loading utilities, center cropping and resizing to 224 × 224 pixels.

## 3. Validation Data and Evaluation Protocol

Both models were evaluated using the same candidate validation dataset.

- Total validation images: **19,100**
- Number of target findings: **6**
- Image ordering: Identical
- Image membership: Identical
- Ground-truth labels: Identical
- Validity masks: Identical

A direct verification was performed using the saved prediction arrays from both experiments. Assertions confirmed that the image paths, ordering, labels and validity masks matched exactly.

For metric calculation, definite positive labels (1) and definite negative labels (0) were evaluated. Uncertain labels (-1) and missing labels were excluded using the corresponding validity masks.

### Evaluation metrics

**AUROC (Area Under the Receiver Operating Characteristic Curve):** Measures the model's ability to distinguish positive cases from negative cases across classification thresholds.

**AUPRC (Area Under the Precision-Recall Curve):** Measures the relationship between precision and recall and is particularly relevant when evaluating findings with imbalanced class distributions.

Both metrics were calculated separately for each finding. Macro averages were obtained by taking the unweighted mean across the six findings.

## 4. Comparative Results

| Finding | ImageNet Fine-tuned AUROC | CheXpert-pretrained AUROC | ImageNet Fine-tuned AUPRC | CheXpert-pretrained AUPRC |
|---|---:|---:|---:|---:|
| Pleural Effusion | 0.9559 | 0.9151 | 0.9804 | 0.9602 |
| Cardiomegaly | 0.9442 | 0.9001 | 0.9790 | 0.9595 |
| Pneumothorax | 0.8268 | 0.7133 | 0.6526 | 0.4781 |
| Atelectasis | 0.6749 | 0.6647 | 0.9786 | 0.9756 |
| Consolidation | 0.9427 | 0.9116 | 0.9225 | 0.8703 |
| Lung Lesion | 0.8091 | 0.6583 | 0.9683 | 0.9333 |
| **Macro Average** | **0.8589** | **0.7939** | **0.9136** | **0.8628** |

### Overall performance

| Metric | ImageNet Fine-tuned | CheXpert-pretrained |
|---|---:|---:|
| Macro AUROC | **0.8589** | 0.7939 |
| Macro AUPRC | **0.9136** | 0.8628 |

The ImageNet-pretrained and fine-tuned model achieved higher macro AUROC and macro AUPRC. It also achieved higher values for both metrics across all six findings in this evaluation.

## 5. Observations

1. **Pleural Effusion and Cardiomegaly:** Both models demonstrated strong performance, with the ImageNet fine-tuned model achieving higher AUROC and AUPRC.

2. **Pneumothorax:** This finding presented a more challenging classification task. The ImageNet fine-tuned model achieved an AUROC of 0.8268 and AUPRC of 0.6526, compared with 0.7133 and 0.4781 respectively for the CheXpert-pretrained model.

3. **Atelectasis:** Both models achieved relatively low AUROC values despite high AUPRC values. This result should be interpreted cautiously because the validation set contained only 105 definite negative samples for this finding.

4. **Consolidation:** Both models demonstrated strong performance, with the ImageNet fine-tuned model achieving higher values for both metrics.

5. **Lung Lesion:** The ImageNet fine-tuned model achieved an AUROC of 0.8091, compared with 0.6583 for the CheXpert-pretrained model. However, only 84 definite negative samples were available, which limits the certainty of the evaluation.

## 6. Limitations

- The two models use different model-specific preprocessing pipelines. Although the underlying validation images, labels and masks are identical, the pixel transformations are not necessarily identical.
- The models have different training histories and initialization strategies. Therefore, the observed performance difference cannot be attributed solely to the pretraining dataset.
- Atelectasis and Lung Lesion have relatively few definite negative validation samples, making their evaluation metrics less stable.
- These results represent baseline classification performance and do not independently establish improvements in abnormality localization or region-specific recapture.

## 7. Conclusion

The comparative evaluation established baseline performance for two DenseNet-121 models on six chest X-ray findings using the same 19,100 validation images.

The ImageNet-pretrained and CheXpert fine-tuned model achieved a macro AUROC of 0.8589 and macro AUPRC of 0.9136, outperforming the TorchXRayVision CheXpert-pretrained model, which achieved 0.7939 and 0.8628 respectively.

These results provide a baseline for subsequent experiments and further investigation into uncertainty-guided, region-specific recapture for chest X-ray abnormality detection.

## 8. Generated Deliverables

The following artifacts were generated and saved:

- `model_performance_comparison.csv`
- `auroc_comparison.png`
- `auprc_comparison.png`
- `comparison_summary.json`

These files contain the comparative metrics, visualizations and summary results for Requirement 3.
