# MVTec Anomaly Detection Results Summary

## Executive Summary
Based on the anomaly detection metrics extracted from the `results_MVTec` directory across various configurations, the optimal setup for this pipeline strongly favors **omitting CAD processing**, using **extreme few-shot configurations (1-4 shots)**, and leveraging the **smaller DINOv2 ViT-Small backbone**. 

## 1. Overall Performance Trends: No CAD vs. CAD Processing
The "No CAD" (`without_cad` or `dinov3_nocad`) approach is the clear winner across all tested configurations. 
Configurations that omit the CAD processing step vastly outperform those using feature concatenation (`concate`) or feature differencing (`diff`).

* **Example Comparison (1-Shot, DINOv2 ViT-B/14):**
  * `without_cad`: **73.02%** AUROC, 70.62% AP
  * `concate`: 62.42% AUROC, 56.83% AP
  * `diff`: 60.99% AUROC, 53.28% AP

The CAD alignment or differencing likely injects noise or misalignment artifacts that confuse the embedding space rather than helping it.

## 2. Impact of "Shot" Count (Few-shot vs. Many-shot)
Surprisingly, increasing the number of reference shots consistently degrades overall detection performance (specifically AUROC and Accuracy) in most methods, indicating a thresholding or sensitivity collapse:

* **Low Shots (1 to 4-shot):** Produce the best, most balanced results. For instance, `without_cad` (DINOv2 ViT-S/14) at 1-shot achieves the peak AUROC of **75.83%** and an F1 of 68.32%.
* **High Shots (40+ shots):** In methods like `concate`, `diff`, `dinov3_concat`, and `dinov3_diff`, providing 40 or more shots causes the **Recall to jump to 100.00%**, but Accuracy plummets to ~44.92%. This effectively means the model starts predicting almost *everything* as anomalous, thus catching all defects but generating massive false positives.

*Takeaway:* The embedding representations used here are optimal for extreme few-shot scenarios. Adding too many references expands the normal boundary too strictly or aggressively, leading to high false-positive rates.

## 3. Backbone Comparison (DINOv2 vs. DINOv3 & Model Sizes)

* **DINOv2 vs DINOv3:** DINOv3 does not show a massive advantage over DINOv2 on this specific MVTec pipeline. 
  * `without_cad` (DINOv2 ViT-B/14) 1-shot AUROC: 73.02%
  * `dinov3_nocad` (DINOv3 ViT-B/16) 1-shot AUROC: 73.07%

* **DINOv2 Model Sizes:** Interestingly, the **ViT-Small (`vits14`) outperforms the larger models (`vitb14` and `vitl14`)**. 
  * `vits14` 1-shot AUROC: **75.83%**
  * `vitb14` 1-shot AUROC: 73.02%
  * `vitl14` 1-shot AUROC: 67.23%

*Takeaway:* For these MVTec anomaly detection tasks, massive backbones (ViT-Large) might be overfitting to general features or capturing too much irrelevant semantic noise, whereas ViT-Small captures more localized, structure-dependent features crucial for finding defects.

## 4. Key Metrics Summary Table

*Note: Metrics are averaged across seeds where multiple seeds were run.*

| Method | Backbone | Shots | AUROC | AP | F1 | Accuracy | Recall |
|--------|----------|-------|-------|----|----|----------|--------|
| **without_cad** | dinov2_vits14_448 | 1 | **75.83** | 73.46 | **68.32** | 64.62 | 84.93 |
| **without_cad** | dinov2_vits14_448 | 4 | 73.79 | 72.75 | 67.29 | 67.69 | 73.97 |
| **without_cad** | dinov2_vitb14_448 | 1 | 73.02 | 70.62 | 67.87 | 67.08 | 77.40 |
| **dinov3_nocad**| dinov3-vitb16 | 1 | 73.07 | **76.26** | 66.01 | **68.00** | 69.18 |
| **dinov3_nocad**| dinov3-vitb16 | 4 | 69.07 | 73.80 | 64.08 | 65.85 | 67.81 |
| **without_cad** | dinov2_vitl14_448 | 1 | 67.23 | 67.68 | 65.10 | 58.77 | 85.62 |
| **concate** | dinov2_vitb14_448 | 4 | 69.58 | 67.56 | 64.77 | 61.85 | 78.08 |
| **concate** | dinov2_vitb14_448 | 60 | 64.56 | 68.26 | 62.00 | 44.92 | 100.00 |
| **diff** | dinov2_vitb14_448 | 4 | 67.39 | 65.92 | 64.91 | 59.08 | 84.25 |
| **dinov3_concat**| dinov3-vitb16 | 4 | 66.60 | 65.00 | 63.89 | 60.00 | 78.77 |
| **dinov3_diff** | dinov3-vitb16 | 4 | 69.46 | 69.77 | 65.31 | 63.38 | 76.71 |

## Recommendations
If deciding on which configuration to prioritize for future experiments or production:
1. **Drop the CAD alignment/differencing** (`concate`/`diff`) as it seems to inject noise that confuses the embeddings. Stick to `without_cad` / `dinov3_nocad`.
2. **Use 1-shot or 4-shot reference setups** instead of higher numbers like 20-100, which clearly degrade the threshold's accuracy and cause massive false-positive spikes.
3. **Adopt DINOv2 ViT-Small (`vits14`)** as it currently yields the strongest baseline (highest AUROC and F1).