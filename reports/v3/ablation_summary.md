# AEGIS (full) vs strongest baseline across all head variants

Note: `quality_only` is a shortcut FLOOR (LR on 3 quality columns), not a competing baseline; it is excluded from best_baseline and reported as floor_auroc / aegis_minus_floor.

| scenario | level | generator | N | head_variant | aegis_accuracy | aegis_auroc | best_baseline | best_accuracy | best_auroc | delta_auroc | delta_accuracy | floor_auroc | aegis_minus_floor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | clean |  |  | clean | 0.9519 | 0.9908 | naive_avg_clean | 0.9725 | 0.9960 | -0.0053 | -0.0206 | 0.6255 | 0.3653 |
| S2 | jpeg_q40 |  |  | clean | 0.7494 | 0.8360 | static_lr_aug | 0.8006 | 0.8970 | -0.0610 | -0.0513 | 0.5876 | 0.2484 |
| S2 | jpeg_q65 |  |  | clean | 0.7231 | 0.8638 | static_lr_aug | 0.8569 | 0.9339 | -0.0701 | -0.1338 | 0.5906 | 0.2732 |
| S2 | jpeg_q85 |  |  | clean | 0.7438 | 0.9134 | static_lr_aug | 0.8950 | 0.9596 | -0.0462 | -0.1512 | 0.5861 | 0.3273 |
| S2 | all |  |  | clean | 0.7388 | 0.8616 | static_lr_aug | 0.8508 | 0.9318 | -0.0702 | -0.1121 | 0.5769 | 0.2847 |
| S3 | blur_k11_s1.5 |  |  | clean | 0.7887 | 0.8627 | naive_avg_aug | 0.8231 | 0.9021 | -0.0394 | -0.0344 | 0.5409 | 0.3218 |
| S3 | blur_k17_s2.5 |  |  | clean | 0.7106 | 0.7789 | naive_avg_aug | 0.7419 | 0.8296 | -0.0507 | -0.0312 | 0.5231 | 0.2559 |
| S3 | resize_0.5 |  |  | clean | 0.6856 | 0.8593 | static_lr_aug | 0.7806 | 0.8931 | -0.0338 | -0.0950 | 0.4928 | 0.3666 |
| S3 | all |  |  | clean | 0.7283 | 0.8210 | naive_avg_aug | 0.7856 | 0.8714 | -0.0504 | -0.0573 | 0.5117 | 0.3094 |
| S4 | all | midjourney | 5 | clean | 0.8171 | 0.9564 | clip_clean | 0.8577 | 0.9603 | -0.0039 | -0.0406 | 0.6518 | 0.3047 |
| S4 | all | midjourney | 20 | clean | 0.8348 | 0.9626 | clip_clean | 0.8577 | 0.9603 | 0.0023 | -0.0230 | 0.6518 | 0.3108 |
| S4 | all | midjourney | 100 | clean | 0.8164 | 0.9568 | clip_clean | 0.8577 | 0.9603 | -0.0036 | -0.0413 | 0.6518 | 0.3050 |
| S4 | all | vqdm | 5 | clean | 0.9751 | 0.9966 | naive_avg_clean | 0.9821 | 0.9989 | -0.0024 | -0.0070 | 0.5962 | 0.4004 |
| S4 | all | vqdm | 20 | clean | 0.9799 | 0.9984 | naive_avg_clean | 0.9821 | 0.9989 | -0.0006 | -0.0022 | 0.5962 | 0.4022 |
| S4 | all | vqdm | 100 | clean | 0.9753 | 0.9967 | naive_avg_clean | 0.9821 | 0.9989 | -0.0022 | -0.0068 | 0.5962 | 0.4006 |
| S4 | all | all | 5 | clean | 0.8961 | 0.9759 | clip_clean | 0.9172 | 0.9790 | -0.0031 | -0.0211 | 0.6239 | 0.3520 |
| S4 | all | all | 20 | clean | 0.9073 | 0.9800 | clip_clean | 0.9172 | 0.9790 | 0.0011 | -0.0099 | 0.6239 | 0.3561 |
| S4 | all | all | 100 | clean | 0.8925 | 0.9757 | clip_clean | 0.9172 | 0.9790 | -0.0032 | -0.0247 | 0.6239 | 0.3518 |
| S1 | clean |  |  | aug | 0.9756 | 0.9962 | naive_avg_clean | 0.9725 | 0.9960 | 0.0002 | 0.0031 | 0.6255 | 0.3708 |
| S2 | jpeg_q40 |  |  | aug | 0.7944 | 0.8885 | static_lr_aug | 0.8006 | 0.8970 | -0.0085 | -0.0062 | 0.5876 | 0.3009 |
| S2 | jpeg_q65 |  |  | aug | 0.8512 | 0.9283 | static_lr_aug | 0.8569 | 0.9339 | -0.0056 | -0.0056 | 0.5906 | 0.3377 |
| S2 | jpeg_q85 |  |  | aug | 0.8962 | 0.9626 | static_lr_aug | 0.8950 | 0.9596 | 0.0029 | 0.0012 | 0.5861 | 0.3764 |
| S2 | all |  |  | aug | 0.8473 | 0.9290 | static_lr_aug | 0.8508 | 0.9318 | -0.0028 | -0.0035 | 0.5769 | 0.3521 |
| S3 | blur_k11_s1.5 |  |  | aug | 0.8213 | 0.8960 | naive_avg_aug | 0.8231 | 0.9021 | -0.0061 | -0.0019 | 0.5409 | 0.3551 |
| S3 | blur_k17_s2.5 |  |  | aug | 0.7475 | 0.8271 | naive_avg_aug | 0.7419 | 0.8296 | -0.0025 | 0.0056 | 0.5231 | 0.3040 |
| S3 | resize_0.5 |  |  | aug | 0.7925 | 0.8925 | static_lr_aug | 0.7806 | 0.8931 | -0.0007 | 0.0119 | 0.4928 | 0.3997 |
| S3 | all |  |  | aug | 0.7871 | 0.8687 | naive_avg_aug | 0.7856 | 0.8714 | -0.0027 | 0.0015 | 0.5117 | 0.3570 |
| S4 | all | midjourney | 5 | aug | 0.7014 | 0.9224 | clip_clean | 0.8577 | 0.9603 | -0.0379 | -0.1563 | 0.6518 | 0.2706 |
| S4 | all | midjourney | 20 | aug | 0.7139 | 0.9346 | clip_clean | 0.8577 | 0.9603 | -0.0258 | -0.1439 | 0.6518 | 0.2828 |
| S4 | all | midjourney | 100 | aug | 0.7269 | 0.9324 | clip_clean | 0.8577 | 0.9603 | -0.0280 | -0.1308 | 0.6518 | 0.2806 |
| S4 | all | vqdm | 5 | aug | 0.9847 | 0.9986 | naive_avg_clean | 0.9821 | 0.9989 | -0.0004 | 0.0026 | 0.5962 | 0.4024 |
| S4 | all | vqdm | 20 | aug | 0.9866 | 0.9986 | naive_avg_clean | 0.9821 | 0.9989 | -0.0003 | 0.0045 | 0.5962 | 0.4024 |
| S4 | all | vqdm | 100 | aug | 0.9851 | 0.9986 | naive_avg_clean | 0.9821 | 0.9989 | -0.0003 | 0.0030 | 0.5962 | 0.4024 |
| S4 | all | all | 5 | aug | 0.8430 | 0.9610 | clip_clean | 0.9172 | 0.9790 | -0.0180 | -0.0742 | 0.6239 | 0.3371 |
| S4 | all | all | 20 | aug | 0.8519 | 0.9680 | clip_clean | 0.9172 | 0.9790 | -0.0110 | -0.0653 | 0.6239 | 0.3441 |
| S4 | all | all | 100 | aug | 0.8491 | 0.9637 | clip_clean | 0.9172 | 0.9790 | -0.0153 | -0.0681 | 0.6239 | 0.3398 |
