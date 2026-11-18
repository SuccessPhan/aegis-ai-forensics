# AEGIS (full) vs strongest baseline across all head variants

Note: `quality_only` is a shortcut FLOOR (LR on 3 quality columns), not a competing baseline; it is excluded from best_baseline and reported as floor_auroc / aegis_minus_floor.

| scenario | level | generator | N | head_variant | aegis_accuracy | aegis_auroc | best_baseline | best_accuracy | best_auroc | delta_auroc | delta_accuracy | floor_auroc | aegis_minus_floor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | clean |  |  | clean | 0.9700 | 0.9958 | naive_avg_clean | 0.9781 | 0.9966 | -0.0008 | -0.0081 | 0.6243 | 0.3715 |
| S2 | jpeg_q40 |  |  | clean | 0.7919 | 0.8693 | static_lr_aug | 0.8075 | 0.9018 | -0.0325 | -0.0156 | 0.6060 | 0.2633 |
| S2 | jpeg_q65 |  |  | clean | 0.7731 | 0.8931 | static_lr_aug | 0.8656 | 0.9373 | -0.0442 | -0.0925 | 0.6051 | 0.2880 |
| S2 | jpeg_q85 |  |  | clean | 0.8000 | 0.9382 | static_lr_aug | 0.8944 | 0.9603 | -0.0221 | -0.0944 | 0.5934 | 0.3448 |
| S2 | all |  |  | clean | 0.7883 | 0.8890 | static_lr_aug | 0.8558 | 0.9336 | -0.0446 | -0.0675 | 0.5908 | 0.2982 |
| S3 | blur_k11_s1.5 |  |  | clean | 0.8000 | 0.8855 | naive_avg_aug | 0.8237 | 0.9050 | -0.0195 | -0.0237 | 0.4986 | 0.3869 |
| S3 | blur_k17_s2.5 |  |  | clean | 0.7425 | 0.8141 | naive_avg_aug | 0.7556 | 0.8410 | -0.0269 | -0.0131 | 0.4896 | 0.3244 |
| S3 | resize_0.5 |  |  | clean | 0.6981 | 0.8835 | static_lr_aug | 0.8013 | 0.9032 | -0.0196 | -0.1031 | 0.4473 | 0.4363 |
| S3 | all |  |  | clean | 0.7469 | 0.8484 | naive_avg_aug | 0.7942 | 0.8813 | -0.0330 | -0.0473 | 0.4842 | 0.3641 |
| S4 | all | glide | 5 | clean | 0.9850 | 0.9979 | naive_avg_clean | 0.9879 | 0.9988 | -0.0010 | -0.0029 | 0.9728 | 0.0251 |
| S4 | all | glide | 20 | clean | 0.9853 | 0.9985 | naive_avg_clean | 0.9879 | 0.9988 | -0.0004 | -0.0026 | 0.9728 | 0.0257 |
| S4 | all | glide | 100 | clean | 0.9844 | 0.9979 | naive_avg_clean | 0.9879 | 0.9988 | -0.0009 | -0.0035 | 0.9728 | 0.0251 |
| S4 | all | midjourney | 5 | clean | 0.8195 | 0.9537 | clip_clean | 0.8470 | 0.9564 | -0.0027 | -0.0275 | 0.6405 | 0.3132 |
| S4 | all | midjourney | 20 | clean | 0.8322 | 0.9578 | clip_clean | 0.8470 | 0.9564 | 0.0015 | -0.0148 | 0.6405 | 0.3174 |
| S4 | all | midjourney | 100 | clean | 0.8261 | 0.9527 | clip_clean | 0.8470 | 0.9564 | -0.0037 | -0.0209 | 0.6405 | 0.3122 |
| S4 | all | all | 5 | clean | 0.9023 | 0.9768 | clip_clean | 0.9147 | 0.9783 | -0.0015 | -0.0125 | 0.8100 | 0.1667 |
| S4 | all | all | 20 | clean | 0.9090 | 0.9792 | clip_clean | 0.9147 | 0.9783 | 0.0009 | -0.0057 | 0.8100 | 0.1692 |
| S4 | all | all | 100 | clean | 0.9032 | 0.9764 | clip_clean | 0.9147 | 0.9783 | -0.0019 | -0.0115 | 0.8100 | 0.1663 |
| S1 | clean |  |  | aug | 0.9731 | 0.9964 | naive_avg_clean | 0.9781 | 0.9966 | -0.0002 | -0.0050 | 0.6243 | 0.3721 |
| S2 | jpeg_q40 |  |  | aug | 0.8087 | 0.8953 | static_lr_aug | 0.8075 | 0.9018 | -0.0066 | 0.0012 | 0.6060 | 0.2892 |
| S2 | jpeg_q65 |  |  | aug | 0.8519 | 0.9309 | static_lr_aug | 0.8656 | 0.9373 | -0.0064 | -0.0137 | 0.6051 | 0.3258 |
| S2 | jpeg_q85 |  |  | aug | 0.8906 | 0.9586 | static_lr_aug | 0.8944 | 0.9603 | -0.0017 | -0.0038 | 0.5934 | 0.3651 |
| S2 | all |  |  | aug | 0.8504 | 0.9298 | static_lr_aug | 0.8558 | 0.9336 | -0.0038 | -0.0054 | 0.5908 | 0.3390 |
| S3 | blur_k11_s1.5 |  |  | aug | 0.8144 | 0.8984 | naive_avg_aug | 0.8237 | 0.9050 | -0.0066 | -0.0094 | 0.4986 | 0.3998 |
| S3 | blur_k17_s2.5 |  |  | aug | 0.7519 | 0.8349 | naive_avg_aug | 0.7556 | 0.8410 | -0.0061 | -0.0038 | 0.4896 | 0.3453 |
| S3 | resize_0.5 |  |  | aug | 0.8037 | 0.8980 | static_lr_aug | 0.8013 | 0.9032 | -0.0052 | 0.0025 | 0.4473 | 0.4507 |
| S3 | all |  |  | aug | 0.7900 | 0.8751 | naive_avg_aug | 0.7942 | 0.8813 | -0.0062 | -0.0042 | 0.4842 | 0.3909 |
| S4 | all | glide | 5 | aug | 0.9722 | 0.9974 | naive_avg_clean | 0.9879 | 0.9988 | -0.0014 | -0.0156 | 0.9728 | 0.0246 |
| S4 | all | glide | 20 | aug | 0.9750 | 0.9972 | naive_avg_clean | 0.9879 | 0.9988 | -0.0016 | -0.0128 | 0.9728 | 0.0244 |
| S4 | all | glide | 100 | aug | 0.9737 | 0.9973 | naive_avg_clean | 0.9879 | 0.9988 | -0.0015 | -0.0141 | 0.9728 | 0.0245 |
| S4 | all | midjourney | 5 | aug | 0.6551 | 0.9105 | clip_clean | 0.8470 | 0.9564 | -0.0459 | -0.1920 | 0.6405 | 0.2700 |
| S4 | all | midjourney | 20 | aug | 0.6771 | 0.9241 | clip_clean | 0.8470 | 0.9564 | -0.0323 | -0.1699 | 0.6405 | 0.2836 |
| S4 | all | midjourney | 100 | aug | 0.7036 | 0.9272 | clip_clean | 0.8470 | 0.9564 | -0.0291 | -0.1434 | 0.6405 | 0.2867 |
| S4 | all | all | 5 | aug | 0.8136 | 0.9563 | clip_clean | 0.9147 | 0.9783 | -0.0220 | -0.1011 | 0.8100 | 0.1462 |
| S4 | all | all | 20 | aug | 0.8271 | 0.9636 | clip_clean | 0.9147 | 0.9783 | -0.0147 | -0.0876 | 0.8100 | 0.1536 |
| S4 | all | all | 100 | aug | 0.8284 | 0.9617 | clip_clean | 0.9147 | 0.9783 | -0.0166 | -0.0864 | 0.8100 | 0.1516 |
