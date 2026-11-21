# Fusion selection (val_gating_fit -> val_gating_verify; test_benchmark never read)

**Selected fusion: `prob`**

Rule: logit only if overall delta AUROC (AEGIS, logit - prob) > 0 with 95% CI excluding 0 and ECE increase <= 0.01.

## head variant = clean: `prob`

prob: overall delta AUROC (logit - prob) is not > 0 with a CI excluding 0

ECE increase (overall AEGIS) = -0.0055; cluster bootstrap = True

| slice | model | metric | prob | logit | delta | CI95 | p | n | ECE prob | ECE logit |
|---|---|---|---|---|---|---|---|---|---|---|
| clean | aegis | auroc | 0.9977 | 0.9976 | -0.0001 | [-0.0005, +0.0002] | 0.613 | 480 | 0.1288 | 0.1230 |
| clean | aegis | accuracy | 0.9646 | 0.9667 | +0.0021 | [+0.0000, +0.0063] | 0.723 | 480 | 0.1288 | 0.1230 |
| clean | static_lr | auroc | 0.9978 | 0.9984 | +0.0005 | [+0.0001, +0.0011] | 0.004 | 480 | 0.0530 | 0.0451 |
| clean | static_lr | accuracy | 0.9292 | 0.9250 | -0.0042 | [-0.0146, +0.0062] | 0.557 | 480 | 0.0530 | 0.0451 |
| overall | aegis | auroc | 0.9195 | 0.9194 | -0.0001 | [-0.0011, +0.0009] | 0.810 | 1440 | 0.0658 | 0.0603 |
| overall | aegis | accuracy | 0.8243 | 0.8292 | +0.0049 | [+0.0014, +0.0090] | 0.015 | 1440 | 0.0658 | 0.0603 |
| overall | static_lr | auroc | 0.9099 | 0.9143 | +0.0043 | [+0.0020, +0.0069] | 0.001 | 1440 | 0.0341 | 0.0281 |
| overall | static_lr | accuracy | 0.8139 | 0.8222 | +0.0083 | [+0.0014, +0.0160] | 0.032 | 1440 | 0.0341 | 0.0281 |

## head variant = aug: `prob`

prob: overall delta AUROC (logit - prob) is not > 0 with a CI excluding 0

ECE increase (overall AEGIS) = -0.0055; cluster bootstrap = True

| slice | model | metric | prob | logit | delta | CI95 | p | n | ECE prob | ECE logit |
|---|---|---|---|---|---|---|---|---|---|---|
| clean | aegis | auroc | 0.9958 | 0.9927 | -0.0031 | [-0.0075, +0.0000] | 0.056 | 480 | 0.0523 | 0.0394 |
| clean | aegis | accuracy | 0.9771 | 0.9771 | +0.0000 | [+0.0000, +0.0000] | 1.000 | 480 | 0.0523 | 0.0394 |
| clean | static_lr | auroc | 0.9979 | 0.9968 | -0.0011 | [-0.0034, +0.0004] | 0.292 | 480 | 0.0325 | 0.0195 |
| clean | static_lr | accuracy | 0.9750 | 0.9708 | -0.0042 | [-0.0146, +0.0042] | 0.523 | 480 | 0.0325 | 0.0195 |
| overall | aegis | auroc | 0.9559 | 0.9547 | -0.0012 | [-0.0030, +0.0004] | 0.149 | 1440 | 0.0406 | 0.0351 |
| overall | aegis | accuracy | 0.8833 | 0.8833 | +0.0000 | [-0.0049, +0.0042] | 1.000 | 1440 | 0.0406 | 0.0351 |
| overall | static_lr | auroc | 0.9575 | 0.9588 | +0.0012 | [-0.0012, +0.0034] | 0.316 | 1440 | 0.0324 | 0.0262 |
| overall | static_lr | accuracy | 0.8771 | 0.8771 | +0.0000 | [-0.0056, +0.0056] | 1.000 | 1440 | 0.0324 | 0.0262 |

