# External test set `newgen`

AI images per generator vs 368 real images. AEGIS image models (`image_models`) and the three machine-learning models trained on the main GenImage splits; no adaptation to the new generators.

## AUROC

| head | features | model | flux | imagegbt | nano | sdxl | seedream | all |
|---|---|---|---|---|---|---|---|---|
| clean | image_models | clip_expert | 0.871 | 0.825 | 0.811 | 0.871 | 0.834 | 0.843 |
| clean | image_models | dinov2_expert | 0.576 | 0.398 | 0.450 | 0.744 | 0.590 | 0.553 |
| clean | image_models | freq_expert | 0.416 | 0.511 | 0.568 | 0.654 | 0.567 | 0.541 |
| clean | image_models | static_lr | 0.798 | 0.668 | 0.669 | 0.884 | 0.777 | 0.760 |
| clean | image_models | aegis | 0.778 | 0.652 | 0.691 | 0.893 | 0.764 | 0.757 |
| clean | handcrafted | rbf_svm | 0.794 | 0.630 | 0.739 | 0.812 | 0.865 | 0.769 |
| clean | handcrafted | random_forest | 0.811 | 0.660 | 0.757 | 0.830 | 0.758 | 0.764 |
| clean | handcrafted | xgboost | 0.796 | 0.633 | 0.749 | 0.805 | 0.779 | 0.754 |
| clean | clip | rbf_svm | 0.842 | 0.829 | 0.809 | 0.854 | 0.836 | 0.834 |
| clean | clip | random_forest | 0.840 | 0.836 | 0.805 | 0.894 | 0.849 | 0.845 |
| clean | clip | xgboost | 0.829 | 0.828 | 0.820 | 0.850 | 0.827 | 0.831 |
| clean | dinov2 | rbf_svm | 0.647 | 0.399 | 0.458 | 0.794 | 0.552 | 0.572 |
| clean | dinov2 | random_forest | 0.524 | 0.437 | 0.448 | 0.698 | 0.473 | 0.516 |
| clean | dinov2 | xgboost | 0.607 | 0.415 | 0.451 | 0.714 | 0.521 | 0.543 |
| aug | image_models | clip_expert | 0.856 | 0.817 | 0.751 | 0.869 | 0.804 | 0.820 |
| aug | image_models | dinov2_expert | 0.551 | 0.414 | 0.474 | 0.756 | 0.600 | 0.560 |
| aug | image_models | freq_expert | 0.392 | 0.522 | 0.611 | 0.696 | 0.633 | 0.568 |
| aug | image_models | static_lr | 0.734 | 0.685 | 0.654 | 0.877 | 0.763 | 0.743 |
| aug | image_models | aegis | 0.677 | 0.686 | 0.682 | 0.882 | 0.735 | 0.732 |
| aug | handcrafted | rbf_svm | 0.776 | 0.616 | 0.684 | 0.762 | 0.819 | 0.733 |
| aug | handcrafted | random_forest | 0.749 | 0.625 | 0.705 | 0.775 | 0.725 | 0.717 |
| aug | handcrafted | xgboost | 0.739 | 0.627 | 0.712 | 0.773 | 0.749 | 0.721 |
| aug | clip | rbf_svm | 0.843 | 0.808 | 0.731 | 0.850 | 0.828 | 0.813 |
| aug | clip | random_forest | 0.831 | 0.806 | 0.763 | 0.872 | 0.820 | 0.819 |
| aug | clip | xgboost | 0.724 | 0.788 | 0.710 | 0.824 | 0.784 | 0.765 |
| aug | dinov2 | rbf_svm | 0.699 | 0.485 | 0.538 | 0.814 | 0.613 | 0.632 |
| aug | dinov2 | random_forest | 0.541 | 0.413 | 0.471 | 0.707 | 0.447 | 0.517 |
| aug | dinov2 | xgboost | 0.682 | 0.489 | 0.470 | 0.740 | 0.537 | 0.586 |

## Balanced accuracy (threshold 0.5)

| head | features | model | flux | imagegbt | nano | sdxl | seedream | all |
|---|---|---|---|---|---|---|---|---|
| clean | image_models | clip_expert | 0.840 | 0.756 | 0.759 | 0.785 | 0.772 | 0.783 |
| clean | image_models | dinov2_expert | 0.547 | 0.455 | 0.477 | 0.687 | 0.569 | 0.548 |
| clean | image_models | freq_expert | 0.463 | 0.479 | 0.491 | 0.570 | 0.505 | 0.501 |
| clean | image_models | static_lr | 0.684 | 0.552 | 0.588 | 0.785 | 0.693 | 0.661 |
| clean | image_models | aegis | 0.694 | 0.602 | 0.571 | 0.795 | 0.676 | 0.668 |
| clean | handcrafted | rbf_svm | 0.580 | 0.497 | 0.575 | 0.615 | 0.667 | 0.587 |
| clean | handcrafted | random_forest | 0.623 | 0.477 | 0.556 | 0.595 | 0.582 | 0.568 |
| clean | handcrafted | xgboost | 0.638 | 0.481 | 0.559 | 0.625 | 0.612 | 0.585 |
| clean | clip | rbf_svm | 0.810 | 0.776 | 0.766 | 0.792 | 0.779 | 0.785 |
| clean | clip | random_forest | 0.829 | 0.826 | 0.761 | 0.840 | 0.827 | 0.817 |
| clean | clip | xgboost | 0.756 | 0.755 | 0.758 | 0.772 | 0.811 | 0.770 |
| clean | dinov2 | rbf_svm | 0.571 | 0.501 | 0.483 | 0.733 | 0.536 | 0.565 |
| clean | dinov2 | random_forest | 0.487 | 0.425 | 0.448 | 0.646 | 0.461 | 0.494 |
| clean | dinov2 | xgboost | 0.532 | 0.448 | 0.457 | 0.628 | 0.457 | 0.505 |
| aug | image_models | clip_expert | 0.859 | 0.735 | 0.686 | 0.791 | 0.725 | 0.761 |
| aug | image_models | dinov2_expert | 0.491 | 0.442 | 0.479 | 0.702 | 0.571 | 0.537 |
| aug | image_models | freq_expert | 0.433 | 0.474 | 0.499 | 0.578 | 0.552 | 0.506 |
| aug | image_models | static_lr | 0.531 | 0.542 | 0.578 | 0.815 | 0.670 | 0.626 |
| aug | image_models | aegis | 0.539 | 0.579 | 0.575 | 0.786 | 0.654 | 0.625 |
| aug | handcrafted | rbf_svm | 0.648 | 0.523 | 0.612 | 0.665 | 0.731 | 0.636 |
| aug | handcrafted | random_forest | 0.674 | 0.571 | 0.619 | 0.698 | 0.658 | 0.645 |
| aug | handcrafted | xgboost | 0.665 | 0.556 | 0.605 | 0.711 | 0.684 | 0.645 |
| aug | clip | rbf_svm | 0.799 | 0.794 | 0.716 | 0.782 | 0.808 | 0.780 |
| aug | clip | random_forest | 0.819 | 0.802 | 0.764 | 0.843 | 0.817 | 0.809 |
| aug | clip | xgboost | 0.678 | 0.719 | 0.669 | 0.734 | 0.761 | 0.712 |
| aug | dinov2 | rbf_svm | 0.611 | 0.555 | 0.550 | 0.747 | 0.563 | 0.606 |
| aug | dinov2 | random_forest | 0.544 | 0.446 | 0.483 | 0.641 | 0.443 | 0.512 |
| aug | dinov2 | xgboost | 0.616 | 0.548 | 0.504 | 0.714 | 0.530 | 0.583 |
