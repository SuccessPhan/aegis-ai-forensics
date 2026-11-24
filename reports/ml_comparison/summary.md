# Machine learning vs the deep image models of AEGIS

Rows with features `image_models` are the AEGIS models (deep learning); the other rows are the machine-learning models on hand-crafted features or on frozen embeddings. AUROC on test_benchmark / s4_eval (S4 without few-shot adaptation for every method). Significance (significance.csv) is ML model minus `aegis`.

| head | features | model | S1 clean | S2 all | S3 all | S4 midjourney | S4 vqdm | S4 all |
|---|---|---|---|---|---|---|---|---|
| clean | image_models | clip_expert | 0.994 | 0.839 | 0.682 | 0.961 | 0.997 | 0.979 |
| clean | image_models | dinov2_expert | 0.891 | 0.774 | 0.805 | 0.802 | 0.944 | 0.874 |
| clean | image_models | freq_expert | 0.985 | 0.548 | 0.541 | 0.829 | 0.991 | 0.909 |
| clean | image_models | static_lr | 0.993 | 0.869 | 0.808 | 0.946 | 0.998 | 0.972 |
| clean | image_models | aegis | 0.993 | 0.865 | 0.822 | 0.957 | 0.998 | 0.978 |
| clean | handcrafted | rbf_svm | 0.948 | 0.746 | 0.544 | 0.789 | 0.918 | 0.854 |
| clean | handcrafted | random_forest | 0.899 | 0.681 | 0.542 | 0.810 | 0.823 | 0.817 |
| clean | handcrafted | xgboost | 0.947 | 0.753 | 0.607 | 0.798 | 0.901 | 0.850 |
| clean | clip | rbf_svm | 0.994 | 0.861 | 0.726 | 0.962 | 0.996 | 0.979 |
| clean | clip | random_forest | 0.969 | 0.804 | 0.765 | 0.930 | 0.974 | 0.952 |
| clean | clip | xgboost | 0.985 | 0.840 | 0.760 | 0.941 | 0.994 | 0.967 |
| clean | dinov2 | rbf_svm | 0.885 | 0.797 | 0.820 | 0.811 | 0.943 | 0.878 |
| clean | dinov2 | random_forest | 0.845 | 0.722 | 0.768 | 0.784 | 0.921 | 0.854 |
| clean | dinov2 | xgboost | 0.837 | 0.762 | 0.781 | 0.760 | 0.905 | 0.832 |
| aug | image_models | clip_expert | 0.987 | 0.899 | 0.823 | 0.928 | 0.995 | 0.961 |
| aug | image_models | dinov2_expert | 0.874 | 0.782 | 0.813 | 0.783 | 0.923 | 0.854 |
| aug | image_models | freq_expert | 0.967 | 0.803 | 0.564 | 0.774 | 0.975 | 0.874 |
| aug | image_models | static_lr | 0.996 | 0.932 | 0.871 | 0.916 | 0.998 | 0.957 |
| aug | image_models | aegis | 0.996 | 0.928 | 0.868 | 0.918 | 0.998 | 0.959 |
| aug | handcrafted | rbf_svm | 0.940 | 0.770 | 0.678 | 0.773 | 0.893 | 0.833 |
| aug | handcrafted | random_forest | 0.895 | 0.737 | 0.657 | 0.791 | 0.809 | 0.800 |
| aug | handcrafted | xgboost | 0.934 | 0.775 | 0.679 | 0.766 | 0.851 | 0.808 |
| aug | clip | rbf_svm | 0.991 | 0.901 | 0.834 | 0.947 | 0.995 | 0.971 |
| aug | clip | random_forest | 0.961 | 0.825 | 0.788 | 0.924 | 0.963 | 0.944 |
| aug | clip | xgboost | 0.981 | 0.874 | 0.808 | 0.917 | 0.990 | 0.954 |
| aug | dinov2 | rbf_svm | 0.859 | 0.794 | 0.812 | 0.775 | 0.917 | 0.847 |
| aug | dinov2 | random_forest | 0.833 | 0.748 | 0.764 | 0.766 | 0.904 | 0.836 |
| aug | dinov2 | xgboost | 0.818 | 0.757 | 0.771 | 0.747 | 0.892 | 0.820 |

## Accuracy

| head | features | model | S1 clean | S2 all | S3 all | S4 midjourney | S4 vqdm | S4 all |
|---|---|---|---|---|---|---|---|---|
| clean | image_models | clip_expert | 0.963 | 0.725 | 0.611 | 0.858 | 0.976 | 0.917 |
| clean | image_models | dinov2_expert | 0.825 | 0.639 | 0.689 | 0.722 | 0.864 | 0.793 |
| clean | image_models | freq_expert | 0.941 | 0.519 | 0.505 | 0.615 | 0.961 | 0.788 |
| clean | image_models | static_lr | 0.932 | 0.759 | 0.728 | 0.763 | 0.973 | 0.868 |
| clean | image_models | aegis | 0.958 | 0.742 | 0.729 | 0.783 | 0.981 | 0.882 |
| clean | handcrafted | rbf_svm | 0.832 | 0.676 | 0.524 | 0.606 | 0.754 | 0.680 |
| clean | handcrafted | random_forest | 0.677 | 0.610 | 0.518 | 0.555 | 0.609 | 0.582 |
| clean | handcrafted | xgboost | 0.769 | 0.675 | 0.511 | 0.566 | 0.689 | 0.628 |
| clean | clip | rbf_svm | 0.954 | 0.763 | 0.662 | 0.819 | 0.971 | 0.895 |
| clean | clip | random_forest | 0.892 | 0.714 | 0.665 | 0.831 | 0.898 | 0.864 |
| clean | clip | xgboost | 0.914 | 0.749 | 0.678 | 0.790 | 0.954 | 0.872 |
| clean | dinov2 | rbf_svm | 0.798 | 0.724 | 0.746 | 0.711 | 0.872 | 0.792 |
| clean | dinov2 | random_forest | 0.748 | 0.656 | 0.698 | 0.684 | 0.840 | 0.762 |
| clean | dinov2 | xgboost | 0.754 | 0.698 | 0.710 | 0.677 | 0.821 | 0.749 |
| aug | image_models | clip_expert | 0.942 | 0.816 | 0.740 | 0.789 | 0.968 | 0.879 |
| aug | image_models | dinov2_expert | 0.794 | 0.707 | 0.738 | 0.689 | 0.849 | 0.769 |
| aug | image_models | freq_expert | 0.894 | 0.694 | 0.539 | 0.584 | 0.921 | 0.753 |
| aug | image_models | static_lr | 0.973 | 0.851 | 0.784 | 0.708 | 0.983 | 0.846 |
| aug | image_models | aegis | 0.977 | 0.847 | 0.789 | 0.717 | 0.982 | 0.849 |
| aug | handcrafted | rbf_svm | 0.864 | 0.699 | 0.630 | 0.669 | 0.796 | 0.732 |
| aug | handcrafted | random_forest | 0.808 | 0.667 | 0.614 | 0.656 | 0.697 | 0.676 |
| aug | handcrafted | xgboost | 0.850 | 0.703 | 0.629 | 0.649 | 0.741 | 0.695 |
| aug | clip | rbf_svm | 0.957 | 0.816 | 0.749 | 0.822 | 0.968 | 0.895 |
| aug | clip | random_forest | 0.876 | 0.745 | 0.693 | 0.826 | 0.884 | 0.855 |
| aug | clip | xgboost | 0.931 | 0.786 | 0.721 | 0.814 | 0.949 | 0.882 |
| aug | dinov2 | rbf_svm | 0.777 | 0.723 | 0.743 | 0.694 | 0.831 | 0.763 |
| aug | dinov2 | random_forest | 0.755 | 0.683 | 0.693 | 0.682 | 0.818 | 0.750 |
| aug | dinov2 | xgboost | 0.736 | 0.692 | 0.701 | 0.672 | 0.806 | 0.739 |

## Models

| model | group | description |
|---|---|---|
| clip_expert | deep learning (AEGIS) | CLIP ViT-L/14 (frozen) + linear head |
| dinov2_expert | deep learning (AEGIS) | DINOv2 ViT-B/14 (frozen) + linear head |
| freq_expert | deep learning (AEGIS) | SRM high-pass CNN (1.17 M params, trained from scratch) |
| static_lr | deep learning (AEGIS) | 3 experts, fixed learned weights (logistic regression on calibrated logits) |
| aegis | deep learning (AEGIS) | 3 experts + per-image gating network (full AEGIS, no few-shot adaptation) |
| rbf_svm | machine learning (kernel) | SVM, Gaussian (RBF) kernel; standardised input (PCA-128 for embeddings); grid {'C': [1.0, 10.0]} |
| random_forest | machine learning (bagging) | random forest, 300 trees, sqrt(d) features per split; grid {'min_samples_leaf': [1, 5]} |
| xgboost | machine learning (boosting) | XGBoost, 400 boosted trees, lr 0.1 (PCA-128 for embeddings); grid {'max_depth': [4, 6]} |
