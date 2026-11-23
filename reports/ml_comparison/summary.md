# Classic ML models on frozen embeddings

Reference model: `linear_head` (the head AEGIS uses). AUROC on test_benchmark / s4_eval.

| head | features | model | S1 clean | S2 all | S3 all | S4 midjourney | S4 vqdm | S4 all |
|---|---|---|---|---|---|---|---|---|
| clean | clip | linear_head | 0.995 | 0.840 | 0.684 | 0.961 | 0.998 | 0.979 |
| clean | clip | rbf_svm | 0.994 | 0.861 | 0.726 | 0.962 | 0.996 | 0.979 |
| clean | clip | random_forest | 0.969 | 0.804 | 0.765 | 0.930 | 0.974 | 0.952 |
| clean | dinov2 | linear_head | 0.892 | 0.774 | 0.806 | 0.803 | 0.944 | 0.875 |
| clean | dinov2 | rbf_svm | 0.885 | 0.797 | 0.820 | 0.811 | 0.943 | 0.878 |
| clean | dinov2 | random_forest | 0.845 | 0.722 | 0.768 | 0.784 | 0.921 | 0.854 |
| clean | clip+dinov2 | linear_head | 0.995 | 0.863 | 0.756 | 0.960 | 0.998 | 0.979 |
| clean | clip+dinov2 | rbf_svm | 0.994 | 0.883 | 0.786 | 0.960 | 0.997 | 0.979 |
| clean | clip+dinov2 | random_forest | 0.969 | 0.808 | 0.785 | 0.929 | 0.977 | 0.954 |
| aug | clip | linear_head | 0.987 | 0.900 | 0.824 | 0.927 | 0.995 | 0.961 |
| aug | clip | rbf_svm | 0.991 | 0.901 | 0.834 | 0.947 | 0.995 | 0.971 |
| aug | clip | random_forest | 0.961 | 0.825 | 0.788 | 0.924 | 0.963 | 0.944 |
| aug | dinov2 | linear_head | 0.874 | 0.782 | 0.813 | 0.782 | 0.922 | 0.853 |
| aug | dinov2 | rbf_svm | 0.859 | 0.794 | 0.812 | 0.775 | 0.917 | 0.847 |
| aug | dinov2 | random_forest | 0.833 | 0.748 | 0.764 | 0.766 | 0.904 | 0.836 |
| aug | clip+dinov2 | linear_head | 0.986 | 0.905 | 0.866 | 0.918 | 0.994 | 0.957 |
| aug | clip+dinov2 | rbf_svm | 0.990 | 0.909 | 0.856 | 0.939 | 0.995 | 0.967 |
| aug | clip+dinov2 | random_forest | 0.961 | 0.832 | 0.809 | 0.917 | 0.966 | 0.942 |

## Accuracy

| head | features | model | S1 clean | S2 all | S3 all | S4 midjourney | S4 vqdm | S4 all |
|---|---|---|---|---|---|---|---|---|
| clean | clip | linear_head | 0.958 | 0.745 | 0.628 | 0.819 | 0.977 | 0.898 |
| clean | clip | rbf_svm | 0.954 | 0.763 | 0.662 | 0.819 | 0.971 | 0.895 |
| clean | clip | random_forest | 0.892 | 0.714 | 0.665 | 0.831 | 0.898 | 0.864 |
| clean | dinov2 | linear_head | 0.789 | 0.703 | 0.729 | 0.675 | 0.870 | 0.772 |
| clean | dinov2 | rbf_svm | 0.798 | 0.724 | 0.746 | 0.711 | 0.872 | 0.792 |
| clean | dinov2 | random_forest | 0.748 | 0.656 | 0.698 | 0.684 | 0.840 | 0.762 |
| clean | clip+dinov2 | linear_head | 0.961 | 0.769 | 0.684 | 0.811 | 0.975 | 0.893 |
| clean | clip+dinov2 | rbf_svm | 0.953 | 0.785 | 0.708 | 0.822 | 0.973 | 0.897 |
| clean | clip+dinov2 | random_forest | 0.879 | 0.710 | 0.686 | 0.828 | 0.906 | 0.867 |
| aug | clip | linear_head | 0.943 | 0.815 | 0.739 | 0.780 | 0.970 | 0.875 |
| aug | clip | rbf_svm | 0.957 | 0.816 | 0.749 | 0.822 | 0.968 | 0.895 |
| aug | clip | random_forest | 0.876 | 0.745 | 0.693 | 0.826 | 0.884 | 0.855 |
| aug | dinov2 | linear_head | 0.785 | 0.713 | 0.743 | 0.687 | 0.848 | 0.768 |
| aug | dinov2 | rbf_svm | 0.777 | 0.723 | 0.743 | 0.694 | 0.831 | 0.763 |
| aug | dinov2 | random_forest | 0.755 | 0.683 | 0.693 | 0.682 | 0.818 | 0.750 |
| aug | clip+dinov2 | linear_head | 0.929 | 0.818 | 0.781 | 0.770 | 0.962 | 0.866 |
| aug | clip+dinov2 | rbf_svm | 0.954 | 0.827 | 0.773 | 0.823 | 0.971 | 0.897 |
| aug | clip+dinov2 | random_forest | 0.885 | 0.755 | 0.709 | 0.820 | 0.889 | 0.854 |

## Models

| model | family | grid | note |
|---|---|---|---|
| linear_head | linear | - | AEGIS head (reference): standardise + Linear + Adam |
| random_forest | ensemble | {'min_samples_leaf': [1, 5]} | random forest, 300 trees |
| rbf_svm | kernel | {'C': [1.0, 10.0]} | RBF-kernel SVM on PCA-128 |
