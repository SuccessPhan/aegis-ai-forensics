# Classic ML models on frozen embeddings

Reference model: `linear_head` (the head AEGIS uses). AUROC on test_benchmark / s4_eval.

| head | features | model | S1 clean | S2 all | S3 all | S4 midjourney | S4 vqdm | S4 all |
|---|---|---|---|---|---|---|---|---|
| clean | clip | linear_head | 0.995 | 0.840 | 0.684 | 0.961 | 0.998 | 0.979 |
| clean | clip | logreg | 0.994 | 0.843 | 0.684 | 0.963 | 0.998 | 0.980 |
| clean | clip | linear_svm | 0.994 | 0.841 | 0.674 | 0.963 | 0.998 | 0.980 |
| clean | clip | lda | 0.986 | 0.847 | 0.728 | 0.952 | 0.992 | 0.972 |
| clean | clip | gaussian_nb | 0.968 | 0.861 | 0.757 | 0.915 | 0.983 | 0.949 |
| clean | clip | knn | 0.942 | 0.803 | 0.762 | 0.893 | 0.964 | 0.928 |
| clean | clip | rbf_svm | 0.994 | 0.861 | 0.726 | 0.962 | 0.996 | 0.979 |
| clean | clip | decision_tree | 0.798 | 0.612 | 0.618 | 0.806 | 0.803 | 0.804 |
| clean | clip | random_forest | 0.969 | 0.804 | 0.765 | 0.930 | 0.974 | 0.952 |
| clean | clip | hist_gb | 0.983 | 0.840 | 0.758 | 0.936 | 0.991 | 0.964 |
| clean | clip | xgboost | 0.985 | 0.840 | 0.760 | 0.941 | 0.994 | 0.967 |
| clean | clip | mlp | 0.994 | 0.846 | 0.693 | 0.961 | 0.997 | 0.979 |
| clean | dinov2 | linear_head | 0.892 | 0.774 | 0.806 | 0.803 | 0.944 | 0.875 |
| clean | dinov2 | logreg | 0.894 | 0.770 | 0.806 | 0.806 | 0.943 | 0.876 |
| clean | dinov2 | linear_svm | 0.894 | 0.768 | 0.805 | 0.807 | 0.942 | 0.877 |
| clean | dinov2 | lda | 0.851 | 0.736 | 0.770 | 0.781 | 0.930 | 0.857 |
| clean | dinov2 | gaussian_nb | 0.761 | 0.705 | 0.726 | 0.655 | 0.855 | 0.755 |
| clean | dinov2 | knn | 0.730 | 0.672 | 0.680 | 0.645 | 0.820 | 0.734 |
| clean | dinov2 | rbf_svm | 0.885 | 0.797 | 0.820 | 0.811 | 0.943 | 0.878 |
| clean | dinov2 | decision_tree | 0.625 | 0.545 | 0.576 | 0.642 | 0.721 | 0.681 |
| clean | dinov2 | random_forest | 0.845 | 0.722 | 0.768 | 0.784 | 0.921 | 0.854 |
| clean | dinov2 | hist_gb | 0.822 | 0.753 | 0.771 | 0.741 | 0.885 | 0.812 |
| clean | dinov2 | xgboost | 0.837 | 0.762 | 0.781 | 0.760 | 0.905 | 0.832 |
| clean | dinov2 | mlp | 0.909 | 0.798 | 0.827 | 0.816 | 0.948 | 0.884 |
| clean | clip+dinov2 | linear_head | 0.995 | 0.863 | 0.756 | 0.960 | 0.998 | 0.979 |
| clean | clip+dinov2 | logreg | 0.996 | 0.864 | 0.751 | 0.965 | 0.998 | 0.981 |
| clean | clip+dinov2 | linear_svm | 0.995 | 0.858 | 0.738 | 0.962 | 0.998 | 0.980 |
| clean | clip+dinov2 | lda | 0.983 | 0.857 | 0.793 | 0.946 | 0.989 | 0.967 |
| clean | clip+dinov2 | gaussian_nb | 0.974 | 0.870 | 0.799 | 0.932 | 0.991 | 0.962 |
| clean | clip+dinov2 | knn | 0.774 | 0.703 | 0.704 | 0.705 | 0.858 | 0.783 |
| clean | clip+dinov2 | rbf_svm | 0.994 | 0.883 | 0.786 | 0.960 | 0.997 | 0.979 |
| clean | clip+dinov2 | decision_tree | 0.790 | 0.611 | 0.619 | 0.803 | 0.804 | 0.803 |
| clean | clip+dinov2 | random_forest | 0.969 | 0.808 | 0.785 | 0.929 | 0.977 | 0.954 |
| clean | clip+dinov2 | hist_gb | 0.897 | 0.800 | 0.807 | 0.809 | 0.949 | 0.880 |
| clean | clip+dinov2 | xgboost | 0.905 | 0.810 | 0.814 | 0.820 | 0.956 | 0.889 |
| clean | clip+dinov2 | mlp | 0.994 | 0.871 | 0.788 | 0.956 | 0.997 | 0.976 |
| aug | clip | linear_head | 0.987 | 0.900 | 0.824 | 0.927 | 0.995 | 0.961 |
| aug | clip | logreg | 0.987 | 0.900 | 0.826 | 0.930 | 0.995 | 0.962 |
| aug | clip | linear_svm | 0.987 | 0.899 | 0.827 | 0.928 | 0.995 | 0.961 |
| aug | clip | lda | 0.988 | 0.881 | 0.800 | 0.932 | 0.994 | 0.963 |
| aug | clip | gaussian_nb | 0.974 | 0.840 | 0.774 | 0.916 | 0.986 | 0.951 |
| aug | clip | knn | 0.936 | 0.795 | 0.727 | 0.888 | 0.951 | 0.919 |
| aug | clip | rbf_svm | 0.991 | 0.901 | 0.834 | 0.947 | 0.995 | 0.971 |
| aug | clip | decision_tree | 0.803 | 0.660 | 0.665 | 0.783 | 0.823 | 0.803 |
| aug | clip | random_forest | 0.961 | 0.825 | 0.788 | 0.924 | 0.963 | 0.944 |
| aug | clip | hist_gb | 0.979 | 0.868 | 0.803 | 0.913 | 0.987 | 0.951 |
| aug | clip | xgboost | 0.981 | 0.874 | 0.808 | 0.917 | 0.990 | 0.954 |
| aug | clip | mlp | 0.991 | 0.918 | 0.846 | 0.937 | 0.995 | 0.966 |
| aug | dinov2 | linear_head | 0.874 | 0.782 | 0.813 | 0.782 | 0.922 | 0.853 |
| aug | dinov2 | logreg | 0.875 | 0.782 | 0.815 | 0.786 | 0.922 | 0.855 |
| aug | dinov2 | linear_svm | 0.876 | 0.783 | 0.816 | 0.787 | 0.921 | 0.855 |
| aug | dinov2 | lda | 0.837 | 0.739 | 0.771 | 0.763 | 0.918 | 0.840 |
| aug | dinov2 | gaussian_nb | 0.737 | 0.693 | 0.710 | 0.630 | 0.828 | 0.728 |
| aug | dinov2 | knn | 0.705 | 0.668 | 0.675 | 0.634 | 0.794 | 0.715 |
| aug | dinov2 | rbf_svm | 0.859 | 0.794 | 0.812 | 0.775 | 0.917 | 0.847 |
| aug | dinov2 | decision_tree | 0.632 | 0.595 | 0.611 | 0.630 | 0.727 | 0.678 |
| aug | dinov2 | random_forest | 0.833 | 0.748 | 0.764 | 0.766 | 0.904 | 0.836 |
| aug | dinov2 | hist_gb | 0.814 | 0.748 | 0.763 | 0.738 | 0.890 | 0.815 |
| aug | dinov2 | xgboost | 0.818 | 0.757 | 0.771 | 0.747 | 0.892 | 0.820 |
| aug | dinov2 | mlp | 0.894 | 0.819 | 0.844 | 0.799 | 0.933 | 0.868 |
| aug | clip+dinov2 | linear_head | 0.986 | 0.905 | 0.866 | 0.918 | 0.994 | 0.957 |
| aug | clip+dinov2 | logreg | 0.987 | 0.906 | 0.867 | 0.923 | 0.994 | 0.959 |
| aug | clip+dinov2 | linear_svm | 0.985 | 0.905 | 0.867 | 0.918 | 0.994 | 0.956 |
| aug | clip+dinov2 | lda | 0.988 | 0.882 | 0.834 | 0.937 | 0.993 | 0.965 |
| aug | clip+dinov2 | gaussian_nb | 0.968 | 0.846 | 0.803 | 0.911 | 0.984 | 0.948 |
| aug | clip+dinov2 | knn | 0.745 | 0.693 | 0.687 | 0.680 | 0.816 | 0.749 |
| aug | clip+dinov2 | rbf_svm | 0.990 | 0.909 | 0.856 | 0.939 | 0.995 | 0.967 |
| aug | clip+dinov2 | decision_tree | 0.800 | 0.653 | 0.668 | 0.781 | 0.817 | 0.799 |
| aug | clip+dinov2 | random_forest | 0.961 | 0.832 | 0.809 | 0.917 | 0.966 | 0.942 |
| aug | clip+dinov2 | hist_gb | 0.865 | 0.787 | 0.791 | 0.798 | 0.923 | 0.860 |
| aug | clip+dinov2 | xgboost | 0.877 | 0.800 | 0.803 | 0.799 | 0.932 | 0.866 |
| aug | clip+dinov2 | mlp | 0.990 | 0.922 | 0.881 | 0.931 | 0.995 | 0.963 |

## Accuracy

| head | features | model | S1 clean | S2 all | S3 all | S4 midjourney | S4 vqdm | S4 all |
|---|---|---|---|---|---|---|---|---|
| clean | clip | linear_head | 0.958 | 0.745 | 0.628 | 0.819 | 0.977 | 0.898 |
| clean | clip | logreg | 0.955 | 0.742 | 0.629 | 0.829 | 0.979 | 0.904 |
| clean | clip | linear_svm | 0.955 | 0.743 | 0.625 | 0.829 | 0.978 | 0.903 |
| clean | clip | lda | 0.917 | 0.743 | 0.658 | 0.812 | 0.945 | 0.879 |
| clean | clip | gaussian_nb | 0.902 | 0.777 | 0.687 | 0.793 | 0.937 | 0.865 |
| clean | clip | knn | 0.854 | 0.713 | 0.684 | 0.784 | 0.887 | 0.835 |
| clean | clip | rbf_svm | 0.954 | 0.763 | 0.662 | 0.819 | 0.971 | 0.895 |
| clean | clip | decision_tree | 0.706 | 0.585 | 0.608 | 0.750 | 0.668 | 0.709 |
| clean | clip | random_forest | 0.892 | 0.714 | 0.665 | 0.831 | 0.898 | 0.864 |
| clean | clip | hist_gb | 0.914 | 0.749 | 0.670 | 0.794 | 0.948 | 0.871 |
| clean | clip | xgboost | 0.914 | 0.749 | 0.678 | 0.790 | 0.954 | 0.872 |
| clean | clip | mlp | 0.960 | 0.748 | 0.632 | 0.820 | 0.977 | 0.898 |
| clean | dinov2 | linear_head | 0.789 | 0.703 | 0.729 | 0.675 | 0.870 | 0.772 |
| clean | dinov2 | logreg | 0.792 | 0.692 | 0.731 | 0.676 | 0.866 | 0.771 |
| clean | dinov2 | linear_svm | 0.791 | 0.686 | 0.729 | 0.674 | 0.864 | 0.769 |
| clean | dinov2 | lda | 0.752 | 0.673 | 0.701 | 0.675 | 0.859 | 0.767 |
| clean | dinov2 | gaussian_nb | 0.677 | 0.649 | 0.666 | 0.585 | 0.787 | 0.686 |
| clean | dinov2 | knn | 0.661 | 0.627 | 0.628 | 0.598 | 0.732 | 0.665 |
| clean | dinov2 | rbf_svm | 0.798 | 0.724 | 0.746 | 0.711 | 0.872 | 0.792 |
| clean | dinov2 | decision_tree | 0.593 | 0.525 | 0.552 | 0.589 | 0.667 | 0.628 |
| clean | dinov2 | random_forest | 0.748 | 0.656 | 0.698 | 0.684 | 0.840 | 0.762 |
| clean | dinov2 | hist_gb | 0.745 | 0.689 | 0.700 | 0.664 | 0.801 | 0.732 |
| clean | dinov2 | xgboost | 0.754 | 0.698 | 0.710 | 0.677 | 0.821 | 0.749 |
| clean | dinov2 | mlp | 0.806 | 0.724 | 0.751 | 0.679 | 0.868 | 0.774 |
| clean | clip+dinov2 | linear_head | 0.961 | 0.769 | 0.684 | 0.811 | 0.975 | 0.893 |
| clean | clip+dinov2 | logreg | 0.966 | 0.765 | 0.677 | 0.828 | 0.978 | 0.903 |
| clean | clip+dinov2 | linear_svm | 0.966 | 0.757 | 0.669 | 0.824 | 0.979 | 0.902 |
| clean | clip+dinov2 | lda | 0.902 | 0.764 | 0.699 | 0.801 | 0.933 | 0.867 |
| clean | clip+dinov2 | gaussian_nb | 0.906 | 0.772 | 0.721 | 0.810 | 0.955 | 0.883 |
| clean | clip+dinov2 | knn | 0.697 | 0.644 | 0.651 | 0.637 | 0.763 | 0.700 |
| clean | clip+dinov2 | rbf_svm | 0.953 | 0.785 | 0.708 | 0.822 | 0.973 | 0.897 |
| clean | clip+dinov2 | decision_tree | 0.704 | 0.584 | 0.609 | 0.751 | 0.671 | 0.711 |
| clean | clip+dinov2 | random_forest | 0.879 | 0.710 | 0.686 | 0.828 | 0.906 | 0.867 |
| clean | clip+dinov2 | hist_gb | 0.805 | 0.720 | 0.729 | 0.698 | 0.871 | 0.784 |
| clean | clip+dinov2 | xgboost | 0.810 | 0.722 | 0.738 | 0.706 | 0.881 | 0.793 |
| clean | clip+dinov2 | mlp | 0.953 | 0.776 | 0.708 | 0.803 | 0.975 | 0.889 |
| aug | clip | linear_head | 0.943 | 0.815 | 0.739 | 0.780 | 0.970 | 0.875 |
| aug | clip | logreg | 0.943 | 0.816 | 0.743 | 0.789 | 0.969 | 0.879 |
| aug | clip | linear_svm | 0.942 | 0.816 | 0.743 | 0.791 | 0.968 | 0.879 |
| aug | clip | lda | 0.948 | 0.791 | 0.716 | 0.803 | 0.958 | 0.881 |
| aug | clip | gaussian_nb | 0.916 | 0.762 | 0.702 | 0.777 | 0.938 | 0.858 |
| aug | clip | knn | 0.856 | 0.708 | 0.653 | 0.793 | 0.875 | 0.834 |
| aug | clip | rbf_svm | 0.957 | 0.816 | 0.749 | 0.822 | 0.968 | 0.895 |
| aug | clip | decision_tree | 0.729 | 0.629 | 0.633 | 0.728 | 0.736 | 0.732 |
| aug | clip | random_forest | 0.876 | 0.745 | 0.693 | 0.826 | 0.884 | 0.855 |
| aug | clip | hist_gb | 0.929 | 0.781 | 0.719 | 0.801 | 0.943 | 0.872 |
| aug | clip | xgboost | 0.931 | 0.786 | 0.721 | 0.814 | 0.949 | 0.882 |
| aug | clip | mlp | 0.950 | 0.834 | 0.765 | 0.806 | 0.972 | 0.889 |
| aug | dinov2 | linear_head | 0.785 | 0.713 | 0.743 | 0.687 | 0.848 | 0.768 |
| aug | dinov2 | logreg | 0.789 | 0.714 | 0.743 | 0.687 | 0.849 | 0.768 |
| aug | dinov2 | linear_svm | 0.789 | 0.713 | 0.741 | 0.685 | 0.848 | 0.767 |
| aug | dinov2 | lda | 0.754 | 0.674 | 0.700 | 0.675 | 0.838 | 0.757 |
| aug | dinov2 | gaussian_nb | 0.671 | 0.640 | 0.654 | 0.579 | 0.762 | 0.670 |
| aug | dinov2 | knn | 0.646 | 0.616 | 0.625 | 0.597 | 0.709 | 0.653 |
| aug | dinov2 | rbf_svm | 0.777 | 0.723 | 0.743 | 0.694 | 0.831 | 0.763 |
| aug | dinov2 | decision_tree | 0.604 | 0.581 | 0.593 | 0.553 | 0.681 | 0.617 |
| aug | dinov2 | random_forest | 0.755 | 0.683 | 0.693 | 0.682 | 0.818 | 0.750 |
| aug | dinov2 | hist_gb | 0.729 | 0.679 | 0.694 | 0.660 | 0.808 | 0.734 |
| aug | dinov2 | xgboost | 0.736 | 0.692 | 0.701 | 0.672 | 0.806 | 0.739 |
| aug | dinov2 | mlp | 0.821 | 0.756 | 0.771 | 0.708 | 0.859 | 0.783 |
| aug | clip+dinov2 | linear_head | 0.929 | 0.818 | 0.781 | 0.770 | 0.962 | 0.866 |
| aug | clip+dinov2 | logreg | 0.933 | 0.818 | 0.782 | 0.778 | 0.962 | 0.870 |
| aug | clip+dinov2 | linear_svm | 0.932 | 0.817 | 0.787 | 0.777 | 0.961 | 0.869 |
| aug | clip+dinov2 | lda | 0.946 | 0.795 | 0.745 | 0.824 | 0.960 | 0.892 |
| aug | clip+dinov2 | gaussian_nb | 0.904 | 0.764 | 0.726 | 0.794 | 0.940 | 0.867 |
| aug | clip+dinov2 | knn | 0.672 | 0.636 | 0.627 | 0.629 | 0.734 | 0.682 |
| aug | clip+dinov2 | rbf_svm | 0.954 | 0.827 | 0.773 | 0.823 | 0.971 | 0.897 |
| aug | clip+dinov2 | decision_tree | 0.723 | 0.625 | 0.633 | 0.721 | 0.727 | 0.724 |
| aug | clip+dinov2 | random_forest | 0.885 | 0.755 | 0.709 | 0.820 | 0.889 | 0.854 |
| aug | clip+dinov2 | hist_gb | 0.786 | 0.716 | 0.714 | 0.703 | 0.839 | 0.771 |
| aug | clip+dinov2 | xgboost | 0.788 | 0.729 | 0.732 | 0.699 | 0.843 | 0.771 |
| aug | clip+dinov2 | mlp | 0.946 | 0.838 | 0.800 | 0.791 | 0.970 | 0.880 |

## Models

| model | family | grid | note |
|---|---|---|---|
| decision_tree | tree | {'max_depth': [5, 10, 20]} | single CART tree |
| gaussian_nb | probabilistic | {'pca': [16, 64]} | Gaussian Naive Bayes on PCA |
| hist_gb | ensemble | {'max_leaf_nodes': [15, 31]} | histogram gradient boosting on PCA-128 |
| knn | instance | {'k': [5, 15, 45]} | kNN, cosine geometry, distance-weighted |
| lda | probabilistic | {'pca': [64, 256]} | LDA with Ledoit-Wolf shrinkage |
| linear_head | linear | - | AEGIS head (reference): standardise + Linear + Adam |
| linear_svm | linear | {'C': [0.001, 0.01, 0.1]} | linear SVM (hinge loss) |
| logreg | linear | {'C': [0.01, 0.1, 1.0]} | L2 logistic regression |
| mlp | neural | {'alpha': [0.0001, 0.01]} | MLP, 1 hidden layer (256), early stopping |
| random_forest | ensemble | {'min_samples_leaf': [1, 5]} | random forest, 300 trees |
| rbf_svm | kernel | {'C': [1.0, 10.0]} | RBF-kernel SVM on PCA-128 |
| xgboost | ensemble | {'max_depth': [4, 6]} | XGBoost on PCA-128 (optional dependency) |
