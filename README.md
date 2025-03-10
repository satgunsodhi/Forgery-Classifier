# Signature Recognition Model Comparison Report

## Executive Summary

This report presents an evaluation of different machine learning models on the CEDAR signatures dataset, focusing on their robustness against Neural Style Transfer (NST) manipulations. Four models were tested both on standard validation data and on a specialized test set containing original signatures and their NST-transformed variants.

The Support Vector Classifier (SVC) demonstrated exceptional performance across both standard validation and NST-transformed data, maintaining perfect scores on the NST test set. In contrast, other models showed significant performance degradation when tested against style-transferred signatures, indicating potential vulnerability to this type of transformation.

## Project Overview

**Objective**: To evaluate the robustness of signature recognition models against Neural Style Transfer transformations.

**Dataset**: CEDAR signatures dataset
- Training and validation split for initial model evaluation
- Specialized test set with 55 original signatures and 55 NST-transformed versions for robustness testing

**Models Evaluated**:
- GaussianNB (Naive Bayes)
- LogisticRegression
- SVC (Support Vector Classifier)
- KNeighborsClassifier

## Performance Metrics

Performance was evaluated using four key metrics:
- **Accuracy**: Overall correctness of predictions
- **Recall**: Ability to identify all relevant instances (sensitivity)
- **Precision**: Accuracy of positive predictions
- **ROC AUC Score**: Model's ability to distinguish between classes

## Results

### Standard Validation Performance

| Model | Accuracy | Recall | Precision | ROC AUC Score |
|-------|----------|--------|-----------|---------------|
| GaussianNB | 0.997 | 0.994 | 1.000 | 0.997 |
| LogisticRegression | 0.805 | 0.826 | 0.800 | 0.805 |
| SVC | 0.994 | 1.000 | 0.989 | 0.994 |
| KNeighborsClassifier | 0.900 | 0.841 | 1.000 | 0.894 |

### Neural Style Transfer Test Performance

| Model | Accuracy | Recall | Precision | ROC AUC Score |
|-------|----------|--------|-----------|---------------|
| GaussianNB | 0.500 | 0.500 | 1.000 | 0.500 |
| LogisticRegression | 0.445 | 0.471 | 0.891 | 0.445 |
| SVC | 1.000 | 1.000 | 1.000 | 1.000 |
| KNeighborsClassifier | 0.500 | 0.500 | 1.000 | 0.500 |

## Analysis

### Model Performance on Standard Validation

- **GaussianNB** performed exceptionally well with near-perfect metrics (accuracy: 0.997, recall: 0.994, precision: 1.000).
- **LogisticRegression** showed the weakest performance among the models with an accuracy of 0.805.
- **SVC** demonstrated outstanding results with perfect recall (1.000) and high precision (0.989).
- **KNeighborsClassifier** achieved good results with perfect precision but lower recall (0.841) compared to GaussianNB and SVC.

### Model Robustness Against NST

- **GaussianNB**: Performance dropped significantly to 50% accuracy when tested against NST data, indicating poor robustness.
- **LogisticRegression**: Similarly struggled with NST data, showing the lowest accuracy (0.445) among all models.
- **SVC**: Demonstrated remarkable robustness, maintaining perfect scores (1.000) across all metrics when tested against NST-transformed signatures.
- **KNeighborsClassifier**: Performance dropped to 50% accuracy, similar to GaussianNB.

## Performance Comparison Visualization

```
                 Standard Validation     NST Testing
                 ------------------     ------------
GaussianNB       █████████▉ (0.997)    █████ (0.500)
LogisticRegr.    ████████ (0.805)      ████▍ (0.445)
SVC              █████████▉ (0.994)    ██████████ (1.000)
KNeighbors       █████████ (0.900)     █████ (0.500)
```

## Key Findings

1. **SVC demonstrates exceptional robustness**: Unlike other models, SVC maintained perfect performance when tested against NST-transformed signatures, suggesting it captures fundamental signature characteristics that remain invariant to style transformations.

2. **Style transfer vulnerability**: Most models (GaussianNB, LogisticRegression, KNeighborsClassifier) showed significant performance degradation when faced with style-transferred signatures, with accuracy dropping to near-random chance levels (0.445-0.500).

3. **Precision remains high**: Despite accuracy and recall drops, precision remained high for all models when tested against NST data, suggesting models maintain good specificity even when overall performance decreases.

## Recommendations

1. **Adopt SVC for signature verification systems**: Based on its exceptional robustness against style transfer attacks, SVC should be the preferred model for signature verification systems where security against manipulations is critical.

2. **Data augmentation with style transfers**: For improving other models, consider incorporating NST-transformed signatures into the training data to enhance robustness.

3. **Ensemble approach**: Consider an ensemble model that leverages SVC's strengths while incorporating complementary features from other models.

4. **Further investigation**: Explore why SVC performs well against NST transformations while other models fail, which could lead to insights for improving model robustness generally.

## Future Work

1. Test against other types of image manipulations beyond NST
2. Evaluate deep learning models (CNNs) on the same dataset
3. Investigate feature importance to understand SVC's robustness
4. Explore model performance on larger and more diverse signature datasets
5. Develop specialized defensive techniques against style transfer attacks

## Conclusion

This study demonstrates that while most traditional machine learning models are vulnerable to Neural Style Transfer manipulations in signature verification tasks, Support Vector Machines display remarkable robustness. This finding has significant implications for securing signature verification systems against potential style-based forgery techniques and provides a foundation for developing more robust signature verification systems.
