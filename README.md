# Signature Recognition Model Comparison Report

## Executive Summary

This report presents an evaluation of different machine learning models on the CEDAR signatures dataset, focusing on their robustness against Neural Style Transfer (NST) manipulations. Four models were tested both on standard validation data and on a specialized test set containing original signatures and their NST-transformed variants.

The Support Vector Classifier (SVC) demonstrated exceptional performance across both standard validation and NST-transformed data, maintaining perfect scores on the NST test set. In contrast, other models showed significant performance degradation when tested against style-transferred signatures, indicating potential vulnerability to this type of transformation.

## Project Overview

**Objective**: To evaluate the robustness of signature recognition models against Neural Style Transfer transformations.

**Dataset**:
- CEDAR signatures dataset from Kaggle (shreelakshmigp/cedardataset)
- ArtBench10 dataset from Kaggle (alexanderliao/artbench10) for style transfer source images

**Models Evaluated**:
- GaussianNB (Naive Bayes)
- LogisticRegression (max_iter=65536)
- SVC (Support Vector Classifier)
- KNeighborsClassifier

## Methodology

### Data Preprocessing

Signature images were processed using the following approach:

```python
def load_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    img = img.reshape(-1)
    return img
```

This preprocessing:
1. Loads each image via OpenCV
2. Resizes to a standardized 256×256 pixel dimension
3. Flattens the image into a one-dimensional vector of raw pixel values

### Neural Style Transfer Implementation

Style transfer was performed using Google's Arbitrary Image Stylization model:

```python
model = hub.load('https://kaggle.com/models/google/arbitrary-image-stylization-v1/frameworks/TensorFlow1/variations/256/versions/1')
```

The style transfer process applied artistic styles from the ArtBench10 dataset to the original signature images:

```python
for img in signatures:
    content_image = img
    style_image = load_img(os.path.join(image_dir, f"{index%60}.jpg"))
    styled_image = model(content_image, style_image)
    cv2.imwrite(os.path.join(output, f"{int(index%60)}.jpg"), 
                cv2.cvtColor(np.squeeze(styled_image)*255, cv2.COLOR_RGB2GRAY))
    index+=1
```

For style transfer, images were loaded using a TensorFlow-compatible approach:

```python
def load_img(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels = 3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img
```

### Model Evaluation

Models were evaluated using scikit-learn's performance metrics:

```python
def ModelEvaluation(model, x_test, y_test):
    predictions = model.predict(x_test)
    accuracy = accuracy_score(predictions, y_test)
    precision = precision_score(predictions, y_test)
    recall = recall_score(predictions, y_test)
    auc = roc_auc_score(y_test, predictions)
    print(classification_report(y_test, predictions))
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, predictions)
    # Visualization of ROC curve and confusion matrix
    # ...
```

Key metrics captured:
- Accuracy: Overall correctness of predictions
- Recall: Ability to identify all relevant instances (sensitivity)
- Precision: Accuracy of positive predictions
- ROC AUC Score: Model's ability to distinguish between classes
- Classification Report: Detailed metrics by class

## Results

### Standard Validation Performance

| Model | Accuracy | Recall | Precision | ROC AUC Score |
|-------|----------|--------|-----------|---------------|
| GaussianNB | 0.997 | 0.994 | 1.000 | 0.997 |
| LogisticRegression | 0.805 | 0.826 | 0.800 | 0.805 |
| SVC | 0.994 | 1.000 | 0.989 | 0.994 |
| KNeighborsClassifier | 0.900 | 0.841 | 1.000 | 0.894 |

#### Detailed Classification Reports (Standard Validation)

**GaussianNB**:
```
              precision    recall  f1-score   support

           0       1.00      0.99      1.00       310
           1       0.99      1.00      1.00       350

    accuracy                           1.00       660
   macro avg       1.00      1.00      1.00       660
weighted avg       1.00      1.00      1.00       660
```

**LogisticRegression**:
```
              precision    recall  f1-score   support

           0       0.78      0.81      0.80       310
           1       0.83      0.80      0.81       350

    accuracy                           0.80       660
   macro avg       0.80      0.80      0.80       660
weighted avg       0.81      0.80      0.80       660
```

**SVC**:
```
              precision    recall  f1-score   support

           0       0.99      1.00      0.99       310
           1       1.00      0.99      0.99       350

    accuracy                           0.99       660
   macro avg       0.99      0.99      0.99       660
weighted avg       0.99      0.99      0.99       660
```

**KNeighborsClassifier**:
```
              precision    recall  f1-score   support

           0       1.00      0.79      0.88       310
           1       0.84      1.00      0.91       350

    accuracy                           0.90       660
   macro avg       0.92      0.89      0.90       660
weighted avg       0.92      0.90      0.90       660
```

### Neural Style Transfer Test Performance

| Model | Accuracy | Recall | Precision | ROC AUC Score |
|-------|----------|--------|-----------|---------------|
| GaussianNB | 0.500 | 0.500 | 1.000 | 0.500 |
| LogisticRegression | 0.445 | 0.471 | 0.891 | 0.445 |
| SVC | 1.000 | 1.000 | 1.000 | 1.000 |
| KNeighborsClassifier | 0.500 | 0.500 | 1.000 | 0.500 |

#### Detailed Classification Reports (NST Testing)

**GaussianNB**:
```
              precision    recall  f1-score   support

           0       0.00      0.00      0.00        55
           1       0.50      1.00      0.67        55

    accuracy                           0.50       110
   macro avg       0.25      0.50      0.33       110
weighted avg       0.25      0.50      0.33       110
```
*Note: Warnings occurred indicating precision is ill-defined for class 0 due to no predicted samples.*

**LogisticRegression**:
```
              precision    recall  f1-score   support

           0       0.00      0.00      0.00        55
           1       0.47      0.89      0.62        55

    accuracy                           0.45       110
   macro avg       0.24      0.45      0.31       110
weighted avg       0.24      0.45      0.31       110
```

**SVC**:
```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        55
           1       1.00      1.00      1.00        55

    accuracy                           1.00       110
   macro avg       1.00      1.00      1.00       110
weighted avg       1.00      1.00      1.00       110
```

**KNeighborsClassifier**:
```
              precision    recall  f1-score   support

           0       0.00      0.00      0.00        55
           1       0.50      1.00      0.67        55

    accuracy                           0.50       110
   macro avg       0.25      0.50      0.33       110
weighted avg       0.25      0.50      0.33       110
```
*Note: Warnings occurred indicating precision is ill-defined for class 0 due to no predicted samples.*

## Analysis

### Model Performance on Standard Validation

- **GaussianNB** performed exceptionally well with near-perfect metrics (accuracy: 0.997, recall: 0.994, precision: 1.000), suggesting the class distributions in pixel space are well-separated for standard signatures.

- **LogisticRegression** showed the weakest performance among the models with an accuracy of 0.805, with relatively balanced precision and recall (0.78-0.83) across both classes.

- **SVC** demonstrated outstanding results with perfect recall (1.000) and high precision (0.989), indicating its effectiveness in handling high-dimensional data through its kernel transformation.

- **KNeighborsClassifier** achieved good results overall (accuracy: 0.90) but showed imbalanced performance between classes - perfect precision but lower recall (0.79) for class 0, versus perfect recall but lower precision (0.84) for class 1.

### Model Robustness Against NST

- **GaussianNB**: Performance dropped dramatically to 50% accuracy when tested against NST data. The classification report reveals the model classified all instances as class 1 (shown by 0.0 precision and recall for class 0), effectively rendering it no better than random guessing.

- **LogisticRegression**: Similarly struggled with NST data, showing the lowest accuracy (0.445) among all models. Like GaussianNB, it failed completely to identify class 0 instances, classifying most examples as class 1 (shown by 0.89 recall for class 1).

- **SVC**: Demonstrated remarkable robustness, maintaining perfect scores (1.000) across all metrics when tested against NST-transformed signatures. The classification report confirms balanced and perfect performance across both classes.

- **KNeighborsClassifier**: Performance dropped to 50% accuracy, with a pattern identical to GaussianNB - classifying all instances as class 1 and completely failing to identify class 0. This indicates that the local neighborhood structure in pixel space is significantly altered by style transfer.

### Class-Specific Impacts of NST

A clear pattern emerges from the classification reports on NST data:

1. For GaussianNB, LogisticRegression, and KNeighborsClassifier, style transfer caused a complete failure to identify class 0 (original signatures), with these models defaulting to predicting most or all samples as class 1.

2. Only SVC maintained the ability to correctly distinguish between both classes after style transfer, suggesting it captures invariant features that persist through artistic style changes.

3. The warnings about ill-defined precision for class 0 confirm that these models predicted no samples as class 0, completely losing their discriminative ability for this class.

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

1. **SVC demonstrates exceptional robustness**: Unlike other models, SVC maintained perfect performance when tested against NST-transformed signatures. This suggests it captures fundamental signature characteristics that remain invariant to style transformations, possibly due to its non-linear kernel mapping that focuses on structural rather than stylistic features.

2. **Class-specific vulnerabilities**: Most models (GaussianNB, LogisticRegression, KNeighborsClassifier) not only showed degraded performance but completely lost the ability to identify one class (class 0) after style transfer, defaulting to predicting everything as class 1.

3. **Raw pixel vulnerability**: Models operating directly on raw pixel values are highly sensitive to the visual changes introduced by NST, with the exception of SVC which likely benefits from its kernel trick to implicitly transform the feature space.

4. **Potential decision boundary collapse**: The classification reports suggest that style transfer caused a collapse of the decision boundaries for most models, pushing all samples to one side of the boundary.

## Recommendations

1. **Adopt SVC for signature verification systems**: Based on its exceptional robustness against style transfer attacks, SVC should be the preferred model for signature verification systems where security against manipulations is critical.

2. **Data augmentation with style transfers**: For improving other models, consider incorporating NST-transformed signatures into the training data to enhance robustness.

3. **Feature engineering**: Consider extracting structural features (like stroke patterns, contours, or topological features) that are more invariant to style changes rather than using raw pixels.

4. **Class balance investigation**: Investigate why most models defaulted to class 1 predictions after style transfer and develop techniques to maintain class balance robustness.

5. **Kernel analysis**: Analyze which specific kernel type and parameters in SVC provide the best robustness, potentially offering insights into what types of transformations preserve signature identity.

## Future Work

1. Test against other types of image manipulations beyond NST (e.g., rotation, scaling, distortion)

2. Evaluate deep learning models (CNNs) on the same dataset, particularly architectures designed for style-invariant feature extraction

3. Investigate feature importance to understand SVC's robustness by analyzing which components of the input space contribute most to its decision function

4. Explore different style transfer techniques and analyze which artistic styles cause the most confusion for various models

5. Develop specialized defensive techniques against style transfer attacks in signature verification systems

6. Conduct analysis on the class-specific impact of style transfer to understand why certain classes become undetectable

## Conclusion

This study demonstrates that while most traditional machine learning models are vulnerable to Neural Style Transfer manipulations in signature verification tasks, Support Vector Machines display remarkable robustness. The exceptional performance of SVC suggests it captures fundamental structural properties of signatures that persist despite artistic style changes.

The detailed classification reports reveal a concerning pattern where style transfer doesn't just reduce model accuracy but can completely eliminate a model's ability to detect one class of signatures, effectively collapsing the decision boundary. This highlights not just performance degradation but a fundamental shift in model behavior under style transformations.

These findings have significant implications for security-critical applications like signature verification systems, emphasizing the importance of thorough testing against potential adversarial manipulations like style transfer. The study provides a foundation for developing more robust signature verification systems capable of withstanding increasingly sophisticated image manipulation techniques.
