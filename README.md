# Zero-Shot Learning with Transductive Approach

## Overview
This repository contains the Python implementation of my thesis project on zero-shot learning, which addresses the challenge of classifying samples from unseen classes. Zero-shot learning is crucial for real-world applications where new, previously unseen classes are often encountered. Traditional models typically exhibit a bias towards seen classes, which can hinder their effectiveness on new classes.

To tackle this issue, my research involves a novel transductive learning technique that utilizes unlabeled data from unseen classes to enhance model performance. This approach uses ensemble learning combined with a discriminator mechanism to optimize training and improve the generalization of the model to new classes.

## Methodology
The methodology consists of several key steps, iteratively applied to refine the model's predictions:
1. **Base Model Training**: Train the base model (Model 1) on all available training data. Identify correctly predicted instances (`pos`) and incorrectly predicted instances (`neg`).
2. **Model 2 Training**: Retrain a copy of Model 1 on the `neg` instances to create Model 2.
3. **Transductive Inference**: Incorporate test samples as transductive data. Both models predict pseudo-labels, with the label having a higher confidence score considered more reliable. Focus is on `hard samples`, where the pseudo-labels from Model 1 and Model 2 differ.
4. **Discriminator Training**: Train a discriminator to determine which model's prediction to trust for each sample, using data classified distinctly by the two models.
5. **Instance Partitioning**: Use the discriminator's predictions to classify all instances (both train and test) into `pos` and `neg`.
6. **Refinement**: Retrain Model 1 on `pos` and Model 2 on `neg`.

## Files and Usage
- `ensemble.py`: Contains the full implementation of the zero-shot learning models and the transductive approach.
- To run the project, ensure that you have Python 3.x installed along with the necessary libraries listed in `requirements.txt`.

## Requirements
- numpy
- pandas
- sklearn
- imblearn
- tensorflow

Please install the required Python packages using:
```bash
pip install -r requirements.txt
