# Ensemble Approach for Zero-Shot Learning

## Project Overview
This repository contains the implementation of my thesis project, which focuses on enhancing zero-shot learning models through an innovative ensemble approach. This methodology aims to improve model generalization to unseen classes, a common challenge in machine learning, particularly in real-world applications.

## Detailed Methodology

### 1. Base Model Training (Model A)
- **Objective**: Train Model A with all available training data.
- **Process**:
  - Identify instances predicted correctly (`pos`) and incorrectly (`neg`).

### 2. Weakness Focused Training (Model B)
- **Objective**: Train Model B specifically on the `neg` instances from Model A's predictions to address its weaknesses.
- **Process**:
  - Focus on instances where Model A's predictions were incorrect.

### 3. Transductive Inference
- **Objective**: Use both Model A and Model B to predict labels for all data, including unseen data.
- **Process**:
  - Evaluate pseudo-labels from both models, considering higher confidence scores as more reliable.

### 4. Discriminator Training
- **Objective**: Train a discriminator to choose between Model A and Model B based on reliability.
- **Process**:
  - Virtual labels are assigned based on which model predicts correctly:
    - Label A if only Model A is correct.
    - Label B if only Model B is correct.
  - Ignore instances where both models agree on incorrect predictions.

### 5. Instance Evaluation and Partitioning
- **Objective**: Use the discriminator's evaluations to partition data for further refinement.
- **Process**:
  - Assign instances to Model A or Model B based on the discriminator's decision.

### 6. Model Refinement
- **Objective**: Retrain Models A and B on newly partitioned data sets to enhance predictive accuracy.
- **Process**:
  - Continuously improve the models based on discriminator feedback.

### Transductive Discriminator
- **Objective**: Mitigate bias and domain shift problems, particularly with unseen data.
- **Process**:
  - Use reliability metrics for each pseudo-label to assess confidence.
  - Assign virtual labels based on which model's pseudo label is deemed more reliable.

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
