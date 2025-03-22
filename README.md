# Earthquake-Damage-Grade-Prediction

## 1. Introduction
Earthquakes cause severe damage to infrastructure, leading to loss of life and economic setbacks. Predicting the extent of damage can help authorities take proactive measures in disaster management and resource allocation. This project aims to classify buildings into different damage categories based on various structural and environmental features using machine learning models.

## 2. Objective
The goal of this project is to develop a machine learning model that can predict the level of damage to buildings caused by an earthquake. Various algorithms are evaluated to determine the most effective model.

## 3. Dataset Overview
The dataset contains multiple features related to building structure, geographic conditions, and construction material. The target variable categorizes buildings into three damage levels:
- **Damage Level 1:** Minimal damage
- **Damage Level 2:** Moderate damage
- **Damage Level 3:** Severe damage

## 4. Methodology
### 4.1 Data Preprocessing & Feature Engineering
- There were no missing values in the dataset.
- **12,319 duplicate rows** were identified and removed.
- **Outliers** were found in three columns: `age`, `area_percentage`, and `height_percentage`. These outliers were analyzed and retained as they represented real earthquake-related values rather than erroneous data.
- **Categorical variables** were encoded using One-Hot Encoding as they were nominal.
- The model training time was significantly high, so **feature selection** was performed. The top 20 most important features were selected, reducing computation time and improving model performance.

### 4.2 Model Selection and Training
The following machine learning and deep learning algorithms were implemented and evaluated:
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree**
- **Random Forest**
- **Gradient Boosting**
- **XGBoost**
- **Naive Bayes**
- **Artificial Neural Networks (ANN)**
- **Convolutional Neural Networks (CNN)**

Each model was trained, and hyperparameter tuning was performed where applicable to optimize performance.

## 5. Model Performance Evaluation
### Accuracy Comparison
| Model                    | Training Accuracy | Testing Accuracy |
|--------------------------|------------------|------------------|
| Logistic Regression      | 0.5712           | 0.5732           |
| K-Nearest Neighbors     | 0.8125           | 0.7164           |
| Decision Tree           | 0.8466           | 0.7660           |
| Random Forest           | 0.9803           | 0.8122           |
| Gradient Boosting       | 0.7793           | 0.7749           |
| XGBoost                | 0.7754           | 0.7733           |
| Naive Bayes            | 0.5321           | 0.5343           |
| Artificial Neural Network (ANN) | 0.7321  | 0.7248           |
| Convolutional Neural Network (CNN) | 0.7441 | 0.7312           |

## 6. Best Performing Model: **Random Forest**
### Best Hyperparameters:
- `n_estimators = 250`
- `max_depth = 30`
- `min_samples_split = 2`
- `min_samples_leaf = 1`
- `max_features = 'sqrt'`
- **Testing Accuracy:** 0.8122
- **Training Accuracy:** 0.9803

**Random Forest** outperformed other models in terms of training accuracy, testing accuracy, generalization, and robustness.

## 7. Hyperparameter Tuning & Optimization
- **RandomizedSearchCV** was used to tune key hyperparameters for models like **Random Forest, XGBoost, ANN, and Gradient Boosting**.
- This process improved accuracy and helped in selecting the best model configurations.
- **Cross-validation** ensured robustness in performance evaluation.

## 8. Model Evaluation
- Performance was assessed using **accuracy, precision, recall, F1-score, and classification reports**.
- **Overfitting** was controlled by analyzing model performance on both training and test datasets.
- **Confusion matrices** provided insights into misclassifications and possible improvements.

## 9. Conclusion
- **Machine learning and deep learning models can effectively predict earthquake damage levels.**
- **Random Forest** achieved the highest accuracy and proved to be the most reliable model.
- **Decision Tree, Gradient Boosting, and CNN** also performed well, but with slightly lower accuracy.
- **Deep learning models (ANN & CNN) showed promising results**, but require further optimization.
