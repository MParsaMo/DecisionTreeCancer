# 🎗️ Breast Cancer Classification with Decision Tree and GridSearchCV

This project uses the **Breast Cancer Wisconsin (Diagnostic)** dataset from scikit-learn to train a **Decision Tree Classifier**. The model is tuned with **GridSearchCV** and evaluated using both cross-validation and a held-out test set. The pipeline includes data loading, preprocessing, hyperparameter tuning, and performance evaluation — making it a great educational and practical example of classical supervised learning.

---

## 📚 Dataset Description

The Breast Cancer Wisconsin dataset is a **binary classification** dataset that includes:

- **569 samples** of breast tumor data
- **30 real-valued input features** computed from digitized images of fine needle aspirates (FNAs)
- **2 target classes**:
  - `malignant` (label `0`)
  - `benign` (label `1`)

This dataset is available directly through `sklearn.datasets.load_breast_cancer()`.

---

## 🧰 Technologies Used

- Python 3.x
- NumPy
- Pandas
- scikit-learn

---

## 🛠️ Features

✅ Load and explore the Breast Cancer dataset  
✅ Split data into training and testing sets  
✅ Use **GridSearchCV** to tune hyperparameters:  
  - `max_depth` of the tree  
  - `criterion`: Gini impurity or entropy  
✅ Evaluate model performance using:
- k-fold cross-validation (default: 10-fold)
- held-out test set (30%)
✅ Report:
- Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)

---

## 📂 Project Structure

├── DecisionTreeCancer.py # Main executable script
├── README.md # This documentation
└── requirements.txt # Optional: list of dependencies


---

## 🧪 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/breast-cancer-dt.git
cd breast-cancer-dt

2. Install Dependencies
pip install -r requirements.txt

3. Run the Script
python DecisionTreeCancer.py

🔍 Grid Search Hyperparameter Tuning
Parameter	Description	Values Tested
max_depth	Maximum tree depth	1 to 10
criterion	Split quality metric	'gini', 'entropy'

Cross-validation (cv=10) is used internally to find the best combination.

📈 Sample Output
Loading Breast Cancer Wisconsin (Diagnostic) dataset...

--- Breast Cancer Data Overview ---
Number of samples: 569
Number of features: 30
Feature names (first 5): ['mean radius' 'mean texture' 'mean perimeter' 'mean area' 'mean smoothness']
Target names (classes): ['malignant' 'benign']
Class distribution: [212 357]

Splitting data into training (70%) and testing (30%) sets...
Training set size: 398 samples
Testing set size: 171 samples

--- Tuning Decision Tree Hyperparameters using GridSearchCV ---
Fitting 10 folds for each of 20 candidates, totalling 200 fits
...
Best parameters found: {'criterion': 'gini', 'max_depth': 4}
Best cross-validation score (accuracy): 0.9397

--- Evaluating Model Performance using 10-Fold Cross-Validation ---
Cross-validation test scores for each fold: [0.95 0.91 0.93 ...]
Mean Cross-validation Accuracy: 0.9387 (+/- 0.0412)

--- Model Performance on Held-Out Test Set ---
Accuracy Score: 0.9474

Confusion Matrix:
            malignant  benign
malignant          59       4
benign              5     103

Classification Report:
              precision    recall  f1-score   support

   malignant       0.92      0.94      0.93        63
      benign       0.96      0.95      0.96       108

    accuracy                           0.95       171
   macro avg       0.94      0.94      0.94       171
weighted avg       0.95      0.95      0.95       171

🧠 Evaluation Strategy
Step	Purpose
GridSearchCV	Selects best hyperparameters via internal cross-validation
evaluate_model_with_cross_validation()	Tests generalization using entire dataset with k-fold CV
make_predictions_on_test_set()	Final validation on completely unseen test set