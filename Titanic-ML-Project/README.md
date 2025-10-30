# Titanic Survival Prediction (Machine Learning Project)

This project explores and models the Titanic dataset to predict passenger survival using
data analysis, feature engineering, and machine learning techniques.

---

## Overview

The goal is to predict whether a passenger survived the Titanic disaster based on features such as:
- Passenger class (Pclass)
- Gender (Sex)
- Age
- Family relations (SibSp, Parch)
- Fare
- Cabin and Embarked port

The notebook performs full end-to-end analysis:
Data understanding → Feature engineering → Model building → Evaluation → Submission file.

---

## Dataset Details

- Training data: train.csv
- Test data: test.csv
- Target variable: Survived (1 = Survived, 0 = Did not survive)
- Total records: 891
- Missing values:
  - Age: 177 missing
  - Cabin: 687 missing
  - Embarked: 2 missing

---

## Data Exploration Highlights

1. Gender and Survival  
   Females had a much higher survival rate than males.  
   Sex is a strong predictor of survival.

2. Passenger Class  
   First-class passengers had higher survival rates.  
   Third-class had the lowest.  
   Socio-economic status clearly affected survival chances.

3. Age and Family  
   Children had higher survival rates.  
   Very small and very large families had lower survival, while small families (2–4) fared best.

4. Cabin and Deck  
   Extracted Deck from the first letter of the Cabin column.  
   Passengers from Decks B, C, D, and E had higher survival.

5. Combined Features  
   Combined Deck + Pclass revealed strong patterns like B_1 (high survival) and U_3 (low survival).  
   Added AgeGroup (Child, Teen, YoungAdult, Adult, Senior) to capture age-based trends.

---

## Feature Engineering

Created new and useful features:

| Feature | Description |
|----------|--------------|
| FamilySize | SibSp + Parch + 1 |
| IsAlone | 1 if FamilySize = 1, else 0 |
| CabinDeck | Extracted first letter from Cabin |
| Deck_Class | Combination of CabinDeck and Pclass |
| Deck_Class_Age | Combination of CabinDeck, Pclass, and AgeGroup |
| AgeGroup | Binned Age into 5 groups (Child → Senior) |

All categorical columns were converted to numeric using label encoding and one-hot encoding.

---

## Data Cleaning Steps

1. Handled Missing Values  
   - Age: Filled using median grouped by Pclass and Sex.  
   - Embarked: Filled using mode.  
   - Cabin: Filled with "Unknown".

2. Encoded Categorical Variables  
   - Sex converted to numeric (male=0, female=1).  
   - Embarked and combined features were one-hot encoded.

3. Dropped Unnecessary Columns  
   Removed Name, Ticket, PassengerId, Cabin, and other redundant fields.

---

## Model Development

Model Used: Random Forest Classifier

Steps:
1. Split data into 80% training and 20% validation sets.
2. Trained the model using RandomForestClassifier (n_estimators=100).
3. Evaluated performance with accuracy score, confusion matrix, classification report, and 5-fold cross validation.

---

## Results

| Metric | Value |
|--------|--------|
| Validation Accuracy | 0.8045 (80.45%) |
| 5-Fold Cross Validation Accuracy | 0.817 (81.7%) |

Confusion Matrix:
[[90 15]  
 [20 54]]

Classification Report:
Precision, recall, and F1-score were around 0.80 overall, indicating a balanced model.

---

## Analysis

- The engineered features (Deck_Class_Age, FamilySize, IsAlone, AgeGroup) improved accuracy.
- The model performs slightly better at predicting non-survivors.
- This serves as a strong baseline for future tuning or model improvement.

---

## Requirements

pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter


---

## Folder Structure

Titanic-ML-Project/
├── data/
│ ├── train.csv
│ └── test.csv
├── notebooks/
│ └── Titanics.ipynb
├── results/
│ └── predictions.csv
├── requirements.txt
└── README.md



---

## How to Run

1. Clone this repository:

git clone https://github.com/AtifMazhar-01/Titanic-ML-Project.git
cd Titanic-ML-Project


2. Install required libraries:

pip install -r requirements.txt


3. Open Jupyter Notebook:

jupyter notebook


4. Run the notebook:

notebooks/Titanics.ipynb



---

## Future Improvements

- Try advanced models like XGBoost or LightGBM
- Perform hyperparameter tuning
- Add feature selection and importance visualization
- Build a simple web interface for predictions

---

## Author

Atif Mazhar  
Engineering Student | Aspiring Data Scientist  
Pune, India



