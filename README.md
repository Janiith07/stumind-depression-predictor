# StuMind 🧠
A machine learning-powered web application for predicting depression risk in students and working professionals using academic, lifestyle, and psychological features. StuMind walks through a complete ML pipeline — from raw data ingestion to a deployable Flask web app with a trained KNN model.

---

## 📁 File Structure

```bash
STUMIND/
├── all_machine_learning_models/
│   ├── ANN.ipynb
│   ├── Decision_Tree.ipynb
│   ├── KNN.ipynb
│   ├── Logistic_Regression.ipynb
│   ├── Random_forest.ipynb
│   └── SVM.ipynb
│
├── data/
│   └── student_depression_dataset.csv
│
├── doc/
│   └── Final Comparison of Best models of the Algorithms.pdf
│
├── models/
│   ├── depression_model.pkl
│   ├── label_encoders.pkl
│   ├── pca_transformer.pkl
│   ├── scaler.pkl
│   └── selected_features.pkl
│
├── templates/
│   └── index.html
│
├── app.py
├── requirements.txt
└── README.md
```

## 📊 Dataset

**Source:** `student_depression_dataset.csv`  
**Target Variable:** `Depression` (1 = Depressed, 0 = Not Depressed)

| Category | Features |
|---|---|
| Demographics | Gender, Age, City, Profession, Degree |
| Academic | Academic Pressure, CGPA, Study Satisfaction |
| Work | Work Pressure, Job Satisfaction, Work/Study Hours |
| Lifestyle | Sleep Duration, Dietary Habits |
| Financial | Financial Stress |
| Psychological | Have you ever had suicidal thoughts?, Family History of Mental Illness |

---

## 🔬 ML Pipeline (`Depression_Prediction.ipynb`)

### 1. Data Loading & Exploration
- Loaded `student_depression_dataset.csv` using Pandas
- Dropped the `id` column (non-informative identifier)
- Inspected dataset shape, column data types, and value distributions
- Checked class balance of the `Depression` target variable

### 2. Data Cleaning
- **Duplicate Detection:** Scanned for duplicate rows — none found
- **Missing Value Handling:** Identified null and `?` entries across all columns
- **Financial Stress column** contained `?` values — replaced with `pd.NA`, converted to numeric, and imputed with the column **median**
- All other numeric nulls filled with median; categorical nulls filled with mode

### 3. Feature Engineering
Created 2 new derived features to capture compound risk signals:

| Feature | Formula / Logic |
|---|---|
| `TotalPressure` | `Academic Pressure + Work Pressure + Financial Stress` |
| `SatisfactionRatio` | `Study Satisfaction / (Job Satisfaction + 1e-6)` clipped to [0, 10] |

### 4. Outlier Handling
- Applied **IQR-based outlier detection** across all numeric columns: `Age`, `CGPA`, `Academic Pressure`, `Work Pressure`, `Work/Study Hours`, `Study Satisfaction`, `Job Satisfaction`, `Financial Stress`, `TotalPressure`, `SatisfactionRatio`
- Used **Winsorization (capping)** — values below `Q1 − 1.5×IQR` are set to the lower bound, values above `Q3 + 1.5×IQR` are set to the upper bound
- Printed a summary of outlier counts per column and the total percentage of data affected

### 5. Feature Scaling
- Applied **MinMaxScaler** to all numeric columns
- Scales all values to the [0, 1] range, preserving relative distributions

### 6. Encoding
- **Label Encoding** for binary/nominal columns:
  `Gender`, `Have you ever had suicidal thoughts?`, `Family History of Mental Illness`, `City`, `Profession`, `Degree`
- Cities with fewer than 10 records were removed to reduce noise before encoding
- **One-Hot Encoding (get_dummies)** for multi-category columns:
  `Sleep Duration`, `Dietary Habits`
- Boolean columns produced by `get_dummies` were cast to `int`

### 7. Feature Selection (RFE)
- Separated features `X` and target `y` from the encoded dataset
- Converted all columns to numeric; filled remaining nulls with column medians
- Applied a **StandardScaler** on features before RFE
- Used **Recursive Feature Elimination (RFE)** with a `LogisticRegression` estimator to rank and select the most predictive features
- Final selected features saved to `selected_features.pkl`

### 8. Feature Extraction (PCA)
- Applied **PCA** on the RFE-selected feature subset
- Fitted full PCA to examine explained variance per component
- Selected the number of components needed to capture **≥ 95% cumulative variance**
- Transformed the dataset using the optimal PCA model
- Saved PCA transformer to `pca_transformer.pkl`
- Final transformed dataset saved as `rfe_pca_df.csv`

### 9. Model Training & Evaluation

**Dataset used:** `rfe_pca_df` (RFE + PCA preprocessed)  
**Train/Test Split:** 80% training / 20% test (`random_state=42`)

#### K-Value Exploration
- Iterated through `k = 1 to 40` to identify the optimal number of neighbours
- Tracked both training and test accuracy for each `k` to detect overfitting
- Selected the `k` value with the highest test accuracy as the final model

#### Cross-Validation
- Performed **5-Fold Cross-Validation** on the best KNN model using the training set
- Reported mean CV accuracy and standard deviation

#### Final Model Performance

| Metric | Score |
|---|---|
| Model | KNeighborsClassifier |
| Best K | Determined by accuracy sweep |
| CV Mean Accuracy | Reported in notebook |
| Evaluation Metrics | Accuracy, Precision, Recall, F1-Score |

### 10. Model Export
All inference assets saved using **Joblib**:

| File | Contents |
|---|---|
| `depression_model.pkl` | Trained KNN classifier |
| `label_encoders.pkl` | LabelEncoder objects per categorical column |
| `scaler.pkl` | Fitted MinMaxScaler |
| `selected_features.pkl` | List of RFE-selected feature names |
| `pca_transformer.pkl` | Fitted PCA transformer |

---

## 🌐 Web Application

A Flask web app (`app.py`) serves predictions via a browser interface:

- **Frontend:** `templates/index.html` — multi-section form with sliders, toggles and dropdowns
- **Backend:** `app.py` loads all `.pkl` assets and runs the full inference pipeline on user input
- Users enter personal, academic, work, and lifestyle data and receive an instant depression risk prediction with probability scores

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3 |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML | Scikit-learn |
| Model Serialization | Joblib |
| Web Framework | Flask |
| Frontend | HTML, CSS, JavaScript (Jinja2 templates) |
| Development | Google Colab |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/StuMind.git
cd StuMind
```

### 2. Create & Activate Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install flask pandas numpy scikit-learn matplotlib seaborn joblib
```

### 4. Run the App
```bash
python app.py
```
Then open your browser and navigate to `http://127.0.0.1:5000`

---

## 📓 Reproducing the ML Pipeline

Open and run `Depression_Prediction.ipynb` in **Google Colab** or Jupyter. Make sure `student_depression_dataset.csv` is accessible (update the file path if running locally). Running all cells will reproduce the full pipeline and regenerate `rfe_pca_df.csv` and all `.pkl` model assets.

---

## 📌 Notes
- The `Financial Stress` column contained erroneous `?` entries that were cleaned during preprocessing
- Cities with fewer than 10 records were removed before encoding to prevent sparse label noise
- Both RFE (feature selection) and PCA (dimensionality reduction) are applied sequentially — the complete preprocessing chain is bundled in the `.pkl` files for consistent inference at runtime
- All scaling and encoding steps are fitted only on training data and applied to the test set to prevent data leakage
