# Breast-Cancer-Prediction

# 🩺 Breast Cancer Prediction Using Machine Learning

This project involves building a classification model to predict whether a breast tumor is malignant or benign based on diagnostic features extracted from digitized images of fine needle aspirate (FNA) of breast masses. The goal is to demonstrate how machine learning can be applied to real-world healthcare datasets for early cancer detection.

## 🔍 Objective
To predict the presence of **breast cancer** using diagnostic data. The target variable is:
- `0` → Malignant
- `1` → Benign

## 📂 Dataset
- **Name:** Breast Cancer Diagnostic Dataset  
- **Download Link:** (https://www.mediafire.com/file/a5gerqz7vd6pdtl/data.csv/file)

## 🛠️ Tools & Technologies
- Python  
- Jupyter Notebook  
- Libraries: `NumPy`, `Pandas`, `Matplotlib`, `Seaborn`, `Scikit-learn`

## ⚙️ Workflow
1. **Data Preprocessing**
   - Dropped irrelevant columns (`Unnamed: 32`)
   - Encoded the target variable (`diagnosis`)
   - Checked for missing values
   - Analyzed feature correlations

2. **Exploratory Data Analysis**
   - Visualized class distribution
   - Used heatmaps and pairplots to understand feature relationships

3. **Model Building & Evaluation**
   - Applied and evaluated multiple classification algorithms:
     - Logistic Regression
     - Support Vector Machine (SVM)
     - K-Nearest Neighbors (KNN)
     - Decision Tree Classifier

   - Evaluated each model using:
     - Accuracy
     - Confusion Matrix
     - Classification Report (Precision, Recall, F1-score)

4. **Sample Input Prediction**
   - Added support for **manual input** of feature values to test the model with sample patient data and observe real-time predictions

## 📊 Model Performance

| Model                     | Accuracy (Test Set) |
|--------------------------|---------------------|
| Logistic Regression      | 97.36%              |
| Support Vector Machine   | 97.36%              |
| K-Nearest Neighbors      | 96.49%              |
| Decision Tree Classifier | 91.22%              |

## 📌 Key Features
- Clean preprocessing pipeline
- Real-time prediction on custom input data
- Visual insights using Matplotlib and Seaborn
- Consistent train-test split using stratification

## ✅ Requirements
Install the required libraries using:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
