# 🩺 Lung Cancer Risk Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-orange?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-yellow?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Made with love](https://img.shields.io/badge/Made%20with-%E2%9D%A4-red)
![Streamlit](https://img.shields.io/badge/Streamlit-App-orange?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Model-yellow?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-darkblue?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Analysis-blue?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-informational?logo=matplotlib)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-blueviolet?logo=seaborn)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-red?logo=xgboost)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-lightgreen?logo=lightgbm)
![Joblib](https://img.shields.io/badge/Joblib-Model%20Saving%20&%20Loading-brightgreen?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Made with love](https://img.shields.io/badge/Made%20with-%E2%9D%A4-red)

![image](https://github.com/user-attachments/assets/b5ed93ab-aa86-4079-85a1-d50df7175e46)

## 📌 Project Overview

This project focuses on **predicting lung cancer risk levels** using a combination of **machine learning models**, **EDA (Exploratory Data Analysis)** and a **Streamlit web app** for deployment.We used patient data containing **clinical symptoms**, **lifestyle factors** (such as smoking and air pollution exposure), and **medical history**.

## 📊 Dataset Description

The dataset contains variables such as:

* Age
* Gender
* Air Pollution Level
* Smoking Level
* Passive Smoker
* Alcohol Use
* Genetic Risk
* Symptoms: Chest Pain, Fatigue, Shortness of Breath, Weight Loss, etc.

**Target variable**: Lung cancer risk level (Low / Medium / High)

## 🧪 Exploratory Data Analysis (EDA)

### 📈 Smoking Level Distribution

![image](https://github.com/user-attachments/assets/c77690d9-ef2b-4098-a60c-c21c075c95f6)


* Majority of patients are **Medium**, **Non-Smoker**, and **Low** levels.
* High levels of smoking are less frequent but highly associated with cancer.

### 🚬 Smoking Level vs Cancer Diagnosis

![image](https://github.com/user-attachments/assets/2437da65-5e3c-41ea-b641-10e335afdddf)

* **Extreme smokers** have the **highest cancer diagnosis rates (100% High Risk)**.
* Non-smokers and Low levels are mostly in the **Low Risk** category.

### 🚬 Passive Smoker vs Cancer

![image](https://github.com/user-attachments/assets/2fcc5f8c-7db6-4818-958d-9bd75d6fb951)


* Passive smokers from level 7 and 8 showed **100% high cancer risk**.
* Indicates that passive smoking exposure is a significant factor.

### 🫁 Smoking vs Shortness of Breath

![image](https://github.com/user-attachments/assets/f0946ee5-8eb0-4c77-a3a9-db1c8a360c20)


* **Shortness of Breath severity increases** with higher smoking levels.
* Patients with smoking level 8 have higher frequency of severe shortness.

### 🌫️ Air Pollution Level Distribution

![image](https://github.com/user-attachments/assets/34c96a0a-21d3-4206-bef2-be133753581d)

* Majority of patients are exposed to **Very High**, **Medium Low**, and **Medium** pollution levels.

### 🫁 Air Pollution vs Cancer Diagnosis

![image](https://github.com/user-attachments/assets/b41d80a0-d89b-4611-abb1-8d86db52dfa5)

* Patients from **Severe** and **Very High** pollution zones show **more High Risk cancer diagnoses**.
* Low pollution areas are associated with **Low/Medium risk**.

### 🌫️ Pollution Level vs Coughing of Blood Severity

![image](https://github.com/user-attachments/assets/f12666d1-24e6-40ea-821f-7e50185fff30)


* Higher air pollution levels result in increased **life-threatening coughing blood severity**.
* Especially in **Very High** and **Severe** pollution zones.

### 📊 Cross-tabulation: Smoking x Air Pollution

![image](https://github.com/user-attachments/assets/6e5dbf82-9a96-43d4-bd56-af3c8708a822)

* Higher **combination of air pollution and smoking** appears in **Very High** pollution zones and **Passive/Non-smoker** categories.

### 🫁 Combined Pollution + Smoking Impact on Cancer Risk

![image](https://github.com/user-attachments/assets/292b8be8-413c-42bd-99c0-9ce60c2087c8)

* The combination **(Very High Pollution + Passive Smoker)** and **(Severe Pollution + Non-smoker)** shows **100% high cancer risk**.

### 📈 Distribution of Symptoms (High Pollution Levels)

![image](https://github.com/user-attachments/assets/32e0f87b-c9fc-4e2f-84c3-f548db0efb23)

* **Shortness of Breath**, **Coughing of Blood**, **Chest Pain**, **Fatigue**, and **Weight Loss** are all elevated in **high pollution groups**.

### 🔍 Feature Importance - XGBoost & LightGBM

**XGBoost:**

![image](https://github.com/user-attachments/assets/e583d21c-8400-473b-8a40-ccba951ea47c)

Top features:

* Coughing of blood
* Alcohol use
* Occupational hazards
* Shortness of breath

**LightGBM:**

![image](https://github.com/user-attachments/assets/382b34b8-60de-44e1-97be-738cee921cbf)

Top features:

* Wheezing
* Passive smoker
* Fatigue
* Obesity

## 🤖 Model Building & Performance

We tested **8 supervised models**:

| Model               | Accuracy | F1-Score | Recall | Precision |
| ------------------- | -------- | -------- | ------ | --------- |
| Logistic Regression | 1.00     | 1.00     | 1.00   | 1.00      |
| Random Forest       | 1.00     | 1.00     | 1.00   | 1.00      |
| Decision Tree       | 1.00     | 1.00     | 1.00   | 1.00      |
| Gradient Boosting   | 1.00     | 1.00     | 1.00   | 1.00      |
| LightGBM            | 1.00     | 1.00     | 1.00   | 1.00      |
| XGBoost             | 1.00     | 1.00     | 1.00   | 1.00      |
| Naive Bayes         | 0.945    | 0.942    | 0.938  | 0.947     |
| KNN                 | 0.825    | 0.813    | 0.816  | 0.818     |
| SVM                 | 0.435    | 0.415    | 0.415  | 0.416     |

### ✅ Confusion Matrices

**Example - Logistic Regression:**

![image](https://github.com/user-attachments/assets/e4de9153-2479-4817-bede-d4b0bf54b833)

**Example - XGBoost:**

![image](https://github.com/user-attachments/assets/efa975fa-4b7c-460b-9e16-b54325d755f4)


### 📈 ROC Curve Comparison

![image](https://github.com/user-attachments/assets/592b3395-d4f2-49b5-b4aa-5b066bca99f0)

Models like **XGBoost**, **Random Forest**, **Gradient Boosting**, **Logistic Regression** and **LightGBM** achieved near **perfect AUC-ROC (1.0)**.

## 🌐 Streamlit Web App (Deployment)

We created a **Streamlit application** to allow users to predict lung cancer risk based on input features.

### ✅ Example Output:

![image](https://github.com/user-attachments/assets/024660cd-59b1-487f-b703-4ab298dacd3f)


### 🧑‍💻 How to Run Locally:

```bash
streamlit run src/app.py
```

## 📌 Project Structure

```
├── src/
│   ├── app.py
├── notebook
│   ├── model_ml.ipynb
├── models/
│   ├── Logistic_Regression.pkl
├── images/
│   ├── 1.png ... 20.png
├── user_predictions_log.csv
├── requirements.txt
├── README.md
```

## ✅ Business Insights & Conclusion

* **Air Pollution and Smoking are the most critical factors** in lung cancer prediction.
* Patients in **high pollution regions** and **extreme smokers or passive smokers** have **almost guaranteed high risk**.
* **Symptoms like coughing of blood, shortness of breath, and chest pain** serve as strong predictive features.
* **All tree-based models and Logistic Regression** performed with near-perfect accuracy.
* This pipeline can support **medical screening tools** for **early lung cancer detection**.

## ✅ Technologies Used

* Python
* Pandas / Scikit-learn / XGBoost / LightGBM
* Streamlit
* Matplotlib / Seaborn
* Jupyter Notebooks

## ✅ Future Improvements

* Test with larger real-world datasets.
* Deploy on cloud (Streamlit Share / Hugging Face Spaces).
* Add Explainable AI (SHAP) to explain individual predictions.

## ✅ Author

Rafael Gallo
