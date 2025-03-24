# Salary Prediction App 🎯💰  

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ml-app-salaryprediction-mk3685.streamlit.app/)

## 🚀 About the Project  
This is a **Machine Learning-powered Salary Prediction App** that estimates salaries based on user input. It is built using **Python, Streamlit, Scikit-learn, Pandas, and NumPy**.

🔗 **Live Demo:** [DevSalary Estimator](https://ml-app-salaryprediction-mk3685.streamlit.app/)

---

## 🛠️ Features  
✔️ User-friendly **web app interface**  
✔️ Predicts salary based on **experience, education, and job role**  
✔️ Uses **machine learning (Decision Tree Regression)** for predictions  
✔️ **Interactive visualization** with Seaborn & Matplotlib  

---

## 📊 Data Source  

The dataset used for this project comes from the **2020 Stack Overflow Developer Survey**.  
- 📂 **Dataset URL:** [Stack Overflow Survey 2020](https://survey.stackoverflow.co/datasets/stack-overflow-developer-survey-2020.zip)  
- 📑 **Preprocessing Steps:**  
  - Filtered for **full-time employed** professionals  
  - Removed **null values** and outliers  
  - Standardized **education levels** and **years of experience**  
  - Used **Label Encoding** for categorical data  

---

## 📌 How It Works  

1️⃣ Enter details like **Years of Experience, Education Level, Job Role**  
2️⃣ Click on **Predict Salary**  
3️⃣ The app runs a **pre-trained ML model** to estimate salary  
4️⃣ The predicted salary is displayed instantly  

---

## 🏗️ Tech Stack  
- **Frontend:** Streamlit  
- **Backend:** Python  
- **ML Model:** Scikit-learn (Decision Tree Regression)  
- **Data Processing:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  

---

## 🧠 Machine Learning Model  

### 🔹 Preprocessing  
- Encoded categorical variables (**Country, Education Level**) using **Label Encoding**  
- Removed outliers (salaries **below $10K or above $250K**)  

### 🔹 Model Training  
We tested multiple models:  
✅ **Linear Regression** – RMSE: High  
✅ **Decision Tree Regressor** – RMSE: **Best Model**  
✅ **Random Forest Regressor** – RMSE: Slightly better, but more complex  

📌 **Final Model Chosen:** `DecisionTreeRegressor` (Tuned using `GridSearchCV`)  

### 🔹 Model Evaluation  
- Used **Root Mean Squared Error (RMSE)** for performance measurement  
- **Final RMSE:** `"${:,.02f}".format(error)`  

📂 **Model Saved as:** `saved_steps.pkl` (Includes trained model & encoders)  

---

## 🔧 Installation & Running Locally  

### 1️⃣ Clone the Repository  
```sh
git clone https://github.com/MK3685/ml-app-salaryprediction.git
cd ml-app-salaryprediction
