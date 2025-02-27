# 🦠 COVID-19 Prediction App  

Welcome to the **COVID-19 Prediction App**, a Flask-based web application that predicts the likelihood of COVID-19 infection based on user symptoms. The app utilizes a **K-Nearest Neighbors (KNN)** model trained on relevant medical features to assess COVID-19 risk.  

## 🚀 Features  

- **User-Friendly Web Interface** – Enter symptoms easily through a form.  
- **Machine Learning Model** – Uses a **KNN classifier** to predict COVID-19 risk.  
- **Instant Predictions** – Receive real-time results upon submission.  
- **Flask Backend** – Lightweight and efficient backend for processing user input.  

## 📊 How It Works  

1. Select **symptoms** from the interactive form (e.g., Fever, Cough, Breathing Problems).  
2. Click the **"Predict"** button to check the likelihood of COVID-19 infection.  
3. View results:  
   - 🟢 **Low Risk** (No COVID-19)  
   - 🔴 **High Risk** (Possible COVID-19)  

## 🛠 Installation & Usage  

To run this Flask app locally, follow these steps:  

### 1️⃣ Clone the Repository  

```bash
git clone https://github.com/machinelearningprodigy/covid-19-detection-system.git
cd covid-19-detection-system
```

### 2️⃣ Install Dependencies  

Ensure you have Python installed, then run:  

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Flask App  

```bash
python app.py
```

The app will be available at `http://127.0.0.1:5000/`.  

## 🏗 Technologies Used  

- **Python 3.x**  
- **Flask** (for the web framework)  
- **Scikit-learn** (for KNN model)  
- **NumPy & Pandas** (for data processing)  
- **HTML/CSS** (for frontend templates)  

## 📦 Requirements  

Ensure you have the following dependencies installed:  

```txt
Flask==2.2.2
numpy==1.23.3
pandas==1.5.0
scikit-learn==1.1.2
```

## 🎯 Future Enhancements  

- ✅ Improve model accuracy with additional data.  
- ✅ Enhance UI with **Bootstrap/TailwindCSS**.  
- ✅ Deploy the app on **Heroku/Render** for public access.  

## 🤝 Contributing  

Want to improve this project? Fork the repo, make changes, and submit a pull request!  

## 📜 License  

This project is open-source and available under the **MIT License**.  

---

📧 **Need help?** Feel free to reach out or raise an issue in the repository! 🚀
