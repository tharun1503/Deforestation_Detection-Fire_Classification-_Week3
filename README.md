# 🔥 Deforestation Detection – Fire Classification Using MODIS Satellite Data

This project is developed as part of the **AICTE-Edunet-Shell Green Skills Internship (July 2025)**. It focuses on classifying fire types using MODIS satellite data from India (2021–2023) and deploying the model as a **Streamlit web app**.

---

## 📌 Project Highlights

- Built a **Random Forest Classification Model** to identify fire types:
  - Vegetation Fire
  - Volcano
  - Static Land Source
  - Offshore Fire

- Developed a **Streamlit Web Application** with user inputs and Folium map visualization.
- Data used: MODIS Fire Detection Data for India (2021, 2022, 2023).

---

## 🧠 How It Works

- Preprocessed fire data and extracted useful time features (month, year, day).
- Encoded categorical variables (`satellite`, `daynight`) using Label Encoding.
- Trained a RandomForestClassifier and evaluated it with ~80% accuracy.
- Built an interactive Streamlit UI for predictions.
- Displayed fire locations on a Folium map.

---

## 📂 Files in This Repository

| File Name | Description |
|-----------|-------------|
| `app.py` | Streamlit Web Application Code |
| `Deforestation_Detection(Fire_Classification).ipynb` | Full Jupyter Notebook |
| `modis_2021_India.csv` | MODIS Fire Data for 2021 |
| `modis_2022_India.csv` | MODIS Fire Data for 2022 |
| `modis_2023_India.csv` | MODIS Fire Data for 2023 |
| `Output_1.png`, `Output_2.png` | Screenshot Outputs |
| `Week_3_Deforestation_Detection(Fire_Classification).pptx` | Final Presentation Slides |
| `README.md` | Project Documentation |

---

## 🧠 Model File Download (Google Drive)

The trained model (`best_fire_detection_model.pkl`) exceeds GitHub's 100MB limit. You can download it here:

🔗 [Download Model from Google Drive](https://drive.google.com/file/d/1Jk1rUkq6VJHgn-Vvfbda9oZWWze4dYR_/view?usp=sharing)

---

## ▶️ Run Locally

Install the required packages:

```bash
pip install -r requirements.txt



Then run the app:

bash
Copy
Edit
streamlit run app.py
👨‍🎓 Author
Name: Badavath Tharun

Student ID: 23N31A3305

AICTE ID: STU682188f9f0bcb1747028217

Internship: Deforestation Detection – Fire Classification

Cohort: AICTE x Edunet x Shell – July 2025

🏆 Outcome
Successfully deployed a working AI model to classify fires.

Visualized incidents interactively on a map.

Learned practical skills in ML, data handling, and web deployment.

⭐ Thank you AICTE, Edunet Foundation & Shell for this wonderful opportunity!

