# 🌍 NeuroAir: Live AI-Driven Air Quality Forecasting

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web_Framework-black?logo=flask&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep_Learning-orange?logo=tensorflow&logoColor=white)
![Open-Meteo](https://img.shields.io/badge/Open--Meteo-Live_API-green)

**NeuroAir** is a multi-layered, hybrid intelligent web application designed to forecast the Air Quality Index (AQI) in major Indian metropolitan cities (Delhi, Mumbai, Chennai, Kolkata, Bangalore). 

Unlike traditional static machine learning models that suffer from recursive feedback loops when processing chaotic environmental data, NeuroAir combines a **1D-CNN + Bi-LSTM** deep learning engine with a deterministic **"Safety Mode" Arbitration Layer** to deliver highly stable, physically realistic, and actionable short-term public health warnings.

---

## ✨ Key Features

* **Live Data Acquisition:** Dynamically fetches real-time satellite data for 5 essential pollutants (PM2.5, PM10, NO2, SO2, CO) via the Open-Meteo API.
* **Hybrid Deep Learning Engine:** * *1D-CNN* for spatial feature extraction and localized chemical relationship mapping.
    * *Bi-LSTM* for bidirectional temporal sequence learning to understand pollution wave momentum.
* **Deterministic Arbitration Layer:** A custom algorithmic "Safety Clamp" that intercepts raw neural network outputs and restricts day-to-day prediction variance, preventing the runaway predictions common in standalone time-series models.
* **Responsive Web Interface:** A modern, glass-morphism UI built with Tailwind CSS that provides users with immediate, color-coded health warnings based on current and future AQI.

---

## 🏗️ System Architecture

1.  **Frontend (UI Layer):** Captures user input (target city) and dynamically updates DOM elements with the latest forecasting data.
2.  **Backend (Flask API):** Acts as the central bridge. Handles geocoding, fetches the last 30 days of hourly weather data, resamples it, and applies MinMax scaling.
3.  **AI Engine:** Processes the 3D data array (Samples, TimeSteps, Features) through the Keras `.keras` model and applies the variance clamp before returning the final JSON response.


---

## 💻 Tech Stack

* **Backend:** Python, Flask, Pandas, NumPy, Scikit-Learn
* **AI/Deep Learning:** TensorFlow, Keras
* **Frontend:** HTML5, CSS3, JavaScript (Vanilla), Tailwind CSS
* **APIs:** Open-Meteo Air Quality API, Open-Meteo Geocoding API

---
