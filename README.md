# 🏠 NYC Airbnb Data Analysis & Price Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**A comprehensive data science project analyzing 48,000+ NYC Airbnb listings with machine learning price prediction and interactive visualizations.**

[Live Demo](#-live-demo) • [Features](#-features) • [Installation](#-installation) • [Documentation](#-documentation)

</div>

---

## 📊 Project Overview

This project performs end-to-end data analysis on NYC Airbnb listings, including:
- 🧹 **Data Cleaning & Preprocessing** - Handle missing values, outliers, and feature engineering
- 📈 **Exploratory Data Analysis** - Comprehensive statistical analysis and visualizations
- 🤖 **Machine Learning Models** - Price prediction using Linear Regression and Random Forest
- 🗺️ **Interactive Dashboard** - Streamlit web app with maps and real-time predictions
- 📊 **Clustering Analysis** - Market segmentation using K-means clustering

### Key Results
- ✅ **48,884 listings** analyzed across 5 NYC boroughs
- ✅ **40.1% R² score** with Random Forest model
- ✅ **$49 average prediction error** (MAE)
- ✅ **5 market segments** identified through clustering

---

## 🎯 Features

### 1. Data Analysis Pipeline
- Automated data cleaning and validation
- Missing value imputation strategies
- Outlier detection and handling
- Feature engineering (20+ features created)

### 2. Machine Learning Models
- **Linear Regression** - Baseline model (R² = 0.126)
- **Random Forest** - Best performer (R² = 0.401)
- **K-means Clustering** - Market segmentation (5 clusters)
- Cross-validation and performance metrics

### 3. Interactive Dashboard
- 🗺️ Interactive map with 10,000+ listings
- 📊 Real-time price predictions
- 🎨 Dynamic filtering by borough, room type, and price
- 📈 Statistical visualizations and insights

### 4. Comprehensive Visualizations
- Geographic distribution maps
- Price distribution analysis
- Correlation heatmaps
- Feature importance charts
- Cluster analysis plots

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip or conda
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Data_Analysis_AirBnB.git
cd Data_Analysis_AirBnB
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the complete pipeline**
```bash
python run_complete_pipeline.py
```

4. **Launch the dashboard**
```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

---

## 📁 Project Structure

```
Data_Analysis_AirBnB/
│
├── 📓 notebooks/
│   ├── airbnb_data_analysis.ipynb    # Main analysis notebook
│   └── AB_NYC_2019.csv                # Dataset (48,895 listings)
│
├── 🐍 src/
│   ├── data_prep.py                   # Data cleaning & preprocessing
│   ├── eda.py                         # Exploratory data analysis
│   └── model.py                       # ML models & training
│
├── 📊 outputs/
│   ├── cleaned_airbnb.csv             # Cleaned dataset
│   ├── model_price_rf.joblib          # Trained Random Forest model
│   ├── model_price_lr.joblib          # Trained Linear Regression model
│   └── figures/                       # Generated visualizations
│
├── 🌐 app.py                          # Streamlit dashboard
├── 🔧 run_complete_pipeline.py        # Pipeline orchestrator
├── 📋 requirements.txt                # Python dependencies
├── 📖 MODEL_BUILDING_REPORT.md        # Detailed ML explanation
├── 🏗️ TECHNICAL_ARCHITECTURE.md       # Technical documentation
└── 📝 README.md                       # This file
```

---

## 💻 Usage

### Option 1: Interactive Dashboard (Recommended)
```bash
streamlit run app.py
```
- Explore data with interactive filters
- View geographic distribution on maps
- Get real-time price predictions
- Analyze market segments

### Option 2: Jupyter Notebook
```bash
jupyter notebook notebooks/airbnb_data_analysis.ipynb
```
- Step-by-step analysis
- Detailed explanations
- Reproducible results

### Option 3: Python Scripts
```bash
# Run complete pipeline
python run_complete_pipeline.py

# Or run individual modules
python -c "from src.data_prep import run_data_cleaning_pipeline; run_data_cleaning_pipeline('notebooks/AB_NYC_2019.csv', 'outputs/cleaned_airbnb.csv')"
```

---

## 📈 Model Performance

| Model | MAE | RMSE | R² Score | Training Time |
|-------|-----|------|----------|---------------|
| **Random Forest** | **$48.82** | **$67.23** | **0.401** | ~30s |
| Linear Regression | $73.01 | $187.08 | 0.126 | ~1s |

### Feature Importance (Top 5)
1. 📍 **Location** (Latitude & Longitude) - 45%
2. 🏠 **Room Type** - 20%
3. ⭐ **Number of Reviews** - 15%
4. 📅 **Availability** - 12%
5. 🌙 **Minimum Nights** - 8%

---

## 🗺️ Market Segments (Clustering)

| Cluster | Avg Price | Characteristics | Target Audience |
|---------|-----------|-----------------|-----------------|
| **Budget** | $80 | High availability, outer boroughs | Students, backpackers |
| **Standard** | $120 | Medium price, mixed locations | Average tourists |
| **Premium** | $250 | Manhattan, entire homes | Families, luxury travelers |
| **Long-Stay** | $150 | High min nights, entire homes | Business, relocations |
| **Super Premium** | $400+ | Prime locations, always booked | Wealthy travelers |

---

## 📚 Documentation

### For Beginners
- 📖 **[MODEL_BUILDING_REPORT.md](MODEL_BUILDING_REPORT.md)** - Complete guide to machine learning (no prior knowledge needed)
  - What is machine learning?
  - How models work
  - Understanding metrics (MAE, RMSE, R²)
  - Practical applications

### For Developers
- 🏗️ **[TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md)** - Technical deep dive
  - How .joblib files are created
  - Model integration in Streamlit
  - Map generation process
  - Complete data flow

---

## 🛠️ Technologies Used

### Data Science Stack
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning models

### Visualization
- **matplotlib** - Static plots
- **seaborn** - Statistical visualizations
- **plotly** - Interactive charts and maps

### Web Framework
- **streamlit** - Interactive dashboard
- **joblib** - Model serialization

---

## 📊 Key Insights

### Geographic Patterns
- 🏙️ Manhattan listings are **2-3x more expensive** than outer boroughs
- 📍 Location is the **#1 price predictor** (45% importance)
- 🗺️ Brooklyn has the **most listings** (20,104)

### Pricing Dynamics
- 🏠 Entire homes cost **50% more** than private rooms
- ⭐ Listings with 100+ reviews charge **15% premium**
- 📅 Low availability correlates with **higher prices** (demand indicator)

### Market Composition
- 52% Entire home/apt
- 45% Private room
- 3% Shared room

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Krishna Chaitanya**
- GitHub: [@letsdoit-sricharan](https://github.com/letsdoit-sricharan)
- Email: krish23306@gmail.com

---

## 🙏 Acknowledgments

- [NYC Open Data](https://opendata.cityofnewyork.us/) for providing the Airbnb dataset
- [Scikit-learn](https://scikit-learn.org/) for machine learning tools
- [Streamlit](https://streamlit.io/) for the amazing dashboard framework
- [Plotly](https://plotly.com/) for interactive visualizations

---



</div>

