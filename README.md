
# Capstone Project

Welcome to the **Capstone Project** repository!  
This project demonstrates an end-to-end data pipeline including data exploration, cleaning, feature engineering, and interactive dashboards built with **Streamlit**.

---

## Project Structure

```
Capstone_Project/
│
├── Phase1/
│   ├── Week1/
│   ├── Week2/
│   └── ...
│
├── Phase2/
│   ├── Week3/
│   ├── Week4/
│   └── ...
│
├── Phase3/
│   ├── Week5/
│   ├── Week6/
│   │   ├── dashboard1.py
│   │   ├── dashboard2.py
│   │   ├── dashboard3.py
│   └── ...
│
├── requirements.txt
└── .gitignore
```

---

## Description

- **Phase 1:** Initial data collection, exploration, and cleaning.
- **Phase 2:** Feature engineering, deeper EDA, and preparation for final outputs.
- **Phase 3:** Interactive dashboards for insights and data presentation.

---

## Setup Instructions

Follow these steps to set up and run the project locally.

### Clone the Repository

git clone https://github.com/maste21/Capstone_Project.git
cd Capstone_Project


### Install Dependencies

pip install -r requirements.txt


---

## How to Run

### Notebooks (Phase 1 & Phase 2)

Open and run notebooks using Jupyter
the code is given in a way that running only the one script will trigger the rest internally and also the dashboards.

open the "Bengaluru Energy Dataset Cleaning and Merging.ipynb" in jupyter notebook and click run taks about 2 to 5 min

---

### Streamlit Dashboards (Phase 3)

Optionally Run dashboards with Streamlit using bash to excecute only dashboards:

streamlit run Phase3/Week6/dashboard1.py
streamlit run Phase3/Week6/dashboard2.py


---

## Requirements

The required Python libraries are listed in `requirements.txt`.  
Core packages include:
- pandas
- numpy
- matplotlib
- seaborn
- streamlit

---

## Notes

- Ensure your **data files** (CSV or other inputs) are in the correct paths as expected by the scripts/notebooks.
- Modify any hardcoded file paths if needed to match your local directory.

---

## Author

Arun
Chandu Srinivas
nalin
nits
akash
---
