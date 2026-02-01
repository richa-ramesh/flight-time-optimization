# Flight Delay Prediction & Flight Time Optimization

## Overview
Flight delays cause major operational disruptions for airlines and inconvenience for passengers.  
This project applies **machine learning and data analysis** to predict flight delays for flights departing from **Chicago O’Hare International Airport (ORD)** and to identify key factors contributing to delays.

The goal is to provide **data-driven insights** that airlines and airport authorities can use to optimize scheduling, resource allocation, and overall operational efficiency.

---

## Problem Statement
How can historical flight and delay data be leveraged to **accurately predict flight delays** and **reduce inefficiencies in airline operations**?

Flight delays cost the aviation industry billions of dollars annually. A reliable predictive system enables proactive decision-making to minimize delays and improve passenger experience.

---

## Dataset
- **Source:** U.S. Department of Transportation – Bureau of Transportation Statistics (BTS)
- **Scope:** Flights originating from ORD (Chicago O’Hare)
- **Size:** 18,901 records
- **Features Used:**
  - ORIGIN, DEST
  - CRS_DEP_TIME
  - DISTANCE
  - CARRIER_DELAY
  - WEATHER_DELAY
  - NAS_DELAY
  - SECURITY_DELAY
  - LATE_AIRCRAFT_DELAY

### Data Preprocessing
- Removed canceled flights and duplicates
- Handled missing values via imputation
- Encoded categorical variables (ORIGIN, DEST)
- Normalized numerical features using `StandardScaler`
- Created a binary target variable:
  - `Delay = 1` → Delayed  
  - `Delay = 0` → On-time  

---

## Methodology
The dataset was split into **80% training** and **20% testing**. Multiple supervised learning models were trained and evaluated:

- **Logistic Regression**
- **Random Forest Classifier**
- **Gradient Boosting Classifier**
- **K-Nearest Neighbors (KNN)** with hyperparameter tuning

Evaluation metrics included:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## Results
| Model | Accuracy |
|------|----------|
| Gradient Boosting | **81%** |
| Logistic Regression | 80% |
| Random Forest | 79% |
| KNN (Optimized) | 71% |

### Key Findings
- **Carrier delays, scheduled departure time, and late aircraft arrivals** were the strongest predictors of delays.
- Weather and air traffic control delays had moderate impact.
- Flight distance and destination airport had minimal influence.

Gradient Boosting performed best, demonstrating strong capability in capturing complex feature relationships.

---

## Visualizations
- Delay distribution analysis
- Feature importance analysis (Random Forest)
- Correlation heatmaps
- Confusion matrices for all models

These visualizations helped interpret model behavior and identify operational bottlenecks.

---

## Practical Implications
**For Airlines:**
- Optimize aircraft turnaround times
- Add buffer time for delay-prone routes
- Use predictive insights for proactive scheduling

**For Airport Authorities:**
- Improve resource allocation during peak congestion hours
- Plan contingencies for weather and traffic-related delays

---

## Future Work
- Incorporate real-time weather and air traffic data
- Address class imbalance using advanced techniques
- Experiment with deep learning models
- Build a dashboard for real-time delay prediction

---

## Tech Stack
- **Languages:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Models:** Logistic Regression, Random Forest, Gradient Boosting, KNN

---

## Contributors
- Richa Rameshkrishna  
- Sohum Bhole  
- Mohammad Nusairat  
- Nahom Yohanes  
- Vageesh Indukuri  
