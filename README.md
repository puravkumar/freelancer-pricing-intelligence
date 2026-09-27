# Freelancer Pricing Intelligence System

An end-to-end machine learning system that predicts reasonable Fiverr package prices for Basic, Standard, and Premium services based on gig and seller characteristics.

## Project Overview

This project uses 8K+ Fiverr gig records to build separate regression models for predicting package-level prices.

The system processes text, categorical, and numerical features such as gig titles, package features, seller level, ratings, delivery time, revisions, and category. The trained models are deployed through a Streamlit web application for real-time price prediction.

## Features

- Predicts Basic, Standard, and Premium package prices
- TF-IDF based text feature extraction
- One-Hot Encoding for categorical features
- Median imputation for missing numerical values
- Feature scaling for numerical features
- Comparison of multiple regression algorithms
- Real-time predictions through Streamlit

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Streamlit
- MySQL
- Git & GitHub

## Machine Learning Approach

### Input Features

- Gig title
- Seller level
- Rating score
- Rating count
- Category
- Package delivery time
- Package revisions
- Unlimited revision indicator
- Package features

### Models Evaluated

- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net

Lasso Regression produced the lowest test MAE among the evaluated models for all three pricing tiers.

## Model Results

| Package | Lasso Test MAE |
|---------|----------------:|
| Basic | 212.86 |
| Standard | 514.47 |
| Premium | 1,045.72 |

MAE (Mean Absolute Error) represents the average absolute difference between the actual and predicted package price.

## Project Structure

```text
freelancer-pricing-intelligence/
│
├── app/
│   └── app.py
│
├── data/
│   ├── gigs_data.csv
│   └── cleaned_gigs.csv
│
├── models/
│   ├── lasso_basic.pkl
│   ├── lasso_standard.pkl
│   ├── lasso_premium.pkl
│   ├── preprocessor_basic.pkl
│   ├── preprocessor_standard.pkl
│   └── preprocessor_premium.pkl
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   └── 02_featureEngineering.ipynb
│
├── .gitignore
├── requirements.txt
└── README.md