# Freelancer Pricing Intelligence System

An end-to-end machine learning project that predicts Fiverr gig prices for Basic, Standard, and Premium packages based on gig and seller characteristics.

## Project Overview

Freelancer Pricing Intelligence System uses machine learning to estimate reasonable package-level prices for Fiverr gigs.

The project covers the complete ML workflow:

- Data cleaning and preprocessing
- Feature engineering
- Exploratory data analysis
- Outlier/price filtering
- Multiple regression models
- 5-fold cross-validation
- Model evaluation
- Random Forest model selection
- Flask web deployment

## Dataset

The project uses a Fiverr gig dataset containing 8K+ gig records and package-level information.

Key information includes:

- Gig title
- Rating score
- Rating count
- Seller level
- Category
- Delivery time
- Revisions
- Package prices
- Package features

The original dataset contained 14 columns.

## Feature Engineering

Separate datasets were created for:

- Basic package
- Standard package
- Premium package

Each model uses 6 numerical features:

- Rating score
- Rating count
- Seller level
- Package delivery days
- Package revision count
- Unlimited revision indicator

Missing numerical revision values are handled using median imputation, followed by feature scaling.

## Models

Five regression algorithms were benchmarked:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Elastic Net
5. Random Forest

Models were evaluated using:

- MAE
- RMSE
- R²
- 5-fold cross-validation MAE

Random Forest was selected for the final deployment based on its performance across the package-level prediction tasks.

## Final Model Performance

| Package | Test MAE | CV MAE | R² |
|---|---:|---:|---:|
| Basic | 45.23 | 45.66 | 0.135 |
| Standard | 106.91 | 104.10 | 0.206 |
| Premium | 181.83 | 181.11 | 0.202 |

## Flask Application

The project includes a Flask web application where users can enter gig and package characteristics and receive real-time price predictions for all three pricing tiers.

The interface collects:

- Gig title
- Category
- Package features
- Rating
- Rating count
- Seller level
- Delivery days
- Revisions
- Unlimited revision options

The current deployed models use the numerical features listed above; text fields such as gig title, category, and package features are collected for the application interface but are not used by the final prediction pipeline.

## Project Structure

```text
Freelancer Project/
├── data/
│   ├── gigs_data.csv
│   ├── cleaned_gigs.csv
│   ├── cleaned_gigs_basic.csv
│   ├── cleaned_gigs_standard.csv
│   └── cleaned_gigs_premium.csv
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   └── 02_featureEngineering.ipynb
│
├── models/
│   ├── best_basic_model.pkl
│   ├── best_standard_model.pkl
│   └── best_premium_model.pkl
│
├── app/
│   ├── app.py
│   ├── static/
│   │   └── style.css
│   └── templates/
│       └── index.html
│
└── requirements.txt