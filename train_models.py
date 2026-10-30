"""Train package-specific Fiverr price models with text + structured features.

Run from the repository root:
    python train_models.py

The script deliberately keeps legitimate high-priced gigs. It uses a log target
transform and reports untouched hold-out performance instead of deleting hard cases.
"""
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.dummy import DummyRegressor

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "cleaned_gigs.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
N_ITER = 20

PACKAGE_CONFIG = {
    "basic": ("basic_price", "basic_delivery_days", "basic_revision",
              "basic_revision_unlimited", "basic_features"),
    "standard": ("standard_price", "standard_delivery_days", "standard_revision",
                 "standard_revision_unlimited", "standard_features"),
    "premium": ("premium_price", "premium_delivery_days", "premium_revision",
                "premium_revision_unlimited", "premium_features"),
}

NUMERIC_COLUMNS = [
    "rating_score", "rating_counts_log", "seller_level",
    "delivery_days", "revision_count", "unlimited_revisions",
]


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Ensure data/cleaned_gigs.csv exists."
        )
    df = pd.read_csv(DATA_PATH)
    required = {
        "gig_title", "category", "rating_score", "rating_counts",
        "seller_level", *[col for config in PACKAGE_CONFIG.values() for col in config]
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df.drop_duplicates().copy()
    df["gig_title"] = df["gig_title"].fillna("").astype(str)
    df["category"] = df["category"].fillna("Unknown").astype(str)
    df["rating_counts"] = pd.to_numeric(df["rating_counts"], errors="coerce").clip(lower=0)
    df["rating_counts_log"] = np.log1p(df["rating_counts"])
    df["rating_score"] = pd.to_numeric(df["rating_score"], errors="coerce")
    df["seller_level"] = pd.to_numeric(df["seller_level"], errors="coerce")
    return df


def build_estimator():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("title_tfidf", TfidfVectorizer(
                max_features=2500, ngram_range=(1, 2), min_df=2,
                sublinear_tf=True, strip_accents="unicode"
            ), "gig_title"),
            ("features_tfidf", TfidfVectorizer(
                max_features=3500, ngram_range=(1, 2), min_df=2,
                sublinear_tf=True, strip_accents="unicode"
            ), "package_features"),
            ("category", OneHotEncoder(
                handle_unknown="ignore", min_frequency=5
            ), ["category"]),
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
        ],
        remainder="drop",
    )

    base_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
    ])

    # Prices are strongly right-skewed. Fit in log space, but evaluate in dollars.
    return TransformedTargetRegressor(
        regressor=base_pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def make_package_frame(df, config):
    target_col, delivery_col, revision_col, unlimited_col, features_col = config
    X = pd.DataFrame({
        "gig_title": df["gig_title"],
        "category": df["category"],
        "package_features": df[features_col].fillna("").astype(str),
        "rating_score": pd.to_numeric(df["rating_score"], errors="coerce"),
        "rating_counts_log": df["rating_counts_log"],
        "seller_level": pd.to_numeric(df["seller_level"], errors="coerce"),
        "delivery_days": pd.to_numeric(df[delivery_col], errors="coerce"),
        "revision_count": pd.to_numeric(df[revision_col], errors="coerce"),
        "unlimited_revisions": pd.to_numeric(df[unlimited_col], errors="coerce").fillna(0),
    })
    y = pd.to_numeric(df[target_col], errors="coerce")
    valid = y.notna() & np.isfinite(y) & (y >= 0)
    return X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)


def train_package(df, package, config):
    X, y = make_package_frame(df, config)
    if len(X) < 100:
        raise ValueError(f"Only {len(X)} valid rows for {package}; at least 100 are required.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    baseline = DummyRegressor(strategy="median")
    baseline.fit(np.zeros((len(y_train), 1)), y_train)
    baseline_pred = baseline.predict(np.zeros((len(y_test), 1)))
    baseline_mae = mean_absolute_error(y_test, baseline_pred)

    estimator = build_estimator()
    param_distributions = {
        "regressor__model": [
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        ],
        "regressor__model__n_estimators": [250, 400, 600],
        "regressor__model__max_depth": [None, 12, 20, 30],
        "regressor__model__min_samples_split": [2, 5, 10],
        "regressor__model__min_samples_leaf": [1, 2, 4],
        "regressor__model__max_features": [0.6, 0.8, 1.0],
    }

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=N_ITER,
        scoring="neg_mean_absolute_error",
        cv=CV_FOLDS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    predictions = np.maximum(0, search.best_estimator_.predict(X_test))

    metrics = {
        "package": package,
        "rows": int(len(X)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "baseline_median_mae": round(float(baseline_mae), 3),
        "cv_mae": round(float(-search.best_score_), 3),
        "test_mae": round(float(mean_absolute_error(y_test, predictions)), 3),
        "test_rmse": round(float(np.sqrt(mean_squared_error(y_test, predictions))), 3),
        "test_r2": round(float(r2_score(y_test, predictions)), 4),
        "best_model": type(search.best_estimator_.regressor_.named_steps["model"]).__name__,
        "best_params": {
            key.replace("regressor__model__", ""): value
            for key, value in search.best_params_.items()
            if key != "regressor__model"
        },
    }

    model_path = MODEL_DIR / f"best_{package}_model.pkl"
    joblib.dump(search.best_estimator_, model_path)
    print(f"\n{package.upper()} RESULTS")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"Saved: {model_path}")
    return metrics


def main():
    df = load_data()
    print(f"Loaded {len(df):,} unique gig rows from {DATA_PATH}")
    all_metrics = []
    for package, config in PACKAGE_CONFIG.items():
        all_metrics.append(train_package(df, package, config))

    metrics_path = MODEL_DIR / "evaluation_metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"\nSaved evaluation report: {metrics_path}")


if __name__ == "__main__":
    main()
