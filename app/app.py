from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

basic_model = joblib.load(BASE_DIR / "models" / "best_basic_model.pkl")
standard_model = joblib.load(BASE_DIR / "models" / "best_standard_model.pkl")
premium_model = joblib.load(BASE_DIR / "models" / "best_premium_model.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    predictions = None
    error = None
    form_data = {}

    if request.method == "POST":
        form_data = request.form.to_dict()

        try:
            rating_score = float(request.form["rating_score"])
            rating_counts = float(request.form["rating_counts"])
            seller_level = int(request.form["seller_level"])

            basic_delivery = float(request.form["basic_delivery_days"])
            standard_delivery = float(request.form["standard_delivery_days"])
            premium_delivery = float(request.form["premium_delivery_days"])

            basic_unlimited = int(request.form.get("basic_unlimited", 0))
            standard_unlimited = int(request.form.get("standard_unlimited", 0))
            premium_unlimited = int(request.form.get("premium_unlimited", 0))

            basic_revision = np.nan if basic_unlimited else float(request.form["basic_revision"])
            standard_revision = np.nan if standard_unlimited else float(request.form["standard_revision"])
            premium_revision = np.nan if premium_unlimited else float(request.form["premium_revision"])

            basic_data = pd.DataFrame([{
                "rating_score": rating_score,
                "rating_counts": rating_counts,
                "seller_level": seller_level,
                "basic_delivery_days": basic_delivery,
                "basic_revision": basic_revision,
                "basic_revision_unlimited": basic_unlimited
            }])

            standard_data = pd.DataFrame([{
                "rating_score": rating_score,
                "rating_counts": rating_counts,
                "seller_level": seller_level,
                "standard_delivery_days": standard_delivery,
                "standard_revision": standard_revision,
                "standard_revision_unlimited": standard_unlimited
            }])

            premium_data = pd.DataFrame([{
                "rating_score": rating_score,
                "rating_counts": rating_counts,
                "seller_level": seller_level,
                "premium_delivery_days": premium_delivery,
                "premium_revision": premium_revision,
                "premium_revision_unlimited": premium_unlimited
            }])

            predictions = {
                "basic": round(max(0, basic_model.predict(basic_data)[0]), 2),
                "standard": round(max(0, standard_model.predict(standard_data)[0]), 2),
                "premium": round(max(0, premium_model.predict(premium_data)[0]), 2)
            }

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        predictions=predictions,
        error=error,
        form_data=form_data
    )


if __name__ == "__main__":
    app.run(debug=True)
