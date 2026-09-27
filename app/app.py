import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Freelancer Pricing Intelligence",
    page_icon="💰",
    layout="wide"
)
st.title("Freelancer Pricing Intelligence 💰")
st.caption("ML-based Fiverr package price prediction")

lasso_basic = joblib.load("models/lasso_basic.pkl")
preprocessor_basic = joblib.load("models/preprocessor_basic.pkl")

lasso_standard = joblib.load("models/lasso_standard.pkl")
preprocessor_standard = joblib.load("models/preprocessor_standard.pkl")

lasso_premium = joblib.load("models/lasso_premium.pkl")
preprocessor_premium = joblib.load("models/preprocessor_premium.pkl")

st.header("Gig Details")
gig_title = st.text_input("Gig Title")
category = st.text_input("Category")

rating_score = st.slider("Select your Rating", min_value=0.0, max_value=5.0, value=4.5)
st.write("Selected Rating", rating_score)

rating_counts = st.number_input("Your Rating Count", min_value = 0, value = 50 )

seller_level = st.selectbox(
    "Seller Level",
    ["New Seller", "Level 1 Seller", "Level 2 Seller", "Top Rated Seller"]
)
seller_level_mapping = {"New Seller": 0, "Level 1 Seller": 1, "Level 2 Seller": 2, "Top Rated Seller": 3}
seller_level = seller_level_mapping[seller_level]
st.divider()

st.subheader("Basic Package")

basic_delivery_days = st.number_input("Basic Delivery Days", min_value=1, value=3)

basic_revision = st.number_input("Basic Revisions", min_value=0, value=1)

basic_revision_unlimited = st.checkbox("Basic Unlimited Revisions")

if basic_revision_unlimited:
    basic_revision = 0

basic_features = st.text_area("Basic Features")
st.divider()

st.subheader("Standard Package")

standard_delivery_days = st.number_input("Standard Delivery Days", min_value=0, value=5)

standard_revision = st.number_input("Standard Revisions", min_value=0, value=2)

standard_revision_unlimited = st.checkbox("Standard Unlimited Revisions")

if standard_revision_unlimited:
    standard_revision = 0

standard_features = st.text_area("Standard Features")
st.divider()

st.subheader("Premium Package")

premium_delivery_days = st.number_input("Premium Delivery Days", min_value=0, value=7)

premium_revision = st.number_input("Premium Revisions", min_value=0, value=3)

premium_revision_unlimited = st.checkbox("Premium Unlimited Revisions")

if premium_revision_unlimited:
    premium_revision = 0

premium_features = st.text_area("Premium Features")

st.divider()
button = st.button("Predict Prices")

#User inputs ko DataFrame mein convert karna
if button :
    basic_input = pd.DataFrame([{
        "gig_title": gig_title,
        "rating_score": rating_score,
        "rating_counts": rating_counts,
        "seller_level": seller_level,
        "category": category,
        "basic_delivery_days": basic_delivery_days,
        "basic_revision": basic_revision,
        "basic_revision_unlimited": int(basic_revision_unlimited),
        "basic_features": basic_features    
    }])

    standard_input = pd.DataFrame ([{
        "gig_title": gig_title,
        "rating_score": rating_score,
        "rating_counts": rating_counts,
        "seller_level": seller_level,
        "category": category,
        "standard_delivery_days": standard_delivery_days,
        "standard_revision": standard_revision,
        "standard_revision_unlimited": int(standard_revision_unlimited),
        "standard_features": standard_features
    }])

    premium_input = pd.DataFrame([{
        "gig_title": gig_title,
        "rating_score": rating_score,
        "rating_counts": rating_counts,
        "seller_level": seller_level,
        "category": category,
        "premium_delivery_days": premium_delivery_days,
        "premium_revision": premium_revision,
        "premium_revision_unlimited": int(premium_revision_unlimited),
        "premium_features": premium_features
    }])

    #ab hum input ko transform karenge
    basic_transformed = preprocessor_basic.transform(basic_input)
    standard_transformed = preprocessor_standard.transform(standard_input)
    premium_transformed = preprocessor_premium.transform(premium_input)

    basic_prediction = lasso_basic.predict(basic_transformed)[0]
    standard_prediction = lasso_standard.predict(standard_transformed)[0]
    premium_prediction = lasso_premium.predict(premium_transformed)[0]

    st.success("Predicted Prices")
    
    col1, col2, col3 = st.columns(3)

    col1.metric("Basic Price", f"${basic_prediction:.2f}")

    col2.metric("Standard Price", f"${standard_prediction:.2f}")

    col3.metric("Premium Price", f"${premium_prediction:.2f}")
        
    st.caption(
        "Prices are treated as USD-equivalent values; "
        " no currency conversion was performed.")