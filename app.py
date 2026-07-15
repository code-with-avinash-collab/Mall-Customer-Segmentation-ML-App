import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load("mall_customer_model.pkl")

# Title
st.title("🛍 Mall Customer Segmentation")

st.write("Enter customer details to find the customer segment.")

# Inputs
income = st.number_input(
    "Annual Income (k$)",
    min_value=10,
    max_value=150,
    value=50
)

score = st.slider(
    "Spending Score (1-100)",
    1,
    100,
    50
)

# Button
if st.button("Predict Customer Segment"):

    # Create DataFrame
    sample = pd.DataFrame(
        [[income, score]],
        columns=[
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ]
    )

    # Predict Cluster
    cluster = model.predict(sample)[0]

    st.success(f"Customer belongs to Cluster {cluster}")

    # Customer Type
    if cluster == 0:
        st.write("😊 This customer has Average Income and Average Spending.")

    elif cluster == 1:
        st.write("👑 This customer has High Income and High Spending.")
        st.write("These are Premium Customers.")

    elif cluster == 2:
        st.write("🛍 This customer has Low Income but High Spending.")

    elif cluster == 3:
        st.write("💰 This customer has High Income but Low Spending.")

    else:
        st.write("💵 This customer has Low Income and Low Spending.")