import streamlit as st
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report


# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Naive Bayes – Iris", layout="centered")

st.markdown("## 🌸 Naive Bayes on Iris Dataset")
st.markdown(
    "<p style='color:gray;'>A simple and clean Naive Bayes classification demo</p>",
    unsafe_allow_html=True
)


# -------------------- LOAD IRIS --------------------
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="target")


# -------------------- DATA PREVIEW --------------------
st.subheader("Dataset Preview")
st.dataframe(X.head(), use_container_width=True)

st.subheader("Class Distribution")
st.write(y.value_counts())


# -------------------- USER INPUT --------------------
test_size = st.slider("Test size", 0.1, 0.5, 0.2)


# -------------------- PREPROCESSING --------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# -------------------- SPLIT --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=test_size,
    random_state=42,
    stratify=y
)


# -------------------- MODEL --------------------
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# -------------------- RESULTS --------------------
st.subheader("Results")

accuracy = accuracy_score(y_test, y_pred)
st.markdown(
    f"<p style='font-size:18px;'>Accuracy: <b>{accuracy:.4f}</b></p>",
    unsafe_allow_html=True
)

st.subheader("Classification Report")
st.code(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    ),
    language="text"
)
