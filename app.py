# =========================================================
# KNN CLASSIFIER - WINE DATASET
# =========================================================

# =========================================================
# STEP 1 - IMPORT LIBRARIES
# =========================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# STREAMLIT PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="KNN Classification Dashboard",
    page_icon="🍷",
    layout="wide"
)

# =========================================================
# LOAD CUSTOM CSS
# =========================================================

def load_css(css_file):
    with open(css_file) as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css("style.css")

# =========================================================
# APPLICATION TITLE
# =========================================================

st.title("🍷 KNN Classification on Wine Dataset")

st.markdown("""
This application demonstrates multiclass classification
using K-Nearest Neighbors (KNN).

### Workflow Included
- Dataset Loading
- Data Cleaning
- Preprocessing
- Exploratory Data Analysis
- Feature Scaling
- KNN Model Training
- Prediction
- Model Evaluation
""")

# =========================================================
# STEP 2 - LOAD DATASET
# =========================================================

st.header("📂 Step 1 : Load Dataset")

wine = load_wine()

df = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
)

df["target"] = wine.target

# =========================================================
# CLASS LABELS
# =========================================================

wine_map = {
    0: "Class 0",
    1: "Class 1",
    2: "Class 2"
}

df["wine_class"] = df["target"].map(wine_map)

# =========================================================
# DATASET PREVIEW
# =========================================================

st.subheader("Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)

# =========================================================
# DATASET METRICS
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rows", df.shape[0])

with col2:
    st.metric("Columns", df.shape[1])

with col3:
    st.metric("Classes", df["target"].nunique())

# =========================================================
# STEP 3 - DATA CLEANING
# =========================================================

st.header("🧹 Step 2 : Data Cleaning")

col1, col2 = st.columns(2)

with col1:

    st.subheader("Missing Values")

    st.dataframe(df.isnull().sum())

with col2:

    st.subheader("Duplicate Records")

    duplicates = df.duplicated().sum()

    st.write(f"Duplicate Rows : {duplicates}")

# Remove duplicates
df.drop_duplicates(inplace=True)

st.success("✅ Data Cleaning Completed Successfully")

# =========================================================
# STEP 4 - DATA PREPROCESSING
# =========================================================

st.header("⚙️ Step 3 : Data Preprocessing")

X = df.drop(
    ["target", "wine_class"],
    axis=1
)

y = df["target"]

# =========================================================
# FEATURE SCALING
# =========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

st.success("Feature Scaling Applied Successfully")

# =========================================================
# STEP 5 - EXPLORATORY DATA ANALYSIS
# =========================================================

st.header("📊 Step 4 : Exploratory Data Analysis")

# =========================================================
# STATISTICAL SUMMARY
# =========================================================

st.subheader("Statistical Summary")

st.dataframe(
    df.describe(),
    use_container_width=True
)

# =========================================================
# CLASS DISTRIBUTION
# =========================================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("Wine Class Distribution")

    fig1, ax1 = plt.subplots(figsize=(7, 4))

    sns.countplot(
        x="wine_class",
        data=df,
        palette="viridis",
        ax=ax1
    )

    ax1.set_xlabel("Wine Class")
    ax1.set_ylabel("Count")

    st.pyplot(fig1)

with col2:

    st.subheader("Correlation Heatmap")

    fig2, ax2 = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        df.drop("wine_class", axis=1).corr(),
        annot=False,
        cmap="coolwarm",
        linewidths=0.5,
        ax=ax2
    )

    st.pyplot(fig2)

# =========================================================
# HISTOGRAM VISUALIZATION
# =========================================================

st.subheader("Feature Distribution")

selected_feature = st.selectbox(
    "Select Feature",
    X.columns
)

fig3, ax3 = plt.subplots(figsize=(8, 5))

sns.histplot(
    df[selected_feature],
    kde=True,
    color="purple",
    ax=ax3
)

ax3.set_xlabel(selected_feature)

st.pyplot(fig3)

# =========================================================
# STEP 6 - TRAIN TEST SPLIT
# =========================================================

st.header("✂️ Step 5 : Train Test Split")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Training Samples",
        X_train.shape[0]
    )

with col2:
    st.metric(
        "Testing Samples",
        X_test.shape[0]
    )

# =========================================================
# STEP 7 - MODEL TRAINING
# =========================================================

st.header("🤖 Step 6 : Model Training")

k = st.slider(
    "Select K Value",
    1,
    20,
    5
)

model = KNeighborsClassifier(
    n_neighbors=k
)

model.fit(
    X_train,
    y_train
)

st.success("KNN Classification Model Trained Successfully")

# =========================================================
# STEP 8 - MODEL PREDICTIONS
# =========================================================

st.header("📌 Step 7 : Predictions")

y_pred = model.predict(X_test)

prediction_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

prediction_df["Actual"] = prediction_df["Actual"].map(wine_map)

prediction_df["Predicted"] = prediction_df["Predicted"].map(wine_map)

st.dataframe(
    prediction_df.head(10),
    use_container_width=True
)

# =========================================================
# STEP 9 - MODEL EVALUATION
# =========================================================

st.header("📉 Step 8 : Model Evaluation")

accuracy = accuracy_score(
    y_test,
    y_pred
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Accuracy Score",
        f"{accuracy:.2f}"
    )

with col2:
    st.metric(
        "Error Rate",
        f"{1 - accuracy:.2f}"
    )

# =========================================================
# CONFUSION MATRIX
# =========================================================

st.subheader("Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred
)

fig4, ax4 = plt.subplots(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=wine_map.values(),
    yticklabels=wine_map.values(),
    ax=ax4
)

ax4.set_xlabel("Predicted Label")
ax4.set_ylabel("Actual Label")

st.pyplot(fig4)

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

st.subheader("Classification Report")

report = classification_report(
    y_test,
    y_pred,
    target_names=wine_map.values(),
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df,
    use_container_width=True
)

# =========================================================
# ACTUAL VS PREDICTED GRAPH
# =========================================================

st.subheader("Actual vs Predicted")

fig5, ax5 = plt.subplots(figsize=(8, 5))

ax5.plot(
    y_test.values,
    label="Actual",
    marker="o"
)

ax5.plot(
    y_pred,
    label="Predicted",
    marker="x"
)

ax5.legend()

st.pyplot(fig5)

# =========================================================
# STEP 10 - USER PREDICTION
# =========================================================

st.header("🎯 Step 9 : Predict Wine Class")

user_input = []

col1, col2 = st.columns(2)

columns = list(X.columns)

for i in range(len(columns)):

    with col1 if i % 2 == 0 else col2:

        val = st.number_input(
            f"Enter {columns[i]}",
            value=float(df[columns[i]].mean())
        )

        user_input.append(val)

# =========================================================
# PREPARE USER INPUT
# =========================================================

scaled_input = scaler.transform([user_input])

prediction = model.predict(scaled_input)[0]

predicted_class = wine_map[prediction]

# =========================================================
# DISPLAY PREDICTION
# =========================================================

if st.button("Predict Wine Class"):

    st.markdown(
        f"""
        <div style="
            background-color:#7c3aed;
            padding:20px;
            border-radius:10px;
            text-align:center;
            color:white;
            font-size:24px;
        ">
            Predicted Wine Class : {predicted_class}
        </div>
        """,
        unsafe_allow_html=True
    )
