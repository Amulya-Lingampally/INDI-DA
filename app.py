
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="SpendWise AI Dashboard", layout="wide")

st.title("💡 SpendWise AI - Smart Financial Insights Dashboard")

st.markdown("### 📊 Business Objective: Identify high-potential users and optimize marketing strategies using data-driven insights.")

uploaded_file = st.file_uploader("📂 Upload Dataset", type=["xlsx","csv"])

if uploaded_file:
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith("xlsx") else pd.read_csv(uploaded_file)

    st.subheader("🔍 Dataset Overview")
    st.dataframe(df.head())

    # Descriptive Analysis
    st.subheader("📊 Descriptive Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x="Income", title="Income Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(df, x="Expenses", title="Expense Distribution")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("👉 Insight: Understanding income and expense distribution helps identify spending capacity of users.")

    # Encoding
    df_model = df.copy()
    le = LabelEncoder()

    for col in df_model.select_dtypes(include='object').columns:
        df_model[col] = le.fit_transform(df_model[col])

    X = df_model.drop("App_Interest", axis=1)
    y = df_model["App_Interest"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Classification
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("🤖 Predictive Analysis (Classification)")

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    st.write(f"Accuracy: {acc:.2f}")
    st.write(f"Precision: {prec:.2f}")
    st.write(f"Recall: {rec:.2f}")
    st.write(f"F1 Score: {f1:.2f}")

    # ROC Curve
    try:
        y_prob = model.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_prob[:,1], pos_label=1)
        roc_auc = auc(fpr, tpr)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"AUC={roc_auc:.2f}"))
        fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("ROC curve works best for binary classification.")

    # Feature importance
    st.subheader("📌 Feature Importance")
    importance = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": X.columns, "Importance": importance})
    fig2 = px.bar(feat_df.sort_values(by="Importance", ascending=False), x="Feature", y="Importance")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("👉 Insight: Features with higher importance strongly influence user adoption decisions.")

    # Clustering
    st.subheader("👥 Customer Segmentation (Clustering)")
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df["Cluster"] = clusters

    fig3 = px.scatter(df, x="Income", y="Expenses", color=df["Cluster"].astype(str), title="Customer Segments")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("👉 Insight: Clustering helps identify user groups like impulsive spenders, balanced users, and disciplined savers.")

    # Association Rules
    st.subheader("🔗 Association Rules")
    basket = pd.get_dummies(df.astype(str))
    freq_items = apriori(basket, min_support=0.1, use_colnames=True)
    rules = association_rules(freq_items, metric="lift", min_threshold=1)

    st.dataframe(rules[['antecedents','consequents','support','confidence','lift']].head())

    st.markdown("👉 Insight: Association rules reveal spending patterns that can guide cross-selling strategies.")

