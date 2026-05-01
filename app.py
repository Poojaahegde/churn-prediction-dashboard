"""
Churn Prediction Dashboard — Streamlit App
Predicts at-risk users with explainable AI and PM-ready retention recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import ChurnModel
from data_generator import generate_user_data

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
      page_title="Churn Prediction Dashboard",
      page_icon="📉",
      layout="wide"
)

st.title("📉 Churn Prediction Dashboard")
st.markdown("*Identify at-risk users before they leave — powered by ML.*")

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
      st.header("⚙️ Configuration")
      n_users = st.slider("Number of users to simulate", 100, 1000, 500, 50)
      churn_rate = st.slider("Simulated churn rate (%)", 10, 40, 22) / 100
      high_risk_threshold = st.slider("High risk threshold", 0.5, 0.9, 0.7, 0.05)
      n_estimators = st.select_slider(
          "Random Forest trees", options=[50, 100, 200], value=100
      )
      st.markdown("---")
      retrain = st.button("🔄 Regenerate Data & Retrain Model", type="primary")

# ─── Data & Model ─────────────────────────────────────────────────────────────
@st.cache_data
def load_and_train(n_users, churn_rate, n_estimators, _seed=42):
      df = generate_user_data(n_users=n_users, churn_rate=churn_rate, seed=_seed)
      model = ChurnModel(n_estimators=n_estimators)
      metrics = model.train(df)
      predictions = model.predict(df)
      importance = model.feature_importance()
      return df, model, metrics, predictions, importance

if retrain:
      st.cache_data.clear()

df, model, metrics, predictions, importance = load_and_train(
      n_users, churn_rate, n_estimators
)

# ─── KPI Row ──────────────────────────────────────────────────────────────────
st.subheader("📊 Model Performance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
col2.metric("Precision", f"{metrics['precision']:.1%}")
col3.metric("Recall", f"{metrics['recall']:.1%}")
col4.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

# ─── Risk Summary ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🚨 Churn Risk Summary")

high_risk = predictions[predictions["risk_level"] == "HIGH"]
medium_risk = predictions[predictions["risk_level"] == "MEDIUM"]
low_risk = predictions[predictions["risk_level"] == "LOW"]

c1, c2, c3 = st.columns(3)
c1.metric(
      "🔴 High Risk Users",
      len(high_risk),
      delta=f"{len(high_risk)/len(predictions):.1%} of users",
      delta_color="inverse"
)
c2.metric(
      "🟠 Medium Risk Users",
      len(medium_risk),
      delta=f"{len(medium_risk)/len(predictions):.1%} of users",
      delta_color="off"
)
c3.metric(
      "🟢 Low Risk Users",
      len(low_risk),
      delta=f"{len(low_risk)/len(predictions):.1%} of users"
)

# ─── Main Tabs ────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
      "🔴 At-Risk Users", "📊 Feature Importance", "💡 Retention Playbook"
])

with tab1:
      st.subheader("At-Risk Users — Sorted by Churn Probability")
      risk_filter = st.multiselect(
          "Filter by risk level",
          ["HIGH", "MEDIUM", "LOW"],
          default=["HIGH", "MEDIUM"]
      )
      filtered = predictions[predictions["risk_level"].isin(risk_filter)].copy()
      filtered = filtered.sort_values("churn_probability", ascending=False)

    display_cols = [
              "user_id", "churn_probability", "risk_level",
              "days_since_last_login", "features_used_count",
              "onboarding_completed", "support_tickets_30d", "top_reason"
    ]
    st.dataframe(
              filtered[display_cols].head(50),
              use_container_width=True,
              hide_index=True
    )

    # Distribution plot
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = {"HIGH": "#e74c3c", "MEDIUM": "#e67e22", "LOW": "#2ecc71"}
    for level, color in colors.items():
              subset = predictions[predictions["risk_level"] == level]
              ax.hist(subset["churn_probability"], bins=20, alpha=0.7,
                      color=color, label=level)
          ax.set_xlabel("Churn Probability")
    ax.set_ylabel("Number of Users")
    ax.set_title("Churn Probability Distribution by Risk Level")
    ax.legend()
    ax.set_facecolor("#0e1117")
    fig.patch.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
              spine.set_edgecolor("white")
          ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.legend(facecolor="#0e1117", labelcolor="white")
    st.pyplot(fig)

with tab2:
      st.subheader("Feature Importance — What Drives Churn?")
      imp_df = importance.sort_values("importance", ascending=True).tail(8)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(imp_df["feature"], imp_df["importance"],
                                      color="#3498db", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Importance Score")
    ax.set_title("Top Features Predicting Churn")
    ax.set_facecolor("#0e1117")
    fig.patch.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
              spine.set_edgecolor("#333")
          st.pyplot(fig)

    st.markdown("**🧠 PM Interpretation:**")
    st.info(
              "The top 3 churn drivers are **recency** (days since last login), "
              "**breadth of adoption** (features used), and **activation** (onboarding completion). "
              "This means retention interventions should focus on re-engagement, "
              "feature discovery, and onboarding completion — not just account health scores."
    )

with tab3:
      st.subheader("💡 Retention Playbook by Risk Segment")

    playbook = {
              "🔴 HIGH RISK (Churn prob > 70%)": {
                            "description": f"{len(high_risk)} users",
                            "signals": "Not logged in 14+ days, 0–2 features used, support friction",
                            "actions": [
                                              "🚨 Immediate personal outreach from CS team (email + in-app message)",
                                              "🎯 Offer 1:1 product walkthrough session",
                                              "💰 Consider proactive discount or plan extension",
                                              "📊 Flag in next sprint review as retention P0",
                            ]
              },
              "🟠 MEDIUM RISK (Churn prob 40–70%)": {
                            "description": f"{len(medium_risk)} users",
                            "signals": "Sporadic login, partial onboarding, occasional support tickets",
                            "actions": [
                                              "📧 Automated re-engagement email with feature highlights",
                                              "🧭 In-app tooltip tour for unused features",
                                              "📅 Invite to upcoming product webinar or office hours",
                                              "🔔 Set retention health alert in CS dashboard",
                            ]
              },
              "🟡 LOW-MEDIUM RISK (Churn prob < 40%)": {
                            "description": f"{len(low_risk)} users",
                            "signals": "Regular login, partial adoption, no support issues",
                            "actions": [
                                              "💡 In-app feature discovery nudges (tooltips, empty states)",
                                              "📰 Monthly product digest with tips for their use case",
                                              "⭐ NPS survey to capture early dissatisfaction signals",
                                              "🔄 Monitor week-over-week for deteriorating signals",
                            ]
              }
    }

    for segment, data in playbook.items():
              with st.expander(f"{segment} — {data['description']}"):
                            st.write(f"**Common signals:** {data['signals']}")
                            st.write("**Recommended actions:**")
                            for action in data["actions"]:
                                              st.write(f"  {action}")

                # ─── Footer ───────────────────────────────────────────────────────────────────
                st.markdown("---")
st.markdown(
      "Built by [Pooja Hegde](https://github.com/Poojaahegde) · "
      "Part of the AI PM Portfolio"
)
