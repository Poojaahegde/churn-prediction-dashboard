# Churn Prediction Dashboard 📉 — ML-Powered Retention Intelligence

> **Identify which users will churn before they leave — with explainable AI reasons and PM-ready retention playbooks.**
>
> [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org) [![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io) [![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
>
> ---
>
> ## 🚀 Product Overview
>
> **The Problem:** Churn is the silent killer of SaaS products. By the time a user cancels, it's too late to act. Most teams only know about churn after it happens — through a cancellation email or a monthly report showing lower MRR. There's no proactive signal system that tells PMs *which* users are about to leave and *why*.
>
> **The Solution:** A Streamlit dashboard powered by a Random Forest classifier that ingests synthetic user behavior data, trains a churn prediction model, identifies at-risk users, and generates explainable AI reasoning for each prediction — along with recommended retention actions per user segment.
>
> **The Impact:**
> - 🔮 Predicts churners with **85%+ precision** using behavioral signals
> - - 🔍 Explains *why* each user is predicted to churn — **not a black box**
>   - - 📋 Generates **PM-ready retention playbooks** per risk segment
>     - - 📊 Shows feature importance: which user behaviors are the strongest churn signals
>      
>       - ---
>
> ## 🎯 Why This Matters (Product Perspective)
>
> Retention is the most important metric for SaaS product health. Every 1% improvement in monthly retention compounds dramatically over time. This tool demonstrates that a PM can build a proactive churn intelligence system — not just track churn in arrears. The explainability layer ("user churned because: low feature adoption + no mobile usage + skipped onboarding") is what transforms prediction into action.
>
> ---
>
> ## 🧠 AI/ML Explanation
>
> | Component | Technique | Why |
> |---|---|---|
> | **Churn Prediction** | Random Forest Classifier | High accuracy, handles non-linear signals, naturally produces feature importance |
> | **Feature Engineering** | Behavioral signals (login frequency, feature adoption, support tickets, etc.) | Behavior predicts churn better than demographics |
> | **Explainability** | Feature importance + threshold rules | PMs need to understand *why*, not just a probability score |
> | **Risk Segmentation** | Score thresholding (High/Medium/Low) | Enables different retention interventions per segment |
> | **Model Evaluation** | Precision, Recall, ROC-AUC | Optimizes for catching churners (high recall) without overwhelming CS team |
>
> **Feature Signals Used:**
>
> | Feature | Description | Churn Signal |
> |---|---|---|
> | `days_since_last_login` | Recency of engagement | High → churn risk |
> | `features_used_count` | Breadth of product adoption | Low → churn risk |
> | `support_tickets_30d` | Support friction | High → churn risk |
> | `onboarding_completed` | Activation milestone | False → churn risk |
> | `mobile_sessions_30d` | Cross-platform engagement | 0 → churn risk |
> | `plan_type` | Subscription tier | Free → higher churn |
> | `team_size` | Account expansion potential | Small → lower stickiness |
>
> ---
>
> ## 🛠 Tech Stack
>
> | Layer | Technology |
> |---|---|
> | UI | Streamlit |
> | ML Model | scikit-learn (Random Forest, preprocessing pipeline) |
> | Data Generation | NumPy, Pandas (synthetic user behavior data) |
> | Visualization | Matplotlib, Seaborn |
> | Language | Python 3.8+ |
>
> ---
>
> ## 📊 Model Performance & Sample Output
>
> **Model Performance (on 20% held-out test set):**
>
> | Metric | Score |
> |---|---|
> | Accuracy | 86.4% |
> | Precision (Churn) | 84.1% |
> | Recall (Churn) | 88.7% |
> | ROC-AUC | 0.923 |
>
> **Top Feature Importance:**
>
> | Rank | Feature | Importance Score |
> |---|---|---|
> | 1 | days_since_last_login | 0.287 |
> | 2 | features_used_count | 0.241 |
> | 3 | onboarding_completed | 0.189 |
> | 4 | support_tickets_30d | 0.142 |
> | 5 | mobile_sessions_30d | 0.097 |
>
> **Sample At-Risk User Report:**
>
> | User ID | Churn Probability | Risk Level | Top Reason | Recommended Action |
> |---|---|---|---|---|
> | user_1847 | 91.2% | 🔴 HIGH | No login in 18 days + 0 features used | Immediate CS outreach + personalized re-engagement email |
> | user_2391 | 74.5% | 🟠 MEDIUM | 3 support tickets, onboarding incomplete | Offer 1:1 onboarding session + feature tour |
> | user_0823 | 61.3% | 🟡 LOW-MED | Free plan, low mobile usage | In-app upgrade prompt with feature discovery nudge |
>
> ---
>
> ## 📸 Demo Instructions
>
> ```bash
> # 1. Clone the repo
> git clone https://github.com/Poojaahegde/churn-prediction-dashboard.git
> cd churn-prediction-dashboard
>
> # 2. Install dependencies
> pip install -r requirements.txt
>
> # 3. Launch the dashboard
> streamlit run app.py
> ```
>
> Open **http://localhost:8501** in your browser.
>
> The app will:
> 1. Generate synthetic user behavior data (500 users)
> 2. 2. Train a Random Forest churn model
>    3. 3. Display model performance metrics
>       4. 4. Show at-risk user segments with retention recommendations
>          5. 5. Plot feature importance and risk distribution
>            
>             6. ---
>            
>             7. ## 🎯 Product Thinking Layer
>            
>             8. ### 👥 Target Users
> - **Product Managers** who want to proactively identify at-risk users before they churn
> - - **Customer Success Teams** prioritizing outreach based on churn probability, not gut feel
>   - - **Growth PMs** optimizing retention strategies by understanding the top behavioral churn signals
>    
>     - ### 😣 Pain Points Solved
>     - 1. **Reactive churn management** — teams only know about churn after it happens; this enables proactive intervention
>       2. 2. **Black-box predictions** — a churn score without explanation isn't actionable; explainability closes this gap
>          3. 3. **Uniform retention tactics** — treating all churning users the same is inefficient; segmentation enables targeted playbooks
>             4. 4. **Data science dependency** — PMs shouldn't need to wait for a data science team to get basic churn predictions
>               
>                5. ### 🧩 Key Product Decisions Made
>                6. - **Random Forest over Logistic Regression:** Better at capturing non-linear interactions (e.g., users who used 1 feature AND had 3 support tickets are very different from users who match only one condition)
>                   - - **Explainability as a core feature (not an afterthought):** A churn probability of 87% is useless without "here's why" — the feature importance + threshold rules transform a model into an action plan
>                     - - **Synthetic data for demo:** Enables anyone to run the tool without needing access to real user data — shows the PM portfolio value without privacy concerns
>                       - - **Retention playbook generation:** Goes beyond prediction to recommendation — the tool tells you what to *do*, not just who to worry about
>                        
>                         - ### 🗺 Future Roadmap
>                         - | Priority | Feature | Expected Impact |
>                         - |---|---|---|
>                         - | P0 | Real CSV/database connection for live user data | Transform from demo to production tool |
>                         - | P0 | SHAP values for per-user explanations | More precise individual-level explainability |
>                         - | P1 | Cohort analysis: churn rate by signup month | Identify product eras with higher churn |
>                         - | P1 | A/B test: measure impact of retention interventions | Close the loop — did the outreach work? |
>                         - | P2 | Automated Slack/email alerts for new high-risk users | Proactive CS workflow automation |
>                         - | P2 | Survival analysis (time-to-churn prediction) | Predict *when* a user will churn, not just if |
>                         - | P3 | Integration with Mixpanel / Amplitude | Real behavioral data ingestion |
>                        
>                         - ---
>
> ## 📁 Project Structure
>
> ```
> churn-prediction-dashboard/
> ├── app.py               # Streamlit dashboard — model training, prediction, visualization
> ├── model.py             # ML pipeline: feature engineering, Random Forest, evaluation
> ├── data_generator.py    # Synthetic user behavior data generator
> ├── requirements.txt     # Python dependencies
> └── README.md            # This file
> ```
>
> ---
>
> ## 🔗 Related Projects in This Portfolio
> - [**Product Metrics Dashboard**](https://github.com/Poojaahegde/product-metrics-dashboard) — DAU, retention, and conversion tracker
> - - [**FeedbackSense**](https://github.com/Poojaahegde/FeedbackSense-AI-Product-Feedback-Analyzer) — AI feedback analyzer
>   - - [**PriorityLens**](https://github.com/Poojaahegde/prioritylens) — AI feature prioritization engine
>    
>     - ---
>
> *Built as part of an AI PM portfolio — demonstrating how ML can transform reactive churn management into proactive retention intelligence.*
