"""
ChurnModel — Random Forest ML pipeline for churn prediction.
Includes training, prediction, evaluation, and feature importance.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, classification_report
)

# Feature columns used in the model
NUMERIC_FEATURES = [
      "days_since_last_login",
      "features_used_count",
      "support_tickets_30d",
      "mobile_sessions_30d",
      "login_count_30d",
      "team_size",
      "account_age_days",
]

CATEGORICAL_FEATURES = ["plan_type", "onboarding_completed"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "churned"


class ChurnModel:
      """
          Random Forest churn prediction pipeline with explainability.

              Usage:
                      model = ChurnModel(n_estimators=100)
                              metrics = model.train(df)
                                      predictions = model.predict(df)
                                              importance = model.feature_importance()
                                                  """

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
              self.n_estimators = n_estimators
              self.random_state = random_state
              self._pipeline = None
              self._feature_names = None
              self._is_trained = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame) -> dict:
              """
                      Train the Random Forest classifier on the provided user dataframe.

                              Args:
                                          df: DataFrame with user behavior features and 'churned' label

                                                  Returns:
                                                              dict of evaluation metrics (accuracy, precision, recall, roc_auc)
                                                                      """
              X = df[ALL_FEATURES]
              y = df[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
                      X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        # Preprocessing pipeline
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        preprocessor = ColumnTransformer(transformers=[
                      ("num", numeric_transformer, NUMERIC_FEATURES),
                      ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ])

        self._pipeline = Pipeline(steps=[
                      ("preprocessor", preprocessor),
                      ("classifier", RandomForestClassifier(
                                        n_estimators=self.n_estimators,
                                        random_state=self.random_state,
                                        class_weight="balanced",
                                        n_jobs=-1,
                      ))
        ])

        self._pipeline.fit(X_train, y_train)
        self._is_trained = True

        # Store feature names after encoding
        cat_feature_names = (
                      self._pipeline.named_steps["preprocessor"]
                      .named_transformers_["cat"]
                      .get_feature_names_out(CATEGORICAL_FEATURES)
                      .tolist()
        )
        self._feature_names = NUMERIC_FEATURES + cat_feature_names

        # Evaluate
        y_pred = self._pipeline.predict(X_test)
        y_prob = self._pipeline.predict_proba(X_test)[:, 1]

        return {
                      "accuracy": accuracy_score(y_test, y_pred),
                      "precision": precision_score(y_test, y_pred, zero_division=0),
                      "recall": recall_score(y_test, y_pred, zero_division=0),
                      "roc_auc": roc_auc_score(y_test, y_prob),
        }

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
              """
                      Generate churn predictions with risk levels and human-readable reasons.

                              Args:
                                          df: Full user DataFrame

                                                  Returns:
                                                              DataFrame with predictions, risk levels, and top reasons
                                                                      """
              assert self._is_trained, "Model must be trained before predicting."

        X = df[ALL_FEATURES]
        probs = self._pipeline.predict_proba(X)[:, 1]

        result = df[["user_id"] + ALL_FEATURES].copy()
        result["churn_probability"] = np.round(probs, 4)
        result["risk_level"] = result["churn_probability"].apply(self._risk_level)
        result["top_reason"] = result.apply(self._generate_reason, axis=1)

        return result

    def feature_importance(self) -> pd.DataFrame:
              """
                      Return feature importances from the trained Random Forest.

                              Returns:
                                          DataFrame with 'feature' and 'importance' columns
                                                  """
              assert self._is_trained, "Model must be trained first."
              importances = (
                  self._pipeline.named_steps["classifier"].feature_importances_
              )
              return pd.DataFrame({
                  "feature": self._feature_names,
                  "importance": importances,
              }).sort_values("importance", ascending=False)

    # ── Private Methods ────────────────────────────────────────────────────────

    @staticmethod
    def _risk_level(prob: float) -> str:
              if prob >= 0.70:
                            return "HIGH"
elif prob >= 0.40:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _generate_reason(row: pd.Series) -> str:
              """Generate a human-readable churn reason based on behavioral signals."""
              reasons = []

        if row["days_since_last_login"] >= 14:
                      reasons.append(f"inactive {int(row['days_since_last_login'])}d")
                  if row["features_used_count"] <= 2:
                                reasons.append(f"low adoption ({int(row['features_used_count'])} features)")
                            if not row["onboarding_completed"]:
                                          reasons.append("onboarding incomplete")
                                      if row["support_tickets_30d"] >= 3:
                                                    reasons.append(f"{int(row['support_tickets_30d'])} support tickets")
                                                if row["mobile_sessions_30d"] == 0:
                                                              reasons.append("no mobile usage")
                                                          if row["plan_type"] == "free":
                                                                        reasons.append("free plan")

        return ", ".join(reasons[:3]) if reasons else "Normal usage patterns"
