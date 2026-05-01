"""
data_generator.py — Synthetic User Behavior Data Generator
Creates realistic SaaS user behavior data for churn prediction modeling.
"""

import numpy as np
import pandas as pd


def generate_user_data(
      n_users: int = 500,
      churn_rate: float = 0.22,
      seed: int = 42,
) -> pd.DataFrame:
      """
          Generate a synthetic SaaS user behavior dataset for churn prediction.

              The churn label is generated deterministically based on behavioral signals
                  with added noise, simulating real-world churn patterns:
                      - High inactivity + low feature adoption → higher churn probability
                          - Incomplete onboarding → higher churn probability
                              - High support tickets → higher churn probability
                                  - Free plan users → higher churn probability

                                      Args:
                                              n_users: Number of users to generate
                                                      churn_rate: Target churn rate (approximate)
                                                              seed: Random seed for reproducibility

                                                                  Returns:
                                                                          DataFrame with user_id, behavioral features, and 'churned' label
                                                                              """
      rng = np.random.default_rng(seed)

    user_ids = [f"user_{i:04d}" for i in range(1, n_users + 1)]

    # ── Plan type (influences churn) ──────────────────────────────────────────
    plan_types = rng.choice(
              ["free", "starter", "pro", "enterprise"],
              size=n_users,
              p=[0.40, 0.30, 0.20, 0.10]
    )

    # ── Onboarding completion ─────────────────────────────────────────────────
    # Free plan users less likely to complete onboarding
    onboarding_prob = np.where(
              plan_types == "free", 0.45,
              np.where(plan_types == "starter", 0.65,
                                        np.where(plan_types == "pro", 0.80, 0.90))
    )
    onboarding_completed = rng.random(n_users) < onboarding_prob

    # ── Days since last login (inactivity signal) ─────────────────────────────
    # Non-churners: mostly active (0-10 days)
    # Churners: mostly inactive (10-30 days)
    # We'll generate base values and corrupt with churn-correlated noise later
    days_since_login_base = rng.integers(0, 30, size=n_users)

    # ── Feature adoption ──────────────────────────────────────────────────────
    max_features = {"free": 5, "starter": 10, "pro": 15, "enterprise": 20}
    features_used = np.array([
              rng.integers(0, max_features[p] + 1)
              for p in plan_types
    ])

    # ── Support tickets (last 30 days) ────────────────────────────────────────
    support_tickets = rng.poisson(lam=0.8, size=n_users)
    support_tickets = np.clip(support_tickets, 0, 8)

    # ── Mobile sessions ───────────────────────────────────────────────────────
    mobile_sessions = np.where(
              rng.random(n_users) < 0.35, 0,
              rng.integers(1, 25, size=n_users)
    )

    # ── Login count (last 30 days) ────────────────────────────────────────────
    login_count = np.where(
              days_since_login_base > 14,
              rng.integers(0, 5, size=n_users),
              rng.integers(5, 30, size=n_users)
    )

    # ── Team size ─────────────────────────────────────────────────────────────
    team_size = np.where(
              plan_types == "free", rng.integers(1, 3, size=n_users),
              np.where(plan_types == "starter", rng.integers(1, 10, size=n_users),
                                        np.where(plan_types == "pro", rng.integers(5, 50, size=n_users),
                                                                           rng.integers(20, 500, size=n_users)))
    )

    # ── Account age (days) ────────────────────────────────────────────────────
    account_age = rng.integers(7, 730, size=n_users)

    # ── Churn label generation ────────────────────────────────────────────────
    # Compute a churn score based on behavioral risk factors
    churn_score = (
              0.30 * (days_since_login_base / 30.0)       # inactivity
              + 0.25 * (1 - features_used / 20.0)          # low adoption
              + 0.20 * (~onboarding_completed).astype(float) # incomplete onboarding
              + 0.15 * (support_tickets / 8.0)              # support friction
              + 0.10 * (plan_types == "free").astype(float) # free plan
    )

    # Add noise and threshold to get binary labels
    noise = rng.normal(0, 0.1, size=n_users)
    churn_score_noisy = np.clip(churn_score + noise, 0, 1)

    # Adjust threshold to achieve target churn rate
    threshold = np.percentile(churn_score_noisy, (1 - churn_rate) * 100)
    churned = (churn_score_noisy >= threshold).astype(int)

    # Adjust days_since_login to be more extreme for churners
    days_since_login_final = np.where(
              churned == 1,
              np.clip(days_since_login_base + rng.integers(5, 15, size=n_users), 0, 30),
              np.clip(days_since_login_base - rng.integers(0, 10, size=n_users), 0, 30)
    )

    df = pd.DataFrame({
              "user_id": user_ids,
              "plan_type": plan_types,
              "onboarding_completed": onboarding_completed,
              "days_since_last_login": days_since_login_final,
              "features_used_count": features_used,
              "support_tickets_30d": support_tickets,
              "mobile_sessions_30d": mobile_sessions,
              "login_count_30d": login_count,
              "team_size": team_size,
              "account_age_days": account_age,
              "churned": churned,
    })

    return df
