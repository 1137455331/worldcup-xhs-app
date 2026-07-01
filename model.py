from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_WEIGHTS = {
    "rank": 0.20,
    "elo": 0.20,
    "squad_value": 0.15,
    "recent_form": 0.25,
    "host": 0.05,
    "offfield": 0.15,
}

RISK_SHRINK_FACTORS = {
    "conservative": 0.75,
    "balanced": 0.90,
    "aggressive": 1.00,
}

LONGSHOT_CUTOFF = 0.20
EXTREME_CUTOFF = 0.12
MIN_PROB = 0.01
LONGSHOT_RISK_PARAMS = {
    "conservative": {"z": 1.15, "base_sigma": 0.070, "longshot_lambda": 0.55, "market_shrink": 0.35},
    "balanced": {"z": 0.80, "base_sigma": 0.055, "longshot_lambda": 0.35, "market_shrink": 0.22},
    "aggressive": {"z": 0.45, "base_sigma": 0.040, "longshot_lambda": 0.18, "market_shrink": 0.10},
}


@dataclass
class MatchModelResult:
    p_factor_a: float
    p_factor_b: float
    p_model_a: float
    p_model_b: float
    score_diff: float
    seed_prob_a: float | None
    contributions: dict[str, float]


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def ensure_finite(value: Any, default: float = 0.0) -> float:
    number = safe_float(value, default)
    return float(number) if np.isfinite(number) else float(default)


def clamp_probability(value: float, lower: float = 0.02, upper: float = 0.98) -> float:
    finite = ensure_finite(value, 0.5)
    return float(np.clip(finite, lower, upper))


def logistic(value: float) -> float:
    finite = ensure_finite(value, 0.0)
    return 1.0 / (1.0 + np.exp(-finite))


def _scaled_rank_diff(row: pd.Series) -> float:
    if pd.notna(row.get("rank_diff")):
        return ensure_finite(row["rank_diff"], 0.0) / 50.0
    rank_a = safe_float(row.get("rank_a"))
    rank_b = safe_float(row.get("rank_b"))
    if np.isnan(rank_a) or np.isnan(rank_b):
        return 0.0
    return (rank_b - rank_a) / 50.0


def _scaled_elo_diff(row: pd.Series) -> float:
    value = safe_float(row.get("elo_diff"))
    if np.isfinite(value):
        return value / 250.0
    elo_a = safe_float(row.get("elo_a"))
    elo_b = safe_float(row.get("elo_b"))
    if np.isnan(elo_a) or np.isnan(elo_b):
        return 0.0
    return (elo_a - elo_b) / 250.0


def _scaled_value_diff(row: pd.Series) -> float:
    diff = safe_float(row.get("squad_value_diff"))
    if np.isfinite(diff):
        base = max(abs(ensure_finite(row.get("squad_value_a"), 0.0)), abs(ensure_finite(row.get("squad_value_b"), 0.0)), 500.0)
        return diff / base
    value_a = safe_float(row.get("squad_value_a"))
    value_b = safe_float(row.get("squad_value_b"))
    if np.isnan(value_a) or np.isnan(value_b):
        return 0.0
    return (value_a - value_b) / max(value_a, value_b, 500.0)


def _scaled_form_diff(row: pd.Series) -> float:
    diff = safe_float(row.get("form_diff"))
    if np.isfinite(diff):
        return diff / 4.0
    form_a = safe_float(row.get("form_a"))
    form_b = safe_float(row.get("form_b"))
    if np.isnan(form_a) or np.isnan(form_b):
        return 0.0
    return (form_a - form_b) / 4.0


def _scaled_host_advantage(row: pd.Series) -> float:
    value = ensure_finite(row.get("host_advantage"), 0.0)
    return float(np.clip(value, -1.0, 1.0))


def _scaled_offfield_diff(row: pd.Series) -> float:
    diff = safe_float(row.get("offfield_diff"))
    if np.isfinite(diff):
        return diff / 100.0
    off_a = safe_float(row.get("offfield_a"))
    off_b = safe_float(row.get("offfield_b"))
    if np.isnan(off_a) or np.isnan(off_b):
        return 0.0
    return (off_a - off_b) / 100.0


def apply_risk_adjustment(probability: float, risk_profile: str) -> float:
    shrink = RISK_SHRINK_FACTORS.get(risk_profile, 0.90)
    adjusted = 0.5 + shrink * (clamp_probability(probability) - 0.5)
    return clamp_probability(adjusted)


def longshot_intensity(q_share: float) -> float:
    q_share = float(np.clip(ensure_finite(q_share, 0.5), 0.0, 1.0))
    return float(np.clip((LONGSHOT_CUTOFF - q_share) / LONGSHOT_CUTOFF, 0.0, 1.0))


def adjust_one_side_for_longshot(probability: float, q_share: float, risk_profile: str) -> float:
    params = LONGSHOT_RISK_PARAMS.get(risk_profile, LONGSHOT_RISK_PARAMS["balanced"])
    base_probability = apply_risk_adjustment(probability, risk_profile)
    q_share = float(np.clip(ensure_finite(q_share, 0.5), MIN_PROB, 0.99))
    intensity = longshot_intensity(q_share)
    sigma = params["base_sigma"] * (1.0 + 1.5 * intensity)
    p_lcb = base_probability - params["z"] * sigma
    p_penalized = p_lcb - params["longshot_lambda"] * intensity * max(0.0, base_probability - q_share)
    shrink = params["market_shrink"] * intensity
    p_final = (1.0 - shrink) * p_penalized + shrink * q_share
    return float(np.clip(p_final, MIN_PROB, 0.99))


def adjust_probability_for_longshot(
    p_a: float,
    p_b: float,
    q_a: float,
    q_b: float,
    risk_profile: str,
) -> tuple[float, float]:
    q_a = float(np.clip(ensure_finite(q_a, 0.5), MIN_PROB, 0.99))
    q_b = float(np.clip(ensure_finite(q_b, 0.5), MIN_PROB, 0.99))
    p_adj_a = adjust_one_side_for_longshot(p_a, q_a, risk_profile)
    p_adj_b = adjust_one_side_for_longshot(p_b, q_b, risk_profile)
    total = p_adj_a + p_adj_b
    if total <= 0:
        return 0.5, 0.5
    return clamp_probability(p_adj_a / total), clamp_probability(p_adj_b / total)


def calculate_match_probabilities(
    match_row: pd.Series,
    weights: dict[str, float] | None = None,
    prior_blend: float = 0.5,
) -> MatchModelResult:
    weights = weights or DEFAULT_WEIGHTS
    factor_values = {
        "rank": _scaled_rank_diff(match_row),
        "elo": _scaled_elo_diff(match_row),
        "squad_value": _scaled_value_diff(match_row),
        "recent_form": _scaled_form_diff(match_row),
        "host": _scaled_host_advantage(match_row),
        "offfield": _scaled_offfield_diff(match_row),
    }
    finite_factors = {key: ensure_finite(value, 0.0) for key, value in factor_values.items()}
    contributions = {key: weights[key] * finite_factors[key] for key in weights}
    score_diff = float(sum(contributions.values()))

    p_factor_a = clamp_probability(logistic(score_diff))
    seed_prob = safe_float(match_row.get("seed_prob_a"))
    if np.isfinite(seed_prob):
        seed_prob = seed_prob / 100.0 if seed_prob > 1 else seed_prob
        seed_prob = clamp_probability(seed_prob)
        p_model_a = clamp_probability(prior_blend * seed_prob + (1 - prior_blend) * p_factor_a)
    else:
        seed_prob = None
        p_model_a = p_factor_a

    return MatchModelResult(
        p_factor_a=p_factor_a,
        p_factor_b=1 - p_factor_a,
        p_model_a=p_model_a,
        p_model_b=1 - p_model_a,
        score_diff=score_diff,
        seed_prob_a=seed_prob,
        contributions=contributions,
    )


def build_explanation(
    team_a: str,
    team_b: str,
    recommendation: str,
    model_result: MatchModelResult,
    ra_prob_a: float,
    ra_prob_b: float,
    ra_ev_a: float,
    ra_ev_b: float,
    score_a: float,
    score_b: float,
    risk_label: str,
    stake: int,
) -> str:
    chosen_team = team_a if recommendation == "A" else team_b
    chosen_ev = ra_ev_a if recommendation == "A" else ra_ev_b
    chosen_score = score_a if recommendation == "A" else score_b
    lead = f"原始模型判断 {team_a} 胜率约 {model_result.p_model_a:.1%}，{team_b} 胜率约 {model_result.p_model_b:.1%}。"
    second = f"按 {risk_label} 风险偏好收缩后，双方风险调整胜率为 {ra_prob_a:.1%} 和 {ra_prob_b:.1%}。"
    if chosen_ev > 0:
        third = f"由于 {chosen_team} 的 risk-adjusted score 更高，风险调整后 EV 为 {chosen_ev:.1%}、score 为 {chosen_score:.1%}，系统推荐该方向，并给出 {stake} 金球档位。"
    else:
        third = f"两边都偏低或为负 EV，但 {chosen_team} 的 risk-adjusted score 仍相对更高，因此作为娱乐参与方向推荐，并给出 {stake} 金球档位。"

    dominant = sorted(model_result.contributions.items(), key=lambda item: abs(item[1]), reverse=True)
    top_factors = [name for name, value in dominant if abs(value) > 0][:2]
    factor_map = {
        "rank": "排名差距",
        "elo": "Elo 强度",
        "squad_value": "阵容身价",
        "recent_form": "近期状态",
        "host": "主场因素",
        "offfield": "场外资本",
    }
    if top_factors:
        fourth = "本场优势主要来自 " + "、".join(factor_map.get(name, name) for name in top_factors) + "。"
    else:
        fourth = "本场特征较中性，因此最终判断更接近五五开。"
    return " ".join([lead, second, third, fourth])
