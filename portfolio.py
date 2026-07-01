from __future__ import annotations

from typing import Any

import numpy as np

from model import EXTREME_CUTOFF, LONGSHOT_CUTOFF, adjust_probability_for_longshot, clamp_probability, ensure_finite, longshot_intensity


RISK_CONFIGS = {
    "conservative": {
        "label": "保守",
        "shrink_factor": 0.75,
        "max_single_bet_pct": 0.03,
        "max_single_bet_abs": 100,
        "max_daily_bet_pct": 0.10,
        "allowed_stakes": [10, 20, 50],
    },
    "balanced": {
        "label": "均衡",
        "shrink_factor": 0.90,
        "max_single_bet_pct": 0.08,
        "max_single_bet_abs": 500,
        "max_daily_bet_pct": 0.20,
        "allowed_stakes": [10, 20, 50, 100, 500],
    },
    "aggressive": {
        "label": "进取",
        "shrink_factor": 1.00,
        "max_single_bet_pct": 0.15,
        "max_single_bet_abs": 500,
        "max_daily_bet_pct": 0.40,
        "allowed_stakes": [10, 20, 50, 100, 500],
    },
}


def normalize_percentages(a_pct: float, b_pct: float) -> tuple[float, float]:
    total = ensure_finite(a_pct, 0.0) + ensure_finite(b_pct, 0.0)
    if total <= 0:
        raise ValueError("比例之和必须大于 0。")
    norm_a = 100.0 * ensure_finite(a_pct, 0.0) / total
    norm_b = 100.0 * ensure_finite(b_pct, 0.0) / total
    return float(np.clip(norm_a, 0.1, 99.9)), float(np.clip(norm_b, 0.1, 99.9))


def sanitize_percentages(a_pct: float, b_pct: float) -> tuple[float, float]:
    a_pct, b_pct = normalize_percentages(a_pct, b_pct)
    total = a_pct + b_pct
    return a_pct * 100.0 / total, b_pct * 100.0 / total


def calc_multiplier(xhs_pct: float) -> float:
    clipped = float(np.clip(ensure_finite(xhs_pct, 50.0), 0.1, 99.9))
    return 100.0 / clipped


def calc_expected_value(probability: float, multiplier: float) -> float:
    probability = clamp_probability(probability)
    multiplier = ensure_finite(multiplier, 1.0)
    value = probability * multiplier - 1.0
    return ensure_finite(value, 0.0)


def calc_returns(probability: float, q_share: float) -> tuple[float, float, float]:
    q_share = float(np.clip(ensure_finite(q_share, 0.5), 1e-6, 0.999999))
    probability = clamp_probability(probability)
    multiplier = 1.0 / q_share
    raw_ev = probability * multiplier - 1.0
    variance = probability * (multiplier - 1.0 - raw_ev) ** 2 + (1.0 - probability) * (-1.0 - raw_ev) ** 2
    vol = variance ** 0.5
    return ensure_finite(multiplier, 1.0), ensure_finite(raw_ev, 0.0), ensure_finite(vol, 0.0)


RISK_AVERSION = {
    "conservative": 0.85,
    "balanced": 0.55,
    "aggressive": 0.30,
}


def risk_adjusted_score(probability: float, q_share: float, risk_profile: str) -> tuple[float, float, float, float]:
    multiplier, ev, vol = calc_returns(probability, q_share)
    intensity = longshot_intensity(q_share)
    gamma = RISK_AVERSION[risk_profile]
    score = ev - gamma * vol * (0.35 + 0.65 * intensity)
    return ensure_finite(score, -1.0), ev, multiplier, vol


def calc_kelly(probability: float, multiplier: float) -> float:
    b = ensure_finite(multiplier, 1.0) - 1.0
    if b <= 0:
        return 0.0
    probability = clamp_probability(probability)
    kelly = (probability * b - (1 - probability)) / b
    return max(0.0, ensure_finite(kelly, 0.0))


def map_to_discrete_stake(theoretical_stake: float, allowed_stakes: list[int]) -> int:
    theoretical = ensure_finite(theoretical_stake, 0.0)
    eligible = [stake for stake in sorted(allowed_stakes) if stake <= theoretical]
    return eligible[-1] if eligible else 0


def _single_bet_cap(balance: float, config: dict[str, Any]) -> float:
    return min(balance * config["max_single_bet_pct"], config["max_single_bet_abs"])


def _stake_candidates_by_ev(ev: float, risk_profile: str) -> list[int]:
    if risk_profile == "conservative":
        if ev <= 0:
            return [10]
        if ev < 0.10:
            return [10, 20]
        if ev < 0.25:
            return [20]
        return [50]
    if risk_profile == "balanced":
        if ev <= 0:
            return [20]
        if ev < 0.10:
            return [20, 50]
        if ev < 0.25:
            return [50, 100]
        return [100, 500]
    if ev <= 0:
        return [50]
    if ev < 0.10:
        return [100, 500]
    if ev < 0.25:
        return [100, 500]
    return [500]


def _apply_longshot_stake_cap(stake: int, q_share: float, probability: float, score: float, risk_profile: str) -> int:
    q_share = float(np.clip(ensure_finite(q_share, 0.5), 0.0, 1.0))
    probability = clamp_probability(probability)
    if q_share >= LONGSHOT_CUTOFF:
        return stake

    if risk_profile == "conservative":
        capped = 10 if probability < 0.25 else min(stake, 20)
    elif risk_profile == "balanced":
        capped = 20 if probability < 0.25 else min(stake, 50)
    else:
        capped = min(stake, 100)
        if probability >= 0.35 and score > 0.25:
            capped = stake

    if q_share < EXTREME_CUTOFF:
        if risk_profile == "aggressive":
            capped = min(capped, 50 if probability < 0.40 else 100)
        elif risk_profile == "balanced":
            capped = min(capped, 20)
        else:
            capped = min(capped, 10)
        if probability < 0.40:
            capped = min(capped, 100)

    if q_share < EXTREME_CUTOFF and probability < 0.40 and capped >= 500:
        capped = 100
    return capped


def _pick_stake(balance: float, risk_profile: str, ev: float, probability: float, multiplier: float, q_share: float, score: float) -> tuple[int, float, float]:
    config = RISK_CONFIGS[risk_profile]
    cap = _single_bet_cap(balance, config)
    kelly = calc_kelly(probability, multiplier)
    theoretical = min(balance * kelly, cap)

    if risk_profile == "aggressive" and balance >= 5000 and ev > 0 and cap >= 500 and q_share >= LONGSHOT_CUTOFF:
        return 500, theoretical, kelly

    candidates = [stake for stake in _stake_candidates_by_ev(ev, risk_profile) if stake in config["allowed_stakes"]]
    feasible = [stake for stake in candidates if stake <= cap]
    if feasible:
        stake = max(feasible)
        return _apply_longshot_stake_cap(stake, q_share, probability, score, risk_profile), theoretical, kelly

    mapped = map_to_discrete_stake(theoretical, config["allowed_stakes"])
    if mapped > 0:
        return _apply_longshot_stake_cap(mapped, q_share, probability, score, risk_profile), theoretical, kelly

    fallback = [stake for stake in config["allowed_stakes"] if stake <= cap]
    if fallback:
        stake = min(fallback)
        return _apply_longshot_stake_cap(stake, q_share, probability, score, risk_profile), theoretical, kelly

    return 0, theoretical, kelly


def recommend_bet(
    p_model_a: float,
    p_model_b: float,
    xhs_pct_a: float,
    xhs_pct_b: float,
    balance: float,
    risk_profile: str,
) -> dict[str, Any]:
    config = RISK_CONFIGS[risk_profile]
    xhs_pct_a, xhs_pct_b = sanitize_percentages(xhs_pct_a, xhs_pct_b)
    q_share_a = xhs_pct_a / 100.0
    q_share_b = xhs_pct_b / 100.0

    ra_prob_a, ra_prob_b = adjust_probability_for_longshot(p_model_a, p_model_b, q_share_a, q_share_b, risk_profile)

    raw_multiplier_a = calc_multiplier(xhs_pct_a)
    raw_multiplier_b = calc_multiplier(xhs_pct_b)
    raw_ev_a = calc_expected_value(p_model_a, raw_multiplier_a)
    raw_ev_b = calc_expected_value(p_model_b, raw_multiplier_b)

    score_a, ra_ev_a, multiplier_a, vol_a = risk_adjusted_score(ra_prob_a, q_share_a, risk_profile)
    score_b, ra_ev_b, multiplier_b, vol_b = risk_adjusted_score(ra_prob_b, q_share_b, risk_profile)

    if score_a >= score_b:
        recommendation = "A"
        probability = ra_prob_a
        multiplier = multiplier_a
        chosen_ev = ra_ev_a
        chosen_score = score_a
        q_share = q_share_a
    else:
        recommendation = "B"
        probability = ra_prob_b
        multiplier = multiplier_b
        chosen_ev = ra_ev_b
        chosen_score = score_b
        q_share = q_share_b

    stake, theoretical, kelly = _pick_stake(float(balance), risk_profile, chosen_ev, probability, multiplier, q_share, chosen_score)
    selected_multiplier = multiplier_a if recommendation == "A" else multiplier_b
    hit_return = stake * selected_multiplier

    note = None
    if q_share < LONGSHOT_CUTOFF:
        note = "该推荐包含低支持率冷门惩罚与概率不确定性折扣，已限制极端冷门的推荐档位。"
    elif chosen_ev <= 0:
        note = "该场为负 EV 或低优势机会，推荐方向仅表示相对更优的一侧。"

    return {
        "risk_profile_label": config["label"],
        "recommendation": recommendation,
        "stake": int(stake),
        "theoretical_stake": float(theoretical),
        "multiplier_a": float(multiplier_a),
        "multiplier_b": float(multiplier_b),
        "raw_prob_a": clamp_probability(p_model_a),
        "raw_prob_b": clamp_probability(p_model_b),
        "raw_ev_a": raw_ev_a,
        "raw_ev_b": raw_ev_b,
        "ra_prob_a": ra_prob_a,
        "ra_prob_b": ra_prob_b,
        "ra_ev_a": ra_ev_a,
        "ra_ev_b": ra_ev_b,
        "score_a": score_a,
        "score_b": score_b,
        "vol_a": vol_a,
        "vol_b": vol_b,
        "ev": chosen_ev,
        "score": chosen_score,
        "kelly": float(kelly),
        "worst_loss": int(stake),
        "hit_return": float(hit_return),
        "xhs_pct_a": xhs_pct_a,
        "xhs_pct_b": xhs_pct_b,
        "warning": note,
    }


def build_daily_portfolio(candidates: list[dict[str, Any]], balance: float, risk_profile: str) -> dict[str, Any]:
    config = RISK_CONFIGS[risk_profile]
    daily_cap = float(balance) * config["max_daily_bet_pct"]
    raw_total = sum(item["decision"]["stake"] for item in candidates)
    scale = min(1.0, daily_cap / raw_total) if raw_total > 0 else 1.0

    adjusted_items: list[dict[str, Any]] = []
    for item in candidates:
        decision = dict(item["decision"])
        stake = decision["stake"]
        scaled = stake * scale
        if scale < 1.0 and stake > 0:
            mapped = map_to_discrete_stake(scaled, config["allowed_stakes"])
            if mapped == 0:
                feasible = [value for value in config["allowed_stakes"] if value <= stake]
                mapped = feasible[0] if feasible else 0
            decision["stake"] = int(mapped)
            decision["worst_loss"] = int(mapped)
            decision["hit_return"] = float(mapped * (decision["multiplier_a"] if decision["recommendation"] == "A" else decision["multiplier_b"]))
        adjusted_items.append({"match": item["match"], "model": item["model"], "decision": decision})

    total_stake = sum(item["decision"]["stake"] for item in adjusted_items)
    return {
        "items": adjusted_items,
        "total_stake": int(total_stake),
        "daily_cap": float(daily_cap),
        "worst_loss": int(total_stake),
        "best_case_return": float(sum(item["decision"]["hit_return"] for item in adjusted_items)),
        "message": "今日组合已按 risk-adjusted score 对每场给出方向，并对低支持率冷门自动限制过激仓位。",
    }


def infer_risk_profile_from_answers(scores: list[int]) -> str:
    avg = sum(scores) / max(len(scores), 1)
    if avg <= 2.2:
        return "conservative"
    if avg <= 3.6:
        return "balanced"
    return "aggressive"
