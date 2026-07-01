from pathlib import Path

from data_loader import load_repository
from model import adjust_probability_for_longshot, apply_risk_adjustment, clamp_probability, calculate_match_probabilities
from portfolio import calc_expected_value, calc_kelly, calc_multiplier, infer_risk_profile_from_answers, map_to_discrete_stake, normalize_percentages, recommend_bet


def test_multiplier():
    assert calc_multiplier(25) == 4.0


def test_expected_value():
    assert round(calc_expected_value(0.6, 2.0), 4) == 0.2


def test_kelly_non_negative():
    assert calc_kelly(0.4, 1.5) >= 0


def test_stake_mapping():
    assert map_to_discrete_stake(67, [10, 20, 50, 100]) == 50
    assert map_to_discrete_stake(8, [10, 20, 50]) == 0


def test_probability_clamp():
    assert clamp_probability(0.001) == 0.02
    assert clamp_probability(0.99) == 0.98


def test_normalize_percentages():
    a, b = normalize_percentages(30, 50)
    assert round(a + b, 10) == 100


def test_recommend_bet_always_has_side():
    decision = recommend_bet(0.51, 0.49, 70, 30, 1000, "balanced")
    assert decision["recommendation"] in {"A", "B"}
    assert decision["stake"] in {10, 20, 50, 100, 500}


def test_risk_quiz():
    assert infer_risk_profile_from_answers([1, 1, 2, 2, 2]) == "conservative"


def test_risk_adjustment_shrinks_toward_even():
    assert apply_risk_adjustment(0.8, "conservative") < 0.8
    assert apply_risk_adjustment(0.8, "aggressive") == 0.8


def test_england_vs_dr_congo_no_nan_and_has_pick():
    repo = load_repository(Path(__file__).resolve().parents[1])
    row = repo.matches[(repo.matches["team_a"] == "England") & (repo.matches["team_b"] == "DR Congo")].iloc[0]
    result = calculate_match_probabilities(row)
    decision = recommend_bet(result.p_model_a, result.p_model_b, 92, 8, 10000, "aggressive")

    assert 0.0 < result.p_model_a < 1.0
    assert 0.0 < result.p_model_b < 1.0
    assert decision["recommendation"] in {"A", "B"}
    assert decision["stake"] in {10, 20, 50, 100, 500}
    assert all(str(decision[key]).lower() != "nan" for key in ["ra_prob_a", "ra_prob_b", "ra_ev_a", "ra_ev_b"])


def test_belgium_vs_senegal_aggressive_has_explicit_pick():
    repo = load_repository(Path(__file__).resolve().parents[1])
    row = repo.matches[(repo.matches["team_a"] == "Belgium") & (repo.matches["team_b"] == "Senegal")].iloc[0]
    result = calculate_match_probabilities(row)
    decision = recommend_bet(result.p_model_a, result.p_model_b, 57, 43, 10000, "aggressive")

    assert decision["recommendation"] in {"A", "B"}
    assert decision["stake"] in {10, 20, 50, 100, 500}
    assert decision["recommendation"] == "B"
    assert decision["stake"] == 500


def test_longshot_probability_adjustment_penalizes_low_support_side():
    p_a, p_b = adjust_probability_for_longshot(0.711, 0.289, 0.90, 0.10, "aggressive")
    assert p_a > p_b
    assert p_b < 0.289


def test_argentina_vs_cape_verde_extreme_longshot_cap():
    repo = load_repository(Path(__file__).resolve().parents[1])
    row = repo.matches[(repo.matches["team_a"] == "Argentina") & (repo.matches["team_b"] == "Cape Verde")].iloc[0]
    result = calculate_match_probabilities(row)
    decision = recommend_bet(result.p_model_a, result.p_model_b, 90, 10, 10000, "aggressive")

    assert decision["recommendation"] in {"A", "B"}
    assert all(str(decision[key]).lower() != "nan" for key in ["ra_prob_a", "ra_prob_b", "raw_ev_a", "raw_ev_b", "score_a", "score_b"])
    if decision["recommendation"] == "B":
        assert decision["stake"] <= 100 or decision["ra_prob_b"] >= 0.35
