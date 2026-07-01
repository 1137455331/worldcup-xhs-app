from __future__ import annotations

import math

import streamlit as st

from data_loader import load_repository
from model import DEFAULT_WEIGHTS, build_explanation, calculate_match_probabilities
from portfolio import RISK_CONFIGS, build_daily_portfolio, infer_risk_profile_from_answers, sanitize_percentages, recommend_bet


st.set_page_config(page_title="2026世界杯小红书金球分析器", layout="wide")


@st.cache_data(show_spinner=False)
def get_repository():
    return load_repository()


def fmt_pct(value: float) -> str:
    return f"{value:.1%}" if math.isfinite(value) else "0.0%"


def fmt_num(value: float) -> str:
    return f"{value:.1f}" if math.isfinite(value) else "0.0"


def team_side_label(row, recommendation: str) -> str:
    return row["team_a"] if recommendation == "A" else row["team_b"]


repo = get_repository()
for warning in repo.warnings:
    st.warning(warning)

st.title("2026 世界杯小红书金球竞猜投资分析器")
st.caption("本地 Streamlit MVP，只读取本地 xlsx，不联网。")

tab_single, tab_portfolio, tab_quiz, tab_explain = st.tabs(["单场分析", "今日组合", "风险测试", "模型解释"])

with tab_single:
    selected_label = st.selectbox("选择比赛", repo.matches["label"].tolist())
    row = repo.matches.loc[repo.matches["label"] == selected_label].iloc[0]

    col1, col2, col3 = st.columns(3)
    with col1:
        xhs_pct_a = st.number_input(f"{row['team_a']} 小红书投注比例%", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
    with col2:
        xhs_pct_b = st.number_input(f"{row['team_b']} 小红书投注比例%", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
    with col3:
        balance = st.number_input("当前金球余额", min_value=0, value=300, step=10)

    risk_profile = st.selectbox(
        "风险偏好",
        list(RISK_CONFIGS.keys()),
        format_func=lambda key: f"{RISK_CONFIGS[key]['label']} ({key})",
    )

    if xhs_pct_a + xhs_pct_b <= 0:
        st.error("A/B 双方比例之和必须大于 0。")
    else:
        raw_a, raw_b = xhs_pct_a, xhs_pct_b
        xhs_pct_a, xhs_pct_b = sanitize_percentages(xhs_pct_a, xhs_pct_b)
        if abs(raw_a + raw_b - 100.0) > 1e-6:
            st.info(f"输入已自动归一化为：{row['team_a']} {xhs_pct_a:.2f}% / {row['team_b']} {xhs_pct_b:.2f}%")

        result = calculate_match_probabilities(row)
        decision = recommend_bet(result.p_model_a, result.p_model_b, xhs_pct_a, xhs_pct_b, float(balance), risk_profile)
        explanation = build_explanation(
            row["team_a"],
            row["team_b"],
            decision["recommendation"],
            result,
            decision["ra_prob_a"],
            decision["ra_prob_b"],
            decision["ra_ev_a"],
            decision["ra_ev_b"],
            decision["score_a"],
            decision["score_b"],
            RISK_CONFIGS[risk_profile]["label"],
            decision["stake"],
        )

        top1, top2, top3, top4 = st.columns(4)
        top1.metric("推荐方向", team_side_label(row, decision["recommendation"]))
        top2.metric("推荐金球", decision["stake"])
        top3.metric("最大可能损失", decision["worst_loss"])
        top4.metric("猜中返还", fmt_num(decision["hit_return"]))

        st.write("**胜率区**")
        prob1, prob2, prob3, prob4 = st.columns(4)
        prob1.metric(f"{row['team_a']} 原始模型胜率", fmt_pct(result.p_model_a))
        prob2.metric(f"{row['team_b']} 原始模型胜率", fmt_pct(result.p_model_b))
        prob3.metric(f"{row['team_a']} 风险调整后胜率", fmt_pct(decision["ra_prob_a"]))
        prob4.metric(f"{row['team_b']} 风险调整后胜率", fmt_pct(decision["ra_prob_b"]))

        st.write("**收益区**")
        gain1, gain2, gain3, gain4 = st.columns(4)
        gain1.metric(f"{row['team_a']} 返还倍率", fmt_num(decision["multiplier_a"]))
        gain2.metric(f"{row['team_b']} 返还倍率", fmt_num(decision["multiplier_b"]))
        gain3.metric(f"{row['team_a']} raw EV", fmt_pct(decision["raw_ev_a"]))
        gain4.metric(f"{row['team_b']} raw EV", fmt_pct(decision["raw_ev_b"]))

        score1, score2, score3, score4 = st.columns(4)
        score1.metric(f"{row['team_a']} 风险调整后 EV", fmt_pct(decision["ra_ev_a"]))
        score2.metric(f"{row['team_b']} 风险调整后 EV", fmt_pct(decision["ra_ev_b"]))
        score3.metric(f"{row['team_a']} risk-adjusted score", fmt_pct(decision["score_a"]))
        score4.metric(f"{row['team_b']} risk-adjusted score", fmt_pct(decision["score_b"]))

        st.write("**解释区**")
        st.write(explanation)
        if decision["warning"]:
            st.warning(decision["warning"])

        with st.expander("查看模型细节"):
            st.json(
                {
                    "match_id": row["match_id"],
                    "weights": DEFAULT_WEIGHTS,
                    "score_diff": result.score_diff,
                    "factor_probability_a": result.p_factor_a,
                    "model_probability_a": result.p_model_a,
                    "risk_adjusted_probability_a": decision["ra_prob_a"],
                    "raw_ev_a": decision["raw_ev_a"],
                    "raw_ev_b": decision["raw_ev_b"],
                    "risk_adjusted_ev_a": decision["ra_ev_a"],
                    "risk_adjusted_ev_b": decision["ra_ev_b"],
                    "risk_adjusted_score_a": decision["score_a"],
                    "risk_adjusted_score_b": decision["score_b"],
                    "contributions": result.contributions,
                }
            )

with tab_portfolio:
    balance = st.number_input("今日组合金球余额", min_value=0, value=500, step=10, key="portfolio_balance")
    risk_profile = st.selectbox(
        "今日组合风险偏好",
        list(RISK_CONFIGS.keys()),
        format_func=lambda key: f"{RISK_CONFIGS[key]['label']} ({key})",
        key="portfolio_risk",
    )
    default_matches = repo.matches["label"].tolist()[: min(5, len(repo.matches))]
    selected_matches = st.multiselect("选择今日要评估的比赛", repo.matches["label"].tolist(), default=default_matches)

    candidates = []
    for idx, label in enumerate(selected_matches):
        row = repo.matches.loc[repo.matches["label"] == label].iloc[0]
        cols = st.columns(4)
        cols[0].write(f"**{row['team_a']} vs {row['team_b']}**")
        pct_a = cols[1].number_input("A%", min_value=0.0, max_value=100.0, value=50.0, step=1.0, key=f"portfolio_a_{idx}")
        pct_b = cols[2].number_input("B%", min_value=0.0, max_value=100.0, value=50.0, step=1.0, key=f"portfolio_b_{idx}")
        cols[3].caption(f"{row['match_id']} / {row['stage']}")
        if pct_a + pct_b > 0:
            pct_a, pct_b = sanitize_percentages(pct_a, pct_b)
            result = calculate_match_probabilities(row)
            decision = recommend_bet(result.p_model_a, result.p_model_b, pct_a, pct_b, float(balance), risk_profile)
            candidates.append({"match": row, "model": result, "decision": decision})

    if st.button("生成今日组合建议", type="primary"):
        portfolio = build_daily_portfolio(candidates, float(balance), risk_profile)
        m1, m2, m3 = st.columns(3)
        m1.metric("今日总投入", portfolio["total_stake"])
        m2.metric("今日最坏亏损", portfolio["worst_loss"])
        m3.metric("若全部猜中返还", fmt_num(portfolio["best_case_return"]))
        st.write(portfolio["message"])

        for item in portfolio["items"]:
            row = item["match"]
            decision = item["decision"]
            st.write(
                f"- {row['team_a']} vs {row['team_b']}：推荐 {team_side_label(row, decision['recommendation'])}，"
                f"金球 {decision['stake']}，risk-adjusted score {decision['score']:.1%}，风险调整后 EV {decision['ev']:.1%}。"
            )
        if not portfolio["items"]:
            st.info("请先至少输入一场比赛的 A/B 比例。")

with tab_quiz:
    questions = [
        "面对波动时，我更愿意少赚一些换稳定。",
        "如果连续两场亏损，我仍能接受按计划执行。",
        "我愿意为了更高 EV 接受更大回撤。",
        "我的金球余额允许我做中等强度试错。",
        "我更在意长期收益，而不是单场输赢体验。",
    ]
    answers = []
    for idx, question in enumerate(questions, start=1):
        answers.append(
            st.slider(
                f"Q{idx}. {question}",
                min_value=1,
                max_value=5,
                value=3,
                help="1 更保守，5 更进取。",
                key=f"quiz_{idx}",
            )
        )
    inferred = infer_risk_profile_from_answers(answers)
    st.success(f"推荐风险偏好：{RISK_CONFIGS[inferred]['label']} ({inferred})")

with tab_explain:
    selected_label = st.selectbox("选择要解释的比赛", repo.matches["label"].tolist(), key="explain_select")
    row = repo.matches.loc[repo.matches["label"] == selected_label].iloc[0]
    result = calculate_match_probabilities(row)

    st.subheader(f"{row['team_a']} vs {row['team_b']}")
    st.table(
        [
            {"模块": "基础排名", "贡献": round(result.contributions.get("rank", 0.0), 4), "A侧原始值": row.get("rank_a"), "B侧原始值": row.get("rank_b")},
            {"模块": "Elo 强度", "贡献": round(result.contributions.get("elo", 0.0), 4), "A侧原始值": row.get("elo_a"), "B侧原始值": row.get("elo_b")},
            {"模块": "阵容身价", "贡献": round(result.contributions.get("squad_value", 0.0), 4), "A侧原始值": row.get("squad_value_a"), "B侧原始值": row.get("squad_value_b")},
            {"模块": "近期状态", "贡献": round(result.contributions.get("recent_form", 0.0), 4), "A侧原始值": row.get("form_a"), "B侧原始值": row.get("form_b")},
            {"模块": "主场因素", "贡献": round(result.contributions.get("host", 0.0), 4), "A侧原始值": row.get("host_advantage"), "B侧原始值": 0},
            {"模块": "场外资本", "贡献": round(result.contributions.get("offfield", 0.0), 4), "A侧原始值": row.get("offfield_a"), "B侧原始值": row.get("offfield_b")},
        ]
    )
    st.write(
        "解释逻辑：先用基础实力、近期状态、主场因素和场外资本合成模型胜率，再按风险偏好把概率向 50% 收缩，最后结合小红书市场比例计算风险调整后 EV。"
    )
