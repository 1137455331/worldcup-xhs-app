from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

import numpy as np
import pandas as pd


DATA_FILE_KEYWORDS = {
    "future": ["未来比赛特征表", "future_match", "complete"],
    "recent": ["近期状态汇总表", "recent_form", "current_complete"],
    "offfield": ["场外因素与资本力量统计表", "offfield", "capital"],
    "static": ["第一阶段与静态实力表", "current_strength", "静态实力"],
}


@dataclass
class DataRepository:
    matches: pd.DataFrame
    team_metrics: pd.DataFrame
    recent_summary: pd.DataFrame
    offfield: pd.DataFrame
    static_strength: pd.DataFrame
    risk_profiles: pd.DataFrame
    warnings: list[str]


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.strip().lower()
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text)


def snake_case(value: Any) -> str:
    text = "" if value is None else str(value).strip().lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def candidate_directories(base_dir: Path | None = None) -> list[Path]:
    root = (base_dir or Path(__file__).resolve().parent).resolve()
    return [root / "data"]


def find_data_file(file_type: str, base_dir: Path | None = None) -> Path | None:
    keywords = DATA_FILE_KEYWORDS[file_type]
    candidates: list[Path] = []
    for folder in candidate_directories(base_dir):
        if not folder.exists():
            continue
        for path in folder.glob("*.xlsx"):
            if path.name.startswith("~$"):
                continue
            name = path.name.lower()
            score = sum(1 for keyword in keywords if keyword.lower() in name)
            if score:
                candidates.append((score, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], len(item[1].name)))
    return candidates[0][1]


def read_workbook(path: Path) -> dict[str, pd.DataFrame]:
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    clean: dict[str, pd.DataFrame] = {}
    for sheet_name, frame in sheets.items():
        sheet = frame.copy()
        sheet.columns = [snake_case(col) or f"col_{idx}" for idx, col in enumerate(sheet.columns)]
        clean[sheet_name] = sheet
    return clean


def find_sheet_name(workbook: dict[str, pd.DataFrame], aliases: list[str]) -> str | None:
    normalized_aliases = [normalize_text(alias) for alias in aliases]
    scored: list[tuple[int, str]] = []
    for name in workbook:
        normalized_name = normalize_text(name)
        score = sum(1 for alias in normalized_aliases if alias and alias in normalized_name)
        if score:
            scored.append((score, name))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], len(item[1])))
    return scored[0][1]


def pick_column(frame: pd.DataFrame, aliases: list[str]) -> str | None:
    norm_aliases = [normalize_text(alias) for alias in aliases]
    scored: list[tuple[int, str]] = []
    for column in frame.columns:
        normalized = normalize_text(column)
        score = sum(1 for alias in norm_aliases if alias and alias in normalized)
        if score:
            scored.append((score, column))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], len(item[1])))
    return scored[0][1]


def coerce_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("€", "", regex=False)
        .str.replace("bn", "000", regex=False)
        .str.replace("m", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def standardize_team_name(series: pd.Series) -> pd.Series:
    aliases = {
        "drcongo": "DR Congo",
        "congodr": "DR Congo",
        "democraticrepublicofthecongo": "DR Congo",
        "drc": "DR Congo",
        "bosniaherzegovina": "Bosnia-Herzegovina",
        "bosniaandherzegovina": "Bosnia-Herzegovina",
        "unitedstates": "United States",
        "usa": "United States",
    }
    cleaned = series.fillna("").astype(str).str.strip()
    normalized = cleaned.map(lambda value: aliases.get(normalize_text(value), value))
    return normalized.astype(str).str.strip()


def prepare_team_tables(
    team_metrics: pd.DataFrame,
    recent_summary: pd.DataFrame,
    offfield: pd.DataFrame,
    static_strength: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for frame in [team_metrics, recent_summary, offfield, static_strength]:
        if not frame.empty:
            team_col = pick_column(frame, ["team", "英文名", "球队", "team_a_input"])
            if team_col and team_col != "team":
                frame["team"] = standardize_team_name(frame[team_col])
            elif team_col == "team":
                frame["team"] = standardize_team_name(frame["team"])

    for col in ["fifa_rank", "elo_rating", "squad_value_eur_m", "adjusted_form_score", "offfield_capital_score"]:
        if col in team_metrics.columns:
            team_metrics[col] = coerce_numeric(team_metrics[col])
    if "adjusted_form_score" in recent_summary.columns:
        recent_summary["adjusted_form_score"] = coerce_numeric(recent_summary["adjusted_form_score"])
    if "offfield_capital_score" in offfield.columns:
        offfield["offfield_capital_score"] = coerce_numeric(offfield["offfield_capital_score"])
    if "rank_by_capital_score" in offfield.columns:
        offfield["rank_by_capital_score"] = coerce_numeric(offfield["rank_by_capital_score"])

    if team_metrics.empty and not static_strength.empty:
        team_metrics = static_strength.rename(
            columns={
                pick_column(static_strength, ["fifa即时排名", "fifa_rank", "rank"]): "fifa_rank",
                pick_column(static_strength, ["elo分", "elo_rating", "elo"]): "elo_rating",
                pick_column(static_strength, ["阵容总身价", "squad_value", "market_value"]): "squad_value_eur_m",
            }
        )
        for col in ["fifa_rank", "elo_rating", "squad_value_eur_m"]:
            if col in team_metrics.columns:
                team_metrics[col] = coerce_numeric(team_metrics[col])

    return team_metrics, recent_summary, offfield, static_strength


def build_team_lookup(
    team_metrics: pd.DataFrame,
    recent_summary: pd.DataFrame,
    offfield: pd.DataFrame,
    static_strength: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not team_metrics.empty:
        frames.append(team_metrics[["team"] + [col for col in team_metrics.columns if col != "team"]])
    if not recent_summary.empty:
        keep = [col for col in ["team", "adjusted_form_score", "points_per_match", "goal_diff", "top20_record", "status"] if col in recent_summary.columns]
        frames.append(recent_summary[keep])
    if not offfield.empty:
        keep = [
            col
            for col in [
                "team",
                "offfield_capital_score",
                "squad_value_score",
                "commercial_attractiveness_score",
                "league_development_score",
            ]
            if col in offfield.columns
        ]
        frames.append(offfield[keep])
    if not static_strength.empty and "team" in static_strength.columns:
        keep = [col for col in ["team", "fifa_rank", "elo_rating", "squad_value_eur_m"] if col in static_strength.columns]
        frames.append(static_strength[keep])

    if not frames:
        return pd.DataFrame(columns=["team"])

    team_lookup = frames[0].copy()
    for frame in frames[1:]:
        team_lookup = team_lookup.merge(frame, on="team", how="outer", suffixes=("", "_dup"))
        dup_cols = [col for col in team_lookup.columns if col.endswith("_dup")]
        for dup_col in dup_cols:
            base_col = dup_col[:-4]
            if base_col in team_lookup.columns:
                team_lookup[base_col] = team_lookup[base_col].combine_first(team_lookup[dup_col])
            else:
                team_lookup[base_col] = team_lookup[dup_col]
        team_lookup = team_lookup.drop(columns=dup_cols)

    team_lookup["team"] = standardize_team_name(team_lookup["team"])
    return team_lookup.drop_duplicates(subset=["team"])


def build_matches(matches_raw: pd.DataFrame, team_lookup: pd.DataFrame) -> pd.DataFrame:
    matches = matches_raw.copy()
    rename_map = {
        pick_column(matches, ["match_id", "编号"]): "match_id",
        pick_column(matches, ["stage", "阶段"]): "stage",
        pick_column(matches, ["taipei_time", "时间", "beijing"]): "match_time",
        pick_column(matches, ["team_a_input", "team_a_slot", "team_a", "a队"]): "team_a",
        pick_column(matches, ["team_b_input", "team_b_slot", "team_b", "b队"]): "team_b",
        pick_column(matches, ["match_status", "状态"]): "match_status",
        pick_column(matches, ["venue_city", "地点"]): "venue_city",
        pick_column(matches, ["venue_country", "国家"]): "venue_country",
    }
    rename_map = {old: new for old, new in rename_map.items() if old}
    matches = matches.rename(columns=rename_map)

    if "team_a" not in matches.columns:
        matches["team_a"] = ""
    if "team_b" not in matches.columns:
        matches["team_b"] = ""
    matches["team_a"] = standardize_team_name(matches["team_a"])
    matches["team_b"] = standardize_team_name(matches["team_b"])

    lookup_a = team_lookup.add_prefix("a_")
    lookup_b = team_lookup.add_prefix("b_")
    matches = matches.merge(lookup_a, left_on="team_a", right_on="a_team", how="left")
    matches = matches.merge(lookup_b, left_on="team_b", right_on="b_team", how="left")

    def ensure_numeric_pair(a_col: str, b_col: str, target_diff: str | None = None) -> None:
        if a_col in matches.columns:
            matches[a_col] = coerce_numeric(matches[a_col])
        if b_col in matches.columns:
            matches[b_col] = coerce_numeric(matches[b_col])
        if target_diff:
            if target_diff not in matches.columns:
                matches[target_diff] = np.nan
            matches[target_diff] = coerce_numeric(matches[target_diff])
            if a_col in matches.columns and b_col in matches.columns:
                matches[target_diff] = matches[target_diff].fillna(matches[a_col] - matches[b_col])

    rename_numeric = {
        pick_column(matches, ["rank_a", "fifa_rank_a"]): "rank_a",
        pick_column(matches, ["rank_b", "fifa_rank_b"]): "rank_b",
        pick_column(matches, ["rank_adv", "rank_diff"]): "rank_diff",
        pick_column(matches, ["elo_rating_a", "elo_a"]): "elo_a",
        pick_column(matches, ["elo_rating_b", "elo_b"]): "elo_b",
        pick_column(matches, ["elo_diff", "elo_adv"]): "elo_diff",
        pick_column(matches, ["squad_value_a", "value_a", "market_value_a"]): "squad_value_a",
        pick_column(matches, ["squad_value_b", "value_b", "market_value_b"]): "squad_value_b",
        pick_column(matches, ["squad_value_diff", "value_diff", "market_value_diff"]): "squad_value_diff",
        pick_column(matches, ["adjusted_form_score_a", "form_a", "recent_form_a"]): "form_a",
        pick_column(matches, ["adjusted_form_score_b", "form_b", "recent_form_b"]): "form_b",
        pick_column(matches, ["adjusted_form_score_diff", "form_diff", "recent_form_diff"]): "form_diff",
        pick_column(matches, ["team_a_offfield_capital_score", "offfield_a"]): "offfield_a",
        pick_column(matches, ["team_b_offfield_capital_score", "offfield_b"]): "offfield_b",
        pick_column(matches, ["offfield_diff", "offfield_capital_diff"]): "offfield_diff",
        pick_column(matches, ["seed_prob", "model_prob", "prior_prob_a"]): "seed_prob_a",
        pick_column(matches, ["host_advantage", "host_adv"]): "host_advantage",
    }
    rename_numeric = {old: new for old, new in rename_numeric.items() if old}
    matches = matches.rename(columns=rename_numeric)

    fallback_map = {
        "rank_a": "a_fifa_rank",
        "rank_b": "b_fifa_rank",
        "elo_a": "a_elo_rating",
        "elo_b": "b_elo_rating",
        "squad_value_a": "a_squad_value_eur_m",
        "squad_value_b": "b_squad_value_eur_m",
        "form_a": "a_adjusted_form_score",
        "form_b": "b_adjusted_form_score",
        "offfield_a": "a_offfield_capital_score",
        "offfield_b": "b_offfield_capital_score",
    }
    for target, source in fallback_map.items():
        if target not in matches.columns:
            matches[target] = np.nan
        if source in matches.columns:
            matches[target] = coerce_numeric(matches[target]).fillna(coerce_numeric(matches[source]))

    level_cols = ["rank_a", "rank_b", "elo_a", "elo_b", "squad_value_a", "squad_value_b", "form_a", "form_b", "offfield_a", "offfield_b"]
    diff_cols = ["rank_diff", "elo_diff", "squad_value_diff", "form_diff", "offfield_diff"]
    for col in level_cols:
        if col not in matches.columns:
            matches[col] = np.nan
        matches[col] = coerce_numeric(matches[col])
        median = matches[col].median(skipna=True)
        matches[col] = matches[col].fillna(0.0 if pd.isna(median) else median)

    ensure_numeric_pair("rank_a", "rank_b", "rank_diff")
    ensure_numeric_pair("elo_a", "elo_b", "elo_diff")
    ensure_numeric_pair("squad_value_a", "squad_value_b", "squad_value_diff")
    ensure_numeric_pair("form_a", "form_b", "form_diff")
    ensure_numeric_pair("offfield_a", "offfield_b", "offfield_diff")
    for col in diff_cols:
        if col not in matches.columns:
            matches[col] = 0.0
        matches[col] = coerce_numeric(matches[col]).fillna(0.0)

    if "host_advantage" not in matches.columns:
        matches["host_advantage"] = np.nan
    matches["host_advantage"] = coerce_numeric(matches["host_advantage"])
    if "a_is_host" in matches.columns:
        host_fill = pd.Series(
            np.where(matches["a_is_host"].astype(str).str.lower().isin(["是", "yes", "true", "1"]), 1.0, 0.0),
            index=matches.index,
        )
    else:
        host_fill = pd.Series(0.0, index=matches.index)
    matches["host_advantage"] = matches["host_advantage"].fillna(host_fill)

    if "seed_prob_a" in matches.columns:
        matches["seed_prob_a"] = coerce_numeric(matches["seed_prob_a"])
        matches["seed_prob_a"] = np.where(matches["seed_prob_a"] > 1, matches["seed_prob_a"] / 100.0, matches["seed_prob_a"])
    else:
        matches["seed_prob_a"] = np.nan

    if "match_time" in matches.columns:
        matches["match_time"] = pd.to_datetime(matches["match_time"], errors="coerce")
    else:
        matches["match_time"] = pd.NaT

    if "match_id" not in matches.columns:
        matches["match_id"] = matches.index.astype(str)
    if "stage" not in matches.columns:
        matches["stage"] = ""
    if "match_status" not in matches.columns:
        matches["match_status"] = ""

    matches["label"] = matches.apply(
        lambda row: f"{row['match_id']} | {row['team_a']} vs {row['team_b']} | {row['stage']}",
        axis=1,
    )
    return matches


def load_repository(base_dir: Path | None = None) -> DataRepository:
    warnings: list[str] = []
    future_path = find_data_file("future", base_dir)
    recent_path = find_data_file("recent", base_dir)
    offfield_path = find_data_file("offfield", base_dir)
    static_path = find_data_file("static", base_dir)

    if future_path is None or recent_path is None or offfield_path is None:
        missing = [name for name, path in [("future", future_path), ("recent", recent_path), ("offfield", offfield_path)] if path is None]
        raise FileNotFoundError(f"缺少必要数据文件: {', '.join(missing)}")

    future_book = read_workbook(future_path)
    recent_book = read_workbook(recent_path)
    offfield_book = read_workbook(offfield_path)
    static_book = read_workbook(static_path) if static_path else {}

    matches_sheet = find_sheet_name(future_book, ["future_match", "未来比赛", "11"])
    team_sheet = find_sheet_name(future_book, ["team_metrics", "metrics", "实力"])
    risk_sheet = find_sheet_name(future_book, ["risk_profiles", "risk", "风险"])
    recent_sheet = find_sheet_name(recent_book, ["recent_form_summary", "form_summary", "状态汇总", "10"])
    offfield_sheet = find_sheet_name(offfield_book, ["offfield_capital_factors", "offfield", "capital"])
    static_sheet = find_sheet_name(static_book, ["current_strength", "静态实力", "strength"]) if static_book else None

    if not matches_sheet:
        raise ValueError("未来比赛工作簿中未找到比赛特征 sheet")

    team_metrics = future_book.get(team_sheet, pd.DataFrame()).copy() if team_sheet else pd.DataFrame()
    recent_summary = recent_book.get(recent_sheet, pd.DataFrame()).copy() if recent_sheet else pd.DataFrame()
    offfield = offfield_book.get(offfield_sheet, pd.DataFrame()).copy() if offfield_sheet else pd.DataFrame()
    static_strength = static_book.get(static_sheet, pd.DataFrame()).copy() if static_sheet else pd.DataFrame()
    risk_profiles = future_book.get(risk_sheet, pd.DataFrame()).copy() if risk_sheet else pd.DataFrame()

    if team_metrics.empty:
        warnings.append("未找到 team_metrics，已尝试使用静态实力表兜底。")
    if recent_summary.empty:
        warnings.append("未找到 recent_form_summary，近期状态将回退为 0。")
    if offfield.empty:
        warnings.append("未找到 off-field 表，场外资本因子将回退为 0。")

    team_metrics, recent_summary, offfield, static_strength = prepare_team_tables(
        team_metrics, recent_summary, offfield, static_strength
    )
    team_lookup = build_team_lookup(team_metrics, recent_summary, offfield, static_strength)
    matches = build_matches(future_book[matches_sheet], team_lookup)

    for col in ["form_a", "form_b", "offfield_a", "offfield_b", "host_advantage"]:
        if col in matches.columns:
            matches[col] = matches[col].fillna(0.0)

    return DataRepository(
        matches=matches.sort_values(["match_time", "match_id"], na_position="last").reset_index(drop=True),
        team_metrics=team_metrics,
        recent_summary=recent_summary,
        offfield=offfield,
        static_strength=static_strength,
        risk_profiles=risk_profiles,
        warnings=warnings,
    )
