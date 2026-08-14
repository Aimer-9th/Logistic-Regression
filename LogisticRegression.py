import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# 0. 기본 설정
# =============================================================================

warnings.filterwarnings("ignore")

DATA_DIR = "/content/drive/MyDrive/𝟐𝟎𝟐𝟔/aimers/9기/open/data"
OUTPUT_DIR = "./eda_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

ID = "row_id"
TARGET = "control_success"
SEASON_COL = "season"
PITCHER_ID_COL = "pitcher_id"

SEASONS = list(range(2019, 2025))


# =============================================================================
# 1. 데이터 로드
# =============================================================================

test_columns = pd.read_csv(
    os.path.join(DATA_DIR, "test.csv"),
    encoding="utf-8-sig",
    nrows=0,
).columns

FEATURES = [
    col
    for col in test_columns
    if col != ID
]

train = pd.read_csv(
    os.path.join(DATA_DIR, "train.csv"),
    encoding="utf-8-sig",
    usecols=FEATURES + [TARGET],
)


# =============================================================================
# 2. Feature Type 정의
#
# 참고 코드의 정의를 그대로 사용
# =============================================================================

CAT_COLS = [
    # 경기 정보
    "game_dayofweek",
    "top_bottom",
    "game_type",
    "base_state",

    # 선수 / 팀 정보
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
]


NUM_COLS = [
    # 시간 / 경기 진행
    "season",
    "game_month",
    "inning",

    # 카운트
    "balls_before",
    "strikes_before",
    "outs_before",

    # 점수
    "run_top_before",
    "run_bot_before",
    "run_total_before",
    "score_diff_home",
    "score_diff_pitcher_team",

    # 주자
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",

    # 경기 중요도
    "home_win_expectancy",
    "away_win_expectancy",
    "li",

    # 투수 누적 이력
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",

    # 투수 최근 경기 이력
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",

    # 타자 과거 이력
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",

    # 투수 구종 구성 이력
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]


# 주요 numeric feature
# 전체 numeric feature 통계는 별도로 저장하고,
# 시각화는 해석 가치가 높은 핵심 feature에 집중
KEY_NUMERIC_COLS = [
    "inning",
    "score_diff_pitcher_team",
    "home_win_expectancy",
    "li",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
]


# =============================================================================
# 3. Feature Type 검증
# =============================================================================

assert set(CAT_COLS).isdisjoint(set(NUM_COLS)), (
    "CAT_COLS와 NUM_COLS에 중복 컬럼이 있습니다."
)

assert set(CAT_COLS) | set(NUM_COLS) == set(FEATURES), (
    "모든 feature가 CAT_COLS 또는 NUM_COLS에 정확히 포함되어야 합니다."
)

assert set(train[TARGET].dropna().unique()).issubset({0, 1}), (
    f"{TARGET}은 binary target이어야 합니다."
)


# =============================================================================
# 4. 전체 데이터 구조 확인
# =============================================================================

print("=" * 80)
print("1. DATA OVERVIEW")
print("=" * 80)

print(f"Rows           : {train.shape[0]:,}")
print(f"Columns        : {train.shape[1]:,}")
print(f"Feature count  : {len(FEATURES):,}")
print(f"Categorical    : {len(CAT_COLS):,}")
print(f"Numeric        : {len(NUM_COLS):,}")
print(
    f"Season range   : "
    f"{train[SEASON_COL].min()} ~ {train[SEASON_COL].max()}"
)
print(f"Pitchers       : {train[PITCHER_ID_COL].nunique():,}")
print(f"Target rate    : {train[TARGET].mean():.4f}")


# =============================================================================
# 5. 시즌별 데이터 수 / Target Rate
# =============================================================================

season_summary = (
    train
    .groupby(SEASON_COL)[TARGET]
    .agg(
        rows="size",
        control_success_count="sum",
        control_success_rate="mean",
    )
    .reindex(SEASONS)
)

season_summary["row_ratio"] = (
    season_summary["rows"] / len(train)
)

print("\n" + "=" * 80)
print("2. SEASON SUMMARY")
print("=" * 80)
print(season_summary.round(4).to_string())

season_summary.to_csv(
    os.path.join(OUTPUT_DIR, "season_summary.csv"),
    encoding="utf-8-sig",
)


# =============================================================================
# 6. 전체 Target Distribution
# =============================================================================

target_summary = (
    train[TARGET]
    .value_counts(dropna=False)
    .sort_index()
    .rename("count")
    .to_frame()
)

target_summary["ratio"] = (
    target_summary["count"] / len(train)
)

print("\n" + "=" * 80)
print("3. TARGET DISTRIBUTION")
print("=" * 80)
print(target_summary.round(4).to_string())

target_summary.to_csv(
    os.path.join(OUTPUT_DIR, "target_distribution.csv"),
    encoding="utf-8-sig",
)


# Target distribution plot
fig, ax = plt.subplots(figsize=(6, 4))

target_counts = (
    train[TARGET]
    .value_counts()
    .sort_index()
)

ax.bar(
    target_counts.index.astype(str),
    target_counts.values,
)

ax.set_title("Control Success Distribution")
ax.set_xlabel("control_success")
ax.set_ylabel("Count")

for i, value in enumerate(target_counts.values):
    ax.text(
        i,
        value,
        f"{value:,}",
        ha="center",
        va="bottom",
    )

plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "target_distribution.png",
    ),
    dpi=150,
    bbox_inches="tight",
)
plt.show()


# =============================================================================
# 7. 시즌별 Target Rate 시각화
# =============================================================================

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(
    season_summary.index,
    season_summary["control_success_rate"],
    marker="o",
)

ax.set_title("Control Success Rate by Season")
ax.set_xlabel("Season")
ax.set_ylabel("Control Success Rate")
ax.set_xticks(SEASONS)

plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "season_target_rate.png",
    ),
    dpi=150,
    bbox_inches="tight",
)
plt.show()


# =============================================================================
# 8. Pitcher별 투구 수 분포
# =============================================================================

pitcher_summary = (
    train
    .groupby(PITCHER_ID_COL)[TARGET]
    .agg(
        pitch_count="size",
        success_count="sum",
        success_rate="mean",
    )
    .sort_values(
        "pitch_count",
        ascending=False,
    )
)

pitcher_count_stats = (
    pitcher_summary["pitch_count"]
    .describe(
        percentiles=[
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
)

n_pitchers = len(pitcher_summary)

top_10pct_n = max(
    1,
    int(np.ceil(n_pitchers * 0.10)),
)

top_10pct_pitch_share = (
    pitcher_summary
    .head(top_10pct_n)["pitch_count"]
    .sum()
    / pitcher_summary["pitch_count"].sum()
)


print("\n" + "=" * 80)
print("4. PITCHER DISTRIBUTION")
print("=" * 80)

print(f"Pitcher count                : {n_pitchers:,}")
print("\nPitch count statistics:")
print(pitcher_count_stats.round(2).to_string())

print(
    f"\nTop 10% pitchers' row share : "
    f"{top_10pct_pitch_share:.2%}"
)

pitcher_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "pitcher_summary.csv",
    ),
    encoding="utf-8-sig",
)


# Pitch count distribution
fig, ax = plt.subplots(figsize=(8, 4))

ax.hist(
    pitcher_summary["pitch_count"],
    bins=50,
)

ax.set_title("Pitch Count Distribution by Pitcher")
ax.set_xlabel("Number of Pitches")
ax.set_ylabel("Number of Pitchers")

plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "pitcher_pitch_count_distribution.png",
    ),
    dpi=150,
    bbox_inches="tight",
)
plt.show()


# =============================================================================
# 9. Pitcher별 Control Success Rate 분포
# =============================================================================

print("\n" + "=" * 80)
print("5. PITCHER CONTROL SUCCESS RATE")
print("=" * 80)

print(
    pitcher_summary["success_rate"]
    .describe(
        percentiles=[
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
        ]
    )
    .round(4)
    .to_string()
)


fig, ax = plt.subplots(figsize=(8, 4))

ax.hist(
    pitcher_summary["success_rate"],
    bins=30,
)

ax.set_title("Control Success Rate Distribution by Pitcher")
ax.set_xlabel("Control Success Rate")
ax.set_ylabel("Number of Pitchers")

plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "pitcher_success_rate_distribution.png",
    ),
    dpi=150,
    bbox_inches="tight",
)
plt.show()


# Pitch count와 success rate 관계
fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(
    pitcher_summary["pitch_count"],
    pitcher_summary["success_rate"],
    alpha=0.5,
)

ax.set_title("Pitch Count vs Control Success Rate")
ax.set_xlabel("Number of Pitches")
ax.set_ylabel("Control Success Rate")

plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "pitcher_count_vs_success_rate.png",
    ),
    dpi=150,
    bbox_inches="tight",
)
plt.show()


# =============================================================================
# 10. Numeric Feature 기본 분포
# =============================================================================

numeric_summary = (
    train[NUM_COLS]
    .describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    )
    .T
)

numeric_summary["missing_count"] = (
    train[NUM_COLS]
    .isna()
    .sum()
)

numeric_summary["missing_rate"] = (
    train[NUM_COLS]
    .isna()
    .mean()
)

numeric_summary["nunique"] = (
    train[NUM_COLS]
    .nunique()
)


print("\n" + "=" * 80)
print("6. NUMERIC FEATURE SUMMARY")
print("=" * 80)

print(
    numeric_summary[
        [
            "count",
            "mean",
            "std",
            "min",
            "1%",
            "50%",
            "99%",
            "max",
            "missing_rate",
            "nunique",
        ]
    ]
    .round(4)
    .to_string()
)

numeric_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "numeric_feature_summary.csv",
    ),
    encoding="utf-8-sig",
)


# 주요 numeric feature 전체 분포
for col in KEY_NUMERIC_COLS:

    if col not in train.columns:
        continue

    fig, ax = plt.subplots(figsize=(8, 4))

    values = train[col].dropna()

    ax.hist(
        values,
        bins=50,
    )

    ax.set_title(f"Distribution: {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"numeric_distribution_{col}.png",
        ),
        dpi=150,
        bbox_inches="tight",
    )

    plt.show()


# =============================================================================
# 11. Categorical Feature category 수 확인
# =============================================================================

categorical_summary = pd.DataFrame(
    {
        "nunique": train[CAT_COLS].nunique(dropna=True),
        "missing_count": train[CAT_COLS].isna().sum(),
        "missing_rate": train[CAT_COLS].isna().mean(),
    }
)

categorical_summary = (
    categorical_summary
    .sort_values(
        "nunique",
        ascending=False,
    )
)


print("\n" + "=" * 80)
print("7. CATEGORICAL FEATURE SUMMARY")
print("=" * 80)

print(
    categorical_summary
    .round(4)
    .to_string()
)

categorical_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "categorical_feature_summary.csv",
    ),
    encoding="utf-8-sig",
)


# =============================================================================
# 12. 결측치 비율 확인
# =============================================================================

missing_summary = pd.DataFrame(
    {
        "missing_count": train.isna().sum(),
        "missing_rate": train.isna().mean(),
    }
)

missing_summary = (
    missing_summary[
        missing_summary["missing_count"] > 0
    ]
    .sort_values(
        "missing_rate",
        ascending=False,
    )
)


print("\n" + "=" * 80)
print("8. MISSING VALUE SUMMARY")
print("=" * 80)

if len(missing_summary) == 0:
    print("결측치가 없습니다.")
else:
    print(
        missing_summary
        .round(4)
        .to_string()
    )

missing_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "missing_value_summary.csv",
    ),
    encoding="utf-8-sig",
)


if len(missing_summary) > 0:

    fig, ax = plt.subplots(
        figsize=(
            10,
            max(
                4,
                len(missing_summary) * 0.3,
            ),
        )
    )

    plot_missing = (
        missing_summary["missing_rate"]
        .sort_values()
    )

    ax.barh(
        plot_missing.index,
        plot_missing.values,
    )

    ax.set_title("Missing Rate by Feature")
    ax.set_xlabel("Missing Rate")
    ax.set_ylabel("Feature")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "missing_rate.png",
        ),
        dpi=150,
        bbox_inches="tight",
    )

    plt.show()


# =============================================================================
# 13. 결측 여부 자체가 Target 정보를 포함하는지 확인
#
# 각 결측 feature에 대해:
# - 값이 존재할 때 target rate
# - 값이 결측일 때 target rate
# - 두 집단의 target rate 차이
# =============================================================================

missing_target_rows = []

for col in missing_summary.index:

    missing_mask = train[col].isna()
    observed_mask = ~missing_mask

    if missing_mask.sum() == 0:
        continue

    missing_target_rate = (
        train.loc[
            missing_mask,
            TARGET,
        ]
        .mean()
    )

    observed_target_rate = (
        train.loc[
            observed_mask,
            TARGET,
        ]
        .mean()
    )

    missing_target_rows.append(
        {
            "feature": col,
            "missing_count": missing_mask.sum(),
            "missing_rate": missing_mask.mean(),
            "target_rate_missing": missing_target_rate,
            "target_rate_observed": observed_target_rate,
            "target_rate_diff":
                missing_target_rate
                - observed_target_rate,
        }
    )


missing_target_summary = pd.DataFrame(
    missing_target_rows
)

if not missing_target_summary.empty:

    missing_target_summary[
        "abs_target_rate_diff"
    ] = (
        missing_target_summary[
            "target_rate_diff"
        ]
        .abs()
    )

    missing_target_summary = (
        missing_target_summary
        .sort_values(
            "abs_target_rate_diff",
            ascending=False,
        )
    )


print("\n" + "=" * 80)
print("9. MISSINGNESS VS TARGET")
print("=" * 80)

if missing_target_summary.empty:
    print("분석할 결측 feature가 없습니다.")
else:
    print(
        missing_target_summary[
            [
                "feature",
                "missing_rate",
                "target_rate_missing",
                "target_rate_observed",
                "target_rate_diff",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

missing_target_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "missingness_target_summary.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


# =============================================================================
# 14. 시즌별 Numeric Feature 분포 확인
#
# 시즌별 mean / median / std를 저장한다.
# =============================================================================

season_numeric_summary = (
    train
    .groupby(SEASON_COL)[NUM_COLS]
    .agg(
        ["mean", "median", "std"]
    )
)

season_numeric_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "season_numeric_summary.csv",
    ),
    encoding="utf-8-sig",
)


# =============================================================================
# 15. 시즌별 Numeric Distribution Shift 후보 탐색
#
# 단순 평균 차이보다 이상치 영향이 적은 median을 사용한다.
#
# shift_score =
#   시즌별 median 최대 차이 / 전체 feature IQR
#
# 값이 클수록 시즌 간 중심 위치가 크게 달라졌을 가능성이 있다.
# 이는 통계적 검정이 아니라 EDA용 후보 탐색 지표이다.
# =============================================================================

shift_rows = []

for col in NUM_COLS:

    if col == SEASON_COL:
        continue

    values = train[col].dropna()

    if values.nunique() <= 1:
        continue

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    iqr = q3 - q1

    season_medians = (
        train
        .groupby(SEASON_COL)[col]
        .median()
        .dropna()
    )

    if len(season_medians) < 2:
        continue

    median_range = (
        season_medians.max()
        - season_medians.min()
    )

    if iqr > 0:
        shift_score = median_range / iqr
    else:
        shift_score = np.nan

    shift_rows.append(
        {
            "feature": col,
            "overall_q1": q1,
            "overall_q3": q3,
            "overall_iqr": iqr,
            "min_season_median": season_medians.min(),
            "max_season_median": season_medians.max(),
            "season_median_range": median_range,
            "shift_score": shift_score,
        }
    )


shift_summary = (
    pd.DataFrame(shift_rows)
    .sort_values(
        "shift_score",
        ascending=False,
        na_position="last",
    )
    .reset_index(drop=True)
)


print("\n" + "=" * 80)
print("10. NUMERIC DISTRIBUTION SHIFT CANDIDATES")
print("=" * 80)

print(
    shift_summary
    .head(15)
    .round(4)
    .to_string(index=False)
)

shift_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "numeric_distribution_shift_candidates.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


# =============================================================================
# 16. 주요 Numeric Feature 시즌별 분포 시각화
#
# 각 시즌의 boxplot을 비교한다.
# =============================================================================

for col in KEY_NUMERIC_COLS:

    if col not in train.columns:
        continue

    season_values = []

    season_labels = []

    for season in SEASONS:

        values = (
            train.loc[
                train[SEASON_COL] == season,
                col,
            ]
            .dropna()
        )

        if len(values) == 0:
            continue

        season_values.append(values)
        season_labels.append(str(season))


    if len(season_values) < 2:
        continue


    fig, ax = plt.subplots(figsize=(9, 5))

    ax.boxplot(
        season_values,
        labels=season_labels,
        showfliers=False,
    )

    ax.set_title(
        f"{col} Distribution by Season"
    )

    ax.set_xlabel("Season")
    ax.set_ylabel(col)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"season_distribution_{col}.png",
        ),
        dpi=150,
        bbox_inches="tight",
    )

    plt.show()


# =============================================================================
# 17. 이상치 후보 탐색
#
# IQR Rule:
# x < Q1 - 1.5 * IQR
# x > Q3 + 1.5 * IQR
#
# binary / 거의 범주형인 numeric 변수는 제외
# =============================================================================

outlier_rows = []

for col in NUM_COLS:

    if col == SEASON_COL:
        continue

    values = train[col].dropna()

    # 0/1 indicator 등은 IQR outlier 분석 대상에서 제외
    if values.nunique() <= 2:
        continue

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    iqr = q3 - q1

    if iqr == 0:
        continue

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (
        (train[col] < lower_bound)
        | (train[col] > upper_bound)
    )

    outlier_count = int(
        outlier_mask.sum()
    )

    non_missing_count = int(
        train[col].notna().sum()
    )

    outlier_rate = (
        outlier_count / non_missing_count
        if non_missing_count > 0
        else np.nan
    )

    outlier_rows.append(
        {
            "feature": col,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_count": outlier_count,
            "outlier_rate": outlier_rate,
            "min": values.min(),
            "max": values.max(),
        }
    )


outlier_summary = (
    pd.DataFrame(outlier_rows)
    .sort_values(
        "outlier_rate",
        ascending=False,
    )
    .reset_index(drop=True)
)


print("\n" + "=" * 80)
print("11. OUTLIER CANDIDATES")
print("=" * 80)

print(
    outlier_summary
    .head(20)
    .round(4)
    .to_string(index=False)
)

outlier_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "outlier_candidates.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


# =============================================================================
# 18. PR 기록용 핵심 요약
# =============================================================================

season_target_min = (
    season_summary[
        "control_success_rate"
    ]
    .min()
)

season_target_max = (
    season_summary[
        "control_success_rate"
    ]
    .max()
)

season_target_gap = (
    season_target_max
    - season_target_min
)

pitch_count_median = (
    pitcher_summary["pitch_count"]
    .median()
)

pitch_count_p90 = (
    pitcher_summary["pitch_count"]
    .quantile(0.90)
)

pitch_count_max = (
    pitcher_summary["pitch_count"]
    .max()
)


print("\n" + "=" * 80)
print("12. PR SUMMARY")
print("=" * 80)

print("\n[시즌별 데이터 수]")
print(
    season_summary[
        ["rows"]
    ]
    .to_string()
)

print("\n[시즌별 target rate]")
print(
    season_summary[
        ["control_success_rate"]
    ]
    .round(4)
    .to_string()
)

print(
    f"\nTarget rate max-min gap : "
    f"{season_target_gap:.4f}"
)

print(
    f"Pitcher count           : "
    f"{n_pitchers:,}"
)

print(
    f"Pitch count median      : "
    f"{pitch_count_median:,.1f}"
)

print(
    f"Pitch count p90         : "
    f"{pitch_count_p90:,.1f}"
)

print(
    f"Pitch count max         : "
    f"{pitch_count_max:,}"
)

print(
    f"Top 10% pitcher share   : "
    f"{top_10pct_pitch_share:.2%}"
)


print("\n[주요 결측 feature]")

if missing_summary.empty:
    print("없음")
else:
    print(
        missing_summary
        .head(10)
        .round(4)
        .to_string()
    )


print("\n[Distribution shift 후보]")

if shift_summary.empty:
    print("없음")
else:
    print(
        shift_summary[
            [
                "feature",
                "shift_score",
            ]
        ]
        .head(10)
        .round(4)
        .to_string(index=False)
    )


print("\n[결측 여부와 Target 관계 후보]")

if missing_target_summary.empty:
    print("없음")
else:
    print(
        missing_target_summary[
            [
                "feature",
                "missing_rate",
                "target_rate_diff",
            ]
        ]
        .head(10)
        .round(4)
        .to_string(index=False)
    )


print("\n[이상치 후보]")

if outlier_summary.empty:
    print("없음")
else:
    print(
        outlier_summary[
            [
                "feature",
                "outlier_count",
                "outlier_rate",
            ]
        ]
        .head(10)
        .round(4)
        .to_string(index=False)
    )


print("\n" + "=" * 80)
print("EDA COMPLETE")
print("=" * 80)

print(f"산출물 저장 위치: {OUTPUT_DIR}")