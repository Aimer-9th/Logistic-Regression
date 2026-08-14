import os
import warnings

import numpy as np
import pandas as pd


# =============================================================================
# 0. 기본 설정
# =============================================================================

warnings.filterwarnings("ignore")

DATA_DIR = "/content/drive/MyDrive/𝟐𝟎𝟐𝟔/aimers/9기/open/data"
OUTPUT_DIR = "./lr02_validation_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

ID = "row_id"
TARGET = "control_success"
SEASON_COL = "season"
PITCHER_ID_COL = "pitcher_id"

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")


# =============================================================================
# 1. 데이터 로드
# =============================================================================

train = pd.read_csv(
    TRAIN_PATH,
    encoding="utf-8-sig",
)

test = pd.read_csv(
    TEST_PATH,
    encoding="utf-8-sig",
)


print("=" * 80)
print("1. DATA LOAD")
print("=" * 80)

print(f"Train shape : {train.shape}")
print(f"Test shape  : {test.shape}")

print(
    f"Train season: "
    f"{train[SEASON_COL].min()} ~ {train[SEASON_COL].max()}"
)

print(
    f"Test season : "
    f"{test[SEASON_COL].min()} ~ {test[SEASON_COL].max()}"
)


# =============================================================================
# 2. Target 검증
#
# - control_success가 0/1인지
# - 결측치 존재 여부
# - class ratio
# =============================================================================

print("\n" + "=" * 80)
print("2. TARGET VALIDATION")
print("=" * 80)

target_unique = sorted(
    train[TARGET]
    .dropna()
    .unique()
    .tolist()
)

target_missing_count = int(
    train[TARGET]
    .isna()
    .sum()
)

target_missing_rate = (
    train[TARGET]
    .isna()
    .mean()
)

target_counts = (
    train[TARGET]
    .value_counts(
        dropna=False,
    )
    .sort_index()
)

target_ratio = (
    train[TARGET]
    .value_counts(
        normalize=True,
        dropna=False,
    )
    .sort_index()
)


target_is_binary = (
    set(target_unique)
    == {0, 1}
)

print(f"Target unique values : {target_unique}")
print(f"Binary target        : {target_is_binary}")

print(
    f"Target missing       : "
    f"{target_missing_count:,} "
    f"({target_missing_rate:.4%})"
)

print("\nClass count:")
print(target_counts.to_string())

print("\nClass ratio:")
print(target_ratio.round(6).to_string())


target_summary = pd.DataFrame(
    {
        "count": target_counts,
        "ratio": target_ratio,
    }
)

target_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "target_summary.csv",
    ),
    encoding="utf-8-sig",
)


# =============================================================================
# 3. 중복 투구 확인
#
# 1차:
#   전체 row 기준 완전 중복
#
# 2차:
#   row_id 중복
#
# 실제 동일 투구 식별 key는 아래에서 별도 탐색
# =============================================================================

print("\n" + "=" * 80)
print("3. DUPLICATE VALIDATION")
print("=" * 80)


full_duplicate_count = int(
    train
    .duplicated()
    .sum()
)

print(
    f"Full-row duplicates : "
    f"{full_duplicate_count:,}"
)


if ID in train.columns:

    row_id_duplicate_count = int(
        train[ID]
        .duplicated()
        .sum()
    )

    print(
        f"{ID} duplicates       : "
        f"{row_id_duplicate_count:,}"
    )

else:

    row_id_duplicate_count = np.nan

    print(
        f"{ID}                : "
        "train에 존재하지 않음"
    )


# =============================================================================
# 4. 동일 투구 식별 Key 후보 탐색
#
# row_id가 유일한 식별자인지 우선 검증한다.
#
# 이후 실제 경기/투구 정보를 조합한 후보 key도 검사한다.
# 존재하지 않는 컬럼은 자동 제외한다.
# =============================================================================

print("\n" + "=" * 80)
print("4. PITCH KEY CANDIDATE VALIDATION")
print("=" * 80)


key_candidates = []


# -----------------------------------------------------------------------------
# Candidate 1: row_id
# -----------------------------------------------------------------------------

if ID in train.columns:

    key_candidates.append(
        {
            "key_name": "row_id",
            "columns": [ID],
        }
    )


# -----------------------------------------------------------------------------
# Candidate 2:
# 데이터에 경기 식별자가 존재할 경우 가능한 조합
#
# 실제 train schema에 존재하는 컬럼만 사용
# -----------------------------------------------------------------------------

possible_game_key_cols = [
    "season",
    "game_id",
    "pitcher_id",
    "batter_id",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
]

available_game_key_cols = [
    col
    for col in possible_game_key_cols
    if col in train.columns
]

if len(available_game_key_cols) >= 2:

    key_candidates.append(
        {
            "key_name": "available_game_context",
            "columns": available_game_key_cols,
        }
    )


key_validation_rows = []


for candidate in key_candidates:

    cols = candidate["columns"]

    duplicate_count = int(
        train
        .duplicated(
            subset=cols,
            keep=False,
        )
        .sum()
    )

    unique_count = (
        train[cols]
        .drop_duplicates()
        .shape[0]
    )

    is_unique = (
        unique_count
        == len(train)
    )

    key_validation_rows.append(
        {
            "key_name":
                candidate["key_name"],

            "columns":
                ", ".join(cols),

            "n_columns":
                len(cols),

            "unique_rows":
                unique_count,

            "duplicate_rows":
                duplicate_count,

            "is_unique":
                is_unique,
        }
    )


key_validation_df = pd.DataFrame(
    key_validation_rows
)


if len(key_validation_df) > 0:

    print(
        key_validation_df
        .to_string(
            index=False,
        )
    )

else:

    print(
        "검증 가능한 key 후보가 없습니다."
    )


key_validation_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "pitch_key_candidates.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


# =============================================================================
# 5. Train / Test Schema 비교
#
# TARGET은 train에만 존재하는 것이 정상
# =============================================================================

print("\n" + "=" * 80)
print("5. TRAIN / TEST SCHEMA")
print("=" * 80)


train_cols = set(
    train.columns
)

test_cols = set(
    test.columns
)


train_only_cols = sorted(
    train_cols
    - test_cols
)

test_only_cols = sorted(
    test_cols
    - train_cols
)


print(
    f"Train columns : "
    f"{len(train.columns)}"
)

print(
    f"Test columns  : "
    f"{len(test.columns)}"
)

print(
    f"\nTrain only    : "
    f"{train_only_cols}"
)

print(
    f"Test only     : "
    f"{test_only_cols}"
)


expected_train_only = {
    TARGET,
}

unexpected_train_only = sorted(
    set(train_only_cols)
    - expected_train_only
)

unexpected_test_only = (
    test_only_cols
)


schema_valid = (
    len(unexpected_train_only) == 0
    and len(unexpected_test_only) == 0
)


print(
    f"\nExpected train-only columns : "
    f"{sorted(expected_train_only)}"
)

print(
    f"Unexpected train-only       : "
    f"{unexpected_train_only}"
)

print(
    f"Unexpected test-only        : "
    f"{unexpected_test_only}"
)

print(
    f"Schema valid                : "
    f"{schema_valid}"
)


schema_summary = pd.DataFrame(
    [
        {
            "item": "train_column_count",
            "value": len(train.columns),
        },
        {
            "item": "test_column_count",
            "value": len(test.columns),
        },
        {
            "item": "train_only",
            "value": ", ".join(train_only_cols),
        },
        {
            "item": "test_only",
            "value": ", ".join(test_only_cols),
        },
        {
            "item": "schema_valid",
            "value": schema_valid,
        },
    ]
)

schema_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "schema_summary.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


# =============================================================================
# 6. dtype 비교
#
# train/test 공통 컬럼 중 dtype이 다른 컬럼 확인
# =============================================================================

print("\n" + "=" * 80)
print("6. DTYPE VALIDATION")
print("=" * 80)


common_cols = sorted(
    train_cols
    & test_cols
)


dtype_rows = []


for col in common_cols:

    train_dtype = str(
        train[col].dtype
    )

    test_dtype = str(
        test[col].dtype
    )

    dtype_rows.append(
        {
            "column": col,
            "train_dtype": train_dtype,
            "test_dtype": test_dtype,
            "same_dtype":
                train_dtype
                == test_dtype,
        }
    )


dtype_comparison = pd.DataFrame(
    dtype_rows
)


dtype_mismatch = (
    dtype_comparison[
        ~dtype_comparison[
            "same_dtype"
        ]
    ]
    .copy()
)


if dtype_mismatch.empty:

    print(
        "Train/Test dtype 차이 없음"
    )

else:

    print(
        dtype_mismatch
        .to_string(
            index=False,
        )
    )


dtype_comparison.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "dtype_comparison.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


# =============================================================================
# 7. Missing Value 비교
#
# train/test 각각의 결측률을 확인
# =============================================================================

print("\n" + "=" * 80)
print("7. MISSING VALUE VALIDATION")
print("=" * 80)


missing_rows = []


for col in common_cols:

    train_missing_count = int(
        train[col]
        .isna()
        .sum()
    )

    test_missing_count = int(
        test[col]
        .isna()
        .sum()
    )

    train_missing_rate = (
        train[col]
        .isna()
        .mean()
    )

    test_missing_rate = (
        test[col]
        .isna()
        .mean()
    )

    missing_rows.append(
        {
            "column": col,

            "train_missing_count":
                train_missing_count,

            "train_missing_rate":
                train_missing_rate,

            "test_missing_count":
                test_missing_count,

            "test_missing_rate":
                test_missing_rate,

            "missing_rate_diff":
                test_missing_rate
                - train_missing_rate,
        }
    )


missing_comparison = (
    pd.DataFrame(
        missing_rows
    )
    .sort_values(
        "test_missing_rate",
        ascending=False,
    )
    .reset_index(
        drop=True,
    )
)


missing_present = (
    missing_comparison[
        (
            missing_comparison[
                "train_missing_count"
            ]
            > 0
        )
        |
        (
            missing_comparison[
                "test_missing_count"
            ]
            > 0
        )
    ]
)


if missing_present.empty:

    print(
        "Train/Test 공통 feature에 "
        "결측치 없음"
    )

else:

    print(
        missing_present
        .round(6)
        .to_string(
            index=False,
        )
    )


missing_comparison.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "missing_value_comparison.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


# =============================================================================
# 8. Historical Feature NaN 패턴
#
# asof_ prefix를 historical feature로 정의
#
# - 각 feature 결측률
# - 한 row에 historical NaN이 몇 개 존재하는지
# - season별 historical NaN 비율
# - pitcher의 첫 등장/적은 history와 연관 가능성 확인
# =============================================================================

print("\n" + "=" * 80)
print("8. HISTORICAL FEATURE NaN PATTERN")
print("=" * 80)


historical_cols = [
    col
    for col in common_cols
    if col.startswith("asof_")
]


print(
    f"Historical feature count : "
    f"{len(historical_cols)}"
)


historical_missing_rows = []


for col in historical_cols:

    historical_missing_rows.append(
        {
            "feature": col,

            "train_missing_count":
                int(
                    train[col]
                    .isna()
                    .sum()
                ),

            "train_missing_rate":
                train[col]
                .isna()
                .mean(),

            "test_missing_count":
                int(
                    test[col]
                    .isna()
                    .sum()
                ),

            "test_missing_rate":
                test[col]
                .isna()
                .mean(),
        }
    )


historical_missing_summary = (
    pd.DataFrame(
        historical_missing_rows
    )
    .sort_values(
        "train_missing_rate",
        ascending=False,
    )
)


print("\n[Historical feature missing rate]")

if historical_missing_summary.empty:

    print(
        "Historical feature 없음"
    )

else:

    print(
        historical_missing_summary
        .round(6)
        .to_string(
            index=False,
        )
    )


historical_missing_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "historical_missing_summary.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


# -----------------------------------------------------------------------------
# Row별 historical NaN 개수
# -----------------------------------------------------------------------------

if historical_cols:

    train_hist_nan_count = (
        train[
            historical_cols
        ]
        .isna()
        .sum(
            axis=1,
        )
    )

    test_hist_nan_count = (
        test[
            historical_cols
        ]
        .isna()
        .sum(
            axis=1,
        )
    )


    print(
        "\n[Train historical NaN count per row]"
    )

    print(
        train_hist_nan_count
        .describe(
            percentiles=[
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
        .round(2)
        .to_string()
    )


    print(
        "\n[Test historical NaN count per row]"
    )

    print(
        test_hist_nan_count
        .describe(
            percentiles=[
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
        .round(2)
        .to_string()
    )


    # -------------------------------------------------------------------------
    # 시즌별 historical NaN 비율
    # -------------------------------------------------------------------------

    train_hist_missing_rate_by_season = (
        train
        .assign(
            historical_nan_count=
                train_hist_nan_count
        )
        .groupby(
            SEASON_COL
        )[
            "historical_nan_count"
        ]
        .agg(
            rows="size",
            mean_nan_count="mean",
            median_nan_count="median",
            rows_with_nan=lambda x:
                (x > 0).sum(),
        )
    )

    train_hist_missing_rate_by_season[
        "rows_with_nan_rate"
    ] = (
        train_hist_missing_rate_by_season[
            "rows_with_nan"
        ]
        / train_hist_missing_rate_by_season[
            "rows"
        ]
    )


    print(
        "\n[Historical NaN by train season]"
    )

    print(
        train_hist_missing_rate_by_season
        .round(4)
        .to_string()
    )


    train_hist_missing_rate_by_season.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "historical_nan_by_season.csv",
        ),
        encoding="utf-8-sig",
    )


# =============================================================================
# 9. Pitcher Train / Test Overlap
# =============================================================================

print("\n" + "=" * 80)
print("9. PITCHER TRAIN / TEST OVERLAP")
print("=" * 80)


train_pitchers = set(
    train[
        PITCHER_ID_COL
    ]
    .dropna()
    .unique()
)

test_pitchers = set(
    test[
        PITCHER_ID_COL
    ]
    .dropna()
    .unique()
)


known_pitchers = (
    train_pitchers
    & test_pitchers
)

unknown_pitchers = (
    test_pitchers
    - train_pitchers
)

train_only_pitchers = (
    train_pitchers
    - test_pitchers
)


print(
    f"Train pitchers       : "
    f"{len(train_pitchers):,}"
)

print(
    f"Test pitchers        : "
    f"{len(test_pitchers):,}"
)

print(
    f"Known test pitchers  : "
    f"{len(known_pitchers):,}"
)

print(
    f"Unknown test pitchers: "
    f"{len(unknown_pitchers):,}"
)

print(
    f"Train-only pitchers  : "
    f"{len(train_only_pitchers):,}"
)


pitcher_overlap_rate = (
    len(known_pitchers)
    / len(test_pitchers)
    if len(test_pitchers) > 0
    else np.nan
)


print(
    f"Pitcher overlap rate : "
    f"{pitcher_overlap_rate:.2%}"
)


# =============================================================================
# 10. Unknown Pitcher 목록 및 통계
# =============================================================================

print("\n" + "=" * 80)
print("10. UNKNOWN PITCHER")
print("=" * 80)


unknown_mask = (
    test[
        PITCHER_ID_COL
    ]
    .isin(
        unknown_pitchers
    )
)


unknown_rows = int(
    unknown_mask
    .sum()
)

unknown_row_rate = (
    unknown_mask
    .mean()
)


print(
    f"Unknown pitcher rows : "
    f"{unknown_rows:,}"
)

print(
    f"Unknown row rate     : "
    f"{unknown_row_rate:.2%}"
)


unknown_pitcher_summary = (
    test.loc[
        unknown_mask
    ]
    .groupby(
        PITCHER_ID_COL
    )
    .size()
    .rename(
        "test_rows"
    )
    .sort_values(
        ascending=False,
    )
    .reset_index()
)


print("\nUnknown pitcher list:")

if unknown_pitcher_summary.empty:

    print(
        "Unknown Pitcher 없음"
    )

else:

    print(
        unknown_pitcher_summary
        .to_string(
            index=False,
        )
    )


unknown_pitcher_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "unknown_pitchers.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


# =============================================================================
# 11. Known vs Unknown Pitcher의 Historical NaN 비교
#
# Unknown Pitcher에서 historical feature가 실제로 더 많이 비어 있는지 확인
# =============================================================================

if historical_cols:

    test_with_pitcher_status = (
        test[
            [
                PITCHER_ID_COL,
                SEASON_COL,
            ]
        ]
        .copy()
    )

    test_with_pitcher_status[
        "pitcher_status"
    ] = np.where(
        unknown_mask,
        "Unknown",
        "Known",
    )

    test_with_pitcher_status[
        "historical_nan_count"
    ] = (
        test[
            historical_cols
        ]
        .isna()
        .sum(
            axis=1,
        )
    )

    pitcher_nan_summary = (
        test_with_pitcher_status
        .groupby(
            "pitcher_status"
        )[
            "historical_nan_count"
        ]
        .agg(
            rows="size",
            mean_nan_count="mean",
            median_nan_count="median",
            rows_with_nan=lambda x:
                (x > 0).sum(),
        )
    )

    pitcher_nan_summary[
        "rows_with_nan_rate"
    ] = (
        pitcher_nan_summary[
            "rows_with_nan"
        ]
        / pitcher_nan_summary[
            "rows"
        ]
    )


    print("\n" + "=" * 80)
    print("11. HISTORICAL NaN: KNOWN vs UNKNOWN")
    print("=" * 80)

    print(
        pitcher_nan_summary
        .round(4)
        .to_string()
    )


    pitcher_nan_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "known_unknown_historical_nan.csv",
        ),
        encoding="utf-8-sig",
    )


# =============================================================================
# 12. 주요 이상값 후보 확인
#
# LR-01 결정에 따라 이상치를 일괄 제거하지 않는다.
#
# 여기서는 데이터 품질 오류 가능성이 높은 값만 점검한다.
# =============================================================================

print("\n" + "=" * 80)
print("12. BASIC INVALID VALUE CHECK")
print("=" * 80)


invalid_checks = {}


# -----------------------------------------------------------------------------
# Binary columns
# -----------------------------------------------------------------------------

binary_cols = [
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
]

for col in binary_cols:

    if col not in train.columns:
        continue

    invalid_count = (
        ~train[col]
        .isna()
        &
        ~train[col]
        .isin(
            [0, 1]
        )
    ).sum()

    invalid_checks[
        f"{col}_not_binary"
    ] = int(
        invalid_count
    )


# -----------------------------------------------------------------------------
# Baseball count 범위
# -----------------------------------------------------------------------------

range_rules = {
    "balls_before": (0, 3),
    "strikes_before": (0, 2),
    "outs_before": (0, 2),
}


for col, (
    min_value,
    max_value,
) in range_rules.items():

    if col not in train.columns:
        continue

    invalid_count = (
        (
            train[col]
            < min_value
        )
        |
        (
            train[col]
            > max_value
        )
    ).sum()

    invalid_checks[
        f"{col}_outside_{min_value}_{max_value}"
    ] = int(
        invalid_count
    )


# -----------------------------------------------------------------------------
# Probability / rate feature
#
# *_rate 및 expectancy는 원칙적으로 [0, 1] 범위인지 확인
# -----------------------------------------------------------------------------

probability_cols = [
    col
    for col in train.columns
    if (
        col.endswith("_rate")
        or col.endswith("_expectancy")
    )
]


for col in probability_cols:

    if not pd.api.types.is_numeric_dtype(
        train[col]
    ):
        continue

    invalid_count = (
        (
            train[col]
            < 0
        )
        |
        (
            train[col]
            > 1
        )
    ).sum()

    invalid_checks[
        f"{col}_outside_0_1"
    ] = int(
        invalid_count
    )


invalid_value_summary = (
    pd.DataFrame(
        [
            {
                "check": key,
                "invalid_count": value,
            }
            for key, value
            in invalid_checks.items()
        ]
    )
    .sort_values(
        "invalid_count",
        ascending=False,
    )
)


if invalid_value_summary.empty:

    print(
        "점검 가능한 이상값 규칙 없음"
    )

else:

    print(
        invalid_value_summary
        .to_string(
            index=False,
        )
    )


invalid_value_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "invalid_value_summary.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


# =============================================================================
# 13. 2025 시점 사용 가능성 검토용 Feature 목록
#
# 이 단계에서는 feature를 제거하지 않는다.
# Leakage 제거는 이후 별도 단계에서 수행한다.
#
# 사람이 확인하기 쉽도록 현재 schema를 분류해 출력한다.
# =============================================================================

print("\n" + "=" * 80)
print("13. FEATURE AVAILABILITY REVIEW")
print("=" * 80)


feature_cols = [
    col
    for col in test.columns
    if col != ID
]


historical_feature_cols = [
    col
    for col in feature_cols
    if col.startswith("asof_")
]


current_context_cols = [
    col
    for col in feature_cols
    if not col.startswith("asof_")
]


print("\n[Historical / as-of features]")
for col in historical_feature_cols:
    print(f"- {col}")


print("\n[Current game / context features]")
for col in current_context_cols:
    print(f"- {col}")


print(
    "\n※ 이 목록은 feature 제거 결과가 아닙니다."
)

print(
    "2025 예측 시점에서 실제 이용 가능한지 "
    "각 feature 생성 로직을 기준으로 별도 검증해야 합니다."
)

print(
    "현재 투구 결과 또는 미래 경기 상황을 사용하는 feature는 "
    "이후 Leakage 제거 단계에서 제외합니다."
)


# =============================================================================
# 14. 전체 Validation 결과 정리
# =============================================================================

print("\n" + "=" * 80)
print("14. FINAL DATA VALIDATION SUMMARY")
print("=" * 80)


validation_summary = {
    "target_binary":
        target_is_binary,

    "target_missing_count":
        target_missing_count,

    "target_positive_rate":
        train[TARGET].mean(),

    "full_duplicate_count":
        full_duplicate_count,

    "row_id_duplicate_count":
        row_id_duplicate_count,

    "schema_valid":
        schema_valid,

    "dtype_mismatch_count":
        len(dtype_mismatch),

    "missing_feature_count_train":
        int(
            train
            .isna()
            .any()
            .sum()
        ),

    "missing_feature_count_test":
        int(
            test
            .isna()
            .any()
            .sum()
        ),

    "train_pitcher_count":
        len(train_pitchers),

    "test_pitcher_count":
        len(test_pitchers),

    "known_pitcher_count":
        len(known_pitchers),

    "unknown_pitcher_count":
        len(unknown_pitchers),

    "unknown_pitcher_row_count":
        unknown_rows,

    "unknown_pitcher_row_rate":
        unknown_row_rate,

    "pitcher_overlap_rate":
        pitcher_overlap_rate,
}


validation_summary_df = (
    pd.DataFrame(
        list(
            validation_summary.items()
        ),
        columns=[
            "item",
            "value",
        ],
    )
)


print(
    validation_summary_df
    .to_string(
        index=False,
    )
)


validation_summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "validation_summary.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


# =============================================================================
# 15. Assertion
#
# 이미 현재까지 확정된 조건 중 학습 자체를 막아야 하는 핵심 조건만 assertion
# =============================================================================

assert target_is_binary, (
    f"{TARGET}이 정확한 binary target이 아닙니다."
)

assert target_missing_count == 0, (
    f"{TARGET}에 결측치가 존재합니다."
)

assert full_duplicate_count == 0, (
    "완전히 동일한 중복 row가 존재합니다."
)

assert schema_valid, (
    "train/test schema에 예상하지 못한 차이가 있습니다."
)


print("\n" + "=" * 80)
print("LR-02 DATA VALIDATION COMPLETE")
print("=" * 80)

print(
    f"산출물 저장 위치: "
    f"{OUTPUT_DIR}"
)