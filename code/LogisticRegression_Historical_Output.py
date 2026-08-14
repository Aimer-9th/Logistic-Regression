import os
import warnings

import numpy as np
import pandas as pd


# =============================================================================
# 0. 기본 설정
# =============================================================================

warnings.filterwarnings("ignore")

DATA_DIR = "/content/drive/MyDrive/𝟐𝟎𝟐𝟔/aimers/9기/open/data"
OUTPUT_DIR = "./lr06_historical_outputs"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

TRAIN_PATH = os.path.join(
    DATA_DIR,
    "train.csv",
)

TEST_PATH = os.path.join(
    DATA_DIR,
    "test.csv",
)

TARGET = "control_success"
PITCHER_ID_COL = "pitcher_id"
SEASON_COL = "season"

ROLLING_WINDOWS = [
    5,
    10,
    20,
    50,
]


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
print("1. DATA")
print("=" * 80)

print(f"Train shape : {train.shape}")
print(f"Test shape  : {test.shape}")


# =============================================================================
# 2. 공식 Historical Feature 확인
#
# 운영 측 제공 asof_* feature는 이미 현재 투구 직전까지의
# 과거 정보만으로 계산되어 있으므로 별도 재계산하지 않는다.
# =============================================================================

OFFICIAL_ASOF_FEATURES = [
    col
    for col in test.columns
    if col.startswith("asof_")
]


print("\n" + "=" * 80)
print("2. OFFICIAL ASOF FEATURES")
print("=" * 80)

print(
    f"Official asof feature count : "
    f"{len(OFFICIAL_ASOF_FEATURES)}"
)

for col in OFFICIAL_ASOF_FEATURES:
    print(f"- {col}")


assert len(OFFICIAL_ASOF_FEATURES) == 19, (
    "공식 asof_* feature 수가 예상과 다릅니다."
)


# =============================================================================
# 3. Main Train에서 Chronological Feature 생성 가능 여부 검증
#
# target 기반 historical feature 생성에는 최소한 다음 정보가 필요하다.
#
# season
# game_date
# game_id
# pitch_order
#
# 현재 train에는 정확한 pitch order가 제공되지 않았으므로
# 원본 row 순서나 row_id로 대체하지 않는다.
# =============================================================================

MAIN_REQUIRED_ORDER_LEVELS = {
    "season": [
        "season",
    ],
    "game_date": [
        "game_date",
    ],
    "game_id": [
        "game_id",
        "trackman_game_id",
    ],
    "pitch_order": [
        "pitch_no",
        "pitch_order",
        "pitch_number",
    ],
}


def resolve_column(
    df,
    candidates,
):
    """
    후보 컬럼 중 실제 존재하는 첫 컬럼을 반환한다.
    """

    for col in candidates:

        if col in df.columns:
            return col

    return None


resolved_main_order = {
    level: resolve_column(
        train,
        candidates,
    )
    for level, candidates
    in MAIN_REQUIRED_ORDER_LEVELS.items()
}


missing_main_order_levels = [
    level
    for level, col
    in resolved_main_order.items()
    if col is None
]


main_chronology_available = (
    len(missing_main_order_levels)
    == 0
)


print("\n" + "=" * 80)
print("3. MAIN TRAIN CHRONOLOGY")
print("=" * 80)

for level, col in resolved_main_order.items():

    print(
        f"{level:<15}: "
        f"{col if col is not None else 'NOT FOUND'}"
    )


print(
    f"\nComplete chronology : "
    f"{main_chronology_available}"
)

print(
    f"Missing levels      : "
    f"{missing_main_order_levels}"
)


# =============================================================================
# 4. Historical Feature 생성 전 공통 Validation
# =============================================================================

def validate_historical_source(
    df,
    target_col,
    group_col,
    order_cols,
    source_name,
):
    """
    target 기반 historical feature를 생성하기 전에
    데이터 source가 안전한지 검증한다.

    원본 train/test row 순서는 chronological order로 인정하지 않는다.
    """

    forbidden_sources = {
        "test",
        "test.csv",
        "evaluation_test",
    }

    if source_name.lower() in forbidden_sources:

        raise ValueError(
            "test.csv 내부 다른 row를 이용해 historical feature를 "
            "생성할 수 없습니다."
        )


    required_cols = (
        [target_col, group_col]
        + order_cols
    )

    missing_cols = [
        col
        for col in required_cols
        if col not in df.columns
    ]


    if missing_cols:

        raise KeyError(
            "Historical feature 생성에 필요한 컬럼이 없습니다: "
            f"{missing_cols}"
        )


    if not order_cols:

        raise ValueError(
            "명시적인 chronological key가 필요합니다. "
            "원본 index 또는 row_id로 대체하지 않습니다."
        )


    order_missing_count = int(
        df[
            order_cols
        ]
        .isna()
        .any(
            axis=1,
        )
        .sum()
    )


    if order_missing_count > 0:

        raise ValueError(
            "Chronological key에 결측값이 존재합니다. "
            f"rows={order_missing_count:,}"
        )


# =============================================================================
# 5. 공통 Chronological Sorting Function
#
# 실제 chronology가 확보된 historical source에만 사용한다.
# =============================================================================

def sort_history(
    df,
    group_col,
    order_cols,
):
    """
    group + chronological order 기준으로 historical data를 정렬한다.

    row_id 또는 원본 index는 chronology 결정에 사용하지 않는다.
    """

    result = df.copy()

    result[
        "_original_position"
    ] = np.arange(
        len(result),
        dtype=np.int64,
    )


    sort_cols = (
        order_cols
        + [group_col]
    )


    result = (
        result
        .sort_values(
            sort_cols,
            kind="mergesort",
        )
        .reset_index(
            drop=True,
        )
    )


    result[
        "_historical_position"
    ] = np.arange(
        len(result),
        dtype=np.int64,
    )


    return result


# =============================================================================
# 6. Pitch-level Historical Feature 생성
#
# 핵심:
#
# current target
#     ↓
# shift(1)
#     ↓
# rolling(...)
#
# 따라서 현재 row target은 절대 현재 row feature에 포함되지 않는다.
# =============================================================================

def add_pitch_level_history(
    df,
    target_col=TARGET,
    pitcher_col=PITCHER_ID_COL,
    windows=ROLLING_WINDOWS,
):
    """
    Pitch-level target history 생성.

    사전에 chronology가 보장된 df에서만 호출해야 한다.
    """

    result = df.copy()


    grouped_target = (
        result
        .groupby(
            pitcher_col,
            sort=False,
        )[target_col]
    )


    # -------------------------------------------------------------------------
    # 직전 투구 결과
    # -------------------------------------------------------------------------

    result[
        "hist_pitcher_prev1_success"
    ] = (
        grouped_target
        .shift(1)
    )


    # -------------------------------------------------------------------------
    # 현재 투구 이전 누적 투구 수
    #
    # 첫 투구: 0
    # 두 번째 투구: 1
    # ...
    # -------------------------------------------------------------------------

    result[
        "hist_pitcher_pitch_count"
    ] = (
        result
        .groupby(
            pitcher_col,
            sort=False,
        )
        .cumcount()
    )


    # -------------------------------------------------------------------------
    # 최근 N구 성공률
    # -------------------------------------------------------------------------

    shifted_target = (
        grouped_target
        .shift(1)
    )


    for window in windows:

        result[
            f"hist_pitcher_last{window}_success_rate"
        ] = (
            shifted_target
            .groupby(
                result[
                    pitcher_col
                ],
                sort=False,
            )
            .transform(
                lambda s:
                    s.rolling(
                        window=window,
                        min_periods=1,
                    )
                    .mean()
            )
        )


        result[
            f"hist_pitcher_last{window}_count"
        ] = (
            result
            .groupby(
                pitcher_col,
                sort=False,
            )
            .cumcount()
            .clip(
                upper=window,
            )
        )


    return result


# =============================================================================
# 7. Season-level Historical Feature
#
# 같은 pitcher + season 내부에서 현재 투구 이전 기록만 사용한다.
#
# 시즌 첫 투구:
# - count = 0
# - rate = NaN
# =============================================================================

def add_season_level_history(
    df,
    target_col=TARGET,
    pitcher_col=PITCHER_ID_COL,
    season_col=SEASON_COL,
):
    """
    Pitcher-season 기준 past-only expanding feature 생성.
    """

    result = df.copy()


    group_cols = [
        pitcher_col,
        season_col,
    ]


    # 현재 row 이전 시즌 누적 투구 수
    result[
        "hist_pitcher_season_pitch_count"
    ] = (
        result
        .groupby(
            group_cols,
            sort=False,
        )
        .cumcount()
    )


    # 현재 row를 제외한 target
    shifted = (
        result
        .groupby(
            group_cols,
            sort=False,
        )[target_col]
        .shift(1)
    )


    result[
        "hist_pitcher_season_success_rate"
    ] = (
        shifted
        .groupby(
            [
                result[pitcher_col],
                result[season_col],
            ],
            sort=False,
        )
        .transform(
            lambda s:
                s.expanding(
                    min_periods=1,
                )
                .mean()
        )
    )


    return result


# =============================================================================
# 8. Conditional Historical Rate
#
# 예:
# pitcher × balls_before 기준 과거 성공률
#
# 반드시 현재 row를 shift(1)로 제외한다.
# =============================================================================

def add_conditional_success_rate(
    df,
    context_col,
    target_col=TARGET,
    pitcher_col=PITCHER_ID_COL,
):
    """
    pitcher + context 기준 과거 success rate / count 생성.

    예:
    context_col = "balls_before"
    """

    result = df.copy()


    group_cols = [
        pitcher_col,
        context_col,
    ]


    shifted = (
        result
        .groupby(
            group_cols,
            sort=False,
        )[target_col]
        .shift(1)
    )


    feature_prefix = (
        f"hist_pitcher_by_{context_col}"
    )


    result[
        f"{feature_prefix}_success_rate"
    ] = (
        shifted
        .groupby(
            [
                result[pitcher_col],
                result[context_col],
            ],
            sort=False,
        )
        .transform(
            lambda s:
                s.expanding(
                    min_periods=1,
                )
                .mean()
        )
    )


    result[
        f"{feature_prefix}_count"
    ] = (
        result
        .groupby(
            group_cols,
            sort=False,
        )
        .cumcount()
    )


    return result


# =============================================================================
# 9. Game-level Historical Feature
#
# 필요한 조건:
# - 실제 game_id
# - 실제 chronological game ordering
#
# 현재 main train에는 game_id가 없으므로 실제 train에는 적용하지 않는다.
# 함수만 안전하게 정의한다.
# =============================================================================

def add_game_level_history(
    df,
    game_col,
    target_col=TARGET,
    pitcher_col=PITCHER_ID_COL,
    recent_game_windows=(1, 3, 5),
):
    """
    과거 경기 단위 success rate를 생성한다.

    입력 df가 이미 실제 chronological order로 정렬되어 있어야 한다.
    """

    result = df.copy()


    # -------------------------------------------------------------------------
    # Pitcher-game 단위 실제 경기 success rate 계산
    #
    # 이 값은 해당 경기가 끝난 이후 다음 경기부터만 사용할 수 있다.
    # -------------------------------------------------------------------------

    game_stats = (
        result
        .groupby(
            [
                pitcher_col,
                game_col,
            ],
            sort=False,
            as_index=False,
        )
        .agg(
            game_success_rate=(
                target_col,
                "mean",
            ),
            game_pitch_count=(
                target_col,
                "size",
            ),
        )
    )


    # -------------------------------------------------------------------------
    # pitcher별 과거 경기만 사용
    # -------------------------------------------------------------------------

    game_stats[
        "hist_prev_game_success_rate"
    ] = (
        game_stats
        .groupby(
            pitcher_col,
            sort=False,
        )[
            "game_success_rate"
        ]
        .shift(1)
    )


    shifted_game_rate = (
        game_stats
        .groupby(
            pitcher_col,
            sort=False,
        )[
            "game_success_rate"
        ]
        .shift(1)
    )


    for window in recent_game_windows:

        game_stats[
            f"hist_prev{window}_games_success_rate"
        ] = (
            shifted_game_rate
            .groupby(
                game_stats[
                    pitcher_col
                ],
                sort=False,
            )
            .transform(
                lambda s:
                    s.rolling(
                        window=window,
                        min_periods=1,
                    )
                    .mean()
            )
        )


    feature_cols = [
        col
        for col in game_stats.columns
        if col.startswith(
            "hist_"
        )
    ]


    result = result.merge(
        game_stats[
            [
                pitcher_col,
                game_col,
            ]
            + feature_cols
        ],
        on=[
            pitcher_col,
            game_col,
        ],
        how="left",
        validate="many_to_one",
    )


    return result


# =============================================================================
# 10. Cold Start 정책
#
# 기본 정책:
#
# rate:
#   NaN 유지
#
# count:
#   0
#
# 이유:
#   "기록 없음"과 "성공률 0%"를 구분해야 한다.
#
# smoothing / default prior는 이번 이슈에서 강제로 적용하지 않는다.
# 모델 preprocessing 단계에서 실험 가능하도록 raw state를 보존한다.
# =============================================================================

COLD_START_POLICY = {
    "historical_rate": "keep_nan",
    "historical_count": "zero",
    "smoothing": "not_applied_by_default",
    "default_prior": "apply_in_model_preprocessing_if_needed",
}


print("\n" + "=" * 80)
print("4. COLD START POLICY")
print("=" * 80)

for key, value in COLD_START_POLICY.items():

    print(
        f"{key:<20}: "
        f"{value}"
    )


# =============================================================================
# 11. Historical Feature 후보 목록
# =============================================================================

PITCH_HISTORY_FEATURES = [
    "hist_pitcher_prev1_success",
    "hist_pitcher_pitch_count",
]


for window in ROLLING_WINDOWS:

    PITCH_HISTORY_FEATURES.extend(
        [
            f"hist_pitcher_last{window}_success_rate",
            f"hist_pitcher_last{window}_count",
        ]
    )


SEASON_HISTORY_FEATURES = [
    "hist_pitcher_season_success_rate",
    "hist_pitcher_season_pitch_count",
]


CONDITIONAL_HISTORY_FEATURES = [
    "hist_pitcher_by_balls_before_success_rate",
    "hist_pitcher_by_balls_before_count",
]


print("\n" + "=" * 80)
print("5. CANDIDATE HISTORICAL FEATURES")
print("=" * 80)

print("\n[Pitch-level]")

for col in PITCH_HISTORY_FEATURES:
    print(f"- {col}")


print("\n[Season-level]")

for col in SEASON_HISTORY_FEATURES:
    print(f"- {col}")


print("\n[Conditional]")

for col in CONDITIONAL_HISTORY_FEATURES:
    print(f"- {col}")


# =============================================================================
# 12. Synthetic Validation Data
#
# 실제 train은 chronology가 없으므로 leakage 검증 자체는
# chronological order가 명확하게 정의된 synthetic data로 수행한다.
#
# 이 테스트는 생성 함수 자체가 현재 target을 포함하지 않는지 검증한다.
# =============================================================================

validation_df = pd.DataFrame(
    {
        "pitcher_id": [
            100,
            100,
            100,
            100,
            200,
            200,
        ],

        "season": [
            2023,
            2023,
            2023,
            2024,
            2024,
            2024,
        ],

        "game_id": [
            1,
            1,
            2,
            3,
            4,
            4,
        ],

        "pitch_no": [
            1,
            2,
            1,
            1,
            1,
            2,
        ],

        "balls_before": [
            0,
            1,
            0,
            0,
            0,
            1,
        ],

        "control_success": [
            1,
            0,
            1,
            0,
            0,
            1,
        ],
    }
)


VALIDATION_ORDER_COLS = [
    "season",
    "game_id",
    "pitch_no",
]


validate_historical_source(
    validation_df,
    target_col=TARGET,
    group_col=PITCHER_ID_COL,
    order_cols=VALIDATION_ORDER_COLS,
    source_name="synthetic_history",
)


validation_sorted = sort_history(
    validation_df,
    group_col=PITCHER_ID_COL,
    order_cols=VALIDATION_ORDER_COLS,
)


validation_features = add_pitch_level_history(
    validation_sorted,
)

validation_features = add_season_level_history(
    validation_features,
)

validation_features = add_conditional_success_rate(
    validation_features,
    context_col="balls_before",
)


# =============================================================================
# 13. Leakage Validation Test
# =============================================================================

print("\n" + "=" * 80)
print("6. LEAKAGE VALIDATION TEST")
print("=" * 80)


# -----------------------------------------------------------------------------
# Test 1:
# 각 pitcher의 첫 투구에는 직전 target이 없어야 한다.
# -----------------------------------------------------------------------------

first_pitch_mask = (
    validation_features
    .groupby(
        PITCHER_ID_COL,
        sort=False,
    )
    .cumcount()
    == 0
)


assert (
    validation_features.loc[
        first_pitch_mask,
        "hist_pitcher_prev1_success",
    ]
    .isna()
    .all()
), (
    "첫 투구에 이전 target 정보가 포함되었습니다."
)


print(
    "✓ Pitcher 첫 투구 prev1 = NaN"
)


# -----------------------------------------------------------------------------
# Test 2:
# 두 번째 투구의 prev1은 첫 번째 투구 target과 동일해야 한다.
# -----------------------------------------------------------------------------

for pitcher_id, pitcher_df in validation_features.groupby(
    PITCHER_ID_COL,
    sort=False,
):

    pitcher_df = pitcher_df.reset_index(
        drop=True
    )

    if len(pitcher_df) < 2:
        continue

    assert (
        pitcher_df.loc[
            1,
            "hist_pitcher_prev1_success",
        ]
        ==
        pitcher_df.loc[
            0,
            TARGET,
        ]
    ), (
        f"Pitcher {pitcher_id}: "
        "두 번째 투구의 prev1이 첫 target과 다릅니다."
    )


print(
    "✓ 두 번째 투구부터 직전 target만 반영"
)


# -----------------------------------------------------------------------------
# Test 3:
# current target 변경이 current row historical feature에 영향을 주면 안 된다.
#
# 동일한 과거를 유지한 채 마지막 row target만 뒤집어서 비교한다.
# -----------------------------------------------------------------------------

original = validation_sorted.copy()

modified = validation_sorted.copy()

modified.loc[
    modified.index[-1],
    TARGET,
] = (
    1
    - modified.loc[
        modified.index[-1],
        TARGET,
    ]
)


original_features = add_pitch_level_history(
    original
)

modified_features = add_pitch_level_history(
    modified
)


historical_rate_cols = [
    col
    for col in original_features.columns
    if (
        col.startswith(
            "hist_"
        )
        and (
            col.endswith(
                "_rate"
            )
            or col.endswith(
                "_success"
            )
        )
    )
]


for col in historical_rate_cols:

    original_value = (
        original_features.loc[
            original_features.index[-1],
            col,
        ]
    )

    modified_value = (
        modified_features.loc[
            modified_features.index[-1],
            col,
        ]
    )


    if (
        pd.isna(original_value)
        and pd.isna(modified_value)
    ):
        continue


    assert (
        original_value
        == modified_value
    ), (
        f"{col}: 현재 row target 변경이 "
        "현재 row historical feature에 영향을 줍니다."
    )


print(
    "✓ Current target 변경이 current historical feature에 영향 없음"
)


# -----------------------------------------------------------------------------
# Test 4:
# 시즌 첫 투구는 시즌 success rate NaN / count 0
# -----------------------------------------------------------------------------

season_first_mask = (
    validation_features
    .groupby(
        [
            PITCHER_ID_COL,
            SEASON_COL,
        ],
        sort=False,
    )
    .cumcount()
    == 0
)


assert (
    validation_features.loc[
        season_first_mask,
        "hist_pitcher_season_success_rate",
    ]
    .isna()
    .all()
)


assert (
    validation_features.loc[
        season_first_mask,
        "hist_pitcher_season_pitch_count",
    ]
    == 0
).all()


print(
    "✓ 시즌 첫 투구 rate=NaN / count=0"
)


# -----------------------------------------------------------------------------
# Test 5:
# 새로운 pitcher는 모든 target-based rate가 cold start 상태
# -----------------------------------------------------------------------------

new_pitcher_df = pd.DataFrame(
    {
        "pitcher_id": [
            999,
        ],
        "season": [
            2025,
        ],
        "game_id": [
            10,
        ],
        "pitch_no": [
            1,
        ],
        "balls_before": [
            0,
        ],
        "control_success": [
            1,
        ],
    }
)


new_pitcher_features = (
    add_pitch_level_history(
        new_pitcher_df
    )
)


assert pd.isna(
    new_pitcher_features.loc[
        0,
        "hist_pitcher_prev1_success",
    ]
)


for window in ROLLING_WINDOWS:

    assert pd.isna(
        new_pitcher_features.loc[
            0,
            f"hist_pitcher_last{window}_success_rate",
        ]
    )

    assert (
        new_pitcher_features.loc[
            0,
            f"hist_pitcher_last{window}_count",
        ]
        == 0
    )


print(
    "✓ Unknown / 신규 Pitcher cold-start 정상"
)


# =============================================================================
# 14. Main Train 적용 여부
#
# 핵심:
# 현재 main train에는 정확한 chronological key가 없으므로
# target-based historical feature를 생성하지 않는다.
# =============================================================================

print("\n" + "=" * 80)
print("7. MAIN TRAIN FEATURE GENERATION")
print("=" * 80)


if not main_chronology_available:

    MAIN_GENERATION_STATUS = "BLOCKED"

    print(
        "BLOCKED"
    )

    print(
        "train.csv에 exact pitch chronology를 구성할 컬럼이 부족합니다."
    )

    print(
        f"Missing levels: "
        f"{missing_main_order_levels}"
    )

    print(
        "row_id 또는 원본 row 순서를 chronological order로 "
        "대체하지 않습니다."
    )

    historical_train = None

else:

    MAIN_GENERATION_STATUS = "READY"

    print(
        "Chronological key가 확인되었습니다."
    )


# =============================================================================
# 15. Test 적용 원칙
#
# 평가 test 각 row는 독립 예측해야 하므로
# test 내부 다른 row를 이용해 target/history feature를 생성하지 않는다.
# =============================================================================

TEST_GENERATION_STATUS = "FORBIDDEN_FROM_TEST_ROWS"


print("\n" + "=" * 80)
print("8. TEST FEATURE GENERATION POLICY")
print("=" * 80)

print(
    "test.csv 내부 다른 row를 이용한 "
    "rolling / expanding / 누적 통계 생성 금지"
)

print(
    "공식 asof_* feature는 그대로 사용 가능"
)


# =============================================================================
# 16. NaN / Cold Start 통계 함수
#
# 실제 historical feature가 생성 가능한 source에 사용할 함수.
# =============================================================================

def summarize_historical_missing(
    df,
    historical_cols,
):
    """
    Historical feature별 NaN / cold-start 비율을 계산한다.
    """

    rows = []


    for col in historical_cols:

        if col not in df.columns:
            continue


        rows.append(
            {
                "feature":
                    col,

                "missing_count":
                    int(
                        df[col]
                        .isna()
                        .sum()
                    ),

                "missing_rate":
                    df[col]
                    .isna()
                    .mean(),
            }
        )


    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "missing_rate",
            ascending=False,
        )
        .reset_index(
            drop=True,
        )
    )


validation_hist_cols = [
    col
    for col in validation_features.columns
    if col.startswith(
        "hist_"
    )
]


validation_missing_summary = (
    summarize_historical_missing(
        validation_features,
        validation_hist_cols,
    )
)


print("\n" + "=" * 80)
print("9. SYNTHETIC COLD START SUMMARY")
print("=" * 80)

print(
    validation_missing_summary
    .round(4)
    .to_string(
        index=False,
    )
)


validation_missing_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "synthetic_historical_missing_summary.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


# =============================================================================
# 17. Feature Generation Status 저장
# =============================================================================

feature_status_rows = []


for col in PITCH_HISTORY_FEATURES:

    feature_status_rows.append(
        {
            "feature": col,
            "group": "pitch_level",
            "implementation": "defined",
            "main_train_generation":
                MAIN_GENERATION_STATUS,
        }
    )


for col in SEASON_HISTORY_FEATURES:

    feature_status_rows.append(
        {
            "feature": col,
            "group": "season_level",
            "implementation": "defined",
            "main_train_generation":
                MAIN_GENERATION_STATUS,
        }
    )


for col in CONDITIONAL_HISTORY_FEATURES:

    feature_status_rows.append(
        {
            "feature": col,
            "group": "conditional",
            "implementation": "defined",
            "main_train_generation":
                MAIN_GENERATION_STATUS,
        }
    )


feature_status_df = pd.DataFrame(
    feature_status_rows
)


feature_status_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "historical_feature_status.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


# =============================================================================
# 18. Final Summary
# =============================================================================

summary = {
    "official_asof_feature_count":
        len(OFFICIAL_ASOF_FEATURES),

    "new_pitch_feature_count":
        len(PITCH_HISTORY_FEATURES),

    "new_season_feature_count":
        len(SEASON_HISTORY_FEATURES),

    "new_conditional_feature_count":
        len(CONDITIONAL_HISTORY_FEATURES),

    "main_train_chronology_available":
        main_chronology_available,

    "main_generation_status":
        MAIN_GENERATION_STATUS,

    "test_row_history_generation":
        TEST_GENERATION_STATUS,

    "cold_start_rate_policy":
        COLD_START_POLICY[
            "historical_rate"
        ],

    "cold_start_count_policy":
        COLD_START_POLICY[
            "historical_count"
        ],

    "synthetic_leakage_tests":
        "PASS",
}


summary_df = pd.DataFrame(
    list(
        summary.items()
    ),
    columns=[
        "item",
        "value",
    ],
)


print("\n" + "=" * 80)
print("10. LR-06 SUMMARY")
print("=" * 80)

print(
    summary_df
    .to_string(
        index=False,
    )
)


summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lr06_summary.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


print("\n" + "=" * 80)
print("LR-06 COMPLETE")
print("=" * 80)

print(
    "Historical feature 생성 함수 및 leakage validation은 정의 완료."
)

print(
    "실제 train target-based feature 생성은 exact chronology 확보 전까지 BLOCKED."
)

print(
    f"Output directory: "
    f"{OUTPUT_DIR}"
)