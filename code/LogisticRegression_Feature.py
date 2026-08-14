import os
import warnings

import numpy as np
import pandas as pd


# =============================================================================
# 0. 기본 설정
# =============================================================================

warnings.filterwarnings("ignore")

DATA_DIR = "/content/drive/MyDrive/𝟐𝟎𝟐𝟔/aimers/9기/open/data"
OUTPUT_DIR = "./lr05_feature_config_outputs"

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

ID = "row_id"
TARGET = "control_success"
PITCHER_ID_COL = "pitcher_id"

EXPECTED_FULL_FEATURE_COUNT = 47
EXPECTED_REDUCED_FEATURE_COUNT = 44


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


# LR-04와 동일하게 test schema를 모델 feature의 source of truth로 사용
ALL_FEATURES = [
    col
    for col in test.columns
    if col != ID
]


print("=" * 80)
print("1. DATA")
print("=" * 80)

print(f"Train shape         : {train.shape}")
print(f"Test shape          : {test.shape}")
print(f"Raw feature count   : {len(ALL_FEATURES)}")


assert TARGET not in ALL_FEATURES, (
    f"{TARGET}이 모델 feature에 포함되어 있습니다."
)

assert ID not in ALL_FEATURES, (
    f"{ID}가 모델 feature에 포함되어 있습니다."
)

assert len(ALL_FEATURES) == EXPECTED_FULL_FEATURE_COUNT, (
    "LR-04에서 확정한 feature 수와 일치하지 않습니다. "
    f"expected={EXPECTED_FULL_FEATURE_COUNT}, "
    f"actual={len(ALL_FEATURES)}"
)


# =============================================================================
# 2. Feature Type 정의
#
# 모든 feature는 아래 네 타입 중 정확히 하나에 포함된다.
#
# numeric
# categorical
# ordinal
# identifier
#
# 이번 LR baseline에서는 순서형 값도 선형적인 수치 의미를 갖는 경우
# numeric으로 처리한다.
#
# 따라서 별도의 ORDINAL_COLS는 현재 없음.
# =============================================================================

CAT_COLS = [
    # 경기 정보
    "game_dayofweek",
    "top_bottom",
    "game_type",
    "base_state",

    # 선수 / 팀 정보
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
]


NUM_COLS = [
    # -------------------------------------------------------------------------
    # 시간 / 경기 진행
    # -------------------------------------------------------------------------
    "season",
    "game_month",
    "inning",

    # -------------------------------------------------------------------------
    # 카운트
    # -------------------------------------------------------------------------
    "balls_before",
    "strikes_before",
    "outs_before",

    # -------------------------------------------------------------------------
    # 점수
    # -------------------------------------------------------------------------
    "run_top_before",
    "run_bot_before",
    "run_total_before",
    "score_diff_home",
    "score_diff_pitcher_team",

    # -------------------------------------------------------------------------
    # 주자
    # -------------------------------------------------------------------------
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",

    # -------------------------------------------------------------------------
    # 경기 중요도
    # -------------------------------------------------------------------------
    "home_win_expectancy",
    "away_win_expectancy",
    "li",

    # -------------------------------------------------------------------------
    # 투수 누적 이력
    # -------------------------------------------------------------------------
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",

    # -------------------------------------------------------------------------
    # 투수 최근 경기 이력
    # -------------------------------------------------------------------------
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",

    # -------------------------------------------------------------------------
    # 타자 과거 이력
    # -------------------------------------------------------------------------
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",

    # -------------------------------------------------------------------------
    # 투수 과거 구종 구성
    # -------------------------------------------------------------------------
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]


# 현재 raw feature에는 별도의 ordinal encoding이 필요한 feature 없음
ORDINAL_COLS = []


# pitcher_id는 별도 실험 대상이므로 일반 categorical과 분리
ID_COLS = [
    "pitcher_id",
]


# =============================================================================
# 3. Feature Type 검증
# =============================================================================

FEATURE_TYPE_GROUPS = {
    "numeric": NUM_COLS,
    "categorical": CAT_COLS,
    "ordinal": ORDINAL_COLS,
    "identifier": ID_COLS,
}


# -----------------------------------------------------------------------------
# 중복 분류 확인
# -----------------------------------------------------------------------------

type_membership = {}

for feature_type, cols in FEATURE_TYPE_GROUPS.items():

    for col in cols:

        type_membership.setdefault(
            col,
            [],
        ).append(
            feature_type
        )


duplicate_type_features = {
    col: types
    for col, types in type_membership.items()
    if len(types) > 1
}


classified_features = set(
    NUM_COLS
    + CAT_COLS
    + ORDINAL_COLS
    + ID_COLS
)

unclassified_features = sorted(
    set(ALL_FEATURES)
    - classified_features
)

unknown_defined_features = sorted(
    classified_features
    - set(ALL_FEATURES)
)


print("\n" + "=" * 80)
print("2. FEATURE TYPE VALIDATION")
print("=" * 80)

print(
    f"Total       : "
    f"{len(ALL_FEATURES)}"
)

print(
    f"Numeric     : "
    f"{len(NUM_COLS)}"
)

print(
    f"Categorical : "
    f"{len(CAT_COLS)}"
)

print(
    f"Ordinal     : "
    f"{len(ORDINAL_COLS)}"
)

print(
    f"Identifier  : "
    f"{len(ID_COLS)}"
)

print(
    f"\nDuplicate classification : "
    f"{duplicate_type_features}"
)

print(
    f"Unclassified features    : "
    f"{unclassified_features}"
)

print(
    f"Unknown defined features : "
    f"{unknown_defined_features}"
)


assert not duplicate_type_features, (
    "둘 이상의 feature type에 중복 분류된 컬럼이 있습니다."
)

assert not unclassified_features, (
    "분류되지 않은 feature가 있습니다: "
    f"{unclassified_features}"
)

assert not unknown_defined_features, (
    "실제 데이터에 없는 feature가 type definition에 포함되어 있습니다: "
    f"{unknown_defined_features}"
)

assert classified_features == set(
    ALL_FEATURES
), (
    "모든 feature가 정확히 하나의 type으로 분류되어야 합니다."
)


# =============================================================================
# 4. Feature Group 정의
#
# Issue에서 정의한 의미적 feature group.
#
# 실제 구종/구속/회전수/무브먼트는 raw train/test에는 없음.
# asof_pitcher_* pitchmix는 과거 투구 구성 정보로 pitcher history에 포함.
# =============================================================================

FEATURE_GROUPS = {
    "pitch_characteristics": [
        # 현재 raw train/test에는 실제 현재 투구의
        # 구종 / 구속 / 회전수 / 무브먼트가 없음
    ],

    "game_context": [
        "game_month",
        "game_dayofweek",
        "inning",
        "top_bottom",
        "game_type",
        "balls_before",
        "strikes_before",
        "outs_before",
        "run_top_before",
        "run_bot_before",
        "run_total_before",
        "score_diff_home",
        "score_diff_pitcher_team",
        "runner_on_1b",
        "runner_on_2b",
        "runner_on_3b",
        "num_runners_on",
        "base_state",
        "home_win_expectancy",
        "away_win_expectancy",
        "li",
    ],

    "pitcher_information": [
        "season",
        "pitcher_hand",
        "pitcher_team_id",

        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate",
        "asof_pitcher_ball_rate",
        "asof_pitcher_strike_rate",

        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",

        "asof_pitcher_pitchmix_n",
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    ],

    "batter_information": [
        "batter_id",
        "batter_hand",
        "batter_team_id",
        "asof_batter_n",
        "asof_batter_success_rate",
        "asof_batter_middle_rate",
    ],

    "identifier": [
        "pitcher_id",
    ],
}


# =============================================================================
# 5. Numeric Scaling 대상 정의
#
# NO_SCALE:
# - 값 범위가 매우 제한적인 count / binary / discrete state
#
# SCALE:
# - 점수, 기대승률, LI
# - 누적 표본 수
# - rate / historical continuous variable
# =============================================================================

NO_SCALE_COLS = [
    "season",
    "game_month",
    "inning",

    "balls_before",
    "strikes_before",
    "outs_before",

    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
]


SCALE_COLS = [
    col
    for col in NUM_COLS
    if col not in NO_SCALE_COLS
]


print("\n" + "=" * 80)
print("3. NUMERIC SCALING")
print("=" * 80)

print(
    f"전체 Numeric      : "
    f"{len(NUM_COLS)}"
)

print(
    f"Scaling 대상      : "
    f"{len(SCALE_COLS)}"
)

print(
    f"Scaling 제외      : "
    f"{len(NO_SCALE_COLS)}"
)

print(
    f"합계              : "
    f"{len(SCALE_COLS) + len(NO_SCALE_COLS)}"
)


scaling_overlap = sorted(
    set(SCALE_COLS)
    & set(NO_SCALE_COLS)
)

unclassified_numeric = sorted(
    set(NUM_COLS)
    - (
        set(SCALE_COLS)
        | set(NO_SCALE_COLS)
    )
)

unknown_scaling_cols = sorted(
    (
        set(SCALE_COLS)
        | set(NO_SCALE_COLS)
    )
    - set(NUM_COLS)
)


print("\n[Scaling / No Scaling 중복]")
print(
    scaling_overlap
)

print("\n[분류되지 않은 Numeric]")
print(
    unclassified_numeric
)

print("\n[NUM_COLS에 없는 Scaling 컬럼]")
print(
    unknown_scaling_cols
)


assert not scaling_overlap
assert not unclassified_numeric
assert not unknown_scaling_cols

assert (
    len(SCALE_COLS)
    + len(NO_SCALE_COLS)
    == len(NUM_COLS)
)


# =============================================================================
# 6. Cardinality 확인
# =============================================================================

print("\n" + "=" * 80)
print("4. CATEGORICAL / IDENTIFIER CARDINALITY")
print("=" * 80)


cardinality_cols = (
    CAT_COLS
    + ID_COLS
)


cardinality_rows = []


for col in cardinality_cols:

    nunique = train[col].nunique(
        dropna=True
    )

    unique_ratio = (
        nunique
        / len(train)
    )

    cardinality_rows.append(
        {
            "feature": col,
            "type": (
                "identifier"
                if col in ID_COLS
                else "categorical"
            ),
            "nunique": nunique,
            "unique_ratio": unique_ratio,
            "high_cardinality":
                nunique >= 50,
        }
    )


cardinality_df = (
    pd.DataFrame(
        cardinality_rows
    )
    .sort_values(
        "nunique",
        ascending=False,
    )
    .reset_index(
        drop=True,
    )
)


print(
    cardinality_df
    .round(6)
    .to_string(
        index=False,
    )
)


print("\n[High-cardinality candidates]")

high_cardinality_df = (
    cardinality_df[
        cardinality_df[
            "high_cardinality"
        ]
    ]
)


if high_cardinality_df.empty:

    print("없음")

else:

    print(
        high_cardinality_df[
            [
                "feature",
                "type",
                "nunique",
            ]
        ]
        .to_string(
            index=False,
        )
    )


cardinality_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "categorical_cardinality.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


# =============================================================================
# 7. Constant / Near-Constant Feature 확인
#
# Near-constant:
# 가장 많이 등장하는 값이 전체 non-missing row의 99.5% 이상
# =============================================================================

NEAR_CONSTANT_THRESHOLD = 0.995


constant_rows = []


for col in ALL_FEATURES:

    non_missing = (
        train[col]
        .dropna()
    )

    nunique = (
        non_missing
        .nunique()
    )


    if len(non_missing) == 0:

        top_frequency = np.nan

    else:

        top_frequency = (
            non_missing
            .value_counts(
                normalize=True,
                dropna=False,
            )
            .iloc[0]
        )


    constant_rows.append(
        {
            "feature": col,
            "nunique": nunique,
            "top_value_ratio":
                top_frequency,
            "is_constant":
                nunique <= 1,
            "is_near_constant":
                (
                    nunique > 1
                    and top_frequency
                    >= NEAR_CONSTANT_THRESHOLD
                ),
        }
    )


constant_df = (
    pd.DataFrame(
        constant_rows
    )
    .sort_values(
        [
            "is_constant",
            "is_near_constant",
            "top_value_ratio",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )
)


constant_candidates = constant_df[
    constant_df[
        "is_constant"
    ]
    |
    constant_df[
        "is_near_constant"
    ]
]


print("\n" + "=" * 80)
print("5. CONSTANT / NEAR-CONSTANT")
print("=" * 80)


if constant_candidates.empty:

    print(
        "Constant / near-constant feature 없음"
    )

else:

    print(
        constant_candidates
        .round(6)
        .to_string(
            index=False,
        )
    )


constant_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "constant_near_constant_summary.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


# =============================================================================
# 8. 명백한 중복 정보 후보 검증
#
# 단순 correlation만으로 제거하지 않고,
# 공식 정의상 동일/파생 관계가 명확한 feature를 검증한다.
#
# 후보:
#
# 1. asof_pitcher_pitchmix_n
#      vs asof_pitcher_n
#
# 2. run_total_before
#      = run_top_before + run_bot_before
#
# 3. num_runners_on
#      = runner_on_1b + runner_on_2b + runner_on_3b
#
# 4. away_win_expectancy
#      ≈ 100 - home_win_expectancy
# =============================================================================

print("\n" + "=" * 80)
print("6. REDUNDANT INFORMATION CHECK")
print("=" * 80)


redundancy_rows = []


# -----------------------------------------------------------------------------
# 8-1. asof_pitcher_pitchmix_n vs asof_pitcher_n
# -----------------------------------------------------------------------------

left = train[
    "asof_pitcher_n"
]

right = train[
    "asof_pitcher_pitchmix_n"
]


both_missing = (
    left.isna()
    & right.isna()
)

both_equal = (
    left.eq(right)
)

pitchmix_n_equal = bool(
    (
        both_missing
        | both_equal
    )
    .all()
)


redundancy_rows.append(
    {
        "feature":
            "asof_pitcher_pitchmix_n",

        "compared_with":
            "asof_pitcher_n",

        "relation":
            "exact equality",

        "validated":
            pitchmix_n_equal,

        "reason":
            (
                "투수 누적 투구 수와 "
                "pitch-mix 표본 수가 동일한지 검증"
            ),
    }
)


# -----------------------------------------------------------------------------
# 8-2. run_total_before
# -----------------------------------------------------------------------------

expected_run_total = (
    train[
        "run_top_before"
    ]
    + train[
        "run_bot_before"
    ]
)


run_total_equal = bool(
    train[
        "run_total_before"
    ]
    .eq(
        expected_run_total
    )
    .all()
)


redundancy_rows.append(
    {
        "feature":
            "run_total_before",

        "compared_with":
            "run_top_before + run_bot_before",

        "relation":
            "exact deterministic sum",

        "validated":
            run_total_equal,

        "reason":
            "두 팀 점수의 단순 합",
    }
)


# -----------------------------------------------------------------------------
# 8-3. num_runners_on
# -----------------------------------------------------------------------------

expected_runner_count = (
    train[
        "runner_on_1b"
    ]
    + train[
        "runner_on_2b"
    ]
    + train[
        "runner_on_3b"
    ]
)


runner_count_equal = bool(
    train[
        "num_runners_on"
    ]
    .eq(
        expected_runner_count
    )
    .all()
)


redundancy_rows.append(
    {
        "feature":
            "num_runners_on",

        "compared_with":
            (
                "runner_on_1b + "
                "runner_on_2b + "
                "runner_on_3b"
            ),

        "relation":
            "exact deterministic sum",

        "validated":
            runner_count_equal,

        "reason":
            "세 base indicator의 단순 합",
    }
)


# -----------------------------------------------------------------------------
# 8-4. home / away win expectancy
#
# 공식 정의가 0~100 scale이므로 합이 100인지 확인
# -----------------------------------------------------------------------------

win_expectancy_sum = (
    train[
        "home_win_expectancy"
    ]
    + train[
        "away_win_expectancy"
    ]
)


win_expectancy_valid = (
    np.isclose(
        win_expectancy_sum.dropna(),
        100.0,
        atol=1e-6,
    )
    .all()
)


max_win_expectancy_error = (
    (
        win_expectancy_sum
        - 100
    )
    .abs()
    .max()
)


redundancy_rows.append(
    {
        "feature":
            "away_win_expectancy",

        "compared_with":
            "100 - home_win_expectancy",

        "relation":
            "deterministic complement",

        "validated":
            bool(
                win_expectancy_valid
            ),

        "reason":
            (
                "home / away 기대승률이 "
                "상호 보완 관계인지 검증"
            ),
    }
)


redundancy_df = pd.DataFrame(
    redundancy_rows
)


print(
    redundancy_df
    .to_string(
        index=False,
    )
)

print(
    "\nMax |home + away - 100| : "
    f"{max_win_expectancy_error:.10f}"
)


redundancy_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "redundant_feature_validation.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


# =============================================================================
# 9. Reduced Feature 제거 후보 정의
#
# deterministic redundancy가 실제 데이터에서 검증된 feature만 제거한다.
# =============================================================================

REDUNDANCY_CANDIDATES = [
    "asof_pitcher_pitchmix_n",
    "run_total_before",
    "num_runners_on",
    "away_win_expectancy",
]


redundancy_validation = (
    redundancy_df
    .set_index("feature")["validated"]
    .to_dict()
)


REDUCED_DROP_COLS = [
    col
    for col in REDUNDANCY_CANDIDATES
    if redundancy_validation.get(
        col,
        False,
    )
]


REDUNDANCY_REJECTED_COLS = [
    col
    for col in REDUNDANCY_CANDIDATES
    if not redundancy_validation.get(
        col,
        False,
    )
]


print("\n" + "=" * 80)
print("7. REDUCED FEATURE CANDIDATES")
print("=" * 80)

print(
    f"Candidates        : "
    f"{REDUNDANCY_CANDIDATES}"
)

print(
    f"Validated drops   : "
    f"{REDUCED_DROP_COLS}"
)

print(
    f"Rejected drops    : "
    f"{REDUNDANCY_REJECTED_COLS}"
)

# =============================================================================
# 10. Full / Reduced Feature Set
# =============================================================================

FULL_FEATURES = ALL_FEATURES.copy()


REDUCED_FEATURES = [
    col
    for col in FULL_FEATURES
    if col not in REDUCED_DROP_COLS
]


print("\n" + "=" * 80)
print("8. FULL / REDUCED FEATURE SET")
print("=" * 80)

print(
    f"Full    : "
    f"{len(FULL_FEATURES)}"
)

print(
    f"Reduced : "
    f"{len(REDUCED_FEATURES)}"
)

print("\n[Removed from Reduced]")

for col in REDUCED_DROP_COLS:
    print(f"- {col}")


assert len(
    FULL_FEATURES
) == EXPECTED_FULL_FEATURE_COUNT


assert len(
    REDUCED_FEATURES
) == EXPECTED_REDUCED_FEATURE_COUNT


assert set(
    REDUCED_FEATURES
).issubset(
    set(
        FULL_FEATURES
    )
)


# =============================================================================
# 11. Pitcher ID 포함 / 제외용 재사용 Feature Set
#
# 이후 모델 실험에서 동일 config를 그대로 사용한다.
# =============================================================================

FULL_FEATURES_PID = (
    FULL_FEATURES.copy()
)

FULL_FEATURES_NO_PID = [
    col
    for col in FULL_FEATURES
    if col != PITCHER_ID_COL
]


REDUCED_FEATURES_PID = (
    REDUCED_FEATURES.copy()
)

REDUCED_FEATURES_NO_PID = [
    col
    for col in REDUCED_FEATURES
    if col != PITCHER_ID_COL
]


FEATURE_SETS = {
    "Full-PID":
        FULL_FEATURES_PID,

    "Full-NoPID":
        FULL_FEATURES_NO_PID,

    "Reduced-PID":
        REDUCED_FEATURES_PID,

    "Reduced-NoPID":
        REDUCED_FEATURES_NO_PID,
}


print("\n" + "=" * 80)
print("9. REUSABLE FEATURE SETS")
print("=" * 80)

for name, features in FEATURE_SETS.items():

    print(
        f"{name:<15} : "
        f"{len(features)} features"
    )


# =============================================================================
# 12. 실험별 Feature Type 반환 함수
# =============================================================================

def get_feature_types(
    feature_list,
):
    """
    특정 실험 feature set에 맞는
    numeric / categorical / ordinal / identifier 목록을 반환한다.
    """

    feature_set = set(
        feature_list
    )

    num_cols = [
        col
        for col in NUM_COLS
        if col in feature_set
    ]

    cat_cols = [
        col
        for col in CAT_COLS
        if col in feature_set
    ]

    ordinal_cols = [
        col
        for col in ORDINAL_COLS
        if col in feature_set
    ]

    id_cols = [
        col
        for col in ID_COLS
        if col in feature_set
    ]


    classified_count = (
        len(num_cols)
        + len(cat_cols)
        + len(ordinal_cols)
        + len(id_cols)
    )


    assert classified_count == len(
        feature_list
    ), (
        "실험 feature set의 일부 컬럼이 type config에 없습니다."
    )


    return {
        "numeric":
            num_cols,

        "categorical":
            cat_cols,

        "ordinal":
            ordinal_cols,

        "identifier":
            id_cols,

        "scale":
            [
                col
                for col in SCALE_COLS
                if col in feature_set
            ],

        "no_scale":
            [
                col
                for col in NO_SCALE_COLS
                if col in feature_set
            ],
    }


# =============================================================================
# 13. 전체 Feature Config Table
# =============================================================================

feature_type_lookup = {}

for feature_type, cols in FEATURE_TYPE_GROUPS.items():

    for col in cols:

        feature_type_lookup[
            col
        ] = feature_type


feature_group_lookup = {}

for feature_group, cols in FEATURE_GROUPS.items():

    for col in cols:

        feature_group_lookup[
            col
        ] = feature_group


feature_config_rows = []


for col in FULL_FEATURES:

    feature_type = (
        feature_type_lookup[
            col
        ]
    )

    feature_config_rows.append(
        {
            "feature":
                col,

            "feature_group":
                feature_group_lookup.get(
                    col,
                    "unassigned",
                ),

            "type":
                feature_type,

            "scaling":
                (
                    "scale"
                    if col in SCALE_COLS
                    else
                    "no_scale"
                    if col in NO_SCALE_COLS
                    else
                    "not_applicable"
                ),

            "high_cardinality":
                (
                    bool(
                        cardinality_df
                        .set_index(
                            "feature"
                        )
                        .loc[
                            col,
                            "high_cardinality",
                        ]
                    )
                    if col in cardinality_cols
                    else False
                ),

            "full_feature":
                True,

            "reduced_feature":
                col in REDUCED_FEATURES,

            "reduced_drop_candidate":
                col in REDUCED_DROP_COLS,
        }
    )


feature_config_df = pd.DataFrame(
    feature_config_rows
)


# 모든 feature group도 명시적으로 할당되었는지 확인
unassigned_groups = (
    feature_config_df.loc[
        feature_config_df[
            "feature_group"
        ]
        == "unassigned",
        "feature",
    ]
    .tolist()
)


assert not unassigned_groups, (
    "Feature Group이 정의되지 않은 컬럼이 있습니다: "
    f"{unassigned_groups}"
)


feature_config_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "feature_config.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


# =============================================================================
# 14. Full / Reduced 목록 저장
# =============================================================================

pd.DataFrame(
    {
        "feature":
            FULL_FEATURES
    }
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "full_features.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


pd.DataFrame(
    {
        "feature":
            REDUCED_FEATURES
    }
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "reduced_features.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


# =============================================================================
# 15. 최종 Summary
# =============================================================================

summary = {
    "total_feature_count":
        len(FULL_FEATURES),

    "numeric_count":
        len(NUM_COLS),

    "categorical_count":
        len(CAT_COLS),

    "ordinal_count":
        len(ORDINAL_COLS),

    "identifier_count":
        len(ID_COLS),

    "scale_count":
        len(SCALE_COLS),

    "no_scale_count":
        len(NO_SCALE_COLS),

    "high_cardinality_count":
        int(
            cardinality_df[
                "high_cardinality"
            ]
            .sum()
        ),

    "constant_count":
        int(
            constant_df[
                "is_constant"
            ]
            .sum()
        ),

    "near_constant_count":
        int(
            constant_df[
                "is_near_constant"
            ]
            .sum()
        ),

    "full_feature_count":
        len(FULL_FEATURES),

    "reduced_feature_count":
        len(REDUCED_FEATURES),

    "reduced_drop_count":
        len(REDUCED_DROP_COLS),
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
print("10. FINAL FEATURE CONFIG SUMMARY")
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
        "feature_config_summary.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


# =============================================================================
# 16. 최종 Assertions
# =============================================================================

# Leakage / non-feature
assert TARGET not in FULL_FEATURES
assert TARGET not in REDUCED_FEATURES

assert ID not in FULL_FEATURES
assert ID not in REDUCED_FEATURES


# 모든 feature 정확히 1개 타입
assert (
    len(NUM_COLS)
    + len(CAT_COLS)
    + len(ORDINAL_COLS)
    + len(ID_COLS)
    == len(FULL_FEATURES)
)


# Numeric scaling 완전 분류
assert (
    set(SCALE_COLS)
    | set(NO_SCALE_COLS)
) == set(NUM_COLS)


assert set(
    SCALE_COLS
).isdisjoint(
    set(
        NO_SCALE_COLS
    )
)


# Full / Reduced
assert len(
    FULL_FEATURES
) == 47

assert len(
    REDUCED_FEATURES
) == 44


print("\n" + "=" * 80)
print("LR-05 FEATURE CONFIG COMPLETE")
print("=" * 80)

print(
    f"FULL_FEATURES    : "
    f"{len(FULL_FEATURES)}"
)

print(
    f"REDUCED_FEATURES : "
    f"{len(REDUCED_FEATURES)}"
)

print(
    f"NUM_COLS         : "
    f"{len(NUM_COLS)}"
)

print(
    f"CAT_COLS         : "
    f"{len(CAT_COLS)}"
)

print(
    f"ORDINAL_COLS     : "
    f"{len(ORDINAL_COLS)}"
)

print(
    f"ID_COLS          : "
    f"{len(ID_COLS)}"
)

print(
    f"SCALE_COLS       : "
    f"{len(SCALE_COLS)}"
)

print(
    f"NO_SCALE_COLS    : "
    f"{len(NO_SCALE_COLS)}"
)

print(
    f"\nOutput directory : "
    f"{OUTPUT_DIR}"
)