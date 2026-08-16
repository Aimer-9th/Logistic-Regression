import os
import warnings

import pandas as pd


# =============================================================================
# 0. 기본 설정
# =============================================================================

warnings.filterwarnings("ignore")

DATA_DIR = "/content/drive/MyDrive/𝟐𝟎𝟐𝟔/aimers/9기/open/data"
OUTPUT_DIR = "./lr07_experiment_outputs"

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

SEASON_COL = "season"
PITCHER_ID_COL = "pitcher_id"

TRAIN_END_SEASON = 2023
VALID_SEASON = 2024

PRIMARY_METRIC = "brier"
SECONDARY_METRICS = [
    "logloss",
    "auc",
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


# LR-05 / LR-04에서 확정한 공식 model feature
ALL_FEATURES = [
    col
    for col in test.columns
    if col != ID
]


print("=" * 80)
print("1. DATA")
print("=" * 80)

print(f"Train shape        : {train.shape}")
print(f"Test shape         : {test.shape}")
print(f"All feature count  : {len(ALL_FEATURES)}")


assert len(ALL_FEATURES) == 47, (
    "LR-05에서 확정한 전체 feature 수와 다릅니다."
)

assert TARGET not in ALL_FEATURES
assert ID not in ALL_FEATURES
assert PITCHER_ID_COL in ALL_FEATURES


# =============================================================================
# 2. 공식 Historical Feature 정의
#
# LR-06에서 확인한 운영 측 제공 asof_* feature 19개.
#
# 신규 target-based rolling / expanding feature는
# 현재 main train chronology 문제로 실제 학습 데이터에는 추가하지 않는다.
# =============================================================================

HISTORICAL_FEATURES = [
    col
    for col in ALL_FEATURES
    if col.startswith("asof_")
]


print("\n" + "=" * 80)
print("2. HISTORICAL FEATURES")
print("=" * 80)

print(
    f"Historical feature count : "
    f"{len(HISTORICAL_FEATURES)}"
)

for col in HISTORICAL_FEATURES:
    print(f"- {col}")


assert len(HISTORICAL_FEATURES) == 19, (
    "공식 asof_* historical feature 수가 예상과 다릅니다."
)


# =============================================================================
# 3. Base Feature 정의
#
# Model A의 feature:
#
# 전체 공식 feature
# - pitcher_id
# - historical feature
#
# 즉 pitcher identity나 pitcher historical statistics 없이도
# 사용 가능한 경기 상황 / 선수 일반 정보만 사용한다.
# =============================================================================

BASE_FEATURES = [
    col
    for col in ALL_FEATURES
    if (
        col != PITCHER_ID_COL
        and col not in HISTORICAL_FEATURES
    )
]


print("\n" + "=" * 80)
print("3. BASE FEATURES")
print("=" * 80)

print(
    f"Base feature count : "
    f"{len(BASE_FEATURES)}"
)

for col in BASE_FEATURES:
    print(f"- {col}")


assert len(BASE_FEATURES) == 27, (
    "Base feature 수가 예상과 다릅니다."
)


# =============================================================================
# 4. Model A
#
# No Pitcher ID
# No Historical
#
# Unknown Pitcher에 가장 안전한 baseline
# =============================================================================

MODEL_A_FEATURES = (
    BASE_FEATURES.copy()
)


# =============================================================================
# 5. Model B
#
# Model A + pitcher_id
#
# pitcher identity 효과만 추가
# =============================================================================

MODEL_B_FEATURES = (
    BASE_FEATURES
    + [
        PITCHER_ID_COL,
    ]
)


# =============================================================================
# 6. Model C
#
# Model A + Historical
#
# pitcher identity 없이 공식 historical information만 추가
# =============================================================================

MODEL_C_FEATURES = (
    BASE_FEATURES
    + HISTORICAL_FEATURES
)


# =============================================================================
# 7. Model D
#
# Model A + pitcher_id + Historical
#
# 전체 47개 공식 feature
# =============================================================================

MODEL_D_FEATURES = (
    BASE_FEATURES
    + [
        PITCHER_ID_COL,
    ]
    + HISTORICAL_FEATURES
)


# =============================================================================
# 8. 실험 Config
# =============================================================================

LR_EXPERIMENTS = {
    "LR-A-NoPitcherID": {
        "features":
            MODEL_A_FEATURES,

        "use_pitcher_id":
            False,

        "use_historical":
            False,

        "description":
            (
                "Pitcher ID와 historical feature를 제외한 "
                "Unknown-safe baseline"
            ),
    },

    "LR-B-PitcherID": {
        "features":
            MODEL_B_FEATURES,

        "use_pitcher_id":
            True,

        "use_historical":
            False,

        "description":
            (
                "Model A + pitcher_id. "
                "Pitcher identity 효과 평가"
            ),
    },

    "LR-C-Historical": {
        "features":
            MODEL_C_FEATURES,

        "use_pitcher_id":
            False,

        "use_historical":
            True,

        "description":
            (
                "Model A + 공식 asof_* historical feature. "
                "Identity 없이 history 효과 평가"
            ),
    },

    "LR-D-PitcherID-Historical": {
        "features":
            MODEL_D_FEATURES,

        "use_pitcher_id":
            True,

        "use_historical":
            True,

        "description":
            (
                "Model A + pitcher_id + historical. "
                "Identity와 history 동시 사용"
            ),
    },
}


# =============================================================================
# 9. Feature Count 검증
# =============================================================================

EXPECTED_FEATURE_COUNTS = {
    "LR-A-NoPitcherID": 27,
    "LR-B-PitcherID": 28,
    "LR-C-Historical": 46,
    "LR-D-PitcherID-Historical": 47,
}


print("\n" + "=" * 100)
print("4. EXPERIMENT CONFIG")
print("=" * 100)

for name, config in LR_EXPERIMENTS.items():

    feature_count = len(
        config["features"]
    )

    print(
        f"{name:<30} | "
        f"PID={str(config['use_pitcher_id']):<5} | "
        f"Historical={str(config['use_historical']):<5} | "
        f"Features={feature_count:>2}"
    )

    assert (
        feature_count
        == EXPECTED_FEATURE_COUNTS[name]
    ), (
        f"{name} feature count가 예상과 다릅니다."
    )


# =============================================================================
# 10. 각 Feature Set 내부 중복 검증
# =============================================================================

for name, config in LR_EXPERIMENTS.items():

    features = config[
        "features"
    ]

    assert (
        len(features)
        == len(set(features))
    ), (
        f"{name}: 중복 feature가 존재합니다."
    )

    assert TARGET not in features, (
        f"{name}: target이 포함되어 있습니다."
    )

    assert ID not in features, (
        f"{name}: row_id가 포함되어 있습니다."
    )


# =============================================================================
# 11. 실험 구조 검증
#
# 2 × 2 factorial design이 정확히 유지되는지 확인
# =============================================================================

A = set(
    MODEL_A_FEATURES
)

B = set(
    MODEL_B_FEATURES
)

C = set(
    MODEL_C_FEATURES
)

D = set(
    MODEL_D_FEATURES
)


# -----------------------------------------------------------------------------
# B - A = pitcher_id 하나만 추가
# -----------------------------------------------------------------------------

assert (
    B - A
) == {
    PITCHER_ID_COL
}, (
    "Model B는 Model A에 pitcher_id만 추가되어야 합니다."
)


# -----------------------------------------------------------------------------
# C - A = historical feature 19개만 추가
# -----------------------------------------------------------------------------

assert (
    C - A
) == set(
    HISTORICAL_FEATURES
), (
    "Model C는 Model A에 historical feature만 추가되어야 합니다."
)


# -----------------------------------------------------------------------------
# D - C = pitcher_id
# -----------------------------------------------------------------------------

assert (
    D - C
) == {
    PITCHER_ID_COL
}, (
    "Model D와 C의 차이는 pitcher_id 하나여야 합니다."
)


# -----------------------------------------------------------------------------
# D - B = historical features
# -----------------------------------------------------------------------------

assert (
    D - B
) == set(
    HISTORICAL_FEATURES
), (
    "Model D와 B의 차이는 historical feature만이어야 합니다."
)


# -----------------------------------------------------------------------------
# D = 전체 공식 feature
# -----------------------------------------------------------------------------

assert D == set(
    ALL_FEATURES
), (
    "Model D는 전체 47개 공식 feature와 일치해야 합니다."
)


print("\n2 × 2 experiment structure validation: PASS")


# =============================================================================
# 12. 동일 Validation Split 정의
#
# 모든 모델:
#
# Train : 2019 ~ 2023
# Valid : 2024
#
# 동일 row를 사용하여 feature 구성만 비교한다.
# =============================================================================

train_mask = (
    train[
        SEASON_COL
    ]
    <= TRAIN_END_SEASON
)

valid_mask = (
    train[
        SEASON_COL
    ]
    == VALID_SEASON
)


train_fold = (
    train.loc[
        train_mask
    ]
    .reset_index(
        drop=True
    )
)

valid_fold = (
    train.loc[
        valid_mask
    ]
    .reset_index(
        drop=True
    )
)


print("\n" + "=" * 80)
print("5. COMMON VALIDATION SPLIT")
print("=" * 80)

print(
    "Train seasons :",
    sorted(
        train_fold[
            SEASON_COL
        ]
        .unique()
    ),
)

print(
    "Valid seasons :",
    sorted(
        valid_fold[
            SEASON_COL
        ]
        .unique()
    ),
)

print(
    f"Train rows     : "
    f"{len(train_fold):,}"
)

print(
    f"Valid rows     : "
    f"{len(valid_fold):,}"
)

print(
    f"Train target   : "
    f"{train_fold[TARGET].mean():.6f}"
)

print(
    f"Valid target   : "
    f"{valid_fold[TARGET].mean():.6f}"
)


# =============================================================================
# 13. Known / Unknown Pitcher Validation Slice
#
# 전체 모델은 동일 validation row를 사용한다.
#
# Model A / C는 pitcher_id를 feature로 사용하지 않더라도
# 성능 분석용 grouping에는 pitcher_id를 사용할 수 있다.
#
# 즉 pitcher_id는 평가 slice 정의용 metadata로만 사용한다.
# =============================================================================

train_pitchers = set(
    train_fold[
        PITCHER_ID_COL
    ]
    .dropna()
    .unique()
)


known_pitcher_mask = (
    valid_fold[
        PITCHER_ID_COL
    ]
    .isin(
        train_pitchers
    )
)

unknown_pitcher_mask = (
    ~known_pitcher_mask
)


print("\n" + "=" * 80)
print("6. KNOWN / UNKNOWN VALIDATION SLICE")
print("=" * 80)

print(
    f"Known rows   : "
    f"{known_pitcher_mask.sum():,} "
    f"({known_pitcher_mask.mean():.2%})"
)

print(
    f"Unknown rows : "
    f"{unknown_pitcher_mask.sum():,} "
    f"({unknown_pitcher_mask.mean():.2%})"
)

print(
    f"Known pitchers   : "
    f"{valid_fold.loc[known_pitcher_mask, PITCHER_ID_COL].nunique():,}"
)

print(
    f"Unknown pitchers : "
    f"{valid_fold.loc[unknown_pitcher_mask, PITCHER_ID_COL].nunique():,}"
)


# =============================================================================
# 14. 동일 Metric 사용 원칙
#
# Primary:
#   Brier Score
#
# Secondary:
#   Log Loss
#   ROC-AUC
#
# 모든 A/B/C/D 모델에서 동일하게 사용한다.
# =============================================================================

print("\n" + "=" * 80)
print("7. COMMON METRIC")
print("=" * 80)

print(
    f"Primary metric    : "
    f"{PRIMARY_METRIC}"
)

print(
    f"Secondary metrics : "
    f"{SECONDARY_METRICS}"
)


# =============================================================================
# 15. 비교 목적 정의
# =============================================================================

EXPERIMENT_COMPARISONS = {
    "Pitcher ID effect without history": {
        "baseline":
            "LR-A-NoPitcherID",

        "comparison":
            "LR-B-PitcherID",

        "difference":
            "pitcher_id only",
    },

    "Historical effect without pitcher ID": {
        "baseline":
            "LR-A-NoPitcherID",

        "comparison":
            "LR-C-Historical",

        "difference":
            "historical features only",
    },

    "Pitcher ID effect with history": {
        "baseline":
            "LR-C-Historical",

        "comparison":
            "LR-D-PitcherID-Historical",

        "difference":
            "pitcher_id only",
    },

    "Historical effect with pitcher ID": {
        "baseline":
            "LR-B-PitcherID",

        "comparison":
            "LR-D-PitcherID-Historical",

        "difference":
            "historical features only",
    },
}


print("\n" + "=" * 100)
print("8. EXPERIMENT COMPARISONS")
print("=" * 100)

for comparison_name, config in EXPERIMENT_COMPARISONS.items():

    print(
        f"\n[{comparison_name}]"
    )

    print(
        f"  {config['baseline']}"
        f" -> "
        f"{config['comparison']}"
    )

    print(
        f"  Difference: "
        f"{config['difference']}"
    )


# =============================================================================
# 16. Experiment Config Table 저장
# =============================================================================

experiment_rows = []


for name, config in LR_EXPERIMENTS.items():

    experiment_rows.append(
        {
            "model":
                name,

            "pitcher_id":
                config[
                    "use_pitcher_id"
                ],

            "historical":
                config[
                    "use_historical"
                ],

            "feature_count":
                len(
                    config[
                        "features"
                    ]
                ),

            "primary_metric":
                PRIMARY_METRIC,

            "train_seasons":
                "2019-2023",

            "valid_season":
                2024,

            "description":
                config[
                    "description"
                ],
        }
    )


experiment_config_df = pd.DataFrame(
    experiment_rows
)


print("\n" + "=" * 100)
print("9. FINAL EXPERIMENT TABLE")
print("=" * 100)

print(
    experiment_config_df[
        [
            "model",
            "pitcher_id",
            "historical",
            "feature_count",
        ]
    ]
    .to_string(
        index=False,
    )
)


experiment_config_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lr07_experiment_config.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


# =============================================================================
# 17. Feature 목록 저장
# =============================================================================

feature_set_rows = []


for model_name, config in LR_EXPERIMENTS.items():

    for feature in config[
        "features"
    ]:

        feature_set_rows.append(
            {
                "model":
                    model_name,

                "feature":
                    feature,

                "is_pitcher_id":
                    feature
                    == PITCHER_ID_COL,

                "is_historical":
                    feature
                    in HISTORICAL_FEATURES,
            }
        )


feature_set_df = pd.DataFrame(
    feature_set_rows
)


feature_set_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lr07_feature_sets.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


# =============================================================================
# 18. 최종 Summary
# =============================================================================

summary = {
    "all_feature_count":
        len(ALL_FEATURES),

    "base_feature_count":
        len(BASE_FEATURES),

    "historical_feature_count":
        len(HISTORICAL_FEATURES),

    "model_a_feature_count":
        len(MODEL_A_FEATURES),

    "model_b_feature_count":
        len(MODEL_B_FEATURES),

    "model_c_feature_count":
        len(MODEL_C_FEATURES),

    "model_d_feature_count":
        len(MODEL_D_FEATURES),

    "train_seasons":
        "2019-2023",

    "valid_season":
        VALID_SEASON,

    "primary_metric":
        PRIMARY_METRIC,

    "factorial_structure_valid":
        True,
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
print("10. LR-07 SUMMARY")
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
        "lr07_summary.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


print("\n" + "=" * 80)
print("LR-07 EXPERIMENT CONFIG COMPLETE")
print("=" * 80)