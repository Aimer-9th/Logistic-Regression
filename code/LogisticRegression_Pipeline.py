import os
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =============================================================================
# 0. 기본 설정
# =============================================================================

warnings.filterwarnings("ignore")

DATA_DIR = "/content/drive/MyDrive/𝟐𝟎𝟐𝟔/aimers/9기/open/data"
OUTPUT_DIR = "./lr08_pipeline_outputs"

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

RANDOM_STATE = 42

LR_C = 1.0
LR_MAX_ITER = 500


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


ALL_FEATURES = [
    col
    for col in test.columns
    if col != ID
]


print("=" * 80)
print("1. DATA")
print("=" * 80)

print(f"Train shape       : {train.shape}")
print(f"Test shape        : {test.shape}")
print(f"All feature count : {len(ALL_FEATURES)}")


assert len(ALL_FEATURES) == 47
assert TARGET not in ALL_FEATURES
assert ID not in ALL_FEATURES


# =============================================================================
# 2. LR-05 Feature Type 정의
# =============================================================================

CAT_COLS = [
    "game_dayofweek",
    "top_bottom",
    "game_type",
    "base_state",

    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
]


ID_COLS = [
    "pitcher_id",
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

    # 타자 historical
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",

    # 투수 pitch mix historical
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]


ORDINAL_COLS = []


# =============================================================================
# 3. Numeric Scaling 정의
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


assert set(SCALE_COLS).isdisjoint(
    set(NO_SCALE_COLS)
)

assert (
    set(SCALE_COLS)
    | set(NO_SCALE_COLS)
) == set(NUM_COLS)


print("\n" + "=" * 80)
print("2. FEATURE TYPE")
print("=" * 80)

print(f"Numeric       : {len(NUM_COLS)}")
print(f"Categorical   : {len(CAT_COLS)}")
print(f"Identifier    : {len(ID_COLS)}")
print(f"Ordinal       : {len(ORDINAL_COLS)}")
print(f"Scale numeric : {len(SCALE_COLS)}")
print(f"No-scale      : {len(NO_SCALE_COLS)}")


# =============================================================================
# 4. LR-07 실험군 정의
# =============================================================================

HISTORICAL_FEATURES = [
    col
    for col in ALL_FEATURES
    if col.startswith("asof_")
]


BASE_FEATURES = [
    col
    for col in ALL_FEATURES
    if (
        col != PITCHER_ID_COL
        and col not in HISTORICAL_FEATURES
    )
]


MODEL_A_FEATURES = (
    BASE_FEATURES.copy()
)

MODEL_B_FEATURES = (
    BASE_FEATURES
    + [
        PITCHER_ID_COL,
    ]
)

MODEL_C_FEATURES = (
    BASE_FEATURES
    + HISTORICAL_FEATURES
)

MODEL_D_FEATURES = (
    BASE_FEATURES
    + [
        PITCHER_ID_COL,
    ]
    + HISTORICAL_FEATURES
)


LR_EXPERIMENTS = {
    "LR-A-NoPitcherID": {
        "features":
            MODEL_A_FEATURES,
    },

    "LR-B-PitcherID": {
        "features":
            MODEL_B_FEATURES,
    },

    "LR-C-Historical": {
        "features":
            MODEL_C_FEATURES,
    },

    "LR-D-PitcherID-Historical": {
        "features":
            MODEL_D_FEATURES,
    },
}


EXPECTED_COUNTS = {
    "LR-A-NoPitcherID": 27,
    "LR-B-PitcherID": 28,
    "LR-C-Historical": 46,
    "LR-D-PitcherID-Historical": 47,
}


for name, config in LR_EXPERIMENTS.items():

    assert (
        len(config["features"])
        == EXPECTED_COUNTS[name]
    )


# =============================================================================
# 5. 특정 Feature Set의 타입 반환
#
# pitcher_id는 identifier로 관리하지만 LR preprocessing에서는
# categorical branch로 전달한다.
# =============================================================================

def get_feature_types(
    feature_list,
):
    """
    선택된 feature set에 맞춰 preprocessing column을 반환한다.
    """

    feature_set = set(
        feature_list
    )


    scale_cols = [
        col
        for col in SCALE_COLS
        if col in feature_set
    ]


    no_scale_cols = [
        col
        for col in NO_SCALE_COLS
        if col in feature_set
    ]


    categorical_cols = [
        col
        for col in CAT_COLS
        if col in feature_set
    ]


    identifier_cols = [
        col
        for col in ID_COLS
        if col in feature_set
    ]


    # identifier는 모델 입력에서 categorical preprocessing
    categorical_model_cols = (
        categorical_cols
        + identifier_cols
    )


    classified_cols = (
        scale_cols
        + no_scale_cols
        + categorical_model_cols
    )


    assert len(
        classified_cols
    ) == len(
        feature_list
    ), (
        "일부 feature가 preprocessing type에 포함되지 않았습니다."
    )


    assert len(
        classified_cols
    ) == len(
        set(classified_cols)
    ), (
        "preprocessing type 간 feature 중복이 있습니다."
    )


    return {
        "scale":
            scale_cols,

        "no_scale":
            no_scale_cols,

        "categorical":
            categorical_model_cols,

        "identifier":
            identifier_cols,
    }


# =============================================================================
# 6. Preprocessing Builder
#
# A. Scale numeric
#    Median Imputer
#    + Missing Indicator
#    + StandardScaler
#
# B. No-scale numeric
#    Median Imputer
#    + Missing Indicator
#
# C. Categorical / identifier
#    Constant missing category
#    + OneHotEncoder(handle_unknown="ignore")
# =============================================================================

def build_preprocessor(
    feature_list,
):
    """
    Logistic Regression용 preprocessing ColumnTransformer 생성.
    """

    feature_types = get_feature_types(
        feature_list
    )


    scale_cols = feature_types[
        "scale"
    ]

    no_scale_cols = feature_types[
        "no_scale"
    ]

    categorical_cols = feature_types[
        "categorical"
    ]


    # -------------------------------------------------------------------------
    # Scaling numeric pipeline
    # -------------------------------------------------------------------------

    scale_numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )


    # -------------------------------------------------------------------------
    # No-scaling numeric pipeline
    # -------------------------------------------------------------------------

    no_scale_numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
        ]
    )


    # -------------------------------------------------------------------------
    # Categorical pipeline
    #
    # missing을 명시적인 category로 유지
    #
    # pitcher_id unknown 역시 OneHotEncoder가 ignore 처리
    # -------------------------------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__MISSING__",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )


    transformers = []


    if scale_cols:

        transformers.append(
            (
                "numeric_scale",
                scale_numeric_pipeline,
                scale_cols,
            )
        )


    if no_scale_cols:

        transformers.append(
            (
                "numeric_no_scale",
                no_scale_numeric_pipeline,
                no_scale_cols,
            )
        )


    if categorical_cols:

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_cols,
            )
        )


    preprocessor = ColumnTransformer(
        transformers=transformers,

        remainder="drop",

        # OneHotEncoder의 sparse output을 유지하기 위한 설정
        sparse_threshold=1.0,

        verbose_feature_names_out=True,
    )


    return preprocessor


# =============================================================================
# 7. Logistic Regression Pipeline Builder
# =============================================================================

def build_lr_pipeline(
    feature_list,
    C=LR_C,
    max_iter=LR_MAX_ITER,
):
    """
    Preprocessing + Logistic Regression 전체 sklearn Pipeline.
    """

    preprocessor = build_preprocessor(
        feature_list
    )


    model = LogisticRegression(
        C=C,
        penalty="l2",

        # sparse matrix 호환
        solver="saga",

        max_iter=max_iter,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )


    return pipeline


# =============================================================================
# 8. Validation Split
#
# LR-07과 완전히 동일
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
print("3. VALIDATION SPLIT")
print("=" * 80)

print(
    f"Train rows : "
    f"{len(train_fold):,}"
)

print(
    f"Valid rows : "
    f"{len(valid_fold):,}"
)


# =============================================================================
# 9. Unit Test용 작은 Sample
#
# 전체 120만 row를 fit하지 않고 preprocessing 동작 자체만 빠르게 검증.
# =============================================================================

UNIT_TEST_TRAIN_SIZE = min(
    50_000,
    len(train_fold),
)

UNIT_TEST_VALID_SIZE = min(
    10_000,
    len(valid_fold),
)


unit_train = (
    train_fold
    .iloc[
        :UNIT_TEST_TRAIN_SIZE
    ]
    .copy()
)

unit_valid = (
    valid_fold
    .iloc[
        :UNIT_TEST_VALID_SIZE
    ]
    .copy()
)


print("\n" + "=" * 80)
print("4. UNIT TEST DATA")
print("=" * 80)

print(
    f"Unit train rows : "
    f"{len(unit_train):,}"
)

print(
    f"Unit valid rows : "
    f"{len(unit_valid):,}"
)


# =============================================================================
# 10. Train Fit / Valid Transform 분리 검증
#
# 핵심:
#
# preprocessor.fit(X_train)
# preprocessor.transform(X_valid)
#
# valid에서 fit 또는 fit_transform 호출하지 않는다.
# =============================================================================

test_feature_set = (
    MODEL_D_FEATURES
)


feature_types = get_feature_types(
    test_feature_set
)


preprocessor = build_preprocessor(
    test_feature_set
)


X_unit_train = unit_train[
    test_feature_set
]

X_unit_valid = unit_valid[
    test_feature_set
]


X_train_transformed = (
    preprocessor
    .fit_transform(
        X_unit_train
    )
)


X_valid_transformed = (
    preprocessor
    .transform(
        X_unit_valid
    )
)


print("\n" + "=" * 80)
print("5. TRAIN FIT / VALID TRANSFORM")
print("=" * 80)

print(
    f"Train transformed shape : "
    f"{X_train_transformed.shape}"
)

print(
    f"Valid transformed shape : "
    f"{X_valid_transformed.shape}"
)


assert (
    X_train_transformed.shape[1]
    == X_valid_transformed.shape[1]
), (
    "Train / Valid transformed column 수가 다릅니다."
)


print(
    "✓ Preprocessor fitted on train only"
)

print(
    "✓ Validation transformed without fit"
)


# =============================================================================
# 11. Sparse / Dense Output 확인
# =============================================================================

from scipy import sparse


train_is_sparse = sparse.issparse(
    X_train_transformed
)

valid_is_sparse = sparse.issparse(
    X_valid_transformed
)


print("\n" + "=" * 80)
print("6. SPARSE / DENSE OUTPUT")
print("=" * 80)

print(
    f"Train output sparse : "
    f"{train_is_sparse}"
)

print(
    f"Valid output sparse : "
    f"{valid_is_sparse}"
)

print(
    f"Train dtype         : "
    f"{X_train_transformed.dtype}"
)

print(
    f"Valid dtype         : "
    f"{X_valid_transformed.dtype}"
)


assert (
    train_is_sparse
    == valid_is_sparse
), (
    "Train / Valid sparse output 형식이 다릅니다."
)


# =============================================================================
# 12. Feature Name 추출
# =============================================================================

print("\n" + "=" * 80)
print("7. FEATURE NAME EXTRACTION")
print("=" * 80)


transformed_feature_names = (
    preprocessor
    .get_feature_names_out()
)


print(
    f"Original feature count    : "
    f"{len(test_feature_set)}"
)

print(
    f"Transformed feature count : "
    f"{len(transformed_feature_names)}"
)


print("\nFirst 30 transformed features:")

for feature_name in transformed_feature_names[:30]:

    print(
        f"- {feature_name}"
    )


assert (
    len(transformed_feature_names)
    == X_train_transformed.shape[1]
)


pd.DataFrame(
    {
        "transformed_feature":
            transformed_feature_names
    }
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "transformed_feature_names.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


# =============================================================================
# 13. Unknown Category Test
#
# validation sample의 categorical value를 일부러 train에 없는 값으로 변경.
# handle_unknown="ignore"가 오류 없이 동작하는지 확인.
# =============================================================================

print("\n" + "=" * 80)
print("8. UNKNOWN CATEGORY TEST")
print("=" * 80)


unknown_test = (
    unit_valid[
        test_feature_set
    ]
    .head(10)
    .copy()
)


categorical_test_cols = feature_types[
    "categorical"
]


if categorical_test_cols:

    test_unknown_col = (
        categorical_test_cols[0]
    )


    # dtype 충돌 방지를 위해 object로 변환
    unknown_test[
        test_unknown_col
    ] = (
        unknown_test[
            test_unknown_col
        ]
        .astype("object")
    )


    unknown_test.loc[
        unknown_test.index[0],
        test_unknown_col,
    ] = "__UNSEEN_CATEGORY__"


    unknown_transformed = (
        preprocessor
        .transform(
            unknown_test
        )
    )


    print(
        f"Test column      : "
        f"{test_unknown_col}"
    )

    print(
        f"Unknown category : "
        f"__UNSEEN_CATEGORY__"
    )

    print(
        f"Transform result : "
        f"PASS"
    )

    print(
        f"Output shape     : "
        f"{unknown_transformed.shape}"
    )


else:

    print(
        "Categorical feature가 없어 test 생략"
    )


# =============================================================================
# 14. Missing Value Test
#
# numeric / categorical 모두 NaN을 넣어도 transform 가능한지 확인.
# =============================================================================

print("\n" + "=" * 80)
print("9. MISSING VALUE TEST")
print("=" * 80)


missing_test = (
    unit_valid[
        test_feature_set
    ]
    .head(10)
    .copy()
)


if feature_types[
    "scale"
]:

    missing_test.loc[
        missing_test.index[0],
        feature_types[
            "scale"
        ][0],
    ] = np.nan


if feature_types[
    "no_scale"
]:

    missing_test.loc[
        missing_test.index[0],
        feature_types[
            "no_scale"
        ][0],
    ] = np.nan


if feature_types[
    "categorical"
]:

    missing_col = feature_types[
        "categorical"
    ][0]

    # integer ID column일 수도 있으므로 object로 변환
    missing_test[
        missing_col
    ] = (
        missing_test[
            missing_col
        ]
        .astype("object")
    )

    missing_test.loc[
        missing_test.index[0],
        missing_col,
    ] = np.nan


missing_transformed = (
    preprocessor
    .transform(
        missing_test
    )
)


print(
    "Missing value transform : PASS"
)

print(
    f"Output shape            : "
    f"{missing_transformed.shape}"
)


# =============================================================================
# 15. Full Pipeline Fit Test
#
# preprocessing + LogisticRegression이 실제 연결되는지 확인
# =============================================================================

print("\n" + "=" * 80)
print("10. FULL PIPELINE FIT TEST")
print("=" * 80)


pipeline = build_lr_pipeline(
    test_feature_set
)


X_train_test = unit_train[
    test_feature_set
]

y_train_test = unit_train[
    TARGET
]

X_valid_test = unit_valid[
    test_feature_set
]


pipeline.fit(
    X_train_test,
    y_train_test,
)


valid_probability = (
    pipeline
    .predict_proba(
        X_valid_test
    )[:, 1]
)


print(
    f"Prediction rows  : "
    f"{len(valid_probability):,}"
)

print(
    f"Prediction min   : "
    f"{valid_probability.min():.6f}"
)

print(
    f"Prediction max   : "
    f"{valid_probability.max():.6f}"
)

print(
    f"Prediction mean  : "
    f"{valid_probability.mean():.6f}"
)


assert np.isfinite(
    valid_probability
).all()

assert (
    valid_probability
    >= 0
).all()

assert (
    valid_probability
    <= 1
).all()


print(
    "✓ Preprocessor -> LogisticRegression 연결 정상"
)


# =============================================================================
# 16. 전체 A/B/C/D Config 재사용 검증
#
# 실제 전체 모델 학습은 하지 않고 각 config에서 preprocessing builder가
# 정상 생성되는지만 확인.
# =============================================================================

print("\n" + "=" * 100)
print("11. EXPERIMENT PIPELINE CONFIG TEST")
print("=" * 100)


experiment_pipeline_rows = []


for name, config in LR_EXPERIMENTS.items():

    features = config[
        "features"
    ]

    types = get_feature_types(
        features
    )

    experiment_pipeline = build_lr_pipeline(
        features
    )


    experiment_pipeline_rows.append(
        {
            "model":
                name,

            "feature_count":
                len(features),

            "scale_numeric":
                len(
                    types["scale"]
                ),

            "no_scale_numeric":
                len(
                    types["no_scale"]
                ),

            "categorical":
                len(
                    types["categorical"]
                ),

            "identifier":
                len(
                    types["identifier"]
                ),

            "pipeline_build":
                "PASS",
        }
    )


experiment_pipeline_df = pd.DataFrame(
    experiment_pipeline_rows
)


print(
    experiment_pipeline_df
    .to_string(
        index=False,
    )
)


experiment_pipeline_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "experiment_pipeline_config.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


# =============================================================================
# 17. Data Leakage 방지 구조 검증
#
# sklearn Pipeline의 의미:
#
# pipeline.fit(X_train)
#   -> imputer.fit(train)
#   -> scaler.fit(train)
#   -> encoder.fit(train)
#   -> LR.fit(train transformed)
#
# pipeline.predict_proba(X_valid)
#   -> 이미 train에서 fit된 객체의 transform만 사용
# =============================================================================

print("\n" + "=" * 80)
print("12. DATA LEAKAGE CHECK")
print("=" * 80)


fitted_preprocessor = (
    pipeline
    .named_steps[
        "preprocessor"
    ]
)


print(
    "Pipeline fit source     : train only"
)

print(
    "Validation preprocessing: transform only"
)

print(
    "Validation refit        : False"
)

print(
    "Test refit              : False"
)


# fitted 여부 sanity check
assert hasattr(
    fitted_preprocessor,
    "transformers_",
)


# =============================================================================
# 18. 최종 Summary
# =============================================================================

summary = {
    "numeric_total":
        len(NUM_COLS),

    "numeric_scaled":
        len(SCALE_COLS),

    "numeric_no_scale":
        len(NO_SCALE_COLS),

    "categorical":
        len(CAT_COLS),

    "identifier":
        len(ID_COLS),

    "numeric_imputer":
        "median + missing indicator",

    "categorical_imputer":
        "constant __MISSING__",

    "categorical_encoder":
        "OneHotEncoder",

    "handle_unknown":
        "ignore",

    "scaled_numeric":
        "StandardScaler",

    "train_fit_valid_transform":
        "PASS",

    "unknown_category_test":
        "PASS",

    "missing_value_test":
        "PASS",

    "feature_name_extraction":
        "PASS",

    "full_pipeline_fit":
        "PASS",

    "sparse_dense_compatible":
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
print("13. LR-08 SUMMARY")
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
        "lr08_pipeline_summary.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


print("\n" + "=" * 80)
print("LR-08 PIPELINE VALIDATION COMPLETE")
print("=" * 80)

print(
    f"Output directory : "
    f"{OUTPUT_DIR}"
)