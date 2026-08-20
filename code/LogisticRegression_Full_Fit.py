import gc
import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn

from scipy import sparse

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


# =============================================================================
# 0. 기본 설정
# =============================================================================

warnings.filterwarnings("ignore")

DATA_DIR = "/content/drive/MyDrive/𝟐𝟎𝟐𝟔/aimers/9기/open/data"

OUTPUT_DIR = (
    "/content/drive/MyDrive/𝟐𝟎𝟐𝟔/aimers/9기/open/"
    "lr17_final_training_outputs"
)

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

RANDOM_STATE = 42


# =============================================================================
# 1. LR-16에서 확정된 Final Configuration
#
# LR-17에서는 절대 다시 tuning / selection 하지 않는다.
# =============================================================================

FINAL_FEATURE_SET_NAME = "M3-Historical-Reduced"

FINAL_RAW_FEATURE_COUNT = 43

FINAL_PITCHER_ID = False
FINAL_HISTORICAL = True
FINAL_ROLLING = False

FINAL_PENALTY = "l1"
FINAL_C = 0.001
FINAL_SOLVER = "saga"
FINAL_MAX_ITER = 500
FINAL_TOL = 1e-3

FINAL_CALIBRATION = "Raw"
FINAL_POST_CALIBRATION = False

TRAIN_START_SEASON = 2019
TRAIN_END_SEASON = 2024

COEF_ZERO_THRESHOLD = 1e-8


# =============================================================================
# 2. Artifact Path
# =============================================================================

FINAL_PIPELINE_PATH = os.path.join(
    OUTPUT_DIR,
    "lr17_final_pipeline.joblib",
)

FINAL_CONFIG_PATH = os.path.join(
    OUTPUT_DIR,
    "lr17_final_config.json",
)

FINAL_FEATURES_PATH = os.path.join(
    OUTPUT_DIR,
    "lr17_final_features.csv",
)

TRAINING_SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "lr17_training_summary.csv",
)

TEST_SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "lr17_test_prediction_summary.csv",
)


# =============================================================================
# 3. 실행 환경
# =============================================================================

print("=" * 80)
print("ENVIRONMENT")
print("=" * 80)

print(f"scikit-learn : {sklearn.__version__}")
print(f"NumPy        : {np.__version__}")
print(f"Pandas       : {pd.__version__}")


print("\n" + "=" * 80)
print("FINAL CONFIGURATION — LOCKED")
print("=" * 80)

print(f"Feature Set : {FINAL_FEATURE_SET_NAME}")
print(f"Raw Features: {FINAL_RAW_FEATURE_COUNT}")
print(f"Pitcher ID  : {FINAL_PITCHER_ID}")
print(f"Historical  : {FINAL_HISTORICAL}")
print(f"Rolling     : {FINAL_ROLLING}")
print(f"Penalty     : {FINAL_PENALTY}")
print(f"C           : {FINAL_C}")
print(f"Solver      : {FINAL_SOLVER}")
print(f"max_iter    : {FINAL_MAX_ITER}")
print(f"tol         : {FINAL_TOL}")
print(f"Calibration : {FINAL_CALIBRATION}")
print(f"Post-Cal    : {FINAL_POST_CALIBRATION}")


# =============================================================================
# 4. Train / Test Schema 확인
#
# 실제 row를 읽기 전에 header만 확인한다.
# =============================================================================

train_columns = pd.read_csv(
    TRAIN_PATH,
    encoding="utf-8-sig",
    nrows=0,
).columns.tolist()


test_columns = pd.read_csv(
    TEST_PATH,
    encoding="utf-8-sig",
    nrows=0,
).columns.tolist()


assert TARGET in train_columns
assert TARGET not in test_columns

assert ID in train_columns
assert ID in test_columns


ALL_FEATURES = [
    col
    for col in test_columns
    if col != ID
]


assert len(ALL_FEATURES) == 47


# =============================================================================
# 5. Feature Type 정의
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

    # 투수 누적 historical
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",

    # 최근 경기 historical
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


assert (
    set(SCALE_COLS)
    | set(NO_SCALE_COLS)
) == set(NUM_COLS)


# =============================================================================
# 6. Final Feature Set
#
# M3 = 전체 47 feature
#      - pitcher_id
#
# Reduced =
#      - run_total_before
#      - num_runners_on
#      - asof_pitcher_pitchmix_n
# =============================================================================

HISTORICAL_FEATURES = [
    col
    for col in ALL_FEATURES
    if col.startswith(
        "asof_"
    )
]


assert len(
    HISTORICAL_FEATURES
) == 19


M3_FULL_FEATURES = [
    col
    for col in ALL_FEATURES
    if col != PITCHER_ID_COL
]


assert len(
    M3_FULL_FEATURES
) == 46


REDUCED_DROP_COLS = [
    "run_total_before",
    "num_runners_on",
    "asof_pitcher_pitchmix_n",
]


FINAL_FEATURES = [
    col
    for col in M3_FULL_FEATURES
    if col not in REDUCED_DROP_COLS
]


# =============================================================================
# 7. Final Feature Assertion
# =============================================================================

assert len(
    FINAL_FEATURES
) == FINAL_RAW_FEATURE_COUNT


assert PITCHER_ID_COL not in FINAL_FEATURES


assert not set(
    REDUCED_DROP_COLS
).intersection(
    FINAL_FEATURES
)


assert set(
    FINAL_FEATURES
).issubset(
    train_columns
)


assert set(
    FINAL_FEATURES
).issubset(
    test_columns
)


assert len(
    FINAL_FEATURES
) == len(
    set(
        FINAL_FEATURES
    )
)


print("\n" + "=" * 80)
print("FINAL FEATURE SET")
print("=" * 80)

print(
    f"M3 Full          : "
    f"{len(M3_FULL_FEATURES)}"
)

print(
    f"Removed          : "
    f"{len(REDUCED_DROP_COLS)}"
)

print(
    f"Final Raw Feature: "
    f"{len(FINAL_FEATURES)}"
)


print("\nRemoved redundant features:")

for col in REDUCED_DROP_COLS:

    print(
        f"  - {col}"
    )


# =============================================================================
# 8. Final Feature List 저장
# =============================================================================

pd.DataFrame(
    {
        "feature":
            FINAL_FEATURES,
    }
).to_csv(
    FINAL_FEATURES_PATH,
    encoding="utf-8-sig",
    index=False,
)


# =============================================================================
# 9. Feature Type 반환
# =============================================================================

def get_feature_types(
    feature_list,
):

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


    classified_cols = (
        scale_cols
        + no_scale_cols
        + categorical_cols
    )


    assert len(
        classified_cols
    ) == len(
        feature_list
    )


    assert len(
        classified_cols
    ) == len(
        set(
            classified_cols
        )
    )


    return {
        "scale":
            scale_cols,

        "no_scale":
            no_scale_cols,

        "categorical":
            categorical_cols,
    }


# =============================================================================
# 10. Final Preprocessor
# =============================================================================

def build_preprocessor(
    feature_list,
):

    feature_types = (
        get_feature_types(
            feature_list
        )
    )


    # -------------------------------------------------------------------------
    # Numeric — Scaling
    # -------------------------------------------------------------------------

    numeric_scale_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    copy=False,
                ),
            ),
            (
                "scaler",
                StandardScaler(
                    copy=False,
                ),
            ),
        ]
    )


    # -------------------------------------------------------------------------
    # Numeric — No Scaling
    # -------------------------------------------------------------------------

    numeric_no_scale_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    copy=False,
                ),
            ),
        ]
    )


    # -------------------------------------------------------------------------
    # Categorical
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
                    dtype=np.float32,
                ),
            ),
        ]
    )


    transformers = []


    if feature_types[
        "scale"
    ]:

        transformers.append(
            (
                "numeric_scale",
                numeric_scale_pipeline,
                feature_types[
                    "scale"
                ],
            )
        )


    if feature_types[
        "no_scale"
    ]:

        transformers.append(
            (
                "numeric_no_scale",
                numeric_no_scale_pipeline,
                feature_types[
                    "no_scale"
                ],
            )
        )


    if feature_types[
        "categorical"
    ]:

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                feature_types[
                    "categorical"
                ],
            )
        )


    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",

        # OHE 결과를 sparse로 유지
        sparse_threshold=1.0,

        verbose_feature_names_out=True,
    )


# =============================================================================
# 11. Final Pipeline
# =============================================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            build_preprocessor(
                FINAL_FEATURES
            ),
        ),
        (
            "model",
            LogisticRegression(
                penalty=FINAL_PENALTY,
                C=FINAL_C,
                solver=FINAL_SOLVER,
                max_iter=FINAL_MAX_ITER,
                tol=FINAL_TOL,
                random_state=RANDOM_STATE,
                verbose=0,
            ),
        ),
    ]
)


# =============================================================================
# 12. 2019~2024 전체 Train 로드
#
# Colab peak RAM을 줄이기 위해 test 실제 row는 아직 읽지 않는다.
# =============================================================================

TRAIN_USE_COLS = (
    FINAL_FEATURES
    + [
        TARGET,
    ]
)


train_load_start = (
    time.perf_counter()
)


train = pd.read_csv(
    TRAIN_PATH,
    encoding="utf-8-sig",
    usecols=TRAIN_USE_COLS,
)


selected_numeric_cols = [
    col
    for col in NUM_COLS
    if col in FINAL_FEATURES
]


train[
    selected_numeric_cols
] = train[
    selected_numeric_cols
].astype(
    np.float32
)


train[
    TARGET
] = train[
    TARGET
].astype(
    np.int8
)


train_load_seconds = (
    time.perf_counter()
    - train_load_start
)


# =============================================================================
# 13. Train 데이터 검증
# =============================================================================

train_seasons = sorted(
    train[
        SEASON_COL
    ]
    .unique()
    .tolist()
)


expected_train_seasons = [
    2019,
    2020,
    2021,
    2022,
    2023,
    2024,
]


assert (
    train_seasons
    == expected_train_seasons
)


assert len(
    train
) == 1_475_092


assert not train[
    TARGET
].isna().any()


assert set(
    train[
        TARGET
    ]
    .unique()
).issubset(
    {
        0,
        1,
    }
)


print("\n" + "=" * 80)
print("FINAL TRAIN DATA")
print("=" * 80)

print(
    f"Train seasons     : "
    f"{train_seasons}"
)

print(
    f"Train rows        : "
    f"{len(train):,}"
)

print(
    f"Raw feature count : "
    f"{len(FINAL_FEATURES)}"
)

print(
    f"Target rate       : "
    f"{train[TARGET].mean():.6f}"
)

print(
    f"Load time         : "
    f"{train_load_seconds:.1f} sec"
)

print(
    f"DataFrame memory  : "
    f"{train.memory_usage(deep=True).sum() / 1024**3:.2f} GB"
)


# =============================================================================
# 14. X / y
# =============================================================================

X_train = (
    train[
        FINAL_FEATURES
    ]
)


y_train = (
    train[
        TARGET
    ]
)


# =============================================================================
# 15. 최종 학습
# =============================================================================

print("\n" + "=" * 90)
print("FINAL TRAINING | 2019~2024")
print("=" * 90)

print(
    f"Model   : "
    f"{FINAL_FEATURE_SET_NAME}"
)

print(
    f"Penalty : "
    f"{FINAL_PENALTY}"
)

print(
    f"C       : "
    f"{FINAL_C}"
)

print("\nFit 시작...")


fit_start = (
    time.perf_counter()
)


with warnings.catch_warnings(
    record=True,
) as caught_warnings:

    warnings.simplefilter(
        "always",
        ConvergenceWarning,
    )


    pipeline.fit(
        X_train,
        y_train,
    )


fit_seconds = (
    time.perf_counter()
    - fit_start
)


# =============================================================================
# 16. Convergence
# =============================================================================

convergence_warnings = [
    warning
    for warning in caught_warnings
    if issubclass(
        warning.category,
        ConvergenceWarning,
    )
]


warning_messages = [
    str(
        warning.message
    )
    for warning
    in convergence_warnings
]


final_model = (
    pipeline
    .named_steps[
        "model"
    ]
)


n_iter = int(
    np.max(
        final_model.n_iter_
    )
)


convergence_warning = (
    len(
        convergence_warnings
    )
    > 0
)


converged = (
    not convergence_warning
    and n_iter
    < FINAL_MAX_ITER
)


print(
    f"\nFit 완료    : "
    f"{fit_seconds:.1f} sec"
)

print(
    f"Iteration   : "
    f"{n_iter}/{FINAL_MAX_ITER}"
)

print(
    f"Converged   : "
    f"{converged}"
)

print(
    f"Warning     : "
    f"{convergence_warning}"
)


if warning_messages:

    for message in warning_messages:

        print(
            f"  - {message}"
        )


# =============================================================================
# 17. Transformed Feature Count
# =============================================================================

preprocessor = (
    pipeline
    .named_steps[
        "preprocessor"
    ]
)


transformed_feature_names = (
    preprocessor
    .get_feature_names_out()
)


transformed_feature_count = len(
    transformed_feature_names
)


assert transformed_feature_count == (
    final_model
    .coef_
    .shape[1]
)


print(
    f"Transformed : "
    f"{transformed_feature_count:,}"
)


# =============================================================================
# 18. Coefficient Sparsity
#
# LR-17의 목적은 coefficient 재해석이 아니라
# 최종 fit 상태 검증이므로 summary만 기록.
# =============================================================================

coefficients = (
    final_model
    .coef_[0]
)


zero_coefficient_count = int(
    (
        np.abs(
            coefficients
        )
        <= COEF_ZERO_THRESHOLD
    )
    .sum()
)


nonzero_coefficient_count = (
    len(
        coefficients
    )
    - zero_coefficient_count
)


coefficient_sparsity = (
    zero_coefficient_count
    / len(
        coefficients
    )
)


print(
    f"Non-zero Coef : "
    f"{nonzero_coefficient_count:,}"
)

print(
    f"Sparsity      : "
    f"{coefficient_sparsity:.2%}"
)


# =============================================================================
# 19. Final Pipeline 저장
#
# 이것이 2025 inference용 최종 fitted pipeline.
# =============================================================================

print("\n" + "=" * 80)
print("SAVE FINAL PIPELINE")
print("=" * 80)


joblib.dump(
    pipeline,
    FINAL_PIPELINE_PATH,
    compress=3,
)


pipeline_size_mb = (
    os.path.getsize(
        FINAL_PIPELINE_PATH
    )
    / 1024**2
)


print(
    f"✓ Pipeline saved : "
    f"{FINAL_PIPELINE_PATH}"
)

print(
    f"Artifact size    : "
    f"{pipeline_size_mb:.2f} MB"
)


# =============================================================================
# 20. Training Summary 저장
# =============================================================================

training_summary = {
    "feature_set":
        FINAL_FEATURE_SET_NAME,

    "raw_feature_count":
        len(
            FINAL_FEATURES
        ),

    "pitcher_id":
        FINAL_PITCHER_ID,

    "historical":
        FINAL_HISTORICAL,

    "rolling":
        FINAL_ROLLING,

    "penalty":
        FINAL_PENALTY,

    "C":
        FINAL_C,

    "solver":
        FINAL_SOLVER,

    "max_iter":
        FINAL_MAX_ITER,

    "tol":
        FINAL_TOL,

    "calibration":
        FINAL_CALIBRATION,

    "train_start":
        TRAIN_START_SEASON,

    "train_end":
        TRAIN_END_SEASON,

    "train_rows":
        len(
            train
        ),

    "target_rate":
        float(
            train[
                TARGET
            ]
            .mean()
        ),

    "transformed_feature_count":
        transformed_feature_count,

    "coefficient_count":
        len(
            coefficients
        ),

    "nonzero_coefficient_count":
        nonzero_coefficient_count,

    "zero_coefficient_count":
        zero_coefficient_count,

    "coefficient_sparsity":
        coefficient_sparsity,

    "n_iter":
        n_iter,

    "converged":
        converged,

    "convergence_warning":
        convergence_warning,

    "warning_message":
        " | ".join(
            warning_messages
        ),

    "fit_seconds":
        fit_seconds,

    "pipeline_size_mb":
        pipeline_size_mb,
}


pd.DataFrame(
    [
        training_summary
    ]
).to_csv(
    TRAINING_SUMMARY_PATH,
    encoding="utf-8-sig",
    index=False,
)


# =============================================================================
# 21. Train 메모리 회수 후 Test 로드
#
# Train + Test를 동시에 오래 보관하지 않아 Colab peak RAM 감소.
# =============================================================================

del X_train
del y_train
del train
del coefficients

gc.collect()


# =============================================================================
# 22. 2025 Test 로드
#
# submission 생성은 LR-18 범위이므로 row_id는 여기서 사용하지 않는다.
# =============================================================================

test_load_start = (
    time.perf_counter()
)


test = pd.read_csv(
    TEST_PATH,
    encoding="utf-8-sig",
    usecols=FINAL_FEATURES,
)


test[
    selected_numeric_cols
] = test[
    selected_numeric_cols
].astype(
    np.float32
)


test_load_seconds = (
    time.perf_counter()
    - test_load_start
)


test_rows = len(
    test
)


print("\n" + "=" * 80)
print("2025 TEST")
print("=" * 80)

print(
    f"Test rows       : "
    f"{test_rows:,}"
)

print(
    f"Raw features    : "
    f"{test.shape[1]}"
)

print(
    f"Load time       : "
    f"{test_load_seconds:.1f} sec"
)

print(
    f"DataFrame memory: "
    f"{test.memory_usage(deep=True).sum() / 1024**3:.2f} GB"
)


# =============================================================================
# 23. Test Schema Compatibility
# =============================================================================

assert list(
    test.columns
) == FINAL_FEATURES


assert test.shape[1] == (
    FINAL_RAW_FEATURE_COUNT
)


# 2025 inference dataset 확인
test_seasons = sorted(
    test[
        SEASON_COL
    ]
    .unique()
    .tolist()
)


print(
    f"Test seasons    : "
    f"{test_seasons}"
)


assert test_seasons == [
    2025
]


# =============================================================================
# 24. Test Preprocessing Transform 검증
#
# Pipeline 안의 이미 fitted preprocessor를 transform만 한다.
# fit / fit_transform 금지.
# =============================================================================

print("\n" + "=" * 80)
print("TEST PREPROCESSING VALIDATION")
print("=" * 80)

transform_start = (
    time.perf_counter()
)


X_test_transformed = (
    preprocessor
    .transform(
        test
    )
)


transform_seconds = (
    time.perf_counter()
    - transform_start
)


assert (
    X_test_transformed
    .shape[0]
    == test_rows
)


assert (
    X_test_transformed
    .shape[1]
    == transformed_feature_count
)


is_sparse = sparse.issparse(
    X_test_transformed
)


if is_sparse:

    transformed_finite = np.isfinite(
        X_test_transformed.data
    ).all()

else:

    transformed_finite = np.isfinite(
        X_test_transformed
    ).all()


assert transformed_finite


print(
    f"Transformed shape : "
    f"{X_test_transformed.shape}"
)

print(
    f"Sparse            : "
    f"{is_sparse}"
)

print(
    f"Finite            : "
    f"{transformed_finite}"
)

print(
    f"Transform time    : "
    f"{transform_seconds:.1f} sec"
)


# =============================================================================
# 25. 2025 Test predict_proba
#
# 이미 transform한 sparse matrix를 재사용해
# 불필요한 두 번째 preprocessing을 피한다.
# =============================================================================

print("\n" + "=" * 80)
print("TEST PREDICT_PROBA VALIDATION")
print("=" * 80)


predict_start = (
    time.perf_counter()
)


test_probability = (
    final_model
    .predict_proba(
        X_test_transformed
    )[:, 1]
    .astype(
        np.float32,
        copy=False,
    )
)


predict_seconds = (
    time.perf_counter()
    - predict_start
)


# =============================================================================
# 26. Prediction Sanity Check
# =============================================================================

prediction_rows = len(
    test_probability
)


nan_count = int(
    np.isnan(
        test_probability
    )
    .sum()
)


inf_count = int(
    np.isinf(
        test_probability
    )
    .sum()
)


finite_check = bool(
    np.isfinite(
        test_probability
    )
    .all()
)


range_check = bool(
    (
        test_probability
        >= 0
    )
    .all()
    and
    (
        test_probability
        <= 1
    )
    .all()
)


row_count_check = (
    prediction_rows
    == test_rows
)


assert row_count_check
assert finite_check
assert nan_count == 0
assert inf_count == 0
assert range_check


# =============================================================================
# 27. Prediction Summary
#
# 진단 목적으로만 기록.
#
# 이 분포를 보고 model / C / feature / calibration을 변경하지 않는다.
# =============================================================================

prediction_mean = float(
    test_probability.mean()
)


prediction_std = float(
    test_probability.std(
        ddof=0
    )
)


prediction_min = float(
    test_probability.min()
)


prediction_max = float(
    test_probability.max()
)


quantiles = np.quantile(
    test_probability,
    [
        0.01,
        0.05,
        0.25,
        0.50,
        0.75,
        0.95,
        0.99,
    ],
)


test_prediction_summary = {
    "test_season":
        2025,

    "test_rows":
        test_rows,

    "prediction_rows":
        prediction_rows,

    "row_count_check":
        row_count_check,

    "prediction_mean":
        prediction_mean,

    "prediction_std":
        prediction_std,

    "prediction_min":
        prediction_min,

    "prediction_max":
        prediction_max,

    "q01":
        float(
            quantiles[0]
        ),

    "q05":
        float(
            quantiles[1]
        ),

    "q25":
        float(
            quantiles[2]
        ),

    "q50":
        float(
            quantiles[3]
        ),

    "q75":
        float(
            quantiles[4]
        ),

    "q95":
        float(
            quantiles[5]
        ),

    "q99":
        float(
            quantiles[6]
        ),

    "nan_count":
        nan_count,

    "inf_count":
        inf_count,

    "finite_check":
        finite_check,

    "range_check":
        range_check,

    "transformed_sparse":
        is_sparse,

    "transform_seconds":
        transform_seconds,

    "predict_seconds":
        predict_seconds,
}


pd.DataFrame(
    [
        test_prediction_summary
    ]
).to_csv(
    TEST_SUMMARY_PATH,
    encoding="utf-8-sig",
    index=False,
)


print(
    f"Test rows       : "
    f"{test_rows:,}"
)

print(
    f"Prediction rows : "
    f"{prediction_rows:,}"
)

print(
    f"Mean            : "
    f"{prediction_mean:.6f}"
)

print(
    f"Std             : "
    f"{prediction_std:.6f}"
)

print(
    f"Min             : "
    f"{prediction_min:.6f}"
)

print(
    f"Max             : "
    f"{prediction_max:.6f}"
)

print(
    f"NaN             : "
    f"{nan_count}"
)

print(
    f"Inf             : "
    f"{inf_count}"
)

print(
    f"Finite          : "
    f"{finite_check}"
)

print(
    f"Range [0,1]     : "
    f"{range_check}"
)

print(
    f"Predict time    : "
    f"{predict_seconds:.1f} sec"
)


print("\nQuantiles:")

for name, value in zip(
    [
        "q01",
        "q05",
        "q25",
        "q50",
        "q75",
        "q95",
        "q99",
    ],
    quantiles,
):

    print(
        f"  {name} : "
        f"{value:.6f}"
    )


# =============================================================================
# 28. Final Configuration JSON
# =============================================================================

final_config = {
    "issue":
        "LR-17",

    "purpose":
        "Final 2019~2024 Logistic Regression refit for 2025 inference",

    "feature_set":
        FINAL_FEATURE_SET_NAME,

    "raw_feature_count":
        FINAL_RAW_FEATURE_COUNT,

    "features":
        FINAL_FEATURES,

    "removed_features":
        REDUCED_DROP_COLS,

    "pitcher_id":
        FINAL_PITCHER_ID,

    "historical":
        FINAL_HISTORICAL,

    "rolling":
        FINAL_ROLLING,

    "penalty":
        FINAL_PENALTY,

    "C":
        FINAL_C,

    "solver":
        FINAL_SOLVER,

    "max_iter":
        FINAL_MAX_ITER,

    "tol":
        FINAL_TOL,

    "random_state":
        RANDOM_STATE,

    "calibration":
        FINAL_CALIBRATION,

    "post_calibration":
        FINAL_POST_CALIBRATION,

    "train_start":
        TRAIN_START_SEASON,

    "train_end":
        TRAIN_END_SEASON,

    "train_rows":
        int(
            training_summary[
                "train_rows"
            ]
        ),

    "transformed_feature_count":
        transformed_feature_count,

    "nonzero_coefficient_count":
        nonzero_coefficient_count,

    "coefficient_sparsity":
        float(
            coefficient_sparsity
        ),

    "n_iter":
        n_iter,

    "converged":
        bool(
            converged
        ),

    "convergence_warning":
        bool(
            convergence_warning
        ),

    "fit_seconds":
        float(
            fit_seconds
        ),

    "pipeline_artifact":
        FINAL_PIPELINE_PATH,

    "test_inference_validated":
        True,

    "test_rows":
        int(
            test_rows
        ),

    "note":
        (
            "Configuration was locked before viewing 2025 "
            "prediction diagnostics. No model selection, tuning, "
            "or calibration changes are performed in LR-17."
        ),
}


with open(
    FINAL_CONFIG_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        final_config,
        file,
        ensure_ascii=False,
        indent=2,
    )


# =============================================================================
# 29. Test 객체 메모리 회수
# =============================================================================

del test
del X_test_transformed
del test_probability

gc.collect()


# =============================================================================
# 30. Final Artifact 확인
# =============================================================================

required_artifacts = [
    FINAL_PIPELINE_PATH,
    FINAL_CONFIG_PATH,
    FINAL_FEATURES_PATH,
    TRAINING_SUMMARY_PATH,
    TEST_SUMMARY_PATH,
]


for path in required_artifacts:

    assert os.path.exists(
        path
    ), (
        f"Missing artifact: "
        f"{path}"
    )


# =============================================================================
# 31. Final Summary
# =============================================================================

print("\n" + "=" * 100)
print("LR-17 FINAL TRAINING COMPLETE")
print("=" * 100)

print(
    f"Feature Set        : "
    f"{FINAL_FEATURE_SET_NAME}"
)

print(
    f"Raw Features       : "
    f"{FINAL_RAW_FEATURE_COUNT}"
)

print(
    f"Train Seasons      : "
    f"{TRAIN_START_SEASON}"
    f"~"
    f"{TRAIN_END_SEASON}"
)

print(
    f"Train Rows         : "
    f"{training_summary['train_rows']:,}"
)

print(
    f"Transformed        : "
    f"{transformed_feature_count:,}"
)

print(
    f"Non-zero Coefs     : "
    f"{nonzero_coefficient_count:,}"
)

print(
    f"Sparsity           : "
    f"{coefficient_sparsity:.2%}"
)

print(
    f"Iteration          : "
    f"{n_iter}/{FINAL_MAX_ITER}"
)

print(
    f"Converged          : "
    f"{converged}"
)

print(
    f"Fit Time           : "
    f"{fit_seconds:.1f} sec"
)

print(
    f"Test Rows          : "
    f"{test_rows:,}"
)

print(
    f"Prediction Rows    : "
    f"{prediction_rows:,}"
)

print(
    f"Prediction Mean    : "
    f"{prediction_mean:.6f}"
)

print(
    f"Prediction Std     : "
    f"{prediction_std:.6f}"
)

print(
    f"Prediction Range   : "
    f"{prediction_min:.6f}"
    f" ~ "
    f"{prediction_max:.6f}"
)

print(
    f"NaN / Inf          : "
    f"{nan_count} / {inf_count}"
)

print(
    f"Range Check        : "
    f"{range_check}"
)

print("\nArtifacts:")

for path in required_artifacts:

    print(
        f"  ✓ {path}"
    )

print(
    "\n※ 2025 prediction diagnostics are validation-only. "
    "Configuration is not changed based on these values."
)

print(
    "※ submission.csv 생성은 LR-18에서 수행합니다."
)