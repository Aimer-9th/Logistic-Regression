import gc
import os
import time
import warnings

import numpy as np
import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
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
OUTPUT_DIR = "./lr10_baseline_outputs"

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


# =============================================================================
# Logistic Regression 설정
#
# Colab CPU 환경에서 대규모 sparse dataset을 고려
# =============================================================================

LR_PENALTY = "l2"
LR_C = 1.0
LR_SOLVER = "saga"

# 충분한 upper bound는 유지
LR_MAX_ITER = 500

# sklearn default 1e-4보다 느슨하게 설정하여 baseline 학습 시간 단축
# LR-10은 hyperparameter tuning 단계가 아니라 기준 모델 구축 단계
LR_TOL = 1e-3


# 전체 feature 목록을 매번 출력하지 않음
PRINT_FEATURE_LIST = False


print("=" * 80)
print("ENVIRONMENT")
print("=" * 80)

print(f"scikit-learn : {sklearn.__version__}")
print(f"NumPy        : {np.__version__}")
print(f"Pandas       : {pd.__version__}")


# =============================================================================
# 1. Test header만 읽어서 feature schema 확보
#
# test.csv 전체 5 rows 여부와 관계없이 실제 학습에는 필요 없음.
# nrows=0으로 컬럼 정보만 가져온다.
# =============================================================================

test_columns = pd.read_csv(
    TEST_PATH,
    encoding="utf-8-sig",
    nrows=0,
).columns


ALL_FEATURES = [
    col
    for col in test_columns
    if col != ID
]


assert len(ALL_FEATURES) == 47

assert TARGET not in ALL_FEATURES
assert ID not in ALL_FEATURES


# =============================================================================
# 2. Feature Type 정의
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

    # 투수 pitch-mix historical
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

assert set(
    SCALE_COLS
).isdisjoint(
    set(NO_SCALE_COLS)
)


# =============================================================================
# 3. Train 데이터만 실제 로드
#
# row_id는 학습에 필요 없으므로 처음부터 읽지 않는다.
# =============================================================================

USE_COLS = (
    ALL_FEATURES
    + [TARGET]
)


load_start = time.perf_counter()


train = pd.read_csv(
    TRAIN_PATH,
    encoding="utf-8-sig",
    usecols=USE_COLS,
)


load_seconds = (
    time.perf_counter()
    - load_start
)


# =============================================================================
# 4. Numeric float32 변환
#
# 37개 numeric feature를 float64보다 절반 크기의 float32로 유지.
#
# SAGA는 float32를 지원한다.
# =============================================================================

train[
    NUM_COLS
] = train[
    NUM_COLS
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


print("\n" + "=" * 80)
print("DATA")
print("=" * 80)

print(
    f"Train shape       : "
    f"{train.shape}"
)

print(
    f"Feature count     : "
    f"{len(ALL_FEATURES)}"
)

print(
    f"Load time         : "
    f"{load_seconds:.1f} sec"
)

print(
    f"DataFrame memory  : "
    f"{train.memory_usage(deep=True).sum() / 1024**3:.2f} GB"
)


# =============================================================================
# 5. LR-07 A/B/C/D Feature Set
# =============================================================================

HISTORICAL_FEATURES = [
    col
    for col in ALL_FEATURES
    if col.startswith(
        "asof_"
    )
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

        "use_pitcher_id":
            False,

        "use_historical":
            False,
    },

    "LR-B-PitcherID": {
        "features":
            MODEL_B_FEATURES,

        "use_pitcher_id":
            True,

        "use_historical":
            False,
    },

    "LR-C-Historical": {
        "features":
            MODEL_C_FEATURES,

        "use_pitcher_id":
            False,

        "use_historical":
            True,
    },

    "LR-D-PitcherID-Historical": {
        "features":
            MODEL_D_FEATURES,

        "use_pitcher_id":
            True,

        "use_historical":
            True,
    },
}


EXPECTED_FEATURE_COUNTS = {
    "LR-A-NoPitcherID": 27,
    "LR-B-PitcherID": 28,
    "LR-C-Historical": 46,
    "LR-D-PitcherID-Historical": 47,
}


for name, config in LR_EXPERIMENTS.items():

    assert (
        len(
            config["features"]
        )
        == EXPECTED_FEATURE_COUNTS[name]
    )


# =============================================================================
# 6. Feature Type 반환
# =============================================================================

def get_feature_types(
    feature_list,
):
    """
    Experiment feature set에 맞는 preprocessing columns 반환.
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
    )

    assert len(
        classified_cols
    ) == len(
        set(classified_cols)
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
# 7. Colab 메모리 최적화 Preprocessor
#
# 핵심:
# - sparse 유지
# - OneHotEncoder output float32
# - 불필요한 dense matrix 방지
# =============================================================================

def build_preprocessor(
    feature_list,
):
    """
    Logistic Regression용 preprocessing.
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
    # Scaling numeric
    # -------------------------------------------------------------------------

    scale_numeric_pipeline = Pipeline(
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
    # No-scale numeric
    # -------------------------------------------------------------------------

    no_scale_numeric_pipeline = Pipeline(
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
    #
    # output dtype=float32로 sparse matrix 메모리 절감
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


    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",

        # One-Hot 결과를 dense로 바꾸지 않도록 유지
        sparse_threshold=1.0,

        verbose_feature_names_out=True,
    )


# =============================================================================
# 8. Logistic Regression Builder
# =============================================================================

def build_lr_pipeline(
    feature_list,
):
    """
    Colab CPU/RAM 환경 최적화 Logistic Regression pipeline.
    """

    preprocessor = build_preprocessor(
        feature_list
    )


    model = LogisticRegression(
        penalty=LR_PENALTY,
        C=LR_C,
        solver=LR_SOLVER,
        max_iter=LR_MAX_ITER,
        tol=LR_TOL,
        random_state=RANDOM_STATE,

        # solver 자체 verbose는 대량 로그를 만들므로 끔
        verbose=0,
    )


    return Pipeline(
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


# =============================================================================
# 9. Validation Mask
#
# train_fold / valid_fold 전체 DataFrame 복사본을 만들지 않는다.
#
# 원본 train 하나만 유지하고 boolean mask로 접근.
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


train_rows = int(
    train_mask.sum()
)

valid_rows = int(
    valid_mask.sum()
)


train_target_rate = float(
    train.loc[
        train_mask,
        TARGET,
    ]
    .mean()
)


valid_target_rate = float(
    train.loc[
        valid_mask,
        TARGET,
    ]
    .mean()
)


print("\n" + "=" * 80)
print("VALIDATION SPLIT")
print("=" * 80)

print(
    f"Train seasons : "
    f"2019 ~ {TRAIN_END_SEASON}"
)

print(
    f"Valid season  : "
    f"{VALID_SEASON}"
)

print(
    f"Train rows    : "
    f"{train_rows:,}"
)

print(
    f"Valid rows    : "
    f"{valid_rows:,}"
)

print(
    f"Train rate    : "
    f"{train_target_rate:.6f}"
)

print(
    f"Valid rate    : "
    f"{valid_target_rate:.6f}"
)


# =============================================================================
# 10. Naive Probability Baseline
# =============================================================================

y_valid = (
    train.loc[
        valid_mask,
        TARGET,
    ]
    .to_numpy(
        dtype=np.int8,
        copy=True,
    )
)


naive_probability = (
    train_target_rate
)


naive_valid_prob = np.full(
    valid_rows,
    naive_probability,
    dtype=np.float32,
)


NAIVE_BRIER = brier_score_loss(
    y_valid,
    naive_valid_prob,
)


NAIVE_LOGLOSS = log_loss(
    y_valid,
    naive_valid_prob,
    labels=[
        0,
        1,
    ],
)


print("\n" + "=" * 80)
print("NAIVE BASELINE")
print("=" * 80)

print(
    f"Naive probability : "
    f"{naive_probability:.6f}"
)

print(
    f"Naive Brier       : "
    f"{NAIVE_BRIER:.6f}"
)

print(
    f"Naive Log Loss    : "
    f"{NAIVE_LOGLOSS:.6f}"
)


# =============================================================================
# 11. 평가 함수
# =============================================================================

def evaluate_predictions(
    y_true,
    y_prob,
):
    """
    확률 예측 metric 계산.
    """

    return {
        "brier":
            brier_score_loss(
                y_true,
                y_prob,
            ),

        "logloss":
            log_loss(
                y_true,
                y_prob,
                labels=[
                    0,
                    1,
                ],
            ),

        "auc":
            roc_auc_score(
                y_true,
                y_prob,
            ),
    }


# =============================================================================
# 12. A/B/C/D Logistic Regression 학습
#
# 메모리 전략:
#
# 1. 한 모델씩 순차 학습
# 2. 결과 숫자만 저장
# 3. pipeline을 4개 모두 메모리에 보관하지 않음
# 4. 모델 완료 후 gc.collect()
# =============================================================================

results = []


print("\n" + "=" * 100)
print("LOGISTIC REGRESSION BASELINE TRAINING")
print("=" * 100)


for experiment_index, (
    experiment_name,
    config,
) in enumerate(
    LR_EXPERIMENTS.items(),
    start=1,
):

    experiment_start = (
        time.perf_counter()
    )


    print("\n" + "#" * 100)

    print(
        f"[{experiment_index}/"
        f"{len(LR_EXPERIMENTS)}] "
        f"{experiment_name}"
    )

    print("#" * 100)


    # -------------------------------------------------------------------------
    # Feature config
    # -------------------------------------------------------------------------

    feature_list = (
        config[
            "features"
        ]
    )


    feature_types = get_feature_types(
        feature_list
    )


    print(
        f"Train       : "
        f"2019~{TRAIN_END_SEASON}"
    )

    print(
        f"Valid       : "
        f"{VALID_SEASON}"
    )

    print(
        f"Train rows  : "
        f"{train_rows:,}"
    )

    print(
        f"Valid rows  : "
        f"{valid_rows:,}"
    )

    print(
        f"Features    : "
        f"{len(feature_list)}"
    )

    print(
        f"Pitcher ID  : "
        f"{config['use_pitcher_id']}"
    )

    print(
        f"Historical  : "
        f"{config['use_historical']}"
    )

    print(
        f"Scale / NoScale / Cat : "
        f"{len(feature_types['scale'])} / "
        f"{len(feature_types['no_scale'])} / "
        f"{len(feature_types['categorical'])}"
    )


    if PRINT_FEATURE_LIST:

        print("\nFeature set:")

        for feature in feature_list:
            print(
                f"  - {feature}"
            )


    # -------------------------------------------------------------------------
    # 필요한 feature만 해당 experiment 시점에 복사
    #
    # train_fold 48-column 전체 복제보다 RAM 사용량이 작음.
    # -------------------------------------------------------------------------

    X_train = (
        train.loc[
            train_mask,
            feature_list,
        ]
    )


    y_train = (
        train.loc[
            train_mask,
            TARGET,
        ]
    )


    X_valid = (
        train.loc[
            valid_mask,
            feature_list,
        ]
    )


    # -------------------------------------------------------------------------
    # Pipeline
    # -------------------------------------------------------------------------

    pipeline = build_lr_pipeline(
        feature_list
    )


    # -------------------------------------------------------------------------
    # Fit
    # -------------------------------------------------------------------------

    print(
        f"\n[{experiment_name}] "
        f"fit 시작"
    )


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
        for warning in convergence_warnings
    ]


    print(
        f"[{experiment_name}] "
        f"fit 완료 | "
        f"{fit_seconds:.1f} sec"
    )


    # -------------------------------------------------------------------------
    # Convergence
    # -------------------------------------------------------------------------

    lr_model = (
        pipeline
        .named_steps[
            "model"
        ]
    )


    n_iter = int(
        np.max(
            lr_model.n_iter_
        )
    )


    convergence_warning = (
        len(
            convergence_warnings
        )
        > 0
    )


    # warning이 없고 max_iter 전에 종료했으면 정상 수렴
    converged = (
        not convergence_warning
        and n_iter < LR_MAX_ITER
    )


    print(
        f"[{experiment_name}] "
        f"Iteration : "
        f"{n_iter}/{LR_MAX_ITER}"
    )


    if convergence_warning:

        print(
            f"[{experiment_name}] "
            f"⚠️ ConvergenceWarning"
        )

    elif n_iter >= LR_MAX_ITER:

        print(
            f"[{experiment_name}] "
            f"⚠️ max_iter 도달"
        )

    else:

        print(
            f"[{experiment_name}] "
            f"✓ 정상 수렴"
        )


    # -------------------------------------------------------------------------
    # Transformed Feature Count
    # -------------------------------------------------------------------------

    preprocessor = (
        pipeline
        .named_steps[
            "preprocessor"
        ]
    )


    transformed_feature_count = len(
        preprocessor
        .get_feature_names_out()
    )


    print(
        f"[{experiment_name}] "
        f"Transformed features : "
        f"{transformed_feature_count:,}"
    )


    # -------------------------------------------------------------------------
    # Validation Prediction
    # -------------------------------------------------------------------------

    predict_start = (
        time.perf_counter()
    )


    valid_prob = (
        pipeline
        .predict_proba(
            X_valid
        )[:, 1]
    )


    predict_seconds = (
        time.perf_counter()
        - predict_start
    )


    # 메모리 절약
    valid_prob = valid_prob.astype(
        np.float32,
        copy=False,
    )


    # -------------------------------------------------------------------------
    # Probability sanity check
    # -------------------------------------------------------------------------

    assert len(
        valid_prob
    ) == valid_rows

    assert np.isfinite(
        valid_prob
    ).all()

    assert (
        valid_prob >= 0
    ).all()

    assert (
        valid_prob <= 1
    ).all()


    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    metrics = evaluate_predictions(
        y_valid,
        valid_prob,
    )


    brier_improvement = (
        NAIVE_BRIER
        - metrics[
            "brier"
        ]
    )


    brier_improvement_pct = (
        brier_improvement
        / NAIVE_BRIER
        * 100
    )


    logloss_improvement = (
        NAIVE_LOGLOSS
        - metrics[
            "logloss"
        ]
    )


    beats_naive = (
        metrics[
            "brier"
        ]
        < NAIVE_BRIER
    )


    total_seconds = (
        time.perf_counter()
        - experiment_start
    )


    # -------------------------------------------------------------------------
    # Result
    # -------------------------------------------------------------------------

    result = {
        "model":
            experiment_name,

        "train_seasons":
            "2019-2023",

        "valid_season":
            VALID_SEASON,

        "train_rows":
            train_rows,

        "valid_rows":
            valid_rows,

        "raw_feature_count":
            len(feature_list),

        "transformed_feature_count":
            transformed_feature_count,

        "pitcher_id":
            config[
                "use_pitcher_id"
            ],

        "historical":
            config[
                "use_historical"
            ],

        "penalty":
            LR_PENALTY,

        "C":
            LR_C,

        "solver":
            LR_SOLVER,

        "tol":
            LR_TOL,

        "max_iter":
            LR_MAX_ITER,

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

        "naive_brier":
            NAIVE_BRIER,

        "brier":
            metrics[
                "brier"
            ],

        "brier_improvement":
            brier_improvement,

        "brier_improvement_pct":
            brier_improvement_pct,

        "beats_naive":
            beats_naive,

        "naive_logloss":
            NAIVE_LOGLOSS,

        "logloss":
            metrics[
                "logloss"
            ],

        "logloss_improvement":
            logloss_improvement,

        "auc":
            metrics[
                "auc"
            ],

        "prediction_mean":
            float(
                valid_prob.mean()
            ),

        "prediction_min":
            float(
                valid_prob.min()
            ),

        "prediction_max":
            float(
                valid_prob.max()
            ),

        "fit_seconds":
            fit_seconds,

        "predict_seconds":
            predict_seconds,

        "total_seconds":
            total_seconds,
    }


    results.append(
        result
    )


    # -------------------------------------------------------------------------
    # 현재 모델 결과
    # -------------------------------------------------------------------------

    print("\n" + "-" * 100)

    print(
        f"{experiment_name} RESULT"
    )

    print("-" * 100)

    print(
        f"Naive Brier       : "
        f"{NAIVE_BRIER:.6f}"
    )

    print(
        f"LR Brier          : "
        f"{metrics['brier']:.6f}"
    )

    print(
        f"Brier Improvement : "
        f"{brier_improvement:+.6f}"
    )

    print(
        f"Improvement (%)   : "
        f"{brier_improvement_pct:+.3f}%"
    )

    print(
        f"Beats Naive       : "
        f"{beats_naive}"
    )

    print(
        f"Log Loss          : "
        f"{metrics['logloss']:.6f}"
    )

    print(
        f"ROC-AUC           : "
        f"{metrics['auc']:.6f}"
    )

    print(
        f"Prediction mean   : "
        f"{valid_prob.mean():.6f}"
    )

    print(
        f"Actual rate       : "
        f"{valid_target_rate:.6f}"
    )

    print(
        f"Fit time          : "
        f"{fit_seconds:.1f} sec"
    )

    print(
        f"Predict time      : "
        f"{predict_seconds:.1f} sec"
    )

    print(
        f"Total time        : "
        f"{total_seconds:.1f} sec"
    )


    # -------------------------------------------------------------------------
    # Colab RAM 회수
    #
    # trained_models에 4개 pipeline을 저장하지 않는다.
    # -------------------------------------------------------------------------

    del X_train
    del y_train
    del X_valid
    del valid_prob
    del pipeline
    del preprocessor
    del lr_model

    gc.collect()


# =============================================================================
# 13. 전체 결과
# =============================================================================

results_df = (
    pd.DataFrame(
        results
    )
    .sort_values(
        "brier",
        ascending=True,
    )
    .reset_index(
        drop=True,
    )
)


DISPLAY_COLS = [
    "model",

    "raw_feature_count",
    "transformed_feature_count",

    "pitcher_id",
    "historical",

    "naive_brier",
    "brier",
    "brier_improvement",
    "brier_improvement_pct",
    "beats_naive",

    "logloss",
    "auc",

    "n_iter",
    "converged",
    "convergence_warning",

    "fit_seconds",
    "predict_seconds",
    "total_seconds",
]


print("\n" + "=" * 180)
print("FINAL LOGISTIC REGRESSION RESULTS")
print("=" * 180)

print(
    results_df[
        DISPLAY_COLS
    ]
    .round(6)
    .to_string(
        index=False,
    )
)


# =============================================================================
# 14. Best Model
# =============================================================================

best_row = (
    results_df
    .iloc[0]
)


best_model_name = (
    best_row[
        "model"
    ]
)


best_brier = float(
    best_row[
        "brier"
    ]
)


best_improvement = float(
    best_row[
        "brier_improvement"
    ]
)


print("\n" + "=" * 80)
print("BEST LR BASELINE")
print("=" * 80)

print(
    f"Best model    : "
    f"{best_model_name}"
)

print(
    f"Naive Brier   : "
    f"{NAIVE_BRIER:.6f}"
)

print(
    f"Best LR Brier : "
    f"{best_brier:.6f}"
)

print(
    f"Improvement   : "
    f"{best_improvement:+.6f}"
)

print(
    f"Beats Naive   : "
    f"{best_brier < NAIVE_BRIER}"
)


# =============================================================================
# 15. A/B/C/D Feature Effect
# =============================================================================

score_table = (
    results_df
    .set_index(
        "model"
    )
)


COMPARISONS = [
    (
        "Pitcher ID effect without history",
        "LR-A-NoPitcherID",
        "LR-B-PitcherID",
    ),

    (
        "Historical effect without pitcher ID",
        "LR-A-NoPitcherID",
        "LR-C-Historical",
    ),

    (
        "Pitcher ID effect with history",
        "LR-C-Historical",
        "LR-D-PitcherID-Historical",
    ),

    (
        "Historical effect with pitcher ID",
        "LR-B-PitcherID",
        "LR-D-PitcherID-Historical",
    ),
]


comparison_rows = []


for (
    comparison_name,
    baseline_model,
    comparison_model,
) in COMPARISONS:

    improvement = (
        score_table.loc[
            baseline_model,
            "brier",
        ]
        -
        score_table.loc[
            comparison_model,
            "brier",
        ]
    )


    comparison_rows.append(
        {
            "comparison":
                comparison_name,

            "baseline_model":
                baseline_model,

            "comparison_model":
                comparison_model,

            "baseline_brier":
                score_table.loc[
                    baseline_model,
                    "brier",
                ],

            "comparison_brier":
                score_table.loc[
                    comparison_model,
                    "brier",
                ],

            "brier_improvement":
                improvement,

            "comparison_better":
                improvement > 0,
        }
    )


comparison_df = pd.DataFrame(
    comparison_rows
)


print("\n" + "=" * 130)
print("A/B/C/D FEATURE EFFECT")
print("=" * 130)

print(
    comparison_df
    .round(6)
    .to_string(
        index=False,
    )
)


# =============================================================================
# 16. 결과 저장
# =============================================================================

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lr10_baseline_results.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


comparison_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lr10_feature_effect_comparison.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


summary_df = pd.DataFrame(
    [
        {
            "train_seasons":
                "2019-2023",

            "valid_season":
                VALID_SEASON,

            "train_rows":
                train_rows,

            "valid_rows":
                valid_rows,

            "penalty":
                LR_PENALTY,

            "C":
                LR_C,

            "solver":
                LR_SOLVER,

            "tol":
                LR_TOL,

            "max_iter":
                LR_MAX_ITER,

            "naive_brier":
                NAIVE_BRIER,

            "naive_logloss":
                NAIVE_LOGLOSS,

            "best_model":
                best_model_name,

            "best_lr_brier":
                best_brier,

            "best_brier_improvement":
                best_improvement,

            "best_beats_naive":
                best_brier
                < NAIVE_BRIER,
        }
    ]
)


summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lr10_summary.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


print("\n" + "=" * 80)
print("LR-10 COMPLETE")
print("=" * 80)

print(
    f"Output directory : "
    f"{OUTPUT_DIR}"
)