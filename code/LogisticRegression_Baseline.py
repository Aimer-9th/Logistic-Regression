import os
import time
import warnings

import numpy as np
import pandas as pd

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

# Logistic Regression baseline
LR_PENALTY = "l2"
LR_C = 1.0
LR_SOLVER = "saga"
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
# 2. Feature Type 정의
#
# LR-05 / LR-08 설정 재사용
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

assert set(SCALE_COLS).isdisjoint(
    set(NO_SCALE_COLS)
)


# =============================================================================
# 3. LR-07 A/B/C/D Feature Set
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


print("\n" + "=" * 100)
print("2. EXPERIMENT CONFIG")
print("=" * 100)

for name, config in LR_EXPERIMENTS.items():

    feature_count = len(
        config["features"]
    )

    print(
        f"{name:<30} | "
        f"features={feature_count:>2} | "
        f"PID={str(config['use_pitcher_id']):<5} | "
        f"Historical={config['use_historical']}"
    )

    assert (
        feature_count
        == EXPECTED_FEATURE_COUNTS[name]
    )


# =============================================================================
# 4. Feature Type 반환
# =============================================================================

def get_feature_types(
    feature_list,
):
    """
    선택된 experiment feature set에 맞는
    preprocessing column 목록을 반환한다.
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


    # pitcher_id는 실제 LR preprocessing에서는 categorical 처리
    categorical_model_cols = (
        categorical_cols
        + identifier_cols
    )


    classified_cols = (
        scale_cols
        + no_scale_cols
        + categorical_model_cols
    )


    assert len(classified_cols) == len(
        feature_list
    )

    assert len(classified_cols) == len(
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
# 5. Preprocessing Builder
#
# LR-08 설정 재사용
# =============================================================================

def build_preprocessor(
    feature_list,
):
    """
    Logistic Regression용 ColumnTransformer 생성.
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
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )


    # -------------------------------------------------------------------------
    # No-scaling numeric
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
    # Categorical / identifier
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


    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )


# =============================================================================
# 6. Logistic Regression Pipeline Builder
# =============================================================================

def build_lr_pipeline(
    feature_list,
    C=LR_C,
    max_iter=LR_MAX_ITER,
):
    """
    Preprocessing + Logistic Regression.
    """

    preprocessor = build_preprocessor(
        feature_list
    )


    model = LogisticRegression(
        penalty=LR_PENALTY,
        C=C,
        solver=LR_SOLVER,
        max_iter=max_iter,
        random_state=RANDOM_STATE,
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
# 7. Validation Split
#
# LR-07 / LR-08과 동일
#
# Train : 2019 ~ 2023
# Valid : 2024
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
    f"Train seasons : "
    f"{sorted(train_fold[SEASON_COL].unique())}"
)

print(
    f"Valid seasons : "
    f"{sorted(valid_fold[SEASON_COL].unique())}"
)

print(
    f"Train rows    : "
    f"{len(train_fold):,}"
)

print(
    f"Valid rows    : "
    f"{len(valid_fold):,}"
)

print(
    f"Train rate    : "
    f"{train_fold[TARGET].mean():.6f}"
)

print(
    f"Valid rate    : "
    f"{valid_fold[TARGET].mean():.6f}"
)


assert train_fold[
    SEASON_COL
].max() < VALID_SEASON

assert (
    valid_fold[
        SEASON_COL
    ]
    == VALID_SEASON
).all()


# =============================================================================
# 8. Naive Probability Baseline
#
# LR-09와 동일:
# 2019~2023 target 평균으로 2024 전체 예측
# =============================================================================

naive_probability = float(
    train_fold[
        TARGET
    ]
    .mean()
)


y_valid = (
    valid_fold[
        TARGET
    ]
    .to_numpy()
)


naive_valid_prob = np.full(
    shape=len(valid_fold),
    fill_value=naive_probability,
    dtype=float,
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
print("4. NAIVE BASELINE")
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


# LR-09 기록값과 sanity check
assert np.isclose(
    NAIVE_BRIER,
    0.251875,
    atol=1e-6,
)

assert np.isclose(
    NAIVE_LOGLOSS,
    0.696904,
    atol=1e-6,
)


# =============================================================================
# 9. 평가 함수
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
# 10. Logistic Regression Baseline 학습
# =============================================================================

results = []
trained_models = {}


print("\n" + "=" * 100)
print("5. LOGISTIC REGRESSION BASELINE TRAINING")
print("=" * 100)


for experiment_index, (
    experiment_name,
    config,
) in enumerate(
    LR_EXPERIMENTS.items(),
    start=1,
):

    total_start = time.perf_counter()


    print("\n" + "#" * 100)

    print(
        f"[{experiment_index}/{len(LR_EXPERIMENTS)}] "
        f"{experiment_name}"
    )

    print("#" * 100)


    # -------------------------------------------------------------------------
    # Feature 설정
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
        f"Train seasons   : "
        f"2019 ~ {TRAIN_END_SEASON}"
    )

    print(
        f"Valid season    : "
        f"{VALID_SEASON}"
    )

    print(
        f"Train rows      : "
        f"{len(train_fold):,}"
    )

    print(
        f"Valid rows      : "
        f"{len(valid_fold):,}"
    )

    print(
        f"Feature count   : "
        f"{len(feature_list)}"
    )

    print(
        f"Pitcher ID      : "
        f"{config['use_pitcher_id']}"
    )

    print(
        f"Historical      : "
        f"{config['use_historical']}"
    )

    print(
        f"Scale numeric   : "
        f"{len(feature_types['scale'])}"
    )

    print(
        f"No-scale numeric: "
        f"{len(feature_types['no_scale'])}"
    )

    print(
        f"Categorical     : "
        f"{len(feature_types['categorical'])}"
    )


    print("\nFeature set:")

    for feature in feature_list:
        print(f"  - {feature}")


    # -------------------------------------------------------------------------
    # X / y
    # -------------------------------------------------------------------------

    X_train = (
        train_fold[
            feature_list
        ]
    )

    y_train = (
        train_fold[
            TARGET
        ]
    )


    X_valid = (
        valid_fold[
            feature_list
        ]
    )

    y_valid_series = (
        valid_fold[
            TARGET
        ]
    )


    # -------------------------------------------------------------------------
    # Pipeline 생성
    # -------------------------------------------------------------------------

    pipeline = build_lr_pipeline(
        feature_list=feature_list,
        C=LR_C,
        max_iter=LR_MAX_ITER,
    )


    # -------------------------------------------------------------------------
    # Fit
    #
    # ConvergenceWarning을 실제로 캡처한다.
    # -------------------------------------------------------------------------

    print(
        f"\n[{experiment_name}] "
        f"fit 시작"
    )


    fit_start = time.perf_counter()


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
    # Convergence 상태
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


    convergence_reached = (
        not convergence_warning
        and n_iter < LR_MAX_ITER
    )


    print(
        f"[{experiment_name}] "
        f"Iteration      : "
        f"{n_iter}/{LR_MAX_ITER}"
    )


    if convergence_warning:

        print(
            f"[{experiment_name}] "
            f"⚠️ ConvergenceWarning 발생"
        )

        for warning_message in warning_messages:

            print(
                f"  - "
                f"{warning_message}"
            )

    elif n_iter >= LR_MAX_ITER:

        print(
            f"[{experiment_name}] "
            "⚠️ max_iter에 도달했습니다."
        )

    else:

        print(
            f"[{experiment_name}] "
            "✓ 정상 수렴"
        )


    # -------------------------------------------------------------------------
    # Transformed Feature Count
    # -------------------------------------------------------------------------

    fitted_preprocessor = (
        pipeline
        .named_steps[
            "preprocessor"
        ]
    )


    transformed_feature_names = (
        fitted_preprocessor
        .get_feature_names_out()
    )


    transformed_feature_count = len(
        transformed_feature_names
    )


    print(
        f"[{experiment_name}] "
        f"Transformed features : "
        f"{transformed_feature_count:,}"
    )


    # -------------------------------------------------------------------------
    # predict_proba
    # -------------------------------------------------------------------------

    print(
        f"[{experiment_name}] "
        f"Validation predict_proba 시작"
    )


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


    print(
        f"[{experiment_name}] "
        f"Validation 예측 완료 | "
        f"{predict_seconds:.1f} sec"
    )


    # -------------------------------------------------------------------------
    # Probability sanity check
    # -------------------------------------------------------------------------

    assert len(
        valid_prob
    ) == len(
        valid_fold
    )

    assert np.isfinite(
        valid_prob
    ).all()

    assert (
        valid_prob >= 0
    ).all()

    assert (
        valid_prob <= 1
    ).all()


    print(
        f"[{experiment_name}] "
        f"Probability range : "
        f"{valid_prob.min():.6f} "
        f"~ "
        f"{valid_prob.max():.6f}"
    )

    print(
        f"[{experiment_name}] "
        f"Mean prediction   : "
        f"{valid_prob.mean():.6f}"
    )

    print(
        f"[{experiment_name}] "
        f"Actual rate       : "
        f"{y_valid_series.mean():.6f}"
    )


    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    metrics = evaluate_predictions(
        y_valid_series,
        valid_prob,
    )


    # Brier는 낮을수록 좋으므로
    #
    # improvement > 0
    #   => LR이 Naive보다 좋음
    #
    # improvement < 0
    #   => LR이 Naive보다 나쁨

    brier_improvement = (
        NAIVE_BRIER
        - metrics[
            "brier"
        ]
    )


    logloss_improvement = (
        NAIVE_LOGLOSS
        - metrics[
            "logloss"
        ]
    )


    brier_improvement_pct = (
        brier_improvement
        / NAIVE_BRIER
        * 100
    )


    beats_naive = (
        metrics[
            "brier"
        ]
        < NAIVE_BRIER
    )


    total_seconds = (
        time.perf_counter()
        - total_start
    )


    # -------------------------------------------------------------------------
    # 결과 저장
    # -------------------------------------------------------------------------

    result = {
        "model":
            experiment_name,

        "train_seasons":
            "2019-2023",

        "valid_season":
            VALID_SEASON,

        "train_rows":
            len(train_fold),

        "valid_rows":
            len(valid_fold),

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

        "max_iter":
            LR_MAX_ITER,

        "n_iter":
            n_iter,

        "converged":
            convergence_reached,

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
            valid_prob.mean(),

        "prediction_min":
            valid_prob.min(),

        "prediction_max":
            valid_prob.max(),

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


    trained_models[
        experiment_name
    ] = pipeline


    # -------------------------------------------------------------------------
    # 현재 모델 결과 로그
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
        f"Naive Log Loss    : "
        f"{NAIVE_LOGLOSS:.6f}"
    )

    print(
        f"LR Log Loss       : "
        f"{metrics['logloss']:.6f}"
    )

    print(
        f"ROC-AUC           : "
        f"{metrics['auc']:.6f}"
    )

    print(
        f"Converged         : "
        f"{convergence_reached}"
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


# =============================================================================
# 11. 전체 결과
# =============================================================================

results_df = (
    pd.DataFrame(
        results
    )
    .sort_values(
        by="brier",
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
print("6. FINAL LOGISTIC REGRESSION BASELINE RESULTS")
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
# 12. Best Model
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
print("7. BEST LR BASELINE")
print("=" * 80)

print(
    f"Best model       : "
    f"{best_model_name}"
)

print(
    f"Naive Brier      : "
    f"{NAIVE_BRIER:.6f}"
)

print(
    f"Best LR Brier    : "
    f"{best_brier:.6f}"
)

print(
    f"Improvement      : "
    f"{best_improvement:+.6f}"
)

print(
    f"Beats Naive      : "
    f"{best_brier < NAIVE_BRIER}"
)


# =============================================================================
# 13. A/B/C/D 효과 비교
# =============================================================================

score_table = (
    results_df
    .set_index(
        "model"
    )
)


comparison_rows = []


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


for (
    comparison_name,
    baseline_model,
    comparison_model,
) in COMPARISONS:

    # positive improvement => comparison model이 더 좋음
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
print("8. A/B/C/D FEATURE EFFECT")
print("=" * 130)

print(
    comparison_df
    .round(6)
    .to_string(
        index=False,
    )
)


# =============================================================================
# 14. 결과 저장
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


# Experiment feature set 저장
feature_rows = []

for name, config in LR_EXPERIMENTS.items():

    for feature in config[
        "features"
    ]:

        feature_rows.append(
            {
                "model":
                    name,

                "feature":
                    feature,
            }
        )


pd.DataFrame(
    feature_rows
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lr10_feature_sets.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


# =============================================================================
# 15. Final Summary
# =============================================================================

summary_df = pd.DataFrame(
    [
        {
            "train_seasons":
                "2019-2023",

            "valid_season":
                2024,

            "train_rows":
                len(train_fold),

            "valid_rows":
                len(valid_fold),

            "penalty":
                LR_PENALTY,

            "C":
                LR_C,

            "solver":
                LR_SOLVER,

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
print("LR-10 LOGISTIC REGRESSION BASELINE COMPLETE")
print("=" * 80)

print(
    f"Output directory : "
    f"{OUTPUT_DIR}"
)