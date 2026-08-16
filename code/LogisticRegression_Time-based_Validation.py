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
OUTPUT_DIR = "./lr11_time_cv_outputs"

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

FIRST_TRAIN_SEASON = 2019

RANDOM_STATE = 42

# Logistic Regression baseline
LR_PENALTY = "l2"
LR_C = 1.0
LR_SOLVER = "saga"
LR_MAX_ITER = 500
LR_TOL = 1e-3

PRINT_FEATURE_LIST = False


# =============================================================================
# 1. Expanding Window 정의
# =============================================================================

FOLD_CONFIGS = [
    {
        "fold": 1,
        "train_start": 2019,
        "train_end": 2021,
        "valid_season": 2022,
    },
    {
        "fold": 2,
        "train_start": 2019,
        "train_end": 2022,
        "valid_season": 2023,
    },
    {
        "fold": 3,
        "train_start": 2019,
        "train_end": 2023,
        "valid_season": 2024,
    },
]


# =============================================================================
# 2. 실행 환경
# =============================================================================

print("=" * 80)
print("ENVIRONMENT")
print("=" * 80)

print(f"scikit-learn : {sklearn.__version__}")
print(f"NumPy        : {np.__version__}")
print(f"Pandas       : {pd.__version__}")


# =============================================================================
# 3. Test schema만 로드
#
# CV 학습에는 test 값 자체가 필요하지 않으므로 header만 사용
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
# 4. Feature Type 정의
#
# LR-05 / LR-08 / LR-10과 동일
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
# 5. Train 데이터 로드
#
# row_id 제외
# 47 feature + target만 로드
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


# Colab RAM 절약
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


load_seconds = (
    time.perf_counter()
    - load_start
)


print("\n" + "=" * 80)
print("DATA")
print("=" * 80)

print(
    f"Train shape      : "
    f"{train.shape}"
)

print(
    f"Feature count    : "
    f"{len(ALL_FEATURES)}"
)

print(
    f"Load time        : "
    f"{load_seconds:.1f} sec"
)

print(
    f"DataFrame memory : "
    f"{train.memory_usage(deep=True).sum() / 1024**3:.2f} GB"
)


# =============================================================================
# 6. A/B/C/D Feature Set
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


for name, config in LR_EXPERIMENTS.items():

    assert (
        len(config["features"])
        == EXPECTED_FEATURE_COUNTS[name]
    )


# =============================================================================
# 7. Historical Feature Boundary 검증
#
# 이번 CV에서 사용하는 historical feature는
# 운영 측 제공 asof_* 19개뿐이다.
#
# Fold 내부 target으로 historical feature를 새로 계산하지 않는다.
# =============================================================================

assert len(
    HISTORICAL_FEATURES
) == 19


assert all(
    col.startswith("asof_")
    for col in HISTORICAL_FEATURES
)


assert TARGET not in HISTORICAL_FEATURES


# LR-06에서 설계했던 신규 hist_* target feature가
# 실수로 입력 schema에 들어왔는지 방어적으로 확인
CUSTOM_TARGET_HISTORY_COLS = [
    col
    for col in ALL_FEATURES
    if col.startswith("hist_")
]


assert not CUSTOM_TARGET_HISTORY_COLS, (
    "공식 asof_* 외에 신규 target-based historical feature가 "
    "입력 schema에 포함되어 있습니다: "
    f"{CUSTOM_TARGET_HISTORY_COLS}"
)


print("\n" + "=" * 80)
print("HISTORICAL FEATURE POLICY")
print("=" * 80)

print(
    f"Official asof features : "
    f"{len(HISTORICAL_FEATURES)}"
)

print(
    "Custom target history  : 0"
)

print(
    "Historical recompute   : False"
)

print(
    "Validation target use  : False"
)


# =============================================================================
# 8. Experiment별 Feature Type
# =============================================================================

def get_feature_types(
    feature_list,
):
    """
    선택된 experiment feature set의 preprocessing column을 반환한다.
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
# 9. Preprocessor Builder
#
# Fold마다 새 객체를 생성하여 train fold에서만 fit
# =============================================================================

def build_preprocessor(
    feature_list,
):
    """
    Fold-safe Logistic Regression preprocessing.
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
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )


# =============================================================================
# 10. LR Pipeline Builder
# =============================================================================

def build_lr_pipeline(
    feature_list,
):
    """
    매 fold / model마다 새로운 Pipeline을 생성한다.
    """

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    feature_list
                ),
            ),
            (
                "model",
                LogisticRegression(
                    penalty=LR_PENALTY,
                    C=LR_C,
                    solver=LR_SOLVER,
                    max_iter=LR_MAX_ITER,
                    tol=LR_TOL,
                    random_state=RANDOM_STATE,
                    verbose=0,
                ),
            ),
        ]
    )


# =============================================================================
# 11. Expanding Window Split Generator
# =============================================================================

def expanding_window_splits(
    data,
    fold_configs,
):
    """
    Expanding-window temporal split generator.

    Random split은 사용하지 않는다.
    """

    for config in fold_configs:

        fold = config[
            "fold"
        ]

        train_start = config[
            "train_start"
        ]

        train_end = config[
            "train_end"
        ]

        valid_season = config[
            "valid_season"
        ]


        train_mask = (
            (
                data[
                    SEASON_COL
                ]
                >= train_start
            )
            &
            (
                data[
                    SEASON_COL
                ]
                <= train_end
            )
        )


        valid_mask = (
            data[
                SEASON_COL
            ]
            == valid_season
        )


        train_rows = int(
            train_mask.sum()
        )

        valid_rows = int(
            valid_mask.sum()
        )


        assert train_rows > 0
        assert valid_rows > 0


        train_seasons = sorted(
            data.loc[
                train_mask,
                SEASON_COL,
            ]
            .unique()
            .tolist()
        )


        valid_seasons = sorted(
            data.loc[
                valid_mask,
                SEASON_COL,
            ]
            .unique()
            .tolist()
        )


        expected_train_seasons = list(
            range(
                train_start,
                train_end + 1,
            )
        )


        assert (
            train_seasons
            == expected_train_seasons
        ), (
            f"Fold {fold}: train season이 예상과 다릅니다."
        )


        assert valid_seasons == [
            valid_season
        ]


        # ---------------------------------------------------------------------
        # 가장 중요한 temporal boundary
        # ---------------------------------------------------------------------

        assert max(
            train_seasons
        ) < min(
            valid_seasons
        ), (
            f"Fold {fold}: 시간 역전이 발생했습니다."
        )


        assert not set(
            train_seasons
        ).intersection(
            valid_seasons
        )


        yield {
            **config,

            "train_mask":
                train_mask,

            "valid_mask":
                valid_mask,

            "train_seasons":
                train_seasons,

            "valid_seasons":
                valid_seasons,

            "train_rows":
                train_rows,

            "valid_rows":
                valid_rows,
        }


# =============================================================================
# 12. Fold 구조 검증 / 기록
# =============================================================================

fold_structures = []


print("\n" + "=" * 100)
print("EXPANDING WINDOW SPLITS")
print("=" * 100)


for fold_data in expanding_window_splits(
    train,
    FOLD_CONFIGS,
):

    train_mask = fold_data[
        "train_mask"
    ]

    valid_mask = fold_data[
        "valid_mask"
    ]


    train_pitcher_count = int(
        train.loc[
            train_mask,
            PITCHER_ID_COL,
        ]
        .nunique()
    )


    valid_pitcher_count = int(
        train.loc[
            valid_mask,
            PITCHER_ID_COL,
        ]
        .nunique()
    )


    fold_structures.append(
        {
            "fold":
                fold_data["fold"],

            "train_period":
                (
                    f"{fold_data['train_start']}"
                    f"~"
                    f"{fold_data['train_end']}"
                ),

            "valid_season":
                fold_data[
                    "valid_season"
                ],

            "train_rows":
                fold_data[
                    "train_rows"
                ],

            "valid_rows":
                fold_data[
                    "valid_rows"
                ],

            "train_pitchers":
                train_pitcher_count,

            "valid_pitchers":
                valid_pitcher_count,

            "temporal_order_valid":
                True,
        }
    )


    print(
        f"Fold {fold_data['fold']} | "
        f"Train "
        f"{fold_data['train_start']}"
        f"~"
        f"{fold_data['train_end']} "
        f"({fold_data['train_rows']:,} rows, "
        f"{train_pitcher_count:,} pitchers) | "
        f"Valid "
        f"{fold_data['valid_season']} "
        f"({fold_data['valid_rows']:,} rows, "
        f"{valid_pitcher_count:,} pitchers)"
    )


fold_structure_df = pd.DataFrame(
    fold_structures
)


fold_structure_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lr11_fold_structure.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


# =============================================================================
# 13. Preprocessing Leakage Unit Test
#
# fit(train sample) 이후 validation transform이
# imputer / scaler / encoder 상태를 변경하지 않는지 확인.
# =============================================================================

def run_preprocessing_leakage_test(
    data,
):
    """
    Validation transform 과정에서 fitted preprocessing state가
    변경되지 않는지 확인한다.
    """

    fold_data = next(
        expanding_window_splits(
            data,
            [
                FOLD_CONFIGS[0]
            ],
        )
    )


    feature_list = (
        MODEL_D_FEATURES
    )


    train_sample = (
        data.loc[
            fold_data[
                "train_mask"
            ],
            feature_list,
        ]
        .head(
            5000
        )
        .copy()
    )


    valid_sample = (
        data.loc[
            fold_data[
                "valid_mask"
            ],
            feature_list,
        ]
        .head(
            1000
        )
        .copy()
    )


    preprocessor = build_preprocessor(
        feature_list
    )


    preprocessor.fit(
        train_sample
    )


    numeric_pipeline = (
        preprocessor
        .named_transformers_[
            "numeric_scale"
        ]
    )


    imputer_before = (
        numeric_pipeline
        .named_steps[
            "imputer"
        ]
        .statistics_
        .copy()
    )


    scaler_before = (
        numeric_pipeline
        .named_steps[
            "scaler"
        ]
        .mean_
        .copy()
    )


    categorical_pipeline = (
        preprocessor
        .named_transformers_[
            "categorical"
        ]
    )


    encoder = (
        categorical_pipeline
        .named_steps[
            "onehot"
        ]
    )


    category_sizes_before = [
        len(categories)
        for categories
        in encoder.categories_
    ]


    # Validation에서는 transform만 호출
    preprocessor.transform(
        valid_sample
    )


    imputer_after = (
        numeric_pipeline
        .named_steps[
            "imputer"
        ]
        .statistics_
    )


    scaler_after = (
        numeric_pipeline
        .named_steps[
            "scaler"
        ]
        .mean_
    )


    category_sizes_after = [
        len(categories)
        for categories
        in encoder.categories_
    ]


    assert np.array_equal(
        imputer_before,
        imputer_after,
        equal_nan=True,
    )


    assert np.array_equal(
        scaler_before,
        scaler_after,
    )


    assert (
        category_sizes_before
        == category_sizes_after
    )


    del train_sample
    del valid_sample
    del preprocessor

    gc.collect()


    return True


preprocessing_leakage_pass = (
    run_preprocessing_leakage_test(
        train
    )
)


print("\n" + "=" * 80)
print("PREPROCESSING LEAKAGE TEST")
print("=" * 80)

print(
    f"Train fit / Valid transform : "
    f"{'PASS' if preprocessing_leakage_pass else 'FAIL'}"
)


# =============================================================================
# 14. Metric 함수
# =============================================================================

def evaluate_predictions(
    y_true,
    y_prob,
):
    """
    동일 metric 구조:
    - Brier
    - Log Loss
    - ROC-AUC
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
# 15. Fold별 Naive Baseline
# =============================================================================

def get_naive_baseline(
    y_train,
    y_valid,
):
    """
    Train target 평균만 사용.
    Validation target은 metric 계산에만 사용.
    """

    probability = float(
        y_train.mean()
    )


    y_prob = np.full(
        len(y_valid),
        probability,
        dtype=np.float32,
    )


    metrics = evaluate_predictions(
        y_valid,
        y_prob,
    )


    return {
        "probability":
            probability,

        "brier":
            metrics[
                "brier"
            ],

        "logloss":
            metrics[
                "logloss"
            ],
    }


# =============================================================================
# 16. CV Runner
#
# 3 folds × 4 models
#
# Colab:
# - 항상 순차 실행
# - 모델 완료 후 pipeline 제거
# - 중간 CSV 저장
# =============================================================================

def run_time_cv(
    data,
    fold_configs,
    experiments,
):
    """
    Expanding-window Logistic Regression CV.
    """

    results = []


    total_experiments = (
        len(
            fold_configs
        )
        * len(
            experiments
        )
    )


    experiment_index = 0


    for fold_data in expanding_window_splits(
        data,
        fold_configs,
    ):

        fold = fold_data[
            "fold"
        ]

        train_mask = fold_data[
            "train_mask"
        ]

        valid_mask = fold_data[
            "valid_mask"
        ]


        # ---------------------------------------------------------------------
        # Fold metadata
        # ---------------------------------------------------------------------

        train_pitcher_count = int(
            data.loc[
                train_mask,
                PITCHER_ID_COL,
            ]
            .nunique()
        )


        valid_pitcher_count = int(
            data.loc[
                valid_mask,
                PITCHER_ID_COL,
            ]
            .nunique()
        )


        y_train_fold = (
            data.loc[
                train_mask,
                TARGET,
            ]
        )


        y_valid = (
            data.loc[
                valid_mask,
                TARGET,
            ]
            .to_numpy(
                dtype=np.int8,
                copy=True,
            )
        )


        train_rate = float(
            y_train_fold.mean()
        )


        valid_rate = float(
            y_valid.mean()
        )


        # ---------------------------------------------------------------------
        # Fold-specific naive baseline
        # ---------------------------------------------------------------------

        naive = get_naive_baseline(
            y_train_fold,
            y_valid,
        )


        print("\n" + "=" * 110)

        print(
            f"FOLD {fold} | "
            f"Train "
            f"{fold_data['train_start']}"
            f"~"
            f"{fold_data['train_end']} "
            f"→ Valid "
            f"{fold_data['valid_season']}"
        )

        print("=" * 110)

        print(
            f"Train rows     : "
            f"{fold_data['train_rows']:,}"
        )

        print(
            f"Valid rows     : "
            f"{fold_data['valid_rows']:,}"
        )

        print(
            f"Train pitchers : "
            f"{train_pitcher_count:,}"
        )

        print(
            f"Valid pitchers : "
            f"{valid_pitcher_count:,}"
        )

        print(
            f"Train rate     : "
            f"{train_rate:.6f}"
        )

        print(
            f"Valid rate     : "
            f"{valid_rate:.6f}"
        )

        print(
            f"Naive Brier    : "
            f"{naive['brier']:.6f}"
        )


        # ---------------------------------------------------------------------
        # A/B/C/D
        # ---------------------------------------------------------------------

        for (
            experiment_name,
            config,
        ) in experiments.items():

            experiment_index += 1


            feature_list = (
                config[
                    "features"
                ]
            )


            feature_types = (
                get_feature_types(
                    feature_list
                )
            )


            experiment_start = (
                time.perf_counter()
            )


            print("\n" + "#" * 110)

            print(
                f"[{experiment_index}/"
                f"{total_experiments}] "
                f"Fold {fold} | "
                f"{experiment_name}"
            )

            print("#" * 110)

            print(
                f"Features       : "
                f"{len(feature_list)}"
            )

            print(
                f"Pitcher ID     : "
                f"{config['use_pitcher_id']}"
            )

            print(
                f"Historical     : "
                f"{config['use_historical']}"
            )

            print(
                f"Scale/NoScale/Cat : "
                f"{len(feature_types['scale'])}/"
                f"{len(feature_types['no_scale'])}/"
                f"{len(feature_types['categorical'])}"
            )


            if PRINT_FEATURE_LIST:

                for feature in feature_list:

                    print(
                        f"  - {feature}"
                    )


            # -----------------------------------------------------------------
            # Fold에 필요한 컬럼만 선택
            # -----------------------------------------------------------------

            X_train = (
                data.loc[
                    train_mask,
                    feature_list,
                ]
            )


            X_valid = (
                data.loc[
                    valid_mask,
                    feature_list,
                ]
            )


            pipeline = build_lr_pipeline(
                feature_list
            )


            # -----------------------------------------------------------------
            # Fit
            # -----------------------------------------------------------------

            print(
                f"[{experiment_name}] "
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
                    y_train_fold,
                )


            fit_seconds = (
                time.perf_counter()
                - fit_start
            )


            convergence_warnings = [
                warning
                for warning
                in caught_warnings
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


            model = (
                pipeline
                .named_steps[
                    "model"
                ]
            )


            n_iter = int(
                np.max(
                    model.n_iter_
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
                < LR_MAX_ITER
            )


            print(
                f"[{experiment_name}] "
                f"fit 완료 | "
                f"{fit_seconds:.1f} sec"
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


            # -----------------------------------------------------------------
            # Transformed feature count
            # -----------------------------------------------------------------

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


            # -----------------------------------------------------------------
            # Validation prediction
            # -----------------------------------------------------------------

            predict_start = (
                time.perf_counter()
            )


            valid_prob = (
                pipeline
                .predict_proba(
                    X_valid
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


            assert len(
                valid_prob
            ) == len(
                y_valid
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


            # -----------------------------------------------------------------
            # Metric
            # -----------------------------------------------------------------

            metrics = evaluate_predictions(
                y_valid,
                valid_prob,
            )


            brier_improvement = (
                naive[
                    "brier"
                ]
                - metrics[
                    "brier"
                ]
            )


            logloss_improvement = (
                naive[
                    "logloss"
                ]
                - metrics[
                    "logloss"
                ]
            )


            total_seconds = (
                time.perf_counter()
                - experiment_start
            )


            # -----------------------------------------------------------------
            # Result
            # -----------------------------------------------------------------

            result = {
                "fold":
                    fold,

                "model":
                    experiment_name,

                "train_start":
                    fold_data[
                        "train_start"
                    ],

                "train_end":
                    fold_data[
                        "train_end"
                    ],

                "train_period":
                    (
                        f"{fold_data['train_start']}"
                        f"~"
                        f"{fold_data['train_end']}"
                    ),

                "valid_season":
                    fold_data[
                        "valid_season"
                    ],

                "train_rows":
                    fold_data[
                        "train_rows"
                    ],

                "valid_rows":
                    fold_data[
                        "valid_rows"
                    ],

                "train_pitchers":
                    train_pitcher_count,

                "valid_pitchers":
                    valid_pitcher_count,

                "train_rate":
                    train_rate,

                "valid_rate":
                    valid_rate,

                "raw_feature_count":
                    len(
                        feature_list
                    ),

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

                "naive_probability":
                    naive[
                        "probability"
                    ],

                "naive_brier":
                    naive[
                        "brier"
                    ],

                "naive_logloss":
                    naive[
                        "logloss"
                    ],

                "brier":
                    metrics[
                        "brier"
                    ],

                "logloss":
                    metrics[
                        "logloss"
                    ],

                "auc":
                    metrics[
                        "auc"
                    ],

                "brier_improvement":
                    brier_improvement,

                "logloss_improvement":
                    logloss_improvement,

                "beats_naive":
                    metrics[
                        "brier"
                    ]
                    < naive[
                        "brier"
                    ],

                "prediction_mean":
                    float(
                        valid_prob.mean()
                    ),

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

                "predict_seconds":
                    predict_seconds,

                "total_seconds":
                    total_seconds,
            }


            results.append(
                result
            )


            print("\n" + "-" * 90)

            print(
                f"Naive Brier       : "
                f"{naive['brier']:.6f}"
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
                f"Log Loss          : "
                f"{metrics['logloss']:.6f}"
            )

            print(
                f"ROC-AUC           : "
                f"{metrics['auc']:.6f}"
            )

            print(
                f"Prediction Mean   : "
                f"{valid_prob.mean():.6f}"
            )

            print(
                f"Total Time        : "
                f"{total_seconds:.1f} sec"
            )


            # -----------------------------------------------------------------
            # Colab 중간 결과 저장
            #
            # Runtime 중단 시 현재까지 계산한 결과를 확인할 수 있게 함.
            # OUTPUT_DIR을 Drive 경로로 바꾸면 영구 보존 가능.
            # -----------------------------------------------------------------

            pd.DataFrame(
                results
            ).to_csv(
                os.path.join(
                    OUTPUT_DIR,
                    "lr11_fold_results_progress.csv",
                ),
                encoding="utf-8-sig",
                index=False,
            )


            # -----------------------------------------------------------------
            # 메모리 회수
            # -----------------------------------------------------------------

            del X_train
            del X_valid
            del pipeline
            del model
            del preprocessor
            del valid_prob

            gc.collect()


        del y_train_fold
        del y_valid

        gc.collect()


    return pd.DataFrame(
        results
    )


# =============================================================================
# 17. CV 실행
# =============================================================================

results_df = run_time_cv(
    data=train,
    fold_configs=FOLD_CONFIGS,
    experiments=LR_EXPERIMENTS,
)


# =============================================================================
# 18. Fold Result 검증
# =============================================================================

expected_result_count = (
    len(
        FOLD_CONFIGS
    )
    * len(
        LR_EXPERIMENTS
    )
)


assert len(
    results_df
) == expected_result_count


assert (
    results_df
    .groupby(
        "model"
    )[
        "fold"
    ]
    .nunique()
    .eq(
        3
    )
    .all()
)


assert (
    results_df[
        "train_end"
    ]
    <
    results_df[
        "valid_season"
    ]
).all()


assert np.isfinite(
    results_df[
        [
            "brier",
            "logloss",
            "auc",
        ]
    ]
    .to_numpy()
).all()


# =============================================================================
# 19. Fold별 최종 결과
# =============================================================================

results_df = (
    results_df
    .sort_values(
        [
            "fold",
            "brier",
        ]
    )
    .reset_index(
        drop=True
    )
)


print("\n" + "=" * 180)
print("FOLD RESULTS")
print("=" * 180)


FOLD_DISPLAY_COLS = [
    "fold",
    "train_period",
    "valid_season",
    "model",

    "raw_feature_count",
    "transformed_feature_count",

    "naive_brier",
    "brier",
    "brier_improvement",
    "beats_naive",

    "logloss",
    "auc",

    "n_iter",
    "converged",

    "fit_seconds",
]


print(
    results_df[
        FOLD_DISPLAY_COLS
    ]
    .round(6)
    .to_string(
        index=False,
    )
)


# =============================================================================
# 20. Model별 CV Mean / Std
#
# std는 LR-09와 동일하게 population std (ddof=0)
# =============================================================================

cv_summary_rows = []


for model_name, model_df in results_df.groupby(
    "model"
):

    cv_summary_rows.append(
        {
            "model":
                model_name,

            "folds":
                len(
                    model_df
                ),

            "mean_brier":
                model_df[
                    "brier"
                ]
                .mean(),

            "std_brier":
                model_df[
                    "brier"
                ]
                .std(
                    ddof=0
                ),

            "mean_logloss":
                model_df[
                    "logloss"
                ]
                .mean(),

            "std_logloss":
                model_df[
                    "logloss"
                ]
                .std(
                    ddof=0
                ),

            "mean_auc":
                model_df[
                    "auc"
                ]
                .mean(),

            "std_auc":
                model_df[
                    "auc"
                ]
                .std(
                    ddof=0
                ),

            "mean_brier_improvement":
                model_df[
                    "brier_improvement"
                ]
                .mean(),

            "std_brier_improvement":
                model_df[
                    "brier_improvement"
                ]
                .std(
                    ddof=0
                ),

            "folds_beating_naive":
                int(
                    model_df[
                        "beats_naive"
                    ]
                    .sum()
                ),

            "mean_prediction":
                model_df[
                    "prediction_mean"
                ]
                .mean(),

            "mean_fit_seconds":
                model_df[
                    "fit_seconds"
                ]
                .mean(),

            "all_converged":
                bool(
                    model_df[
                        "converged"
                    ]
                    .all()
                ),
        }
    )


cv_summary_df = (
    pd.DataFrame(
        cv_summary_rows
    )
    .sort_values(
        "mean_brier",
        ascending=True,
    )
    .reset_index(
        drop=True
    )
)


print("\n" + "=" * 140)
print("CV SUMMARY")
print("=" * 140)

print(
    cv_summary_df
    .round(6)
    .to_string(
        index=False,
    )
)


# =============================================================================
# 21. Naive CV Summary
#
# 모델별 중복 저장된 naive 값을 fold당 한 번만 사용
# =============================================================================

naive_fold_df = (
    results_df[
        [
            "fold",
            "train_period",
            "valid_season",
            "naive_probability",
            "naive_brier",
            "naive_logloss",
        ]
    ]
    .drop_duplicates(
        subset=[
            "fold"
        ]
    )
    .sort_values(
        "fold"
    )
)


NAIVE_MEAN_BRIER = (
    naive_fold_df[
        "naive_brier"
    ]
    .mean()
)


NAIVE_STD_BRIER = (
    naive_fold_df[
        "naive_brier"
    ]
    .std(
        ddof=0
    )
)


NAIVE_MEAN_LOGLOSS = (
    naive_fold_df[
        "naive_logloss"
    ]
    .mean()
)


NAIVE_STD_LOGLOSS = (
    naive_fold_df[
        "naive_logloss"
    ]
    .std(
        ddof=0
    )
)


print("\n" + "=" * 80)
print("NAIVE CV SUMMARY")
print("=" * 80)

print(
    f"Mean Brier   : "
    f"{NAIVE_MEAN_BRIER:.6f}"
)

print(
    f"Std Brier    : "
    f"{NAIVE_STD_BRIER:.6f}"
)

print(
    f"Mean LogLoss : "
    f"{NAIVE_MEAN_LOGLOSS:.6f}"
)

print(
    f"Std LogLoss  : "
    f"{NAIVE_STD_LOGLOSS:.6f}"
)


# =============================================================================
# 22. Best CV Model
# =============================================================================

best_row = (
    cv_summary_df
    .iloc[0]
)


print("\n" + "=" * 80)
print("BEST CV MODEL")
print("=" * 80)

print(
    f"Model        : "
    f"{best_row['model']}"
)

print(
    f"Mean Brier   : "
    f"{best_row['mean_brier']:.6f}"
)

print(
    f"Std Brier    : "
    f"{best_row['std_brier']:.6f}"
)

print(
    f"Mean LogLoss : "
    f"{best_row['mean_logloss']:.6f}"
)

print(
    f"Mean AUC     : "
    f"{best_row['mean_auc']:.6f}"
)

print(
    f"Naive beats  : "
    f"{int(best_row['folds_beating_naive'])}/3 folds"
)


# =============================================================================
# 23. Leakage Validation Summary
# =============================================================================

leakage_validation_df = pd.DataFrame(
    [
        {
            "check":
                "temporal_boundary",

            "result":
                "PASS",

            "detail":
                "max(train season) < valid season for all folds",
        },

        {
            "check":
                "preprocessing",

            "result":
                "PASS",

            "detail":
                (
                    "new pipeline per fold; "
                    "fit on fold train only; "
                    "valid uses transform/predict only"
                ),
        },

        {
            "check":
                "preprocessing_state",

            "result":
                (
                    "PASS"
                    if preprocessing_leakage_pass
                    else "FAIL"
                ),

            "detail":
                (
                    "imputer/scaler/encoder state unchanged "
                    "after validation transform"
                ),
        },

        {
            "check":
                "historical_features",

            "result":
                "PASS",

            "detail":
                (
                    "official asof_* 19 only; "
                    "not recomputed from fold target"
                ),
        },

        {
            "check":
                "rolling",

            "result":
                "PASS",

            "detail":
                (
                    "no custom target-based rolling feature "
                    "is generated in LR-11"
                ),
        },

        {
            "check":
                "expanding",

            "result":
                "PASS",

            "detail":
                (
                    "no custom target-based expanding feature "
                    "is generated in LR-11"
                ),
        },

        {
            "check":
                "validation_target",

            "result":
                "PASS",

            "detail":
                (
                    "validation target is used only for "
                    "metric calculation"
                ),
        },
    ]
)


print("\n" + "=" * 100)
print("LEAKAGE VALIDATION")
print("=" * 100)

print(
    leakage_validation_df
    .to_string(
        index=False,
    )
)


# =============================================================================
# 24. 결과 저장
# =============================================================================

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lr11_fold_results.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


cv_summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lr11_cv_summary.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


naive_fold_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lr11_naive_fold_results.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


leakage_validation_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lr11_leakage_validation.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


# 최종 완료 후 progress 파일은 삭제 가능
progress_path = os.path.join(
    OUTPUT_DIR,
    "lr11_fold_results_progress.csv",
)

if os.path.exists(
    progress_path
):
    os.remove(
        progress_path
    )


print("\n" + "=" * 80)
print("LR-11 EXPANDING WINDOW CV COMPLETE")
print("=" * 80)

print(
    f"Output directory : "
    f"{OUTPUT_DIR}"
)