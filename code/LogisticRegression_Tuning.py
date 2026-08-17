import gc
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import sklearn

from scipy import sparse

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
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

# Colab runtime 종료에도 결과를 보존하도록 Drive에 저장
OUTPUT_DIR = (
    "/content/drive/MyDrive/𝟐𝟎𝟐𝟔/aimers/9기/open/"
    "lr13_regularization_outputs"
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
# 1. Logistic Regression 공통 설정
#
# L1 / L2 / ElasticNet을 하나의 solver로 동일하게 비교하기 위해 saga 사용
# =============================================================================

LR_SOLVER = "saga"
LR_MAX_ITER = 500
LR_TOL = 1e-3

# 같은 penalty에서 C를 순차 탐색할 때 이전 coefficient를 초기값으로 재사용
USE_WARM_START = True

# practical coefficient sparsity 판정
COEF_ZERO_THRESHOLD = 1e-8


# =============================================================================
# 2. Hyperparameter Grid
# =============================================================================

C_GRID = [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]


# 0은 L2, 1은 L1과 중복되므로 제외
ELASTICNET_L1_RATIOS = [
    0.25,
    0.50,
    0.75,
]


PENALTY_SPECS = [
    {
        "penalty": "l2",
        "l1_ratio": None,
    },
    {
        "penalty": "l1",
        "l1_ratio": None,
    },
    {
        "penalty": "elasticnet",
        "l1_ratio": 0.25,
    },
    {
        "penalty": "elasticnet",
        "l1_ratio": 0.50,
    },
    {
        "penalty": "elasticnet",
        "l1_ratio": 0.75,
    },
]


# saga가 세 penalty 모두 지원해야 함
assert LR_SOLVER == "saga"

assert {
    spec["penalty"]
    for spec in PENALTY_SPECS
}.issubset(
    {
        "l1",
        "l2",
        "elasticnet",
    }
)


# =============================================================================
# 3. Expanding Window
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
# 4. 실행 환경
# =============================================================================

print("=" * 80)
print("ENVIRONMENT")
print("=" * 80)

print(f"scikit-learn : {sklearn.__version__}")
print(f"NumPy        : {np.__version__}")
print(f"Pandas       : {pd.__version__}")

print("\n" + "=" * 80)
print("TUNING GRID")
print("=" * 80)

print(f"C Grid              : {C_GRID}")
print(f"ElasticNet l1_ratio : {ELASTICNET_L1_RATIOS}")
print(f"Solver              : {LR_SOLVER}")
print(f"Warm Start          : {USE_WARM_START}")


# =============================================================================
# 5. Test는 schema만 로드
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
# 6. Feature Type
#
# LR-05 ~ LR-12와 동일
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
    "season",
    "game_month",
    "inning",

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

    "home_win_expectancy",
    "away_win_expectancy",
    "li",

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

    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",

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
# 7. Train 로드
#
# Colab 최적화:
# - row_id 제외
# - 47 features + target만 로드
# - numeric float32
# - target int8
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
    f"Load time        : "
    f"{load_seconds:.1f} sec"
)

print(
    f"DataFrame memory : "
    f"{train.memory_usage(deep=True).sum() / 1024**3:.2f} GB"
)


# =============================================================================
# 8. Model C / D만 Tuning
#
# LR-12 결과:
# C와 D가 A/B보다 명확히 우수하고,
# C/D 차이는 매우 작으므로 둘 다 유지
# =============================================================================

HISTORICAL_FEATURES = [
    col
    for col in ALL_FEATURES
    if col.startswith("asof_")
]


assert len(
    HISTORICAL_FEATURES
) == 19


BASE_FEATURES = [
    col
    for col in ALL_FEATURES
    if (
        col != PITCHER_ID_COL
        and col not in HISTORICAL_FEATURES
    )
]


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


TUNING_MODELS = {
    "LR-C-Historical": {
        "features":
            MODEL_C_FEATURES,

        "use_pitcher_id":
            False,
    },

    "LR-D-PitcherID-Historical": {
        "features":
            MODEL_D_FEATURES,

        "use_pitcher_id":
            True,
    },
}


assert len(
    MODEL_C_FEATURES
) == 46

assert len(
    MODEL_D_FEATURES
) == 47


# =============================================================================
# 9. 전체 실험 수
# =============================================================================

CONFIGS_PER_MODEL = (
    len(C_GRID)
    * len(PENALTY_SPECS)
)

TOTAL_FITS = (
    len(FOLD_CONFIGS)
    * len(TUNING_MODELS)
    * CONFIGS_PER_MODEL
)


print("\n" + "=" * 80)
print("EXPERIMENT SIZE")
print("=" * 80)

print(
    f"Configs / model : "
    f"{CONFIGS_PER_MODEL}"
)

print(
    f"Models          : "
    f"{len(TUNING_MODELS)}"
)

print(
    f"Folds           : "
    f"{len(FOLD_CONFIGS)}"
)

print(
    f"Total LR fits   : "
    f"{TOTAL_FITS}"
)


# =============================================================================
# 10. Historical Boundary 검증
# =============================================================================

CUSTOM_TARGET_HISTORY_COLS = [
    col
    for col in ALL_FEATURES
    if col.startswith("hist_")
]


assert not CUSTOM_TARGET_HISTORY_COLS

assert TARGET not in HISTORICAL_FEATURES


print("\nHistorical policy:")
print(
    f"- Official asof_* : "
    f"{len(HISTORICAL_FEATURES)}"
)

print(
    "- Custom target rolling / expanding : 0"
)

print(
    "- 2025 data used for tuning          : False"
)


# =============================================================================
# 11. Feature Type 반환
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
    }


# =============================================================================
# 12. Preprocessor
#
# 매우 중요한 최적화:
#
# hyperparameter마다 fit하지 않는다.
#
# fold × model마다 단 한 번 fit:
#
# 3 folds × 2 models = 6 preprocessing fits
#
# 이후 180 LR fit에서 같은 transformed sparse matrix 재사용
# =============================================================================

def build_preprocessor(
    feature_list,
):
    feature_types = (
        get_feature_types(
            feature_list
        )
    )


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


    if feature_types[
        "scale"
    ]:

        transformers.append(
            (
                "numeric_scale",
                scale_numeric_pipeline,
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
                no_scale_numeric_pipeline,
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
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )


# =============================================================================
# 13. Fold Generator
# =============================================================================

def expanding_window_splits(
    data,
):
    for config in FOLD_CONFIGS:

        train_mask = (
            (
                data[
                    SEASON_COL
                ]
                >= config[
                    "train_start"
                ]
            )
            &
            (
                data[
                    SEASON_COL
                ]
                <= config[
                    "train_end"
                ]
            )
        )


        valid_mask = (
            data[
                SEASON_COL
            ]
            == config[
                "valid_season"
            ]
        )


        assert train_mask.sum() > 0
        assert valid_mask.sum() > 0


        assert (
            data.loc[
                train_mask,
                SEASON_COL,
            ]
            .max()
            <
            config[
                "valid_season"
            ]
        )


        yield {
            **config,

            "train_mask":
                train_mask,

            "valid_mask":
                valid_mask,

            "train_rows":
                int(
                    train_mask.sum()
                ),

            "valid_rows":
                int(
                    valid_mask.sum()
                ),
        }


# =============================================================================
# 14. Known / Unknown Mask
# =============================================================================

def make_pitcher_masks(
    data,
    train_mask,
    valid_mask,
):
    train_pitchers = set(
        data.loc[
            train_mask,
            PITCHER_ID_COL,
        ]
        .dropna()
        .unique()
    )


    valid_pitcher = (
        data.loc[
            valid_mask,
            PITCHER_ID_COL,
        ]
        .reset_index(
            drop=True
        )
    )


    known_mask = (
        valid_pitcher
        .isin(
            train_pitchers
        )
        .to_numpy()
    )


    unknown_mask = (
        ~known_mask
    )


    assert (
        known_mask
        | unknown_mask
    ).all()


    assert not (
        known_mask
        & unknown_mask
    ).any()


    return (
        known_mask,
        unknown_mask,
    )


# =============================================================================
# 15. Metric
# =============================================================================

def evaluate_brier(
    y_true,
    y_prob,
):
    return brier_score_loss(
        y_true,
        y_prob,
    )


def evaluate_logloss(
    y_true,
    y_prob,
):
    return log_loss(
        y_true,
        y_prob,
        labels=[
            0,
            1,
        ],
    )


# =============================================================================
# 16. Sparse float32 보장
# =============================================================================

def ensure_float32_matrix(
    matrix,
):
    if sparse.issparse(
        matrix
    ):

        return (
            matrix
            .tocsr()
            .astype(
                np.float32,
                copy=False,
            )
        )


    return np.asarray(
        matrix,
        dtype=np.float32,
    )


# =============================================================================
# 17. Logistic Regression Builder
# =============================================================================

def build_logistic_model(
    penalty,
    C,
    l1_ratio,
):
    kwargs = {
        "penalty":
            penalty,

        "C":
            C,

        "solver":
            LR_SOLVER,

        "max_iter":
            LR_MAX_ITER,

        "tol":
            LR_TOL,

        "random_state":
            RANDOM_STATE,

        "warm_start":
            USE_WARM_START,

        "verbose":
            0,
    }


    if penalty == "elasticnet":

        kwargs[
            "l1_ratio"
        ] = l1_ratio


    return LogisticRegression(
        **kwargs
    )


# =============================================================================
# 18. Checkpoint / Resume
# =============================================================================

PROGRESS_PATH = os.path.join(
    OUTPUT_DIR,
    "lr13_tuning_progress.csv",
)


def normalize_l1_ratio(
    value,
):
    if value is None:

        return None


    if pd.isna(
        value
    ):

        return None


    return float(
        value
    )


def make_result_key(
    fold,
    model,
    penalty,
    C,
    l1_ratio,
):
    normalized_ratio = (
        normalize_l1_ratio(
            l1_ratio
        )
    )


    ratio_text = (
        "NA"
        if normalized_ratio is None
        else f"{normalized_ratio:.6f}"
    )


    return (
        f"{int(fold)}|"
        f"{model}|"
        f"{penalty}|"
        f"{float(C):.12g}|"
        f"{ratio_text}"
    )


if os.path.exists(
    PROGRESS_PATH
):

    previous_df = pd.read_csv(
        PROGRESS_PATH
    )


    results = (
        previous_df
        .to_dict(
            "records"
        )
    )


    completed_keys = {
        make_result_key(
            row[
                "fold"
            ],
            row[
                "model"
            ],
            row[
                "penalty"
            ],
            row[
                "C"
            ],
            row.get(
                "l1_ratio",
                None,
            ),
        )
        for row
        in results
    }


    print(
        f"\nResume checkpoint : "
        f"{len(completed_keys)}/{TOTAL_FITS} fits completed"
    )

else:

    results = []
    completed_keys = set()


# =============================================================================
# 19. Tuning Runner
# =============================================================================

run_counter = len(
    completed_keys
)


for fold_data in expanding_window_splits(
    train
):

    fold = (
        fold_data[
            "fold"
        ]
    )


    train_mask = (
        fold_data[
            "train_mask"
        ]
    )


    valid_mask = (
        fold_data[
            "valid_mask"
        ]
    )


    y_train = (
        train.loc[
            train_mask,
            TARGET,
        ]
        .to_numpy(
            dtype=np.int8,
            copy=True,
        )
    )


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


    (
        known_mask,
        unknown_mask,
    ) = make_pitcher_masks(
        data=train,
        train_mask=train_mask,
        valid_mask=valid_mask,
    )


    print("\n" + "=" * 110)

    print(
        f"FOLD {fold} | "
        f"{fold_data['train_start']}"
        f"~"
        f"{fold_data['train_end']} "
        f"→ "
        f"{fold_data['valid_season']}"
    )

    print("=" * 110)

    print(
        f"Train rows   : "
        f"{len(y_train):,}"
    )

    print(
        f"Valid rows   : "
        f"{len(y_valid):,}"
    )

    print(
        f"Known rows   : "
        f"{known_mask.sum():,}"
    )

    print(
        f"Unknown rows : "
        f"{unknown_mask.sum():,}"
    )


    for (
        model_name,
        model_config,
    ) in TUNING_MODELS.items():

        feature_list = (
            model_config[
                "features"
            ]
        )


        # ---------------------------------------------------------------------
        # 이 fold/model의 모든 config가 이미 완료됐으면 preprocessing도 skip
        # ---------------------------------------------------------------------

        expected_keys = {
            make_result_key(
                fold=fold,
                model=model_name,
                penalty=spec[
                    "penalty"
                ],
                C=C,
                l1_ratio=spec[
                    "l1_ratio"
                ],
            )
            for spec
            in PENALTY_SPECS
            for C
            in C_GRID
        }


        if expected_keys.issubset(
            completed_keys
        ):

            print(
                f"\n{model_name}: "
                f"all configs already completed → skip"
            )

            continue


        # ---------------------------------------------------------------------
        # Preprocessing
        #
        # hyperparameter별 재-fit 금지.
        # fold/model당 정확히 1번 train fit.
        # ---------------------------------------------------------------------

        print("\n" + "-" * 110)

        print(
            f"Fold {fold} | "
            f"{model_name} | "
            f"preprocessing 시작"
        )

        print("-" * 110)


        preprocess_start = (
            time.perf_counter()
        )


        X_train = (
            train.loc[
                train_mask,
                feature_list,
            ]
        )


        X_valid = (
            train.loc[
                valid_mask,
                feature_list,
            ]
        )


        preprocessor = (
            build_preprocessor(
                feature_list
            )
        )


        X_train_transformed = (
            preprocessor
            .fit_transform(
                X_train
            )
        )


        X_valid_transformed = (
            preprocessor
            .transform(
                X_valid
            )
        )


        X_train_transformed = (
            ensure_float32_matrix(
                X_train_transformed
            )
        )


        X_valid_transformed = (
            ensure_float32_matrix(
                X_valid_transformed
            )
        )


        transformed_feature_count = (
            X_train_transformed
            .shape[1]
        )


        preprocess_seconds = (
            time.perf_counter()
            - preprocess_start
        )


        assert (
            X_train_transformed
            .shape[1]
            ==
            X_valid_transformed
            .shape[1]
        )


        print(
            f"Raw features         : "
            f"{len(feature_list)}"
        )

        print(
            f"Transformed features : "
            f"{transformed_feature_count:,}"
        )

        print(
            f"Preprocessing time   : "
            f"{preprocess_seconds:.1f} sec"
        )


        # Raw DataFrame은 이제 필요 없음
        del X_train
        del X_valid
        del preprocessor

        gc.collect()


        # ---------------------------------------------------------------------
        # Penalty별 warm-start sequence
        # ---------------------------------------------------------------------

        for spec in PENALTY_SPECS:

            penalty = (
                spec[
                    "penalty"
                ]
            )


            l1_ratio = (
                spec[
                    "l1_ratio"
                ]
            )


            # 같은 penalty / l1_ratio 내에서 C ascending
            # 하나의 model 객체를 재사용
            lr_model = (
                build_logistic_model(
                    penalty=penalty,
                    C=C_GRID[0],
                    l1_ratio=l1_ratio,
                )
            )


            for C in C_GRID:

                result_key = (
                    make_result_key(
                        fold=fold,
                        model=model_name,
                        penalty=penalty,
                        C=C,
                        l1_ratio=l1_ratio,
                    )
                )


                if result_key in completed_keys:

                    continue


                run_counter += 1


                lr_model.set_params(
                    C=C
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


                    lr_model.fit(
                        X_train_transformed,
                        y_train,
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


                n_iter = int(
                    np.max(
                        lr_model.n_iter_
                    )
                )


                converged = (
                    len(
                        convergence_warnings
                    )
                    == 0
                    and n_iter
                    < LR_MAX_ITER
                )


                # -------------------------------------------------------------
                # Prediction
                # -------------------------------------------------------------

                predict_start = (
                    time.perf_counter()
                )


                valid_prob = (
                    lr_model
                    .predict_proba(
                        X_valid_transformed
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


                assert np.isfinite(
                    valid_prob
                ).all()


                assert (
                    valid_prob >= 0
                ).all()


                assert (
                    valid_prob <= 1
                ).all()


                # -------------------------------------------------------------
                # Overall / Known / Unknown
                # -------------------------------------------------------------

                overall_brier = (
                    evaluate_brier(
                        y_valid,
                        valid_prob,
                    )
                )


                known_brier = (
                    evaluate_brier(
                        y_valid[
                            known_mask
                        ],
                        valid_prob[
                            known_mask
                        ],
                    )
                )


                unknown_brier = (
                    evaluate_brier(
                        y_valid[
                            unknown_mask
                        ],
                        valid_prob[
                            unknown_mask
                        ],
                    )
                )


                overall_logloss = (
                    evaluate_logloss(
                        y_valid,
                        valid_prob,
                    )
                )


                # -------------------------------------------------------------
                # Coefficient Sparsity
                # -------------------------------------------------------------

                coef = (
                    lr_model
                    .coef_
                    .ravel()
                )


                coef_zero_count = int(
                    (
                        np.abs(
                            coef
                        )
                        <= COEF_ZERO_THRESHOLD
                    )
                    .sum()
                )


                coef_total_count = int(
                    coef.size
                )


                coef_nonzero_count = (
                    coef_total_count
                    - coef_zero_count
                )


                coefficient_sparsity = (
                    coef_zero_count
                    / coef_total_count
                )


                result = {
                    "fold":
                        fold,

                    "model":
                        model_name,

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

                    "raw_feature_count":
                        len(
                            feature_list
                        ),

                    "transformed_feature_count":
                        transformed_feature_count,

                    "pitcher_id":
                        model_config[
                            "use_pitcher_id"
                        ],

                    "penalty":
                        penalty,

                    "C":
                        C,

                    "l1_ratio":
                        (
                            np.nan
                            if l1_ratio is None
                            else l1_ratio
                        ),

                    "solver":
                        LR_SOLVER,

                    "overall_brier":
                        overall_brier,

                    "known_brier":
                        known_brier,

                    "unknown_brier":
                        unknown_brier,

                    "overall_logloss":
                        overall_logloss,

                    "coef_total_count":
                        coef_total_count,

                    "coef_nonzero_count":
                        coef_nonzero_count,

                    "coef_zero_count":
                        coef_zero_count,

                    "coefficient_sparsity":
                        coefficient_sparsity,

                    "n_iter":
                        n_iter,

                    "converged":
                        converged,

                    "convergence_warning":
                        len(
                            convergence_warnings
                        )
                        > 0,

                    "warning_message":
                        " | ".join(
                            str(
                                warning.message
                            )
                            for warning
                            in convergence_warnings
                        ),

                    "preprocess_seconds":
                        preprocess_seconds,

                    "fit_seconds":
                        fit_seconds,

                    "predict_seconds":
                        predict_seconds,
                }


                results.append(
                    result
                )


                completed_keys.add(
                    result_key
                )


                # -------------------------------------------------------------
                # Checkpoint
                # -------------------------------------------------------------

                progress_df = (
                    pd.DataFrame(
                        results
                    )
                    .drop_duplicates(
                        subset=[
                            "fold",
                            "model",
                            "penalty",
                            "C",
                            "l1_ratio",
                        ],
                        keep="last",
                    )
                )


                progress_df.to_csv(
                    PROGRESS_PATH,
                    encoding="utf-8-sig",
                    index=False,
                )


                ratio_text = (
                    "-"
                    if l1_ratio is None
                    else f"{l1_ratio:.2f}"
                )


                print(
                    f"[{run_counter:03d}/{TOTAL_FITS}] "
                    f"F{fold} | "
                    f"{model_name} | "
                    f"{penalty:<10} | "
                    f"C={C:<7g} | "
                    f"l1={ratio_text:<4} | "
                    f"Brier={overall_brier:.6f} | "
                    f"K={known_brier:.6f} | "
                    f"U={unknown_brier:.6f} | "
                    f"Sparsity={coefficient_sparsity:.2%} | "
                    f"Iter={n_iter:<3d} | "
                    f"{fit_seconds:.1f}s"
                )


                del valid_prob
                del coef

                gc.collect()


            del lr_model

            gc.collect()


        # ---------------------------------------------------------------------
        # fold/model transformed matrix 해제
        # ---------------------------------------------------------------------

        del X_train_transformed
        del X_valid_transformed

        gc.collect()


    del y_train
    del y_valid
    del known_mask
    del unknown_mask

    gc.collect()


# =============================================================================
# 20. 최종 Result 정리
# =============================================================================

results_df = (
    pd.DataFrame(
        results
    )
    .drop_duplicates(
        subset=[
            "fold",
            "model",
            "penalty",
            "C",
            "l1_ratio",
        ],
        keep="last",
    )
    .sort_values(
        [
            "model",
            "penalty",
            "l1_ratio",
            "C",
            "fold",
        ],
        na_position="first",
    )
    .reset_index(
        drop=True
    )
)


assert len(
    results_df
) == TOTAL_FITS, (
    f"Expected {TOTAL_FITS} results, "
    f"got {len(results_df)}"
)


# 모든 config가 정확히 3 folds
config_fold_counts = (
    results_df
    .groupby(
        [
            "model",
            "penalty",
            "C",
            "l1_ratio",
        ],
        dropna=False,
    )[
        "fold"
    ]
    .nunique()
)


assert (
    config_fold_counts
    == 3
).all()


# =============================================================================
# 21. CV Summary
# =============================================================================

summary_rows = []


for (
    model_name,
    penalty,
    C,
    l1_ratio,
), group in results_df.groupby(
    [
        "model",
        "penalty",
        "C",
        "l1_ratio",
    ],
    dropna=False,
):

    summary_rows.append(
        {
            "model":
                model_name,

            "penalty":
                penalty,

            "C":
                C,

            "l1_ratio":
                l1_ratio,

            "mean_overall_brier":
                group[
                    "overall_brier"
                ]
                .mean(),

            "std_overall_brier":
                group[
                    "overall_brier"
                ]
                .std(
                    ddof=0
                ),

            "mean_known_brier":
                group[
                    "known_brier"
                ]
                .mean(),

            "std_known_brier":
                group[
                    "known_brier"
                ]
                .std(
                    ddof=0
                ),

            "mean_unknown_brier":
                group[
                    "unknown_brier"
                ]
                .mean(),

            "std_unknown_brier":
                group[
                    "unknown_brier"
                ]
                .std(
                    ddof=0
                ),

            "mean_logloss":
                group[
                    "overall_logloss"
                ]
                .mean(),

            "mean_coefficient_sparsity":
                group[
                    "coefficient_sparsity"
                ]
                .mean(),

            "mean_nonzero_coef":
                group[
                    "coef_nonzero_count"
                ]
                .mean(),

            "mean_n_iter":
                group[
                    "n_iter"
                ]
                .mean(),

            "convergence_warning_folds":
                int(
                    group[
                        "convergence_warning"
                    ]
                    .sum()
                ),

            "all_converged":
                bool(
                    group[
                        "converged"
                    ]
                    .all()
                ),

            "mean_fit_seconds":
                group[
                    "fit_seconds"
                ]
                .mean(),
        }
    )


summary_df = (
    pd.DataFrame(
        summary_rows
    )
    .sort_values(
        [
            "mean_overall_brier",
            "mean_unknown_brier",
            "std_overall_brier",
            "mean_nonzero_coef",
            "C",
        ],
        ascending=[
            True,
            True,
            True,
            True,
            True,
        ],
    )
    .reset_index(
        drop=True
    )
)


# =============================================================================
# 22. Penalty별 Best
#
# 각 Model × Penalty에서 Mean Overall Brier가 가장 낮은 config
# =============================================================================

penalty_best_df = (
    summary_df
    .sort_values(
        [
            "model",
            "penalty",
            "mean_overall_brier",
            "mean_unknown_brier",
            "std_overall_brier",
        ]
    )
    .groupby(
        [
            "model",
            "penalty",
        ],
        as_index=False,
        sort=False,
    )
    .head(1)
    .reset_index(
        drop=True
    )
)


# =============================================================================
# 23. Global Best
#
# 1차 기준:
#   Mean Overall Brier
#
# 동률 판단용:
#   Unknown Brier
#   Overall Std
#   coefficient complexity
# =============================================================================

best_row = (
    summary_df
    .iloc[0]
)


best_l1_ratio = (
    None
    if pd.isna(
        best_row[
            "l1_ratio"
        ]
    )
    else float(
        best_row[
            "l1_ratio"
        ]
    )
)


BEST_CONFIG = {
    "model":
        best_row[
            "model"
        ],

    "penalty":
        best_row[
            "penalty"
        ],

    "C":
        float(
            best_row[
                "C"
            ]
        ),

    "l1_ratio":
        best_l1_ratio,

    "solver":
        LR_SOLVER,

    "mean_brier":
        float(
            best_row[
                "mean_overall_brier"
            ]
        ),

    "std_brier":
        float(
            best_row[
                "std_overall_brier"
            ]
        ),

    "known_brier":
        float(
            best_row[
                "mean_known_brier"
            ]
        ),

    "unknown_brier":
        float(
            best_row[
                "mean_unknown_brier"
            ]
        ),

    "mean_coefficient_sparsity":
        float(
            best_row[
                "mean_coefficient_sparsity"
            ]
        ),
}


# =============================================================================
# 24. LR-12 Baseline Config 확인
#
# L2 / C=1.0의 tuning 결과를 따로 표시
# =============================================================================

baseline_reference_df = (
    summary_df[
        (
            summary_df[
                "penalty"
            ]
            == "l2"
        )
        &
        (
            np.isclose(
                summary_df[
                    "C"
                ],
                1.0,
            )
        )
    ]
    .sort_values(
        "model"
    )
    .reset_index(
        drop=True
    )
)


# =============================================================================
# 25. 출력
# =============================================================================

print("\n" + "=" * 160)
print("PENALTY BEST RESULTS")
print("=" * 160)

print(
    penalty_best_df[
        [
            "model",
            "penalty",
            "C",
            "l1_ratio",
            "mean_overall_brier",
            "std_overall_brier",
            "mean_known_brier",
            "mean_unknown_brier",
            "mean_coefficient_sparsity",
            "convergence_warning_folds",
        ]
    ]
    .round(6)
    .to_string(
        index=False
    )
)


print("\n" + "=" * 120)
print("L2 C=1.0 BASELINE REFERENCE")
print("=" * 120)

print(
    baseline_reference_df[
        [
            "model",
            "mean_overall_brier",
            "mean_known_brier",
            "mean_unknown_brier",
            "std_overall_brier",
        ]
    ]
    .round(6)
    .to_string(
        index=False
    )
)


print("\n" + "=" * 80)
print("BEST CONFIG")
print("=" * 80)

for key, value in BEST_CONFIG.items():

    print(
        f"{key:<28}: "
        f"{value}"
    )


# =============================================================================
# 26. 결과 저장
# =============================================================================

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lr13_fold_results.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lr13_cv_summary.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


penalty_best_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lr13_penalty_best.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


baseline_reference_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lr13_l2_c1_baseline_reference.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


with open(
    os.path.join(
        OUTPUT_DIR,
        "lr13_best_config.json",
    ),
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        BEST_CONFIG,
        file,
        ensure_ascii=False,
        indent=2,
    )


# 최종 완료 시 progress도 최종 결과와 일치하도록 갱신
results_df.to_csv(
    PROGRESS_PATH,
    encoding="utf-8-sig",
    index=False,
)


print("\n" + "=" * 80)
print("LR-13 REGULARIZATION TUNING COMPLETE")
print("=" * 80)

print(
    f"Total fits      : "
    f"{len(results_df)}"
)

print(
    f"Output directory: "
    f"{OUTPUT_DIR}"
)