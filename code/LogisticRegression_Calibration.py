import gc
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
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

OUTPUT_DIR = (
    "/content/drive/MyDrive/𝟐𝟎𝟐𝟔/aimers/9기/open/"
    "lr15_calibration_outputs"
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
# 1. LR-13 / LR-14 최종 후보 설정
#
# Performance-first candidate:
# M3-Historical
#
# pitcher_id 없음
# official asof_* 포함
# =============================================================================

MODEL_NAME = "M3-Historical"

LR_PENALTY = "l1"
LR_C = 0.001
LR_SOLVER = "saga"
LR_MAX_ITER = 500
LR_TOL = 1e-3


# Calibration
CALIBRATION_BINS = 10
HISTOGRAM_BINS = 30

# "예측확률 약 0.7" 질문 확인용
PROBABILITY_BAND_LOW = 0.65
PROBABILITY_BAND_HIGH = 0.75


# =============================================================================
# 2. Expanding Window
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
# 3. 환경
# =============================================================================

print("=" * 80)
print("ENVIRONMENT")
print("=" * 80)

print(f"scikit-learn : {sklearn.__version__}")
print(f"NumPy        : {np.__version__}")
print(f"Pandas       : {pd.__version__}")

print("\nFinal candidate:")
print(f"Model   : {MODEL_NAME}")
print(f"Penalty : {LR_PENALTY}")
print(f"C       : {LR_C}")


# =============================================================================
# 4. Test schema만 로드
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


# M3 = 전체 공식 feature - pitcher_id
FINAL_FEATURES = [
    col
    for col in ALL_FEATURES
    if col != PITCHER_ID_COL
]


assert len(FINAL_FEATURES) == 46
assert PITCHER_ID_COL not in FINAL_FEATURES


# =============================================================================
# 5. Feature Type
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
# 6. Train 로드
#
# Calibration 분석에는 Known/Unknown 평가용 pitcher_id도 필요.
# =============================================================================

USE_COLS = list(
    dict.fromkeys(
        FINAL_FEATURES
        + [
            PITCHER_ID_COL,
            TARGET,
        ]
    )
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

print(f"Train shape      : {train.shape}")
print(f"Feature count    : {len(FINAL_FEATURES)}")
print(f"Load time        : {load_seconds:.1f} sec")

print(
    f"DataFrame memory : "
    f"{train.memory_usage(deep=True).sum() / 1024**3:.2f} GB"
)


# =============================================================================
# 7. Preprocessor
# =============================================================================

def build_preprocessor(
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


    classified = (
        scale_cols
        + no_scale_cols
        + categorical_cols
    )


    assert len(classified) == len(
        feature_list
    )


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
                numeric_scale_pipeline,
                scale_cols,
            )
        )


    if no_scale_cols:

        transformers.append(
            (
                "numeric_no_scale",
                numeric_no_scale_pipeline,
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
# 8. LR Pipeline
# =============================================================================

def build_lr_pipeline(
    feature_list,
):

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
# 9. Fold Generator
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
        }


# =============================================================================
# 10. Known / Unknown
# =============================================================================

def make_known_unknown_masks(
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


    valid_pitchers = (
        data.loc[
            valid_mask,
            PITCHER_ID_COL,
        ]
        .reset_index(
            drop=True
        )
    )


    known_mask = (
        valid_pitchers
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
# 11. Metric
# =============================================================================

def evaluate_probability(
    y_true,
    y_prob,
):

    y_true = np.asarray(
        y_true
    )

    y_prob = np.asarray(
        y_prob
    )


    result = {
        "rows":
            len(
                y_true
            ),

        "target_rate":
            float(
                y_true.mean()
            ),

        "prediction_mean":
            float(
                y_prob.mean()
            ),

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
    }


    if len(
        np.unique(
            y_true
        )
    ) == 2:

        result[
            "roc_auc"
        ] = roc_auc_score(
            y_true,
            y_prob,
        )


        result[
            "pr_auc"
        ] = average_precision_score(
            y_true,
            y_prob,
        )

    else:

        result[
            "roc_auc"
        ] = np.nan

        result[
            "pr_auc"
        ] = np.nan


    return result


# =============================================================================
# 12. Calibration Table
#
# Uniform probability bins
# =============================================================================

def build_calibration_table(
    y_true,
    y_prob,
    n_bins=10,
):

    y_true = np.asarray(
        y_true
    )

    y_prob = np.asarray(
        y_prob
    )


    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )


    # 0 ~ n_bins-1
    bin_id = np.digitize(
        y_prob,
        edges[
            1:-1
        ],
        right=False,
    )


    rows = []


    for index in range(
        n_bins
    ):

        mask = (
            bin_id
            == index
        )


        if not mask.any():

            continue


        mean_pred = float(
            y_prob[
                mask
            ]
            .mean()
        )


        actual_rate = float(
            y_true[
                mask
            ]
            .mean()
        )


        calibration_gap = (
            mean_pred
            - actual_rate
        )


        # probability가 실제 empirical rate보다
        # 0.5에서 더 멀리 떨어져 있으면 overconfidence heuristic
        confidence_gap = (
            abs(
                mean_pred
                - 0.5
            )
            -
            abs(
                actual_rate
                - 0.5
            )
        )


        rows.append(
            {
                "bin":
                    index + 1,

                "bin_left":
                    edges[
                        index
                    ],

                "bin_right":
                    edges[
                        index + 1
                    ],

                "count":
                    int(
                        mask.sum()
                    ),

                "mean_prediction":
                    mean_pred,

                "actual_rate":
                    actual_rate,

                "calibration_gap":
                    calibration_gap,

                "abs_calibration_gap":
                    abs(
                        calibration_gap
                    ),

                "confidence_gap":
                    confidence_gap,
            }
        )


    return pd.DataFrame(
        rows
    )


# =============================================================================
# 13. Calibration Summary
#
# ECE / MCE + over/under-confidence heuristic
# =============================================================================

def summarize_calibration(
    calibration_df,
):

    total = (
        calibration_df[
            "count"
        ]
        .sum()
    )


    weights = (
        calibration_df[
            "count"
        ]
        / total
    )


    ece = float(
        (
            weights
            * calibration_df[
                "abs_calibration_gap"
            ]
        )
        .sum()
    )


    mce = float(
        calibration_df[
            "abs_calibration_gap"
        ]
        .max()
    )


    overconfidence_index = float(
        (
            weights
            * calibration_df[
                "confidence_gap"
            ]
            .clip(
                lower=0
            )
        )
        .sum()
    )


    underconfidence_index = float(
        (
            weights
            * (
                -calibration_df[
                    "confidence_gap"
                ]
            )
            .clip(
                lower=0
            )
        )
        .sum()
    )


    return {
        "ece":
            ece,

        "mce":
            mce,

        "overconfidence_index":
            overconfidence_index,

        "underconfidence_index":
            underconfidence_index,
    }


# =============================================================================
# 14. 0.7 Probability Band
# =============================================================================

def evaluate_probability_band(
    y_true,
    y_prob,
    lower=0.65,
    upper=0.75,
):

    mask = (
        (y_prob >= lower)
        &
        (y_prob < upper)
    )


    if mask.sum() == 0:

        return {
            "rows":
                0,

            "mean_prediction":
                np.nan,

            "actual_rate":
                np.nan,

            "gap":
                np.nan,
        }


    mean_prediction = float(
        y_prob[
            mask
        ]
        .mean()
    )


    actual_rate = float(
        y_true[
            mask
        ]
        .mean()
    )


    return {
        "rows":
            int(
                mask.sum()
            ),

        "mean_prediction":
            mean_prediction,

        "actual_rate":
            actual_rate,

        "gap":
            mean_prediction
            - actual_rate,
    }


# =============================================================================
# 15. Fold OOF Prediction 생성
#
# Colab checkpoint:
# fold prediction이 이미 있으면 다시 학습하지 않음.
# =============================================================================

oof_parts = []
fold_metric_rows = []
fold_calibration_tables = []


for fold_data in expanding_window_splits(
    train
):

    fold = (
        fold_data[
            "fold"
        ]
    )


    checkpoint_path = os.path.join(
        OUTPUT_DIR,
        f"lr15_fold_{fold}_predictions.npz",
    )


    # -------------------------------------------------------------------------
    # Resume
    # -------------------------------------------------------------------------

    if os.path.exists(
        checkpoint_path
    ):

        checkpoint = np.load(
            checkpoint_path
        )


        y_valid = (
            checkpoint[
                "y_true"
            ]
            .astype(
                np.int8
            )
        )


        valid_prob = (
            checkpoint[
                "y_prob"
            ]
            .astype(
                np.float32
            )
        )


        known_mask = (
            checkpoint[
                "known_mask"
            ]
            .astype(
                bool
            )
        )


        valid_season = int(
            checkpoint[
                "valid_season"
            ]
        )


        n_iter = int(
            checkpoint[
                "n_iter"
            ]
        )


        fit_seconds = float(
            checkpoint[
                "fit_seconds"
            ]
        )


        print(
            f"\nFold {fold}: "
            f"checkpoint loaded"
        )


    # -------------------------------------------------------------------------
    # 신규 학습
    # -------------------------------------------------------------------------

    else:

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


        (
            known_mask,
            unknown_mask,
        ) = make_known_unknown_masks(
            data=train,
            train_mask=train_mask,
            valid_mask=valid_mask,
        )


        X_train = (
            train.loc[
                train_mask,
                FINAL_FEATURES,
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
                FINAL_FEATURES,
            ]
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


        pipeline = (
            build_lr_pipeline(
                FINAL_FEATURES
            )
        )


        print("\n" + "=" * 100)

        print(
            f"FOLD {fold} | "
            f"{fold_data['train_start']}"
            f"~"
            f"{fold_data['train_end']} "
            f"→ "
            f"{fold_data['valid_season']}"
        )

        print("=" * 100)


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


        if convergence_warnings:

            print(
                "⚠️ ConvergenceWarning"
            )

        else:

            print(
                f"✓ converged "
                f"{n_iter}/{LR_MAX_ITER}"
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


        assert np.isfinite(
            valid_prob
        ).all()


        valid_season = (
            fold_data[
                "valid_season"
            ]
        )


        np.savez_compressed(
            checkpoint_path,

            y_true=
                y_valid,

            y_prob=
                valid_prob,

            known_mask=
                known_mask,

            valid_season=
                np.int16(
                    valid_season
                ),

            n_iter=
                np.int16(
                    n_iter
                ),

            fit_seconds=
                np.float32(
                    fit_seconds
                ),
        )


        del X_train
        del y_train
        del X_valid
        del pipeline
        del model

        gc.collect()


    unknown_mask = (
        ~known_mask
    )


    # -------------------------------------------------------------------------
    # Fold metric
    # -------------------------------------------------------------------------

    subgroup_masks = {
        "Overall":
            np.ones(
                len(
                    y_valid
                ),
                dtype=bool,
            ),

        "Known":
            known_mask,

        "Unknown":
            unknown_mask,
    }


    for (
        subgroup,
        subgroup_mask,
    ) in subgroup_masks.items():

        metrics = (
            evaluate_probability(
                y_true=y_valid[
                    subgroup_mask
                ],
                y_prob=valid_prob[
                    subgroup_mask
                ],
            )
        )


        calibration_df = (
            build_calibration_table(
                y_true=y_valid[
                    subgroup_mask
                ],
                y_prob=valid_prob[
                    subgroup_mask
                ],
                n_bins=CALIBRATION_BINS,
            )
        )


        calibration_summary = (
            summarize_calibration(
                calibration_df
            )
        )


        fold_metric_rows.append(
            {
                "fold":
                    fold,

                "valid_season":
                    valid_season,

                "subgroup":
                    subgroup,

                **metrics,

                **calibration_summary,

                "prediction_bias":
                    (
                        metrics[
                            "prediction_mean"
                        ]
                        -
                        metrics[
                            "target_rate"
                        ]
                    ),

                "n_iter":
                    n_iter,

                "fit_seconds":
                    fit_seconds,
            }
        )


        calibration_df[
            "fold"
        ] = fold


        calibration_df[
            "valid_season"
        ] = valid_season


        calibration_df[
            "subgroup"
        ] = subgroup


        fold_calibration_tables.append(
            calibration_df
        )


    # -------------------------------------------------------------------------
    # OOF 저장
    # -------------------------------------------------------------------------

    oof_parts.append(
        pd.DataFrame(
            {
                "fold":
                    fold,

                "valid_season":
                    valid_season,

                "y_true":
                    y_valid,

                "y_prob":
                    valid_prob,

                "known":
                    known_mask,
            }
        )
    )


    del y_valid
    del valid_prob
    del known_mask
    del unknown_mask

    gc.collect()


# =============================================================================
# 16. 전체 Temporal OOF
# =============================================================================

oof_df = pd.concat(
    oof_parts,
    ignore_index=True,
)


fold_metrics_df = pd.DataFrame(
    fold_metric_rows
)


fold_calibration_df = pd.concat(
    fold_calibration_tables,
    ignore_index=True,
)


print("\n" + "=" * 140)
print("FOLD METRICS")
print("=" * 140)

print(
    fold_metrics_df
    .round(6)
    .to_string(
        index=False
    )
)


# =============================================================================
# 17. Pooled OOF Calibration
#
# 2022 / 2023 / 2024는 모두 해당 시점보다 과거 데이터로 학습된 OOF 예측.
# =============================================================================

pooled_metric_rows = []
pooled_calibration_tables = []
band_rows = []


POOLED_MASKS = {
    "Overall":
        np.ones(
            len(
                oof_df
            ),
            dtype=bool,
        ),

    "Known":
        oof_df[
            "known"
        ]
        .to_numpy(),

    "Unknown":
        ~oof_df[
            "known"
        ]
        .to_numpy(),
}


for (
    subgroup,
    subgroup_mask,
) in POOLED_MASKS.items():

    y_true = (
        oof_df.loc[
            subgroup_mask,
            "y_true",
        ]
        .to_numpy()
    )


    y_prob = (
        oof_df.loc[
            subgroup_mask,
            "y_prob",
        ]
        .to_numpy()
    )


    metrics = (
        evaluate_probability(
            y_true,
            y_prob,
        )
    )


    calibration_df = (
        build_calibration_table(
            y_true,
            y_prob,
            n_bins=CALIBRATION_BINS,
        )
    )


    calibration_summary = (
        summarize_calibration(
            calibration_df
        )
    )


    pooled_metric_rows.append(
        {
            "subgroup":
                subgroup,

            **metrics,

            **calibration_summary,

            "prediction_bias":
                (
                    metrics[
                        "prediction_mean"
                    ]
                    -
                    metrics[
                        "target_rate"
                    ]
                ),
        }
    )


    calibration_df[
        "subgroup"
    ] = subgroup


    pooled_calibration_tables.append(
        calibration_df
    )


    # -------------------------------------------------------------------------
    # 예측확률 약 0.7 구간
    # -------------------------------------------------------------------------

    band = evaluate_probability_band(
        y_true=y_true,
        y_prob=y_prob,
        lower=PROBABILITY_BAND_LOW,
        upper=PROBABILITY_BAND_HIGH,
    )


    band_rows.append(
        {
            "subgroup":
                subgroup,

            "lower":
                PROBABILITY_BAND_LOW,

            "upper":
                PROBABILITY_BAND_HIGH,

            **band,
        }
    )


pooled_metrics_df = pd.DataFrame(
    pooled_metric_rows
)


pooled_calibration_df = pd.concat(
    pooled_calibration_tables,
    ignore_index=True,
)


probability_band_df = pd.DataFrame(
    band_rows
)


print("\n" + "=" * 120)
print("POOLED TEMPORAL OOF METRICS")
print("=" * 120)

print(
    pooled_metrics_df
    .round(6)
    .to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("PROBABILITY 0.65 ~ 0.75")
print("=" * 100)

print(
    probability_band_df
    .round(6)
    .to_string(
        index=False
    )
)


# =============================================================================
# 18. Calibration Curve — Fold별
# =============================================================================

fig, ax = plt.subplots(
    figsize=(
        8,
        7,
    )
)


ax.plot(
    [
        0,
        1,
    ],
    [
        0,
        1,
    ],
    linestyle="--",
    label="Perfect Calibration",
)


for fold in sorted(
    oof_df[
        "fold"
    ]
    .unique()
):

    fold_cal = (
        fold_calibration_df[
            (
                fold_calibration_df[
                    "fold"
                ]
                == fold
            )
            &
            (
                fold_calibration_df[
                    "subgroup"
                ]
                == "Overall"
            )
        ]
    )


    ax.plot(
        fold_cal[
            "mean_prediction"
        ],
        fold_cal[
            "actual_rate"
        ],
        marker="o",
        label=f"Fold {fold}",
    )


ax.set_xlabel(
    "Mean Predicted Probability"
)

ax.set_ylabel(
    "Observed Success Rate"
)

ax.set_title(
    "Calibration Curve by Fold"
)

ax.legend()

plt.tight_layout()


fold_curve_path = os.path.join(
    OUTPUT_DIR,
    "lr15_calibration_curve_by_fold.png",
)


plt.savefig(
    fold_curve_path,
    dpi=150,
    bbox_inches="tight",
)

plt.show()


# =============================================================================
# 19. Calibration Curve — Overall / Known / Unknown
# =============================================================================

fig, ax = plt.subplots(
    figsize=(
        8,
        7,
    )
)


ax.plot(
    [
        0,
        1,
    ],
    [
        0,
        1,
    ],
    linestyle="--",
    label="Perfect Calibration",
)


for subgroup in [
    "Overall",
    "Known",
    "Unknown",
]:

    subgroup_cal = (
        pooled_calibration_df[
            pooled_calibration_df[
                "subgroup"
            ]
            == subgroup
        ]
    )


    ax.plot(
        subgroup_cal[
            "mean_prediction"
        ],
        subgroup_cal[
            "actual_rate"
        ],
        marker="o",
        label=subgroup,
    )


ax.set_xlabel(
    "Mean Predicted Probability"
)

ax.set_ylabel(
    "Observed Success Rate"
)

ax.set_title(
    "Pooled Temporal OOF Calibration"
)

ax.legend()

plt.tight_layout()


subgroup_curve_path = os.path.join(
    OUTPUT_DIR,
    "lr15_calibration_curve_subgroups.png",
)


plt.savefig(
    subgroup_curve_path,
    dpi=150,
    bbox_inches="tight",
)

plt.show()


# =============================================================================
# 20. Probability Histogram
# =============================================================================

fig, ax = plt.subplots(
    figsize=(
        9,
        5,
    )
)


ax.hist(
    oof_df[
        "y_prob"
    ],
    bins=HISTOGRAM_BINS,
)


ax.set_xlabel(
    "Predicted Probability"
)

ax.set_ylabel(
    "Number of Pitches"
)

ax.set_title(
    "Predicted Probability Histogram — Temporal OOF"
)

plt.tight_layout()


histogram_path = os.path.join(
    OUTPUT_DIR,
    "lr15_probability_histogram.png",
)


plt.savefig(
    histogram_path,
    dpi=150,
    bbox_inches="tight",
)

plt.show()


# =============================================================================
# 21. Known / Unknown Histogram
# =============================================================================

fig, ax = plt.subplots(
    figsize=(
        9,
        5,
    )
)


ax.hist(
    [
        oof_df.loc[
            oof_df[
                "known"
            ],
            "y_prob",
        ],

        oof_df.loc[
            ~oof_df[
                "known"
            ],
            "y_prob",
        ],
    ],
    bins=HISTOGRAM_BINS,
    label=[
        "Known",
        "Unknown",
    ],
    histtype="step",
)


ax.set_xlabel(
    "Predicted Probability"
)

ax.set_ylabel(
    "Number of Pitches"
)

ax.set_title(
    "Predicted Probability — Known vs Unknown"
)

ax.legend()

plt.tight_layout()


subgroup_histogram_path = os.path.join(
    OUTPUT_DIR,
    "lr15_probability_histogram_subgroups.png",
)


plt.savefig(
    subgroup_histogram_path,
    dpi=150,
    bbox_inches="tight",
)

plt.show()


# =============================================================================
# 22. Post-Calibration
#
# 매우 중요한 leakage 방지:
#
# 2023 보정:
#   calibrator train = 2022 OOF
#   evaluation       = 2023
#
# 2024 보정:
#   calibrator train = 2022 + 2023 OOF
#   evaluation       = 2024
#
# 동일 validation fold의 target으로 calibrator를 fit하지 않는다.
# =============================================================================

def safe_logit(
    probability,
):

    clipped = np.clip(
        probability,
        1e-6,
        1.0 - 1e-6,
    )


    return np.log(
        clipped
        / (
            1.0
            - clipped
        )
    )


def fit_platt(
    y_prob,
    y_true,
):

    x = (
        safe_logit(
            y_prob
        )
        .reshape(
            -1,
            1,
        )
    )


    calibrator = LogisticRegression(
        penalty="l2",
        C=1e6,
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )


    calibrator.fit(
        x,
        y_true,
    )


    return calibrator


def predict_platt(
    calibrator,
    y_prob,
):

    x = (
        safe_logit(
            y_prob
        )
        .reshape(
            -1,
            1,
        )
    )


    return (
        calibrator
        .predict_proba(
            x
        )[:, 1]
    )


def fit_isotonic(
    y_prob,
    y_true,
):

    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )


    calibrator.fit(
        y_prob,
        y_true,
    )


    return calibrator


post_calibration_rows = []
post_prediction_parts = []


# Fold 1은 이전 OOF calibration data가 없으므로 raw only.
for target_fold in [
    2,
    3,
]:

    calibration_train = (
        oof_df[
            oof_df[
                "fold"
            ]
            < target_fold
        ]
    )


    evaluation = (
        oof_df[
            oof_df[
                "fold"
            ]
            == target_fold
        ]
        .copy()
    )


    assert (
        calibration_train[
            "valid_season"
        ]
        .max()
        <
        evaluation[
            "valid_season"
        ]
        .min()
    )


    platt = fit_platt(
        y_prob=
            calibration_train[
                "y_prob"
            ]
            .to_numpy(),

        y_true=
            calibration_train[
                "y_true"
            ]
            .to_numpy(),
    )


    isotonic = fit_isotonic(
        y_prob=
            calibration_train[
                "y_prob"
            ]
            .to_numpy(),

        y_true=
            calibration_train[
                "y_true"
            ]
            .to_numpy(),
    )


    predictions = {
        "Raw":
            evaluation[
                "y_prob"
            ]
            .to_numpy(),

        "Platt":
            predict_platt(
                platt,
                evaluation[
                    "y_prob"
                ]
                .to_numpy(),
            ),

        "Isotonic":
            isotonic.predict(
                evaluation[
                    "y_prob"
                ]
                .to_numpy()
            ),
    }


    known_mask = (
        evaluation[
            "known"
        ]
        .to_numpy()
    )


    subgroup_masks = {
        "Overall":
            np.ones(
                len(
                    evaluation
                ),
                dtype=bool,
            ),

        "Known":
            known_mask,

        "Unknown":
            ~known_mask,
    }


    for (
        method,
        calibrated_prob,
    ) in predictions.items():

        for (
            subgroup,
            subgroup_mask,
        ) in subgroup_masks.items():

            metrics = evaluate_probability(
                y_true=
                    evaluation.loc[
                        subgroup_mask,
                        "y_true",
                    ]
                    .to_numpy(),

                y_prob=
                    calibrated_prob[
                        subgroup_mask
                    ],
            )


            calibration_table = (
                build_calibration_table(
                    y_true=
                        evaluation.loc[
                            subgroup_mask,
                            "y_true",
                        ]
                        .to_numpy(),

                    y_prob=
                        calibrated_prob[
                            subgroup_mask
                        ],

                    n_bins=
                        CALIBRATION_BINS,
                )
            )


            calibration_summary = (
                summarize_calibration(
                    calibration_table
                )
            )


            post_calibration_rows.append(
                {
                    "fold":
                        target_fold,

                    "valid_season":
                        int(
                            evaluation[
                                "valid_season"
                            ]
                            .iloc[0]
                        ),

                    "calibration_train_seasons":
                        (
                            f"{calibration_train['valid_season'].min()}"
                            f"~"
                            f"{calibration_train['valid_season'].max()}"
                        ),

                    "method":
                        method,

                    "subgroup":
                        subgroup,

                    **metrics,

                    **calibration_summary,
                }
            )


        post_prediction_parts.append(
            pd.DataFrame(
                {
                    "fold":
                        target_fold,

                    "valid_season":
                        evaluation[
                            "valid_season"
                        ]
                        .to_numpy(),

                    "y_true":
                        evaluation[
                            "y_true"
                        ]
                        .to_numpy(),

                    "method":
                        method,

                    "y_prob":
                        calibrated_prob,
                }
            )
        )


    del platt
    del isotonic

    gc.collect()


post_calibration_df = pd.DataFrame(
    post_calibration_rows
)


post_predictions_df = pd.concat(
    post_prediction_parts,
    ignore_index=True,
)


# =============================================================================
# 23. Post-Calibration Summary
#
# 2023 / 2024만 비교
# =============================================================================

post_summary_df = (
    post_calibration_df
    .groupby(
        [
            "method",
            "subgroup",
        ],
        as_index=False,
    )
    .agg(
        mean_brier=(
            "brier",
            "mean",
        ),

        mean_logloss=(
            "logloss",
            "mean",
        ),

        mean_ece=(
            "ece",
            "mean",
        ),

        mean_mce=(
            "mce",
            "mean",
        ),

        mean_roc_auc=(
            "roc_auc",
            "mean",
        ),

        mean_pr_auc=(
            "pr_auc",
            "mean",
        ),
    )
)


print("\n" + "=" * 130)
print("POST-CALIBRATION SUMMARY — 2023/2024")
print("=" * 130)

print(
    post_summary_df
    .round(6)
    .to_string(
        index=False
    )
)


# =============================================================================
# 24. Post-Calibration 적용 여부 판단
# =============================================================================

overall_post = (
    post_summary_df[
        post_summary_df[
            "subgroup"
        ]
        == "Overall"
    ]
    .sort_values(
        "mean_brier"
    )
    .reset_index(
        drop=True
    )
)


best_method = (
    overall_post
    .iloc[0][
        "method"
    ]
)


raw_brier = float(
    overall_post.loc[
        overall_post[
            "method"
        ]
        == "Raw",
        "mean_brier",
    ]
    .iloc[0]
)


best_brier = float(
    overall_post
    .iloc[0][
        "mean_brier"
    ]
)


post_calibration_improvement = (
    raw_brier
    - best_brier
)


apply_post_calibration = (
    best_method
    != "Raw"
    and post_calibration_improvement
    > 0
)


post_decision_df = pd.DataFrame(
    [
        {
            "best_method":
                best_method,

            "raw_mean_brier":
                raw_brier,

            "best_mean_brier":
                best_brier,

            "brier_improvement":
                post_calibration_improvement,

            "apply_post_calibration":
                apply_post_calibration,
        }
    ]
)


print("\n" + "=" * 80)
print("POST-CALIBRATION DECISION")
print("=" * 80)

print(
    post_decision_df
    .round(6)
    .to_string(
        index=False
    )
)


# =============================================================================
# 25. Raw vs Post-Calibration Curve
#
# 2023/2024 pooled comparison
# =============================================================================

fig, ax = plt.subplots(
    figsize=(
        8,
        7,
    )
)


ax.plot(
    [
        0,
        1,
    ],
    [
        0,
        1,
    ],
    linestyle="--",
    label="Perfect Calibration",
)


for method in [
    "Raw",
    "Platt",
    "Isotonic",
]:

    method_df = (
        post_predictions_df[
            post_predictions_df[
                "method"
            ]
            == method
        ]
    )


    method_calibration = (
        build_calibration_table(
            y_true=
                method_df[
                    "y_true"
                ]
                .to_numpy(),

            y_prob=
                method_df[
                    "y_prob"
                ]
                .to_numpy(),

            n_bins=
                CALIBRATION_BINS,
        )
    )


    ax.plot(
        method_calibration[
            "mean_prediction"
        ],
        method_calibration[
            "actual_rate"
        ],
        marker="o",
        label=method,
    )


ax.set_xlabel(
    "Mean Predicted Probability"
)

ax.set_ylabel(
    "Observed Success Rate"
)

ax.set_title(
    "Raw vs Temporal Post-Calibration"
)

ax.legend()

plt.tight_layout()


post_curve_path = os.path.join(
    OUTPUT_DIR,
    "lr15_post_calibration_curve.png",
)


plt.savefig(
    post_curve_path,
    dpi=150,
    bbox_inches="tight",
)

plt.show()


# =============================================================================
# 26. 저장
# =============================================================================

fold_metrics_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lr15_fold_metrics.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


fold_calibration_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lr15_fold_calibration_table.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


pooled_metrics_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lr15_pooled_oof_metrics.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


pooled_calibration_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lr15_pooled_calibration_table.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


probability_band_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lr15_probability_065_075.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


post_calibration_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lr15_post_calibration_fold_results.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


post_summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lr15_post_calibration_summary.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


post_decision_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lr15_post_calibration_decision.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


# 대용량 OOF는 CSV 대신 압축 NPZ
np.savez_compressed(
    os.path.join(
        OUTPUT_DIR,
        "lr15_oof_predictions.npz",
    ),

    fold=
        oof_df[
            "fold"
        ]
        .to_numpy(
            dtype=np.int8
        ),

    valid_season=
        oof_df[
            "valid_season"
        ]
        .to_numpy(
            dtype=np.int16
        ),

    y_true=
        oof_df[
            "y_true"
        ]
        .to_numpy(
            dtype=np.int8
        ),

    y_prob=
        oof_df[
            "y_prob"
        ]
        .to_numpy(
            dtype=np.float32
        ),

    known=
        oof_df[
            "known"
        ]
        .to_numpy(
            dtype=bool
        ),
)


print("\n" + "=" * 80)
print("LR-15 CALIBRATION ANALYSIS COMPLETE")
print("=" * 80)

print(
    f"Output directory : "
    f"{OUTPUT_DIR}"
)

print(
    f"Best post-calibration method : "
    f"{best_method}"
)

print(
    f"Apply post-calibration       : "
    f"{apply_post_calibration}"
)