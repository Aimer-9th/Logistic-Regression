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
OUTPUT_DIR = "./lr12_known_unknown_outputs"

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

# LR-10 / LR-11과 동일
LR_PENALTY = "l2"
LR_C = 1.0
LR_SOLVER = "saga"
LR_MAX_ITER = 500
LR_TOL = 1e-3

PRINT_FEATURE_LIST = False


# =============================================================================
# 1. Expanding Window
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
# 2. 환경
# =============================================================================

print("=" * 80)
print("ENVIRONMENT")
print("=" * 80)

print(f"scikit-learn : {sklearn.__version__}")
print(f"NumPy        : {np.__version__}")
print(f"Pandas       : {pd.__version__}")


# =============================================================================
# 3. Test schema만 로드
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
# 4. Feature Type
#
# LR-05 / LR-08 / LR-10 / LR-11과 동일
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
# 5. Train 로드
#
# Colab RAM 절약:
# - row_id 제외
# - 필요한 47 features + target만 로드
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
# 6. A/B/C/D Feature Set
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
        == EXPECTED_FEATURE_COUNTS[
            name
        ]
    )


# =============================================================================
# 7. Feature Type 반환
# =============================================================================

def get_feature_types(
    feature_list,
):
    """
    Experiment별 preprocessing column 구성.
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
# 8. Preprocessor
# =============================================================================

def build_preprocessor(
    feature_list,
):
    """
    Fold train에서만 fit되는 preprocessing.
    """

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


    if feature_types["scale"]:

        transformers.append(
            (
                "numeric_scale",
                scale_numeric_pipeline,
                feature_types["scale"],
            )
        )


    if feature_types["no_scale"]:

        transformers.append(
            (
                "numeric_no_scale",
                no_scale_numeric_pipeline,
                feature_types["no_scale"],
            )
        )


    if feature_types["categorical"]:

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
# 9. LR Pipeline
# =============================================================================

def build_lr_pipeline(
    feature_list,
):
    """
    Fold/model마다 새로운 Pipeline 생성.
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
# 10. Expanding Window Generator
# =============================================================================

def expanding_window_splits(
    data,
    fold_configs,
):

    for config in fold_configs:

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


        assert max(
            train_seasons
        ) < min(
            valid_seasons
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
# 11. Known / Unknown Mask
# =============================================================================

def make_pitcher_masks(
    data,
    train_mask,
    valid_mask,
):
    """
    Validation pitcher를 fold train 등장 여부로 분리.
    """

    train_pitcher_ids = set(
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
            train_pitcher_ids
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


    known_pitchers = set(
        valid_pitcher[
            known_mask
        ]
        .dropna()
        .unique()
    )


    unknown_pitchers = set(
        valid_pitcher[
            unknown_mask
        ]
        .dropna()
        .unique()
    )


    assert not (
        known_pitchers
        & unknown_pitchers
    )


    return {
        "train_pitcher_ids":
            train_pitcher_ids,

        "known_mask":
            known_mask,

        "unknown_mask":
            unknown_mask,

        "known_pitchers":
            known_pitchers,

        "unknown_pitchers":
            unknown_pitchers,
    }


# =============================================================================
# 12. Subgroup Metric
#
# Overall / Known / Unknown 모두 동일 함수 사용.
#
# subset에 class가 하나뿐이면
# ROC-AUC / PR-AUC는 NaN으로 기록.
# =============================================================================

def evaluate_probability(
    y_true,
    y_prob,
):
    """
    Brier / LogLoss / ROC-AUC / PR-AUC
    """

    y_true = np.asarray(
        y_true
    )

    y_prob = np.asarray(
        y_prob
    )


    if len(
        y_true
    ) == 0:

        return {
            "rows":
                0,

            "positive_rate":
                np.nan,

            "brier":
                np.nan,

            "logloss":
                np.nan,

            "auc":
                np.nan,

            "pr_auc":
                np.nan,
        }


    result = {
        "rows":
            len(
                y_true
            ),

        "positive_rate":
            float(
                y_true.mean()
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
            "auc"
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
            "auc"
        ] = np.nan

        result[
            "pr_auc"
        ] = np.nan


    return result


# =============================================================================
# 13. Historical NaN / Cold-start Summary
#
# Raw official asof_* 상태를
# Overall / Known / Unknown으로 분리.
#
# Rate NaN을 임의 0으로 해석하지 않는다.
# =============================================================================

def summarize_historical_cold_start(
    valid_df,
    known_mask,
    unknown_mask,
    fold,
):
    """
    Official historical feature availability 확인.
    """

    subgroup_masks = {
        "Overall":
            np.ones(
                len(valid_df),
                dtype=bool,
            ),

        "Known":
            known_mask,

        "Unknown":
            unknown_mask,
    }


    rows = []


    for (
        subgroup,
        subgroup_mask,
    ) in subgroup_masks.items():

        subgroup_df = (
            valid_df.loc[
                subgroup_mask,
                HISTORICAL_FEATURES,
            ]
        )


        if len(
            subgroup_df
        ) == 0:

            continue


        nan_count_per_row = (
            subgroup_df
            .isna()
            .sum(
                axis=1
            )
        )


        row_has_nan = (
            nan_count_per_row
            > 0
        )


        rows.append(
            {
                "fold":
                    fold,

                "subgroup":
                    subgroup,

                "rows":
                    len(
                        subgroup_df
                    ),

                "rows_with_hist_nan":
                    int(
                        row_has_nan.sum()
                    ),

                "hist_nan_row_rate":
                    float(
                        row_has_nan.mean()
                    ),

                "mean_hist_nan_count":
                    float(
                        nan_count_per_row.mean()
                    ),

                "median_hist_nan_count":
                    float(
                        nan_count_per_row.median()
                    ),

                "pitcher_n_zero_rate":
                    float(
                        valid_df.loc[
                            subgroup_mask,
                            "asof_pitcher_n",
                        ]
                        .eq(0)
                        .mean()
                    ),

                "pitcher_n_missing_rate":
                    float(
                        valid_df.loc[
                            subgroup_mask,
                            "asof_pitcher_n",
                        ]
                        .isna()
                        .mean()
                    ),

                "batter_n_zero_rate":
                    float(
                        valid_df.loc[
                            subgroup_mask,
                            "asof_batter_n",
                        ]
                        .eq(0)
                        .mean()
                    ),
            }
        )


    return rows


# =============================================================================
# 14. Pitcher ID Encoder 검증
#
# Model B/D에서 validation Unknown pitcher가
# train-fitted OneHotEncoder categories에 없는지 확인.
#
# handle_unknown="ignore"이므로 transform/predict는 정상이어야 함.
# =============================================================================

def validate_pitcher_encoder(
    pipeline,
    feature_list,
    unknown_pitcher_ids,
):
    """
    Returns:
        status
        n_unknown_pitchers_outside_encoder
    """

    if PITCHER_ID_COL not in feature_list:

        return (
            "NOT_APPLICABLE",
            0,
        )


    feature_types = (
        get_feature_types(
            feature_list
        )
    )


    categorical_cols = (
        feature_types[
            "categorical"
        ]
    )


    pitcher_index = (
        categorical_cols.index(
            PITCHER_ID_COL
        )
    )


    encoder = (
        pipeline
        .named_steps[
            "preprocessor"
        ]
        .named_transformers_[
            "categorical"
        ]
        .named_steps[
            "onehot"
        ]
    )


    pitcher_categories = set(
        encoder.categories_[
            pitcher_index
        ]
        .tolist()
    )


    unseen_pitchers = {
        pitcher_id
        for pitcher_id
        in unknown_pitcher_ids
        if pitcher_id
        not in pitcher_categories
    }


    assert unseen_pitchers == set(
        unknown_pitcher_ids
    ), (
        "Unknown Pitcher 중 일부가 encoder training categories에 "
        "포함되어 있습니다."
    )


    return (
        "PASS",
        len(
            unseen_pitchers
        ),
    )


# =============================================================================
# 15. CV Runner
#
# 3 folds × 4 models
#
# 각 prediction을 Overall / Known / Unknown으로 분리 평가.
# =============================================================================

def run_known_unknown_cv(
    data,
):

    result_rows = []
    distribution_rows = []
    cold_start_rows = []
    encoder_rows = []


    total_runs = (
        len(
            FOLD_CONFIGS
        )
        * len(
            LR_EXPERIMENTS
        )
    )


    run_index = 0


    for fold_data in expanding_window_splits(
        data,
        FOLD_CONFIGS,
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


        # ---------------------------------------------------------------------
        # Known / Unknown
        # ---------------------------------------------------------------------

        pitcher_split = (
            make_pitcher_masks(
                data=data,
                train_mask=train_mask,
                valid_mask=valid_mask,
            )
        )


        known_mask = (
            pitcher_split[
                "known_mask"
            ]
        )

        unknown_mask = (
            pitcher_split[
                "unknown_mask"
            ]
        )


        known_rows = int(
            known_mask.sum()
        )

        unknown_rows = int(
            unknown_mask.sum()
        )


        known_pitcher_count = len(
            pitcher_split[
                "known_pitchers"
            ]
        )

        unknown_pitcher_count = len(
            pitcher_split[
                "unknown_pitchers"
            ]
        )


        unknown_row_rate = (
            unknown_rows
            / fold_data[
                "valid_rows"
            ]
        )


        valid_pitcher_count = (
            known_pitcher_count
            + unknown_pitcher_count
        )


        unknown_pitcher_rate = (
            unknown_pitcher_count
            / valid_pitcher_count
            if valid_pitcher_count > 0
            else np.nan
        )


        distribution_rows.append(
            {
                "fold":
                    fold,

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

                "train_pitchers":
                    len(
                        pitcher_split[
                            "train_pitcher_ids"
                        ]
                    ),

                "valid_pitchers":
                    valid_pitcher_count,

                "known_pitchers":
                    known_pitcher_count,

                "unknown_pitchers":
                    unknown_pitcher_count,

                "valid_rows":
                    fold_data[
                        "valid_rows"
                    ],

                "known_rows":
                    known_rows,

                "unknown_rows":
                    unknown_rows,

                "known_row_rate":
                    known_rows
                    / fold_data[
                        "valid_rows"
                    ],

                "unknown_row_rate":
                    unknown_row_rate,

                "unknown_pitcher_rate":
                    unknown_pitcher_rate,
            }
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
            f"Valid rows       : "
            f"{fold_data['valid_rows']:,}"
        )

        print(
            f"Known rows       : "
            f"{known_rows:,} "
            f"({known_rows / fold_data['valid_rows']:.2%})"
        )

        print(
            f"Unknown rows     : "
            f"{unknown_rows:,} "
            f"({unknown_row_rate:.2%})"
        )

        print(
            f"Known pitchers   : "
            f"{known_pitcher_count:,}"
        )

        print(
            f"Unknown pitchers : "
            f"{unknown_pitcher_count:,}"
        )


        # ---------------------------------------------------------------------
        # Raw historical cold-start statistics
        #
        # fold당 한 번만 계산
        # ---------------------------------------------------------------------

        valid_history_df = (
            data.loc[
                valid_mask,
                HISTORICAL_FEATURES
                + [
                    "asof_pitcher_n",
                    "asof_batter_n",
                ],
            ]
            .reset_index(
                drop=True
            )
        )


        # 중복 column 방지
        valid_history_df = (
            valid_history_df.loc[
                :,
                ~valid_history_df
                .columns
                .duplicated()
            ]
        )


        cold_start_rows.extend(
            summarize_historical_cold_start(
                valid_df=valid_history_df,
                known_mask=known_mask,
                unknown_mask=unknown_mask,
                fold=fold,
            )
        )


        # ---------------------------------------------------------------------
        # Target
        # ---------------------------------------------------------------------

        y_train = (
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


        # ---------------------------------------------------------------------
        # Model A/B/C/D
        # ---------------------------------------------------------------------

        for (
            experiment_name,
            config,
        ) in LR_EXPERIMENTS.items():

            run_index += 1


            feature_list = (
                config[
                    "features"
                ]
            )


            run_start = (
                time.perf_counter()
            )


            print("\n" + "#" * 110)

            print(
                f"[{run_index}/{total_runs}] "
                f"Fold {fold} | "
                f"{experiment_name}"
            )

            print("#" * 110)


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


            pipeline = (
                build_lr_pipeline(
                    feature_list
                )
            )


            # -----------------------------------------------------------------
            # Fit
            # -----------------------------------------------------------------

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
                for warning
                in caught_warnings
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


            converged = (
                len(
                    convergence_warnings
                )
                == 0
                and n_iter
                < LR_MAX_ITER
            )


            # -----------------------------------------------------------------
            # Unknown pitcher encoder
            # -----------------------------------------------------------------

            (
                encoder_status,
                encoder_unseen_count,
            ) = validate_pitcher_encoder(
                pipeline=pipeline,
                feature_list=feature_list,
                unknown_pitcher_ids=(
                    pitcher_split[
                        "unknown_pitchers"
                    ]
                ),
            )


            encoder_rows.append(
                {
                    "fold":
                        fold,

                    "model":
                        experiment_name,

                    "uses_pitcher_id":
                        config[
                            "use_pitcher_id"
                        ],

                    "unknown_pitchers":
                        unknown_pitcher_count,

                    "encoder_status":
                        encoder_status,

                    "unseen_pitchers_confirmed":
                        encoder_unseen_count,
                }
            )


            # -----------------------------------------------------------------
            # Prediction
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


            assert np.isfinite(
                valid_prob
            ).all()


            assert (
                valid_prob >= 0
            ).all()


            assert (
                valid_prob <= 1
            ).all()


            # Unknown prediction도 finite
            if unknown_rows > 0:

                assert np.isfinite(
                    valid_prob[
                        unknown_mask
                    ]
                ).all()


            # -----------------------------------------------------------------
            # Subgroup metric
            # -----------------------------------------------------------------

            overall_metrics = (
                evaluate_probability(
                    y_true=y_valid,
                    y_prob=valid_prob,
                )
            )


            known_metrics = (
                evaluate_probability(
                    y_true=y_valid[
                        known_mask
                    ],
                    y_prob=valid_prob[
                        known_mask
                    ],
                )
            )


            unknown_metrics = (
                evaluate_probability(
                    y_true=y_valid[
                        unknown_mask
                    ],
                    y_prob=valid_prob[
                        unknown_mask
                    ],
                )
            )


            total_seconds = (
                time.perf_counter()
                - run_start
            )


            result_rows.append(
                {
                    "fold":
                        fold,

                    "model":
                        experiment_name,

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

                    "pitcher_id":
                        config[
                            "use_pitcher_id"
                        ],

                    "historical":
                        config[
                            "use_historical"
                        ],

                    "raw_feature_count":
                        len(
                            feature_list
                        ),

                    # -----------------------------------------
                    # Distribution
                    # -----------------------------------------

                    "overall_rows":
                        fold_data[
                            "valid_rows"
                        ],

                    "known_rows":
                        known_rows,

                    "unknown_rows":
                        unknown_rows,

                    "known_pitchers":
                        known_pitcher_count,

                    "unknown_pitchers":
                        unknown_pitcher_count,

                    "unknown_row_rate":
                        unknown_row_rate,

                    # -----------------------------------------
                    # Overall
                    # -----------------------------------------

                    "overall_brier":
                        overall_metrics[
                            "brier"
                        ],

                    "overall_logloss":
                        overall_metrics[
                            "logloss"
                        ],

                    "overall_auc":
                        overall_metrics[
                            "auc"
                        ],

                    "overall_pr_auc":
                        overall_metrics[
                            "pr_auc"
                        ],

                    # -----------------------------------------
                    # Known
                    # -----------------------------------------

                    "known_brier":
                        known_metrics[
                            "brier"
                        ],

                    "known_logloss":
                        known_metrics[
                            "logloss"
                        ],

                    "known_auc":
                        known_metrics[
                            "auc"
                        ],

                    "known_pr_auc":
                        known_metrics[
                            "pr_auc"
                        ],

                    "known_target_rate":
                        known_metrics[
                            "positive_rate"
                        ],

                    # -----------------------------------------
                    # Unknown
                    # -----------------------------------------

                    "unknown_brier":
                        unknown_metrics[
                            "brier"
                        ],

                    "unknown_logloss":
                        unknown_metrics[
                            "logloss"
                        ],

                    "unknown_auc":
                        unknown_metrics[
                            "auc"
                        ],

                    "unknown_pr_auc":
                        unknown_metrics[
                            "pr_auc"
                        ],

                    "unknown_target_rate":
                        unknown_metrics[
                            "positive_rate"
                        ],

                    # -----------------------------------------
                    # Pipeline / status
                    # -----------------------------------------

                    "unknown_encoder_status":
                        encoder_status,

                    "n_iter":
                        n_iter,

                    "converged":
                        converged,

                    "convergence_warning":
                        len(
                            convergence_warnings
                        )
                        > 0,

                    "fit_seconds":
                        fit_seconds,

                    "predict_seconds":
                        predict_seconds,

                    "total_seconds":
                        total_seconds,
                }
            )


            print(
                f"Overall Brier : "
                f"{overall_metrics['brier']:.6f}"
            )

            print(
                f"Known Brier   : "
                f"{known_metrics['brier']:.6f}"
            )

            print(
                f"Unknown Brier : "
                f"{unknown_metrics['brier']:.6f}"
            )

            print(
                f"Unknown OHE   : "
                f"{encoder_status}"
            )

            print(
                f"Fit time      : "
                f"{fit_seconds:.1f} sec"
            )


            # -----------------------------------------------------------------
            # 중간 checkpoint
            # -----------------------------------------------------------------

            pd.DataFrame(
                result_rows
            ).to_csv(
                os.path.join(
                    OUTPUT_DIR,
                    "lr12_subgroup_results_progress.csv",
                ),
                encoding="utf-8-sig",
                index=False,
            )


            # -----------------------------------------------------------------
            # Colab 메모리 회수
            # -----------------------------------------------------------------

            del X_train
            del X_valid
            del pipeline
            del model
            del valid_prob

            gc.collect()


        del y_train
        del y_valid
        del valid_history_df

        gc.collect()


    return (
        pd.DataFrame(
            result_rows
        ),
        pd.DataFrame(
            distribution_rows
        ),
        pd.DataFrame(
            cold_start_rows
        ),
        pd.DataFrame(
            encoder_rows
        ),
    )


# =============================================================================
# 16. 실행
# =============================================================================

(
    subgroup_results_df,
    unknown_distribution_df,
    cold_start_df,
    encoder_validation_df,
) = run_known_unknown_cv(
    train
)


# =============================================================================
# 17. 결과 구조 검증
# =============================================================================

assert len(
    subgroup_results_df
) == 12


assert (
    subgroup_results_df
    .groupby(
        "model"
    )[
        "fold"
    ]
    .nunique()
    .eq(3)
    .all()
)


assert (
    subgroup_results_df[
        "known_rows"
    ]
    +
    subgroup_results_df[
        "unknown_rows"
    ]
    ==
    subgroup_results_df[
        "overall_rows"
    ]
).all()


# PID model은 Unknown encoder PASS
pid_encoder_results = (
    encoder_validation_df[
        encoder_validation_df[
            "uses_pitcher_id"
        ]
    ]
)


assert (
    pid_encoder_results[
        "encoder_status"
    ]
    == "PASS"
).all()


# 모든 model probability가 정상 생성됐기 때문에
# Unknown prediction도 정상 처리된 것으로 확인
assert (
    subgroup_results_df[
        "converged"
    ]
).all()


# =============================================================================
# 18. Fold별 Known / Unknown Distribution
# =============================================================================

print("\n" + "=" * 120)
print("KNOWN / UNKNOWN DISTRIBUTION")
print("=" * 120)

print(
    unknown_distribution_df
    .round(6)
    .to_string(
        index=False,
    )
)


# =============================================================================
# 19. Fold별 Subgroup Result
# =============================================================================

DISPLAY_COLS = [
    "fold",
    "model",

    "overall_rows",
    "known_rows",
    "unknown_rows",
    "unknown_row_rate",

    "overall_brier",
    "known_brier",
    "unknown_brier",

    "overall_logloss",
    "known_logloss",
    "unknown_logloss",

    "overall_auc",
    "known_auc",
    "unknown_auc",

    "overall_pr_auc",
    "known_pr_auc",
    "unknown_pr_auc",
]


print("\n" + "=" * 180)
print("FOLD SUBGROUP RESULTS")
print("=" * 180)

print(
    subgroup_results_df[
        DISPLAY_COLS
    ]
    .round(6)
    .to_string(
        index=False,
    )
)


# =============================================================================
# 20. CV Subgroup Summary
#
# 모델별 3개 fold Mean / Std
# =============================================================================

summary_rows = []


for (
    model_name,
    model_df,
) in subgroup_results_df.groupby(
    "model"
):

    summary_rows.append(
        {
            "model":
                model_name,

            "pitcher_id":
                bool(
                    model_df[
                        "pitcher_id"
                    ]
                    .iloc[0]
                ),

            "historical":
                bool(
                    model_df[
                        "historical"
                    ]
                    .iloc[0]
                ),

            # Overall
            "mean_overall_brier":
                model_df[
                    "overall_brier"
                ]
                .mean(),

            "std_overall_brier":
                model_df[
                    "overall_brier"
                ]
                .std(
                    ddof=0
                ),

            # Known
            "mean_known_brier":
                model_df[
                    "known_brier"
                ]
                .mean(),

            "std_known_brier":
                model_df[
                    "known_brier"
                ]
                .std(
                    ddof=0
                ),

            # Unknown
            "mean_unknown_brier":
                model_df[
                    "unknown_brier"
                ]
                .mean(),

            "std_unknown_brier":
                model_df[
                    "unknown_brier"
                ]
                .std(
                    ddof=0
                ),

            # LogLoss
            "mean_known_logloss":
                model_df[
                    "known_logloss"
                ]
                .mean(),

            "mean_unknown_logloss":
                model_df[
                    "unknown_logloss"
                ]
                .mean(),

            # ROC-AUC
            "mean_known_auc":
                model_df[
                    "known_auc"
                ]
                .mean(),

            "mean_unknown_auc":
                model_df[
                    "unknown_auc"
                ]
                .mean(),

            # PR-AUC
            "mean_known_pr_auc":
                model_df[
                    "known_pr_auc"
                ]
                .mean(),

            "mean_unknown_pr_auc":
                model_df[
                    "unknown_pr_auc"
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


cv_subgroup_summary_df = (
    pd.DataFrame(
        summary_rows
    )
    .sort_values(
        "mean_overall_brier"
    )
    .reset_index(
        drop=True
    )
)


print("\n" + "=" * 150)
print("CV SUBGROUP SUMMARY")
print("=" * 150)

print(
    cv_subgroup_summary_df
    .round(6)
    .to_string(
        index=False,
    )
)


# =============================================================================
# 21. Pitcher ID 효과
#
# B - A / D - C를 Known과 Unknown에서 별도로 비교.
#
# improvement > 0
# → pitcher_id 추가 모델의 Brier가 더 낮음
# =============================================================================

summary_table = (
    cv_subgroup_summary_df
    .set_index(
        "model"
    )
)


PID_COMPARISONS = [
    (
        "PID effect without history",
        "LR-A-NoPitcherID",
        "LR-B-PitcherID",
    ),
    (
        "PID effect with history",
        "LR-C-Historical",
        "LR-D-PitcherID-Historical",
    ),
]


pid_effect_rows = []


for (
    comparison,
    baseline_model,
    pid_model,
) in PID_COMPARISONS:

    pid_effect_rows.append(
        {
            "comparison":
                comparison,

            "baseline_model":
                baseline_model,

            "pid_model":
                pid_model,

            "overall_brier_improvement":
                (
                    summary_table.loc[
                        baseline_model,
                        "mean_overall_brier",
                    ]
                    -
                    summary_table.loc[
                        pid_model,
                        "mean_overall_brier",
                    ]
                ),

            "known_brier_improvement":
                (
                    summary_table.loc[
                        baseline_model,
                        "mean_known_brier",
                    ]
                    -
                    summary_table.loc[
                        pid_model,
                        "mean_known_brier",
                    ]
                ),

            "unknown_brier_improvement":
                (
                    summary_table.loc[
                        baseline_model,
                        "mean_unknown_brier",
                    ]
                    -
                    summary_table.loc[
                        pid_model,
                        "mean_unknown_brier",
                    ]
                ),
        }
    )


pid_effect_df = pd.DataFrame(
    pid_effect_rows
)


print("\n" + "=" * 120)
print("PITCHER ID EFFECT")
print("=" * 120)

print(
    pid_effect_df
    .round(6)
    .to_string(
        index=False,
    )
)


# =============================================================================
# 22. Historical Cold-start Summary
# =============================================================================

print("\n" + "=" * 140)
print("HISTORICAL NaN / COLD START")
print("=" * 140)

print(
    cold_start_df
    .round(6)
    .to_string(
        index=False,
    )
)


# =============================================================================
# 23. Historical Cold-start 검증
#
# 여기서는 NaN이 없어야 한다고 assertion하지 않는다.
#
# NaN은 공식적으로 cold-start를 나타낼 수 있는 정상 상태.
#
# 대신 C/D가 이러한 NaN을 포함한 Unknown row에서도
# 오류 없이 probability를 생성했는지를 검증한다.
# =============================================================================

historical_models = (
    subgroup_results_df[
        subgroup_results_df[
            "historical"
        ]
    ]
)


assert (
    historical_models[
        "unknown_rows"
    ]
    > 0
).all()


assert (
    historical_models[
        "unknown_brier"
    ]
    .notna()
).all()


historical_cold_start_status = "PASS"


# =============================================================================
# 24. Unknown 표본 충분성 참고
#
# 자동으로 Group Holdout을 추가하지 않는다.
# 자연 발생 Unknown 분포만 기록.
# =============================================================================

print("\n" + "=" * 80)
print("UNKNOWN SAMPLE CHECK")
print("=" * 80)

for _, row in unknown_distribution_df.iterrows():

    print(
        f"Fold {int(row['fold'])}: "
        f"{int(row['unknown_pitchers']):,} pitchers | "
        f"{int(row['unknown_rows']):,} rows | "
        f"{row['unknown_row_rate']:.2%}"
    )


# =============================================================================
# 25. 결과 저장
# =============================================================================

subgroup_results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lr12_fold_subgroup_results.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


cv_subgroup_summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lr12_cv_subgroup_summary.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


unknown_distribution_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lr12_unknown_distribution.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


cold_start_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lr12_historical_cold_start.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


encoder_validation_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lr12_unknown_encoder_validation.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


pid_effect_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lr12_pitcher_id_effect.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


# 완료 후 progress file 삭제
progress_path = os.path.join(
    OUTPUT_DIR,
    "lr12_subgroup_results_progress.csv",
)

if os.path.exists(
    progress_path
):
    os.remove(
        progress_path
    )


# =============================================================================
# 26. Final Validation Summary
# =============================================================================

validation_summary_df = pd.DataFrame(
    [
        {
            "check":
                "known_unknown_partition",

            "result":
                "PASS",

            "detail":
                "known + unknown = validation rows",
        },

        {
            "check":
                "unknown_pitcher_encoder",

            "result":
                "PASS",

            "detail":
                (
                    "Model B/D unseen pitcher_id handled by "
                    "OneHotEncoder(handle_unknown='ignore')"
                ),
        },

        {
            "check":
                "historical_cold_start",

            "result":
                historical_cold_start_status,

            "detail":
                (
                    "official asof_* NaN retained in raw data; "
                    "pipeline imputation handles Unknown rows"
                ),
        },

        {
            "check":
                "overall_known_unknown_metric",

            "result":
                "PASS",

            "detail":
                (
                    "Brier / LogLoss / ROC-AUC / PR-AUC "
                    "evaluated separately"
                ),
        },

        {
            "check":
                "temporal_boundary",

            "result":
                "PASS",

            "detail":
                (
                    "Known/Unknown membership uses fold train "
                    "pitcher set only"
                ),
        },
    ]
)


validation_summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lr12_validation_summary.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


print("\n" + "=" * 100)
print("VALIDATION SUMMARY")
print("=" * 100)

print(
    validation_summary_df
    .to_string(
        index=False,
    )
)


print("\n" + "=" * 80)
print("LR-12 KNOWN / UNKNOWN VALIDATION COMPLETE")
print("=" * 80)

print(
    f"Output directory : "
    f"{OUTPUT_DIR}"
)