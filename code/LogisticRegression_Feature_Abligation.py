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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
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
    "lr14_feature_ablation_outputs"
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
# 1. LR-13 Best Regularization 고정
#
# Feature Ablation에서는 regularization을 다시 tuning하지 않는다.
# Feature 구성만 변경한다.
# =============================================================================

LR_PENALTY = "l1"
LR_C = 0.001
LR_SOLVER = "saga"
LR_MAX_ITER = 500
LR_TOL = 1e-3


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
# 3. 실행 환경
# =============================================================================

print("=" * 80)
print("ENVIRONMENT")
print("=" * 80)

print(f"scikit-learn : {sklearn.__version__}")
print(f"NumPy        : {np.__version__}")
print(f"Pandas       : {pd.__version__}")

print("\nBest LR-13 config:")
print(f"penalty : {LR_PENALTY}")
print(f"C       : {LR_C}")
print(f"solver  : {LR_SOLVER}")


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

    # 투수 누적 history
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",

    # 투수 최근 경기 history
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",

    # 타자 history
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",

    # 과거 pitch mix
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
# 6. Train 데이터 로드
#
# Colab:
# - row_id 제외
# - 필요한 feature + target만 로드
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
# 7. Feature Group 정의
# =============================================================================

HISTORICAL_FEATURES = [
    col
    for col in ALL_FEATURES
    if col.startswith("asof_")
]


assert len(
    HISTORICAL_FEATURES
) == 19


# 현재 공식 train/test에는
# 현재 투구의 실제 구종/구속/회전수/무브먼트가 없음.
PITCH_CHARACTERISTIC_FEATURES = []


# LR-06에서 신규 target rolling은 실제 train에 생성하지 않았음.
ROLLING_FEATURES = []


# pitcher_id와 official historical을 제외한 27개
# 경기 상황 + 시점 + batter/team/static 정보
CONTEXT_FEATURES = [
    col
    for col in ALL_FEATURES
    if (
        col != PITCHER_ID_COL
        and col not in HISTORICAL_FEATURES
    )
]


assert len(
    CONTEXT_FEATURES
) == 27


# =============================================================================
# 8. Reduced Feature 정의
#
# LR-05에서 실제 deterministic redundancy가 확인된 3개만 제거.
#
# away_win_expectancy는 complement 검증 실패로 유지.
# =============================================================================

REDUCED_DROP_COLS = [
    "run_total_before",
    "num_runners_on",
    "asof_pitcher_pitchmix_n",
]


assert all(
    col in ALL_FEATURES
    for col in REDUCED_DROP_COLS
)


# =============================================================================
# 9. Ablation Stage
#
# M1 / M5는 현재 데이터 조건에서 실제 실행하지 않는다.
# =============================================================================

M2_FEATURES = (
    CONTEXT_FEATURES.copy()
)


M3_FEATURES = (
    CONTEXT_FEATURES
    + HISTORICAL_FEATURES
)


M4_FEATURES = (
    CONTEXT_FEATURES
    + HISTORICAL_FEATURES
    + [
        PITCHER_ID_COL,
    ]
)


M3_REDUCED_FEATURES = [
    col
    for col in M3_FEATURES
    if col not in REDUCED_DROP_COLS
]


M4_REDUCED_FEATURES = [
    col
    for col in M4_FEATURES
    if col not in REDUCED_DROP_COLS
]


assert len(M2_FEATURES) == 27
assert len(M3_FEATURES) == 46
assert len(M4_FEATURES) == 47
assert len(M3_REDUCED_FEATURES) == 43
assert len(M4_REDUCED_FEATURES) == 44


ABLATION_EXPERIMENTS = {
    "M2-Context": {
        "features":
            M2_FEATURES,

        "stage":
            "M2",

        "description":
            "Context + available static player/team information",
    },

    "M3-Historical": {
        "features":
            M3_FEATURES,

        "stage":
            "M3",

        "description":
            "M2 + official asof_* historical features",
    },

    "M4-PitcherID": {
        "features":
            M4_FEATURES,

        "stage":
            "M4",

        "description":
            "M3 + pitcher_id",
    },

    "M3-Historical-Reduced": {
        "features":
            M3_REDUCED_FEATURES,

        "stage":
            "Reduced",

        "description":
            "M3 - validated redundant features",
    },

    "M4-PitcherID-Reduced": {
        "features":
            M4_REDUCED_FEATURES,

        "stage":
            "Reduced",

        "description":
            "M4 - validated redundant features",
    },
}


# =============================================================================
# 10. Stage Status
# =============================================================================

stage_status_df = pd.DataFrame(
    [
        {
            "stage": "M0",
            "name": "Mean",
            "status": "EXECUTABLE",
            "reason": "Fold training target mean",
        },
        {
            "stage": "M1",
            "name": "Pitch Characteristics",
            "status": "NOT_AVAILABLE",
            "reason": (
                "Current pitch type/speed/spin/movement are not "
                "available as official prediction-time features"
            ),
        },
        {
            "stage": "M2",
            "name": "Context",
            "status": "EXECUTABLE",
            "reason": "27 official context/static features",
        },
        {
            "stage": "M3",
            "name": "Historical",
            "status": "EXECUTABLE",
            "reason": "M2 + 19 official past-only asof_* features",
        },
        {
            "stage": "M4",
            "name": "Pitcher ID",
            "status": "EXECUTABLE",
            "reason": "M3 + pitcher_id",
        },
        {
            "stage": "M5",
            "name": "Rolling",
            "status": "BLOCKED",
            "reason": (
                "Custom target rolling features were not generated "
                "because exact main-data chronology is unavailable"
            ),
        },
    ]
)


print("\n" + "=" * 100)
print("ABLATION STAGE STATUS")
print("=" * 100)

print(
    stage_status_df
    .to_string(
        index=False
    )
)


stage_status_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lr14_stage_status.csv",
    ),
    encoding="utf-8-sig",
    index=False,
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
# =============================================================================

def build_preprocessor(
    feature_list,
):
    feature_types = (
        get_feature_types(
            feature_list
        )
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


    if feature_types["scale"]:

        transformers.append(
            (
                "numeric_scale",
                numeric_scale_pipeline,
                feature_types[
                    "scale"
                ],
            )
        )


    if feature_types["no_scale"]:

        transformers.append(
            (
                "numeric_no_scale",
                numeric_no_scale_pipeline,
                feature_types[
                    "no_scale"
                ],
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
# 13. Logistic Regression
#
# LR-13 Best config 고정
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
# 14. Expanding Window
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
# 15. Known / Unknown
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
# 16. Brier
# =============================================================================

def get_brier(
    y_true,
    y_prob,
):
    return brier_score_loss(
        y_true,
        y_prob,
    )


# =============================================================================
# 17. Checkpoint
# =============================================================================

PROGRESS_PATH = os.path.join(
    OUTPUT_DIR,
    "lr14_ablation_progress.csv",
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
        (
            int(row["fold"]),
            row["model"],
        )
        for row in results
    }


    print(
        f"\nResume: "
        f"{len(completed_keys)} experiments already completed"
    )

else:

    results = []
    completed_keys = set()


# =============================================================================
# 18. CV 실행
# =============================================================================

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


    # =========================================================================
    # M0 — Fold Train Mean
    # =========================================================================

    m0_key = (
        fold,
        "M0-Mean",
    )


    if m0_key not in completed_keys:

        naive_probability = float(
            y_train.mean()
        )


        naive_prob = np.full(
            len(
                y_valid
            ),
            naive_probability,
            dtype=np.float32,
        )


        results.append(
            {
                "fold":
                    fold,

                "model":
                    "M0-Mean",

                "stage":
                    "M0",

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

                "feature_count":
                    0,

                "overall_brier":
                    get_brier(
                        y_valid,
                        naive_prob,
                    ),

                "known_brier":
                    get_brier(
                        y_valid[
                            known_mask
                        ],
                        naive_prob[
                            known_mask
                        ],
                    ),

                "unknown_brier":
                    get_brier(
                        y_valid[
                            unknown_mask
                        ],
                        naive_prob[
                            unknown_mask
                        ],
                    ),

                "n_iter":
                    0,

                "converged":
                    True,

                "fit_seconds":
                    0.0,
            }
        )


        completed_keys.add(
            m0_key
        )


        del naive_prob


    # =========================================================================
    # LR Ablation
    # =========================================================================

    for (
        experiment_name,
        config,
    ) in ABLATION_EXPERIMENTS.items():

        result_key = (
            fold,
            experiment_name,
        )


        if result_key in completed_keys:

            continue


        feature_list = (
            config[
                "features"
            ]
        )


        print("\n" + "=" * 110)

        print(
            f"Fold {fold} | "
            f"{experiment_name}"
        )

        print("=" * 110)

        print(
            f"Features : "
            f"{len(feature_list)}"
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


        pipeline = (
            build_lr_pipeline(
                feature_list
            )
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


        converged = (
            len(
                convergence_warnings
            )
            == 0
            and n_iter
            < LR_MAX_ITER
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


        overall_brier = (
            get_brier(
                y_valid,
                valid_prob,
            )
        )


        known_brier = (
            get_brier(
                y_valid[
                    known_mask
                ],
                valid_prob[
                    known_mask
                ],
            )
        )


        unknown_brier = (
            get_brier(
                y_valid[
                    unknown_mask
                ],
                valid_prob[
                    unknown_mask
                ],
            )
        )


        results.append(
            {
                "fold":
                    fold,

                "model":
                    experiment_name,

                "stage":
                    config[
                        "stage"
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

                "feature_count":
                    len(
                        feature_list
                    ),

                "overall_brier":
                    overall_brier,

                "known_brier":
                    known_brier,

                "unknown_brier":
                    unknown_brier,

                "n_iter":
                    n_iter,

                "converged":
                    converged,

                "fit_seconds":
                    fit_seconds,
            }
        )


        completed_keys.add(
            result_key
        )


        print(
            f"Overall : "
            f"{overall_brier:.6f}"
        )

        print(
            f"Known   : "
            f"{known_brier:.6f}"
        )

        print(
            f"Unknown : "
            f"{unknown_brier:.6f}"
        )

        print(
            f"Iter    : "
            f"{n_iter}/{LR_MAX_ITER}"
        )

        print(
            f"Fit     : "
            f"{fit_seconds:.1f} sec"
        )


        # ---------------------------------------------------------------------
        # checkpoint
        # ---------------------------------------------------------------------

        pd.DataFrame(
            results
        ).to_csv(
            PROGRESS_PATH,
            encoding="utf-8-sig",
            index=False,
        )


        # ---------------------------------------------------------------------
        # Colab RAM 회수
        # ---------------------------------------------------------------------

        del X_train
        del X_valid
        del pipeline
        del lr_model
        del valid_prob

        gc.collect()


    del y_train
    del y_valid
    del known_mask
    del unknown_mask

    gc.collect()


# =============================================================================
# 19. 결과 정리
# =============================================================================

results_df = (
    pd.DataFrame(
        results
    )
    .drop_duplicates(
        subset=[
            "fold",
            "model",
        ],
        keep="last",
    )
    .sort_values(
        [
            "fold",
            "model",
        ]
    )
    .reset_index(
        drop=True
    )
)


EXPECTED_RESULT_COUNT = (
    len(
        FOLD_CONFIGS
    )
    * (
        1
        + len(
            ABLATION_EXPERIMENTS
        )
    )
)


assert len(
    results_df
) == EXPECTED_RESULT_COUNT


# =============================================================================
# 20. CV Summary
# =============================================================================

summary_rows = []


for (
    model_name,
    model_df,
) in results_df.groupby(
    "model"
):

    summary_rows.append(
        {
            "model":
                model_name,

            "stage":
                model_df[
                    "stage"
                ]
                .iloc[0],

            "features":
                int(
                    model_df[
                        "feature_count"
                    ]
                    .iloc[0]
                ),

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

            "mean_known_brier":
                model_df[
                    "known_brier"
                ]
                .mean(),

            "mean_unknown_brier":
                model_df[
                    "unknown_brier"
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


summary_df = pd.DataFrame(
    summary_rows
)


# =============================================================================
# 21. Incremental Contribution
#
# M0 → M2 → M3 → M4
#
# Positive Δ Brier = next model improves
# =============================================================================

MAIN_STAGE_ORDER = [
    "M0-Mean",
    "M2-Context",
    "M3-Historical",
    "M4-PitcherID",
]


summary_lookup = (
    summary_df
    .set_index(
        "model"
    )
)


contribution_rows = []


for index in range(
    1,
    len(
        MAIN_STAGE_ORDER
    ),
):

    previous_model = (
        MAIN_STAGE_ORDER[
            index - 1
        ]
    )


    current_model = (
        MAIN_STAGE_ORDER[
            index
        ]
    )


    contribution_rows.append(
        {
            "from_model":
                previous_model,

            "to_model":
                current_model,

            "added_group":
                {
                    "M2-Context":
                        "Context / static information",

                    "M3-Historical":
                        "Official historical features",

                    "M4-PitcherID":
                        "pitcher_id",
                }[
                    current_model
                ],

            "overall_brier_improvement":
                (
                    summary_lookup.loc[
                        previous_model,
                        "mean_overall_brier",
                    ]
                    -
                    summary_lookup.loc[
                        current_model,
                        "mean_overall_brier",
                    ]
                ),

            "known_brier_improvement":
                (
                    summary_lookup.loc[
                        previous_model,
                        "mean_known_brier",
                    ]
                    -
                    summary_lookup.loc[
                        current_model,
                        "mean_known_brier",
                    ]
                ),

            "unknown_brier_improvement":
                (
                    summary_lookup.loc[
                        previous_model,
                        "mean_unknown_brier",
                    ]
                    -
                    summary_lookup.loc[
                        current_model,
                        "mean_unknown_brier",
                    ]
                ),
        }
    )


contribution_df = pd.DataFrame(
    contribution_rows
)


# =============================================================================
# 22. Full vs Reduced
#
# positive improvement:
# Reduced가 Full보다 좋음
# =============================================================================

REDUCTION_PAIRS = [
    (
        "M3-Historical",
        "M3-Historical-Reduced",
    ),
    (
        "M4-PitcherID",
        "M4-PitcherID-Reduced",
    ),
]


reduction_rows = []


for (
    full_model,
    reduced_model,
) in REDUCTION_PAIRS:

    reduction_rows.append(
        {
            "full_model":
                full_model,

            "reduced_model":
                reduced_model,

            "full_features":
                int(
                    summary_lookup.loc[
                        full_model,
                        "features",
                    ]
                ),

            "reduced_features":
                int(
                    summary_lookup.loc[
                        reduced_model,
                        "features",
                    ]
                ),

            "overall_brier_improvement":
                (
                    summary_lookup.loc[
                        full_model,
                        "mean_overall_brier",
                    ]
                    -
                    summary_lookup.loc[
                        reduced_model,
                        "mean_overall_brier",
                    ]
                ),

            "known_brier_improvement":
                (
                    summary_lookup.loc[
                        full_model,
                        "mean_known_brier",
                    ]
                    -
                    summary_lookup.loc[
                        reduced_model,
                        "mean_known_brier",
                    ]
                ),

            "unknown_brier_improvement":
                (
                    summary_lookup.loc[
                        full_model,
                        "mean_unknown_brier",
                    ]
                    -
                    summary_lookup.loc[
                        reduced_model,
                        "mean_unknown_brier",
                    ]
                ),
        }
    )


reduction_df = pd.DataFrame(
    reduction_rows
)


# =============================================================================
# 23. 출력
# =============================================================================

print("\n" + "=" * 140)
print("ABLATION CV SUMMARY")
print("=" * 140)

print(
    summary_df
    .sort_values(
        "mean_overall_brier"
    )
    .round(6)
    .to_string(
        index=False
    )
)


print("\n" + "=" * 120)
print("INCREMENTAL FEATURE CONTRIBUTION")
print("=" * 120)

print(
    contribution_df
    .round(6)
    .to_string(
        index=False
    )
)


print("\n" + "=" * 120)
print("FULL VS REDUCED")
print("=" * 120)

print(
    reduction_df
    .round(6)
    .to_string(
        index=False
    )
)


# =============================================================================
# 24. Model Comparison Plot
# =============================================================================

plot_models = [
    "M0-Mean",
    "M2-Context",
    "M3-Historical",
    "M4-PitcherID",
]


plot_df = (
    summary_df
    .set_index(
        "model"
    )
    .loc[
        plot_models
    ]
    .reset_index()
)


x = np.arange(
    len(
        plot_df
    )
)

width = 0.25


fig, ax = plt.subplots(
    figsize=(
        11,
        6,
    )
)


ax.bar(
    x - width,
    plot_df[
        "mean_overall_brier"
    ],
    width,
    label="Overall",
)


ax.bar(
    x,
    plot_df[
        "mean_known_brier"
    ],
    width,
    label="Known",
)


ax.bar(
    x + width,
    plot_df[
        "mean_unknown_brier"
    ],
    width,
    label="Unknown",
)


ax.set_xticks(
    x
)

ax.set_xticklabels(
    plot_df[
        "model"
    ],
    rotation=15,
)


ax.set_ylabel(
    "Mean Brier Score"
)

ax.set_title(
    "LR-14 Feature Ablation — 3-Fold CV"
)

ax.legend()


plt.tight_layout()


plot_path = os.path.join(
    OUTPUT_DIR,
    "lr14_ablation_brier.png",
)


plt.savefig(
    plot_path,
    dpi=150,
    bbox_inches="tight",
)


plt.show()


# =============================================================================
# 25. Feature 구성 저장
# =============================================================================

feature_rows = []


for (
    model_name,
    config,
) in ABLATION_EXPERIMENTS.items():

    for feature in config[
        "features"
    ]:

        feature_rows.append(
            {
                "model":
                    model_name,

                "feature":
                    feature,
            }
        )


feature_config_df = pd.DataFrame(
    feature_rows
)


# =============================================================================
# 26. 결과 저장
# =============================================================================

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lr14_fold_results.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lr14_cv_summary.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


contribution_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lr14_feature_contribution.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


reduction_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lr14_full_vs_reduced.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


feature_config_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lr14_feature_sets.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


# 최종 결과와 progress 동기화
results_df.to_csv(
    PROGRESS_PATH,
    encoding="utf-8-sig",
    index=False,
)


print("\n" + "=" * 80)
print("LR-14 FEATURE ABLATION COMPLETE")
print("=" * 80)

print(
    f"Output directory : "
    f"{OUTPUT_DIR}"
)

print(
    f"Comparison plot  : "
    f"{plot_path}"
)