import os
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
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

ID = "row_id"
TARGET = "control_success"
PITCHER_ID_COL = "pitcher_id"

TRAIN_END_SEASON = 2023
VALID_SEASON = 2024

RANDOM_STATE = 42

LR_C = 1.0
LR_MAX_ITER = 500


# =============================================================================
# 1. 데이터 로드
# =============================================================================

test_columns = pd.read_csv(
    os.path.join(DATA_DIR, "test.csv"),
    encoding="utf-8-sig",
    nrows=0,
).columns

FEATURES = [
    col
    for col in test_columns
    if col != ID
]

train = pd.read_csv(
    os.path.join(DATA_DIR, "train.csv"),
    encoding="utf-8-sig",
    usecols=FEATURES + [TARGET],
)


print("=" * 80)
print("1. DATA")
print("=" * 80)

print(f"Train shape   : {train.shape}")
print(f"Feature count : {len(FEATURES)}")
print(
    f"Season        : "
    f"{train['season'].min()} ~ {train['season'].max()}"
)
print(
    f"Target rate   : "
    f"{train[TARGET].mean():.4f}"
)


# =============================================================================
# 2. Feature Type 정의
# =============================================================================

CAT_COLS = [
    # 경기 정보
    "game_dayofweek",
    "top_bottom",
    "game_type",
    "base_state",

    # 선수 / 팀 정보
    "pitcher_id",
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

    # 타자 과거 이력
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",

    # 투수 구종 구성 이력
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]


# =============================================================================
# 3. Feature Type 검증
# =============================================================================

assert set(CAT_COLS).isdisjoint(set(NUM_COLS)), (
    "CAT_COLS와 NUM_COLS에 중복 컬럼이 있습니다."
)

assert (
    set(CAT_COLS) | set(NUM_COLS)
) == set(FEATURES), (
    "모든 feature가 CAT/NUM 중 하나에 정확히 포함되어야 합니다."
)


print("\n" + "=" * 80)
print("2. FEATURE TYPE")
print("=" * 80)

print(f"Total       : {len(FEATURES)}")
print(f"Categorical : {len(CAT_COLS)}")
print(f"Numeric     : {len(NUM_COLS)}")


# =============================================================================
# 4. 명백한 중복 Feature 정의
# =============================================================================

DROP_REDUNDANT_COLS = [
    # asof_pitcher_n과 완전히 동일
    "asof_pitcher_pitchmix_n",

    # run_top_before + run_bot_before
    "run_total_before",

    # runner_on_1b + runner_on_2b + runner_on_3b
    "num_runners_on",

    # home_win_expectancy와 사실상 동일 정보
    "away_win_expectancy",
]


print("\n" + "=" * 80)
print("3. REDUNDANT FEATURES")
print("=" * 80)

for col in DROP_REDUNDANT_COLS:
    print(f"- {col}")


# =============================================================================
# 5. 4개 Logistic Regression 실험군 정의
# =============================================================================

FEATURES_FULL_PID = FEATURES.copy()

FEATURES_FULL_NO_PID = [
    col
    for col in FEATURES
    if col != PITCHER_ID_COL
]

FEATURES_REDUCED_PID = [
    col
    for col in FEATURES
    if col not in DROP_REDUNDANT_COLS
]

FEATURES_REDUCED_NO_PID = [
    col
    for col in FEATURES
    if (
        col not in DROP_REDUNDANT_COLS
        and col != PITCHER_ID_COL
    )
]


LR_EXPERIMENTS = {
    "LR-Full-PID": {
        "features": FEATURES_FULL_PID,
        "use_pitcher_id": True,
        "reduced": False,
    },

    "LR-Full-NoPID": {
        "features": FEATURES_FULL_NO_PID,
        "use_pitcher_id": False,
        "reduced": False,
    },

    "LR-Reduced-PID": {
        "features": FEATURES_REDUCED_PID,
        "use_pitcher_id": True,
        "reduced": True,
    },

    "LR-Reduced-NoPID": {
        "features": FEATURES_REDUCED_NO_PID,
        "use_pitcher_id": False,
        "reduced": True,
    },
}


print("\n" + "=" * 80)
print("4. EXPERIMENTS")
print("=" * 80)

for name, config in LR_EXPERIMENTS.items():
    print(
        f"{name:<20} | "
        f"features={len(config['features']):>2} | "
        f"PID={str(config['use_pitcher_id']):<5} | "
        f"Reduced={config['reduced']}"
    )


# =============================================================================
# 6. 실험별 Feature Type 반환 함수
# =============================================================================

def get_feature_types(feature_list):
    """
    실험에 사용되는 feature 목록에 맞춰
    categorical / numeric column 목록을 생성한다.
    """

    cat_cols = [
        col
        for col in CAT_COLS
        if col in feature_list
    ]

    num_cols = [
        col
        for col in NUM_COLS
        if col in feature_list
    ]

    assert (
        len(cat_cols) + len(num_cols)
        == len(feature_list)
    )

    return cat_cols, num_cols


# =============================================================================
# 7. Logistic Regression Pipeline
# =============================================================================

def build_lr_pipeline(
    cat_cols,
    num_cols,
    C=1.0,
    max_iter=500,
):
    """
    Logistic Regression Pipeline

    Categorical
        -> Most Frequent Imputation
        -> OneHotEncoder

    Numeric
        -> Median Imputation
        -> Missing Indicator
        -> StandardScaler

    Model
        -> L2 Logistic Regression
    """

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
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

    numeric_pipeline = Pipeline(
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

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                cat_cols,
            ),
            (
                "numeric",
                numeric_pipeline,
                num_cols,
            ),
        ],
        remainder="drop",
    )

    model = LogisticRegression(
        C=C,
        penalty="l2",
        solver="saga",
        max_iter=max_iter,
        n_jobs=-1,
        random_state=RANDOM_STATE,

        # solver 내부 로그는 끔.
        # 대신 아래 학습 루프에서 깔끔한 로그를 출력함.
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
# 8. Time-based Validation Split
#
# Train : 2019 ~ 2023
# Valid : 2024
# =============================================================================

train_mask = (
    train["season"]
    <= TRAIN_END_SEASON
)

valid_mask = (
    train["season"]
    == VALID_SEASON
)


train_fold = (
    train.loc[train_mask]
    .reset_index(drop=True)
)

valid_fold = (
    train.loc[valid_mask]
    .reset_index(drop=True)
)


print("\n" + "=" * 80)
print("5. VALIDATION SPLIT")
print("=" * 80)

print(
    "Train seasons :",
    sorted(train_fold["season"].unique()),
)

print(
    "Valid seasons :",
    sorted(valid_fold["season"].unique()),
)

print(
    f"Train rows     : {len(train_fold):,}"
)

print(
    f"Valid rows     : {len(valid_fold):,}"
)

print(
    f"Train rate     : {train_fold[TARGET].mean():.4f}"
)

print(
    f"Valid rate     : {valid_fold[TARGET].mean():.4f}"
)


# =============================================================================
# 9. Known / Unknown Pitcher 정의
# =============================================================================

train_pitchers = set(
    train_fold[PITCHER_ID_COL]
    .dropna()
    .unique()
)


known_mask = (
    valid_fold[PITCHER_ID_COL]
    .isin(train_pitchers)
)

unknown_mask = (
    ~valid_fold[PITCHER_ID_COL]
    .isin(train_pitchers)
)


print("\n" + "=" * 80)
print("6. KNOWN / UNKNOWN PITCHER")
print("=" * 80)

print(
    f"Known rows      : "
    f"{known_mask.sum():,} "
    f"({known_mask.mean():.2%})"
)

print(
    f"Unknown rows    : "
    f"{unknown_mask.sum():,} "
    f"({unknown_mask.mean():.2%})"
)

print(
    f"Known pitchers  : "
    f"{valid_fold.loc[known_mask, PITCHER_ID_COL].nunique():,}"
)

print(
    f"Unknown pitchers: "
    f"{valid_fold.loc[unknown_mask, PITCHER_ID_COL].nunique():,}"
)


# =============================================================================
# 10. 평가 함수
# =============================================================================

def evaluate_predictions(y_true, y_prob):
    """
    확률 예측 성능 평가.

    Primary
        Brier Score

    Secondary
        Log Loss
        ROC-AUC
    """

    if len(y_true) == 0:
        return {
            "brier": np.nan,
            "logloss": np.nan,
            "auc": np.nan,
        }

    result = {
        "brier": brier_score_loss(
            y_true,
            y_prob,
        ),

        "logloss": log_loss(
            y_true,
            y_prob,
        ),
    }

    if len(np.unique(y_true)) == 2:
        result["auc"] = roc_auc_score(
            y_true,
            y_prob,
        )
    else:
        result["auc"] = np.nan

    return result


# =============================================================================
# 11. 모델 학습
# =============================================================================

results = []
trained_models = {}


print("\n" + "=" * 80)
print("7. MODEL TRAINING START")
print("=" * 80)


for experiment_index, (experiment_name, config) in enumerate(
    LR_EXPERIMENTS.items(),
    start=1,
):

    total_start = time.time()

    print("\n" + "#" * 80)
    print(
        f"[{experiment_index}/{len(LR_EXPERIMENTS)}] "
        f"{experiment_name}"
    )
    print("#" * 80)


    # -------------------------------------------------------------------------
    # Feature 설정
    # -------------------------------------------------------------------------

    feature_list = config["features"]

    cat_cols, num_cols = get_feature_types(
        feature_list
    )

    print(
        f"Features        : {len(feature_list)}"
    )
    print(
        f"Categorical     : {len(cat_cols)}"
    )
    print(
        f"Numeric         : {len(num_cols)}"
    )
    print(
        f"Pitcher ID      : {config['use_pitcher_id']}"
    )
    print(
        f"Reduced         : {config['reduced']}"
    )


    # -------------------------------------------------------------------------
    # 데이터 생성
    # -------------------------------------------------------------------------

    X_train = train_fold[
        feature_list
    ]

    y_train = train_fold[
        TARGET
    ]

    X_valid = valid_fold[
        feature_list
    ]

    y_valid = valid_fold[
        TARGET
    ]


    print(
        f"Train rows      : {len(X_train):,}"
    )
    print(
        f"Valid rows      : {len(X_valid):,}"
    )


    # -------------------------------------------------------------------------
    # Pipeline 생성
    # -------------------------------------------------------------------------

    pipeline = build_lr_pipeline(
        cat_cols=cat_cols,
        num_cols=num_cols,
        C=LR_C,
        max_iter=LR_MAX_ITER,
    )


    # -------------------------------------------------------------------------
    # 학습
    # -------------------------------------------------------------------------

    print(
        f"\n[{experiment_name}] "
        "전처리 + 모델 학습 시작"
    )

    fit_start = time.time()

    pipeline.fit(
        X_train,
        y_train,
    )

    fit_seconds = (
        time.time()
        - fit_start
    )

    print(
        f"[{experiment_name}] "
        f"학습 완료 | "
        f"{fit_seconds:.1f} sec"
    )


    # -------------------------------------------------------------------------
    # 수렴 확인
    # -------------------------------------------------------------------------

    lr_model = (
        pipeline
        .named_steps["model"]
    )

    n_iter = int(
        lr_model.n_iter_[0]
    )

    print(
        f"[{experiment_name}] "
        f"Iteration: "
        f"{n_iter}/{lr_model.max_iter}"
    )

    if n_iter >= lr_model.max_iter:
        print(
            f"[{experiment_name}] "
            "⚠️ max_iter 도달 → "
            "완전히 수렴하지 않았을 가능성 있음"
        )
    else:
        print(
            f"[{experiment_name}] "
            "✓ max_iter 이전 수렴"
        )


    # -------------------------------------------------------------------------
    # Validation 예측
    # -------------------------------------------------------------------------

    print(
        f"[{experiment_name}] "
        "Validation 확률 예측 시작"
    )

    predict_start = time.time()

    valid_prob = (
        pipeline
        .predict_proba(X_valid)[:, 1]
    )

    predict_seconds = (
        time.time()
        - predict_start
    )

    print(
        f"[{experiment_name}] "
        f"Validation 예측 완료 | "
        f"{predict_seconds:.1f} sec"
    )


    # -------------------------------------------------------------------------
    # 예측 확률 sanity check
    # -------------------------------------------------------------------------

    print(
        f"[{experiment_name}] "
        f"Prediction range: "
        f"{valid_prob.min():.6f} ~ "
        f"{valid_prob.max():.6f}"
    )

    print(
        f"[{experiment_name}] "
        f"Mean prediction : "
        f"{valid_prob.mean():.6f}"
    )

    print(
        f"[{experiment_name}] "
        f"Actual rate     : "
        f"{y_valid.mean():.6f}"
    )


    # -------------------------------------------------------------------------
    # Overall 평가
    # -------------------------------------------------------------------------

    overall_metrics = evaluate_predictions(
        y_valid,
        valid_prob,
    )


    # -------------------------------------------------------------------------
    # Known Pitcher 평가
    # -------------------------------------------------------------------------

    known_idx = known_mask.to_numpy()

    known_metrics = evaluate_predictions(
        y_valid.to_numpy()[known_idx],
        valid_prob[known_idx],
    )


    # -------------------------------------------------------------------------
    # Unknown Pitcher 평가
    # -------------------------------------------------------------------------

    unknown_idx = unknown_mask.to_numpy()

    unknown_metrics = evaluate_predictions(
        y_valid.to_numpy()[unknown_idx],
        valid_prob[unknown_idx],
    )


    # -------------------------------------------------------------------------
    # 전체 시간
    # -------------------------------------------------------------------------

    total_seconds = (
        time.time()
        - total_start
    )


    # -------------------------------------------------------------------------
    # 결과 저장
    # -------------------------------------------------------------------------

    result = {
        "model": experiment_name,

        "n_features": len(feature_list),
        "n_cat": len(cat_cols),
        "n_num": len(num_cols),

        "pitcher_id": config["use_pitcher_id"],
        "reduced": config["reduced"],

        "overall_brier": overall_metrics["brier"],
        "overall_logloss": overall_metrics["logloss"],
        "overall_auc": overall_metrics["auc"],

        "known_brier": known_metrics["brier"],
        "known_logloss": known_metrics["logloss"],
        "known_auc": known_metrics["auc"],

        "unknown_brier": unknown_metrics["brier"],
        "unknown_logloss": unknown_metrics["logloss"],
        "unknown_auc": unknown_metrics["auc"],

        "n_iter": n_iter,

        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "total_seconds": total_seconds,
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

    print("\n" + "-" * 80)

    print(
        f"{experiment_name} RESULT"
    )

    print("-" * 80)

    print(
        f"Overall Brier   : "
        f"{overall_metrics['brier']:.6f}"
    )

    print(
        f"Known Brier     : "
        f"{known_metrics['brier']:.6f}"
    )

    print(
        f"Unknown Brier   : "
        f"{unknown_metrics['brier']:.6f}"
    )

    print(
        f"Overall LogLoss : "
        f"{overall_metrics['logloss']:.6f}"
    )

    print(
        f"Overall AUC     : "
        f"{overall_metrics['auc']:.6f}"
    )

    print(
        f"Fit time        : "
        f"{fit_seconds:.1f} sec"
    )

    print(
        f"Predict time    : "
        f"{predict_seconds:.1f} sec"
    )

    print(
        f"Total time      : "
        f"{total_seconds:.1f} sec"
    )


# =============================================================================
# 12. 전체 결과 DataFrame
# =============================================================================

results_df = (
    pd.DataFrame(results)
    .sort_values(
        by="overall_brier",
        ascending=True,
    )
    .reset_index(drop=True)
)


DISPLAY_COLS = [
    "model",

    "n_features",
    "pitcher_id",
    "reduced",

    "overall_brier",
    "known_brier",
    "unknown_brier",

    "overall_logloss",
    "overall_auc",

    "n_iter",

    "fit_seconds",
    "predict_seconds",
    "total_seconds",
]


print("\n" + "=" * 140)
print("8. FINAL LOGISTIC REGRESSION RESULTS")
print("=" * 140)

print(
    results_df[
        DISPLAY_COLS
    ]
    .round(6)
    .to_string(index=False)
)


# =============================================================================
# 13. Best Model
# =============================================================================

best_model_name = (
    results_df
    .iloc[0]["model"]
)

best_brier = (
    results_df
    .iloc[0]["overall_brier"]
)


print("\n" + "=" * 80)
print("9. BEST MODEL")
print("=" * 80)

print(
    f"Best model        : {best_model_name}"
)

print(
    f"Best Overall Brier: {best_brier:.6f}"
)


# =============================================================================
# 14. Pitcher ID 효과 비교
# =============================================================================

score_table = (
    results_df
    .set_index("model")
)


pid_comparisons = []


for feature_type in [
    "Full",
    "Reduced",
]:

    pid_model = (
        f"LR-{feature_type}-PID"
    )

    no_pid_model = (
        f"LR-{feature_type}-NoPID"
    )

    pid_comparisons.append(
        {
            "comparison":
                f"{feature_type}: PID - NoPID",

            "overall_brier_diff":
                score_table.loc[
                    pid_model,
                    "overall_brier",
                ]
                -
                score_table.loc[
                    no_pid_model,
                    "overall_brier",
                ],

            "known_brier_diff":
                score_table.loc[
                    pid_model,
                    "known_brier",
                ]
                -
                score_table.loc[
                    no_pid_model,
                    "known_brier",
                ],

            "unknown_brier_diff":
                score_table.loc[
                    pid_model,
                    "unknown_brier",
                ]
                -
                score_table.loc[
                    no_pid_model,
                    "unknown_brier",
                ],
        }
    )


pid_comparison_df = pd.DataFrame(
    pid_comparisons
)


print("\n" + "=" * 100)
print("10. PITCHER ID EFFECT")
print("=" * 100)

print(
    pid_comparison_df
    .round(6)
    .to_string(index=False)
)

print(
    "\nBrier diff < 0 : pitcher_id 포함이 더 좋음"
)
print(
    "Brier diff > 0 : pitcher_id 제외가 더 좋음"
)


# =============================================================================
# 15. Full vs Reduced 비교
# =============================================================================

reduction_comparisons = []


for pid_type in [
    "PID",
    "NoPID",
]:

    full_model = (
        f"LR-Full-{pid_type}"
    )

    reduced_model = (
        f"LR-Reduced-{pid_type}"
    )

    reduction_comparisons.append(
        {
            "comparison":
                f"{pid_type}: Reduced - Full",

            "overall_brier_diff":
                score_table.loc[
                    reduced_model,
                    "overall_brier",
                ]
                -
                score_table.loc[
                    full_model,
                    "overall_brier",
                ],

            "known_brier_diff":
                score_table.loc[
                    reduced_model,
                    "known_brier",
                ]
                -
                score_table.loc[
                    full_model,
                    "known_brier",
                ],

            "unknown_brier_diff":
                score_table.loc[
                    reduced_model,
                    "unknown_brier",
                ]
                -
                score_table.loc[
                    full_model,
                    "unknown_brier",
                ],
        }
    )


reduction_comparison_df = pd.DataFrame(
    reduction_comparisons
)


print("\n" + "=" * 100)
print("11. REDUNDANCY REMOVAL EFFECT")
print("=" * 100)

print(
    reduction_comparison_df
    .round(6)
    .to_string(index=False)
)

print(
    "\nBrier diff < 0 : Reduced가 더 좋음"
)
print(
    "Brier diff > 0 : Full이 더 좋음"
)