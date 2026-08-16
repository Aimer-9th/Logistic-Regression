import os
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    brier_score_loss,
    log_loss,
)


# =============================================================================
# 0. 기본 설정
# =============================================================================

warnings.filterwarnings("ignore")

DATA_DIR = "/content/drive/MyDrive/𝟐𝟎𝟐𝟔/aimers/9기/open/data"
OUTPUT_DIR = "./lr09_naive_baseline_outputs"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

TRAIN_PATH = os.path.join(
    DATA_DIR,
    "train.csv",
)

TARGET = "control_success"
SEASON_COL = "season"

FIRST_TRAIN_SEASON = 2019

VALID_SEASONS = [
    2022,
    2023,
    2024,
]


# =============================================================================
# 1. 데이터 로드
# =============================================================================

train = pd.read_csv(
    TRAIN_PATH,
    encoding="utf-8-sig",
)


print("=" * 80)
print("1. DATA")
print("=" * 80)

print(f"Train shape : {train.shape}")

print(
    f"Season      : "
    f"{train[SEASON_COL].min()} "
    f"~ "
    f"{train[SEASON_COL].max()}"
)

print(
    f"Target rate : "
    f"{train[TARGET].mean():.6f}"
)


# =============================================================================
# 2. Target 기본 검증
# =============================================================================

target_values = set(
    train[TARGET]
    .dropna()
    .unique()
)


assert target_values == {0, 1}, (
    f"{TARGET}은 정확한 binary target이어야 합니다."
)

assert train[TARGET].isna().sum() == 0, (
    f"{TARGET}에 결측치가 존재합니다."
)


# =============================================================================
# 3. Naive Probability Fit 함수
#
# 핵심:
# probability는 training target만 입력받아 계산한다.
#
# validation target은 이 함수에 전달되지 않는다.
# =============================================================================

def fit_naive_probability(
    y_train,
):
    """
    Training set의 positive rate를 naive probability로 사용한다.

    P(control_success=1)
        = mean(y_train)
    """

    probability = float(
        np.mean(y_train)
    )

    assert 0.0 < probability < 1.0, (
        "Training target 평균이 (0, 1) 범위를 벗어났습니다."
    )

    return probability


# =============================================================================
# 4. 평가 함수
#
# validation target은 metric 계산에만 사용한다.
# baseline probability 계산에는 사용하지 않는다.
# =============================================================================

def evaluate_constant_probability(
    y_valid,
    probability,
):
    """
    모든 validation row에 동일한 확률을 예측하고
    Brier Score / Log Loss를 계산한다.
    """

    y_valid = np.asarray(
        y_valid
    )

    y_prob = np.full(
        shape=len(y_valid),
        fill_value=probability,
        dtype=float,
    )


    brier = brier_score_loss(
        y_valid,
        y_prob,
    )


    logloss = log_loss(
        y_valid,
        y_prob,
        labels=[
            0,
            1,
        ],
    )


    return {
        "brier":
            brier,

        "logloss":
            logloss,

        "prediction_mean":
            y_prob.mean(),
    }


# =============================================================================
# 5. Expanding Time Split 정의
#
# Valid 2022:
#   Train 2019 ~ 2021
#
# Valid 2023:
#   Train 2019 ~ 2022
#
# Valid 2024:
#   Train 2019 ~ 2023
# =============================================================================

def make_time_fold(
    data,
    valid_season,
):
    """
    validation season 이전 시즌만 training fold에 포함한다.
    """

    train_mask = (
        (data[SEASON_COL] >= FIRST_TRAIN_SEASON)
        &
        (data[SEASON_COL] < valid_season)
    )

    valid_mask = (
        data[SEASON_COL]
        == valid_season
    )


    train_fold = (
        data.loc[
            train_mask
        ]
        .copy()
    )

    valid_fold = (
        data.loc[
            valid_mask
        ]
        .copy()
    )


    assert len(train_fold) > 0, (
        f"{valid_season} fold의 train이 비어 있습니다."
    )

    assert len(valid_fold) > 0, (
        f"{valid_season} fold의 validation이 비어 있습니다."
    )


    # -------------------------------------------------------------------------
    # 시간 leakage 검증
    # -------------------------------------------------------------------------

    assert (
        train_fold[SEASON_COL].max()
        < valid_season
    ), (
        "Validation season 또는 미래 season이 "
        "training fold에 포함되었습니다."
    )

    assert (
        valid_fold[SEASON_COL]
        == valid_season
    ).all()


    return (
        train_fold,
        valid_fold,
    )


# =============================================================================
# 6. Naive Baseline CV
# =============================================================================

results = []


print("\n" + "=" * 100)
print("2. NAIVE PROBABILITY BASELINE")
print("=" * 100)


for fold_index, valid_season in enumerate(
    VALID_SEASONS,
    start=1,
):

    train_fold, valid_fold = make_time_fold(
        train,
        valid_season,
    )


    # -------------------------------------------------------------------------
    # Train / Valid target
    # -------------------------------------------------------------------------

    y_train = train_fold[
        TARGET
    ]

    y_valid = valid_fold[
        TARGET
    ]


    # -------------------------------------------------------------------------
    # Baseline probability
    #
    # 오직 y_train만 사용한다.
    # -------------------------------------------------------------------------

    baseline_probability = (
        fit_naive_probability(
            y_train
        )
    )


    # -------------------------------------------------------------------------
    # Validation 평가
    # -------------------------------------------------------------------------

    metrics = (
        evaluate_constant_probability(
            y_valid,
            baseline_probability,
        )
    )


    train_start_season = int(
        train_fold[
            SEASON_COL
        ]
        .min()
    )

    train_end_season = int(
        train_fold[
            SEASON_COL
        ]
        .max()
    )


    train_rate = float(
        y_train.mean()
    )

    valid_rate = float(
        y_valid.mean()
    )


    result = {
        "fold":
            fold_index,

        "train_start":
            train_start_season,

        "train_end":
            train_end_season,

        "valid_season":
            valid_season,

        "train_rows":
            len(train_fold),

        "valid_rows":
            len(valid_fold),

        "train_rate":
            train_rate,

        # valid_rate는 결과 해석용.
        # baseline probability 계산에는 사용하지 않는다.
        "valid_rate":
            valid_rate,

        "baseline_probability":
            baseline_probability,

        "brier":
            metrics[
                "brier"
            ],

        "logloss":
            metrics[
                "logloss"
            ],
    }


    results.append(
        result
    )


    # -------------------------------------------------------------------------
    # Fold 로그
    # -------------------------------------------------------------------------

    print("\n" + "-" * 100)

    print(
        f"Fold {fold_index}"
    )

    print("-" * 100)

    print(
        f"Train seasons        : "
        f"{train_start_season}"
        f" ~ "
        f"{train_end_season}"
    )

    print(
        f"Valid season         : "
        f"{valid_season}"
    )

    print(
        f"Train rows           : "
        f"{len(train_fold):,}"
    )

    print(
        f"Valid rows           : "
        f"{len(valid_fold):,}"
    )

    print(
        f"Train rate           : "
        f"{train_rate:.6f}"
    )

    print(
        f"Baseline probability : "
        f"{baseline_probability:.6f}"
    )

    print(
        f"Valid rate           : "
        f"{valid_rate:.6f}"
    )

    print(
        f"Brier Score          : "
        f"{metrics['brier']:.6f}"
    )

    print(
        f"Log Loss             : "
        f"{metrics['logloss']:.6f}"
    )


# =============================================================================
# 7. Fold 결과 DataFrame
# =============================================================================

results_df = pd.DataFrame(
    results
)


results_df[
    "train_period"
] = (
    results_df[
        "train_start"
    ]
    .astype(str)
    + "~"
    + results_df[
        "train_end"
    ]
    .astype(str)
)


DISPLAY_COLS = [
    "fold",
    "train_period",
    "valid_season",
    "train_rows",
    "valid_rows",
    "train_rate",
    "valid_rate",
    "brier",
    "logloss",
]


print("\n" + "=" * 120)
print("3. FOLD RESULTS")
print("=" * 120)

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
# 8. 전체 CV 평균
#
# Issue의 "전체 CV 평균"은 fold별 metric의 단순 산술평균으로 정의한다.
# =============================================================================

mean_brier = float(
    results_df[
        "brier"
    ]
    .mean()
)

mean_logloss = float(
    results_df[
        "logloss"
    ]
    .mean()
)


std_brier = float(
    results_df[
        "brier"
    ]
    .std(
        ddof=0
    )
)

std_logloss = float(
    results_df[
        "logloss"
    ]
    .std(
        ddof=0
    )
)


print("\n" + "=" * 80)
print("4. CV SUMMARY")
print("=" * 80)

print(
    f"Mean Brier   : "
    f"{mean_brier:.6f}"
)

print(
    f"Mean LogLoss : "
    f"{mean_logloss:.6f}"
)

print(
    f"Std Brier    : "
    f"{std_brier:.6f}"
)

print(
    f"Std LogLoss  : "
    f"{std_logloss:.6f}"
)


# =============================================================================
# 9. Leakage Validation
#
# baseline_probability와 train_rate가 정확히 같아야 한다.
#
# Valid Rate가 probability 계산에 사용되지 않았음을 구조적으로 검증한다.
# =============================================================================

print("\n" + "=" * 80)
print("5. LEAKAGE VALIDATION")
print("=" * 80)


probability_matches_train_rate = (
    np.isclose(
        results_df[
            "baseline_probability"
        ],
        results_df[
            "train_rate"
        ],
        atol=1e-15,
    )
    .all()
)


temporal_order_valid = (
    results_df[
        "train_end"
    ]
    .lt(
        results_df[
            "valid_season"
        ]
    )
    .all()
)


assert probability_matches_train_rate, (
    "Baseline probability가 train target 평균과 다릅니다."
)

assert temporal_order_valid, (
    "Training fold에 validation/future season이 포함되었습니다."
)


print(
    "✓ Baseline probability = training target mean"
)

print(
    "✓ Validation target is used only for metric calculation"
)

print(
    "✓ Every training period ends before validation season"
)


# =============================================================================
# 10. 결과 저장
# =============================================================================

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "naive_baseline_fold_results.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


summary_df = pd.DataFrame(
    [
        {
            "baseline":
                "GlobalTrainMean",

            "n_folds":
                len(results_df),

            "mean_brier":
                mean_brier,

            "std_brier":
                std_brier,

            "mean_logloss":
                mean_logloss,

            "std_logloss":
                std_logloss,
        }
    ]
)


summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "naive_baseline_summary.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


# =============================================================================
# 11. PR 기록용 결과
# =============================================================================

print("\n" + "=" * 100)
print("6. PR RESULT TABLE")
print("=" * 100)


pr_table = (
    results_df[
        [
            "train_period",
            "valid_season",
            "train_rate",
            "brier",
            "logloss",
        ]
    ]
    .copy()
)


print(
    pr_table
    .round(6)
    .to_string(
        index=False,
    )
)


print("\n[Mean]")

print(
    f"Brier    : "
    f"{mean_brier:.6f}"
)

print(
    f"Log Loss : "
    f"{mean_logloss:.6f}"
)


print("\n" + "=" * 80)
print("LR-09 NAIVE BASELINE COMPLETE")
print("=" * 80)

print(
    f"Output directory : "
    f"{OUTPUT_DIR}"
)