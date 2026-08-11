import gc
import os
import time
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =============================================================================
# 0. 기본 설정
# =============================================================================

warnings.filterwarnings("ignore")

# 기본값: 현재 폴더의 ./data
# 예:
# DATA_DIR=/home/smwu/ml/data python LogisticRegression_Xavier_GPU.py
DATA_DIR = os.environ.get("DATA_DIR", "./data")

ID = "row_id"
TARGET = "control_success"
PITCHER_ID_COL = "pitcher_id"

TRAIN_END_SEASON = 2023
VALID_SEASON = 2024
RANDOM_STATE = 42

# -----------------------------------------------------------------------------
# Xavier GPU 학습 설정
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GPU_BATCH_SIZE = int(os.environ.get("GPU_BATCH_SIZE", "32768"))
GPU_EPOCHS = int(os.environ.get("GPU_EPOCHS", "40"))
GPU_LR = float(os.environ.get("GPU_LR", "0.03"))
GPU_PATIENCE = int(os.environ.get("GPU_PATIENCE", "5"))
GPU_TOL = float(os.environ.get("GPU_TOL", "1e-5"))

# 예측 시에도 한 번에 너무 큰 sparse matrix를 GPU로 보내지 않음
PRED_BATCH_SIZE = int(os.environ.get("PRED_BATCH_SIZE", "65536"))

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

print("=" * 80)
print("Runtime")
print("=" * 80)
print(f"Device         : {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
    print(f"CUDA           : {torch.version.cuda}")
    print(f"cuDNN          : {torch.backends.cudnn.version()}")
    print(f"Capability     : {torch.cuda.get_device_capability(0)}")

print(f"Batch size     : {GPU_BATCH_SIZE:,}")
print(f"Epochs         : {GPU_EPOCHS}")
print(f"Learning rate  : {GPU_LR}")


# =============================================================================
# 1. 데이터 로드
# =============================================================================

test_columns = pd.read_csv(
    os.path.join(DATA_DIR, "test.csv"),
    encoding="utf-8-sig",
    nrows=0,
).columns

FEATURES = [col for col in test_columns if col != ID]

train = pd.read_csv(
    os.path.join(DATA_DIR, "train.csv"),
    encoding="utf-8-sig",
    usecols=FEATURES + [TARGET],
    low_memory=False,
)

print("\n" + "=" * 80)
print("Data")
print("=" * 80)

print(f"Train shape    : {train.shape}")
print(f"Feature count  : {len(FEATURES)}")
print(f"Season         : {train['season'].min()} ~ {train['season'].max()}")
print(f"Target rate    : {train[TARGET].mean():.4f}")


# =============================================================================
# 2. Feature type 정의
# =============================================================================

CAT_COLS = [
    "game_dayofweek",
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_id",
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

classified_cols = set(CAT_COLS) | set(NUM_COLS)

assert set(CAT_COLS).isdisjoint(NUM_COLS)
assert classified_cols == set(FEATURES)


# -----------------------------------------------------------------------------
# 원본 DataFrame 메모리 절약
# -----------------------------------------------------------------------------

def optimize_dataframe_memory(df):
    before_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

    for col in NUM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            ).astype(np.float32)

    # category dtype은 메모리를 크게 줄일 수 있음.
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    df[TARGET] = df[TARGET].astype(np.float32)

    after_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

    print(
        f"DataFrame RAM  : {before_mb:,.1f} MB -> "
        f"{after_mb:,.1f} MB"
    )

    return df


train = optimize_dataframe_memory(train)

print("\n" + "=" * 80)
print("Feature Type")
print("=" * 80)
print(f"Total          : {len(FEATURES)}")
print(f"Categorical    : {len(CAT_COLS)}")
print(f"Numeric        : {len(NUM_COLS)}")


# =============================================================================
# 3. 명백한 중복 Feature 정의
# =============================================================================

DROP_REDUNDANT_COLS = [
    "asof_pitcher_pitchmix_n",
    "run_total_before",
    "num_runners_on",
    "away_win_expectancy",
]


# =============================================================================
# 4. 4개 Logistic Regression 실험군
# =============================================================================

FEATURES_FULL_PID = FEATURES.copy()

FEATURES_FULL_NO_PID = [
    col for col in FEATURES
    if col != PITCHER_ID_COL
]

FEATURES_REDUCED_PID = [
    col for col in FEATURES
    if col not in DROP_REDUNDANT_COLS
]

FEATURES_REDUCED_NO_PID = [
    col for col in FEATURES
    if col not in DROP_REDUNDANT_COLS
    and col != PITCHER_ID_COL
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
print("Experiments")
print("=" * 80)

for name, config in LR_EXPERIMENTS.items():
    print(
        f"{name:<20} "
        f"| Features: {len(config['features']):>2} "
        f"| PID: {config['use_pitcher_id']} "
        f"| Reduced: {config['reduced']}"
    )


# =============================================================================
# 5. Feature type 자동 생성
# =============================================================================

def get_feature_types(feature_list):
    cat_cols = [
        col for col in CAT_COLS
        if col in feature_list
    ]

    num_cols = [
        col for col in NUM_COLS
        if col in feature_list
    ]

    assert len(cat_cols) + len(num_cols) == len(feature_list)

    return cat_cols, num_cols


# =============================================================================
# 6. sklearn preprocessing
# =============================================================================

def build_preprocessor(cat_cols, num_cols):
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
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

        # categorical + numeric을 최종적으로 CSR sparse로 유지
        sparse_threshold=1.0,
        n_jobs=1,
    )

    return preprocessor


# =============================================================================
# 7. scipy sparse -> PyTorch CUDA sparse 변환
# =============================================================================

def scipy_csr_to_torch_sparse(matrix, device):
    """
    scipy CSR matrix를 torch sparse COO tensor로 변환.

    전체 데이터가 아니라 mini-batch 단위로 호출해서
    Xavier의 unified memory 사용량을 제한한다.
    """
    if not sp.isspmatrix_csr(matrix):
        matrix = matrix.tocsr()

    coo = matrix.tocoo(copy=False)

    indices_np = np.vstack(
        (coo.row, coo.col)
    ).astype(np.int64, copy=False)

    indices = torch.from_numpy(indices_np).to(
        device=device,
        non_blocking=False,
    )

    values = torch.from_numpy(
        coo.data.astype(np.float32, copy=False)
    ).to(
        device=device,
        non_blocking=False,
    )

    x = torch.sparse_coo_tensor(
        indices,
        values,
        size=coo.shape,
        dtype=torch.float32,
        device=device,
    )

    return x.coalesce()


# =============================================================================
# 8. PyTorch CUDA Logistic Regression
# =============================================================================

class TorchSparseLogisticRegression:
    """
    L2 Logistic Regression implemented with PyTorch.

    - 입력: scipy CSR sparse matrix
    - 학습: CUDA sparse matrix multiplication
    - optimizer: Adam
    - mini-batch: CSR row blocks

    주의:
    sklearn의 SAGA solver와 optimizer는 다르지만,
    학습하는 모델 자체는 동일한 선형 logistic regression이다.
    """

    def __init__(
        self,
        C=1.0,
        lr=GPU_LR,
        epochs=GPU_EPOCHS,
        batch_size=GPU_BATCH_SIZE,
        patience=GPU_PATIENCE,
        tol=GPU_TOL,
        device=DEVICE,
    ):
        self.C = float(C)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        self.tol = float(tol)
        self.device = torch.device(device)

        self.weight = None
        self.bias = None
        self.n_features_in_ = None

    def fit(self, X, y):
        if not sp.isspmatrix_csr(X):
            X = sp.csr_matrix(X)

        X = X.astype(np.float32, copy=False)
        y = np.asarray(y, dtype=np.float32)

        n_samples, n_features = X.shape

        self.n_features_in_ = n_features

        self.weight = torch.zeros(
            n_features,
            device=self.device,
            dtype=torch.float32,
            requires_grad=True,
        )

        self.bias = torch.zeros(
            1,
            device=self.device,
            dtype=torch.float32,
            requires_grad=True,
        )

        optimizer = torch.optim.Adam(
            [self.weight, self.bias],
            lr=self.lr,
        )

        # sklearn LogisticRegression L2 objective에 가깝게
        # sample 수에 따라 regularization scale을 조정.
        l2_strength = 1.0 / (
            max(self.C, 1e-12) * n_samples
        )

        n_batches = (
            n_samples + self.batch_size - 1
        ) // self.batch_size

        best_loss = np.inf
        bad_epochs = 0

        print(
            f"Encoded shape  : "
            f"{n_samples:,} x {n_features:,}"
        )
        print(
            f"Encoded nnz    : {X.nnz:,}"
        )
        print(
            f"Encoded RAM    : "
            f"{(X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / 1024**2:,.1f} MB"
        )

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # Row-block의 순서만 shuffle.
            # 전체 sparse matrix fancy-indexing보다 훨씬 저렴하다.
            blocks = list(range(n_batches))
            rng = np.random.default_rng(
                RANDOM_STATE + epoch
            )
            rng.shuffle(blocks)

            total_data_loss = 0.0
            total_seen = 0

            for block_idx in blocks:
                start = block_idx * self.batch_size
                end = min(
                    start + self.batch_size,
                    n_samples,
                )

                X_batch = X[start:end]
                y_batch_np = y[start:end]

                x_t = scipy_csr_to_torch_sparse(
                    X_batch,
                    self.device,
                )

                y_t = torch.from_numpy(
                    y_batch_np
                ).to(
                    self.device,
                    dtype=torch.float32,
                )

                optimizer.zero_grad(set_to_none=True)

                logits = torch.sparse.mm(
                    x_t,
                    self.weight.unsqueeze(1),
                ).squeeze(1)

                logits = logits + self.bias

                data_loss = F.binary_cross_entropy_with_logits(
                    logits,
                    y_t,
                    reduction="mean",
                )

                l2_loss = (
                    0.5
                    * l2_strength
                    * torch.sum(self.weight * self.weight)
                )

                loss = data_loss + l2_loss

                loss.backward()
                optimizer.step()

                batch_n = end - start
                total_data_loss += (
                    float(data_loss.detach().cpu())
                    * batch_n
                )
                total_seen += batch_n

                del x_t, y_t, logits, data_loss, l2_loss, loss

            epoch_loss = (
                total_data_loss / max(total_seen, 1)
            )

            elapsed = time.time() - epoch_start

            if torch.cuda.is_available():
                allocated_mb = (
                    torch.cuda.memory_allocated()
                    / 1024**2
                )
            else:
                allocated_mb = 0.0

            print(
                f"Epoch {epoch:03d}/{self.epochs} "
                f"| BCE={epoch_loss:.6f} "
                f"| {elapsed:.1f}s "
                f"| CUDA alloc={allocated_mb:.0f} MB"
            )

            # Early stopping은 학습 BCE의 개선량 기준.
            improvement = best_loss - epoch_loss

            if improvement > self.tol:
                best_loss = epoch_loss
                bad_epochs = 0
            else:
                bad_epochs += 1

            if bad_epochs >= self.patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(best BCE={best_loss:.6f})"
                )
                break

        return self

    @torch.no_grad()
    def predict_proba(self, X):
        if not sp.isspmatrix_csr(X):
            X = sp.csr_matrix(X)

        X = X.astype(np.float32, copy=False)

        n_samples = X.shape[0]
        probabilities = np.empty(
            n_samples,
            dtype=np.float32,
        )

        self.weight.requires_grad_(False)
        self.bias.requires_grad_(False)

        for start in range(
            0,
            n_samples,
            PRED_BATCH_SIZE,
        ):
            end = min(
                start + PRED_BATCH_SIZE,
                n_samples,
            )

            x_t = scipy_csr_to_torch_sparse(
                X[start:end],
                self.device,
            )

            logits = torch.sparse.mm(
                x_t,
                self.weight.unsqueeze(1),
            ).squeeze(1)

            logits = logits + self.bias

            probs = torch.sigmoid(logits)

            probabilities[start:end] = (
                probs.cpu().numpy()
            )

            del x_t, logits, probs

        return np.column_stack(
            (
                1.0 - probabilities,
                probabilities,
            )
        )


# =============================================================================
# 9. sklearn preprocessing + GPU model wrapper
# =============================================================================

class GPULogisticPipeline:
    """
    sklearn preprocessor + PyTorch CUDA Logistic Regression.

    원본 pipeline과 유사하게:
        fit(X, y)
        predict_proba(X)
    를 제공한다.
    """

    def __init__(
        self,
        cat_cols,
        num_cols,
        C=1.0,
    ):
        self.preprocessor = build_preprocessor(
            cat_cols,
            num_cols,
        )

        self.model = TorchSparseLogisticRegression(
            C=C,
        )

    def fit(self, X, y):
        print("Fitting preprocessing...")

        X_encoded = self.preprocessor.fit_transform(
            X
        )

        if not sp.issparse(X_encoded):
            X_encoded = sp.csr_matrix(
                X_encoded,
                dtype=np.float32,
            )
        else:
            X_encoded = X_encoded.tocsr().astype(
                np.float32,
                copy=False,
            )

        print("Training CUDA Logistic Regression...")

        self.model.fit(
            X_encoded,
            np.asarray(y, dtype=np.float32),
        )

        del X_encoded
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self

    def predict_proba(self, X):
        X_encoded = self.preprocessor.transform(X)

        if not sp.issparse(X_encoded):
            X_encoded = sp.csr_matrix(
                X_encoded,
                dtype=np.float32,
            )
        else:
            X_encoded = X_encoded.tocsr().astype(
                np.float32,
                copy=False,
            )

        probs = self.model.predict_proba(
            X_encoded
        )

        del X_encoded
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return probs


# =============================================================================
# 10. Time-based Validation Split
# =============================================================================

train_mask = train["season"] <= TRAIN_END_SEASON
valid_mask = train["season"] == VALID_SEASON

# reset_index를 하지 않아 불필요한 대형 복사를 피한다.
train_fold = train.loc[train_mask]
valid_fold = train.loc[valid_mask]

print("\n" + "=" * 80)
print("Validation Split")
print("=" * 80)

print(
    "Train seasons :",
    sorted(train_fold["season"].unique()),
)
print(
    "Valid seasons :",
    sorted(valid_fold["season"].unique()),
)
print(f"Train rows     : {len(train_fold):,}")
print(f"Valid rows     : {len(valid_fold):,}")
print(
    f"Train rate     : "
    f"{train_fold[TARGET].mean():.4f}"
)
print(
    f"Valid rate     : "
    f"{valid_fold[TARGET].mean():.4f}"
)


# =============================================================================
# 11. Known / Unknown Pitcher
# =============================================================================

train_pitchers = set(
    train_fold[PITCHER_ID_COL]
    .dropna()
    .unique()
)

known_mask = valid_fold[
    PITCHER_ID_COL
].isin(train_pitchers)

unknown_mask = ~known_mask

print("\n" + "=" * 80)
print("Known / Unknown Pitcher")
print("=" * 80)

print(
    f"Known rows     : "
    f"{known_mask.sum():,} "
    f"({known_mask.mean():.2%})"
)
print(
    f"Unknown rows   : "
    f"{unknown_mask.sum():,} "
    f"({unknown_mask.mean():.2%})"
)
print(
    f"Known pitchers : "
    f"{valid_fold.loc[known_mask, PITCHER_ID_COL].nunique():,}"
)
print(
    f"Unknown pitchers: "
    f"{valid_fold.loc[unknown_mask, PITCHER_ID_COL].nunique():,}"
)


# =============================================================================
# 12. 평가 함수
# =============================================================================

def evaluate_predictions(y_true, y_prob):
    if len(y_true) == 0:
        return {
            "brier": np.nan,
            "logloss": np.nan,
            "auc": np.nan,
        }

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # 극단 확률 때문에 log_loss에서 문제가 생기는 것을 방지.
    y_prob = np.clip(
        y_prob,
        1e-7,
        1.0 - 1e-7,
    )

    metrics = {
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
        metrics["auc"] = roc_auc_score(
            y_true,
            y_prob,
        )
    else:
        metrics["auc"] = np.nan

    return metrics


# =============================================================================
# 13. 4개 GPU Logistic Regression 학습
# =============================================================================

results = []
trained_models = {}

known_idx = known_mask.to_numpy()
unknown_idx = unknown_mask.to_numpy()

for experiment_name, config in LR_EXPERIMENTS.items():

    print("\n" + "=" * 80)
    print(f"Training: {experiment_name}")
    print("=" * 80)

    start_time = time.time()

    feature_list = config["features"]

    cat_cols, num_cols = get_feature_types(
        feature_list
    )

    print(f"Features       : {len(feature_list)}")
    print(f"Categorical    : {len(cat_cols)}")
    print(f"Numeric        : {len(num_cols)}")

    X_train = train_fold[feature_list]
    y_train = train_fold[TARGET]

    X_valid = valid_fold[feature_list]
    y_valid = valid_fold[TARGET]

    pipeline = GPULogisticPipeline(
        cat_cols=cat_cols,
        num_cols=num_cols,
        C=1.0,
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    valid_prob = (
        pipeline.predict_proba(X_valid)[:, 1]
    )

    overall_metrics = evaluate_predictions(
        y_valid,
        valid_prob,
    )

    known_metrics = evaluate_predictions(
        y_valid.iloc[np.where(known_idx)[0]],
        valid_prob[known_idx],
    )

    unknown_metrics = evaluate_predictions(
        y_valid.iloc[np.where(unknown_idx)[0]],
        valid_prob[unknown_idx],
    )

    elapsed_seconds = (
        time.time() - start_time
    )

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

        "train_seconds": elapsed_seconds,
    }

    results.append(result)
    trained_models[experiment_name] = pipeline

    print()
    print(
        f"Overall Brier  : "
        f"{overall_metrics['brier']:.6f}"
    )
    print(
        f"Known Brier    : "
        f"{known_metrics['brier']:.6f}"
    )
    print(
        f"Unknown Brier  : "
        f"{unknown_metrics['brier']:.6f}"
    )
    print(
        f"Overall AUC    : "
        f"{overall_metrics['auc']:.6f}"
    )
    print(
        f"Train time     : "
        f"{elapsed_seconds:.1f} sec"
    )

    del (
        X_train,
        X_valid,
        y_train,
        y_valid,
        valid_prob,
    )

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# 14. 전체 결과 정리
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
    "train_seconds",
]

print("\n" + "=" * 120)
print("GPU Logistic Regression Results")
print("=" * 120)

print(
    results_df[
        DISPLAY_COLS
    ]
    .round(6)
    .to_string(index=False)
)


# =============================================================================
# 15. PID 포함 효과 비교
# =============================================================================

score_table = results_df.set_index("model")
comparison_results = []

for feature_type in ["Full", "Reduced"]:

    pid_model = f"LR-{feature_type}-PID"
    no_pid_model = f"LR-{feature_type}-NoPID"

    comparison_results.append(
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

comparison_df = pd.DataFrame(
    comparison_results
)

print("\n" + "=" * 120)
print("Pitcher ID Effect")
print("=" * 120)

print(
    comparison_df
    .round(6)
    .to_string(index=False)
)

print(
    "\n※ Brier diff < 0 : pitcher_id 포함 모델이 더 좋음"
)
print(
    "※ Brier diff > 0 : pitcher_id 제외 모델이 더 좋음"
)


# =============================================================================
# 16. Full vs Reduced 효과 비교
# =============================================================================

reduction_results = []

for pid_type in ["PID", "NoPID"]:

    full_model = f"LR-Full-{pid_type}"
    reduced_model = f"LR-Reduced-{pid_type}"

    reduction_results.append(
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

reduction_df = pd.DataFrame(
    reduction_results
)

print("\n" + "=" * 120)
print("Redundancy Removal Effect")
print("=" * 120)

print(
    reduction_df
    .round(6)
    .to_string(index=False)
)

print(
    "\n※ Brier diff < 0 : Reduced 모델이 더 좋음"
)
print(
    "※ Brier diff > 0 : Full 모델이 더 좋음"
)
