import os
import warnings

import numpy as np
import pandas as pd


# =============================================================================
# 0. 기본 설정
# =============================================================================

warnings.filterwarnings("ignore")

DATA_DIR = "/content/drive/MyDrive/𝟐𝟎𝟐𝟔/aimers/9기/open/data"
OUTPUT_DIR = "./lr03_ordering_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

ID = "row_id"
TARGET = "control_success"

SEASON_COL = "season"
PITCHER_ID_COL = "pitcher_id"


# =============================================================================
# 1. Chronological Key 정의
#
# 목표:
# season
# -> game_date
# -> game_id
# -> inning
# -> plate appearance
# -> pitch order
#
# 실제 컬럼명이 다를 수 있으므로 아래 alias 중 존재하는 컬럼을 탐색한다.
# =============================================================================

CHRONOLOGICAL_LEVELS = {
    "season": [
        "season",
    ],
    "game_date": [
        "game_date",
        "date",
        "game_dt",
    ],
    "game_id": [
        "game_id",
        "game_pk",
        "game_no",
    ],
    "inning": [
        "inning",
    ],
    "plate_appearance": [
        "plate_appearance",
        "plate_appearance_id",
        "pa_id",
        "pa_number",
        "at_bat_number",
        "atbat_number",
    ],
    "pitch_order": [
        "pitch_order",
        "pitch_number",
        "pitch_no",
        "pitch_seq",
        "pitch_sequence",
    ],
}


# =============================================================================
# 2. 데이터 로드
# =============================================================================

train = pd.read_csv(
    TRAIN_PATH,
    encoding="utf-8-sig",
)

test = pd.read_csv(
    TEST_PATH,
    encoding="utf-8-sig",
)


print("=" * 80)
print("1. DATA")
print("=" * 80)

print(f"Train shape : {train.shape}")
print(f"Test shape  : {test.shape}")

print(
    f"Train season: "
    f"{train[SEASON_COL].min()} ~ {train[SEASON_COL].max()}"
)

print(
    f"Test season : "
    f"{test[SEASON_COL].min()} ~ {test[SEASON_COL].max()}"
)


# =============================================================================
# 3. 사용 가능한 시간 관련 컬럼 확인
# =============================================================================

print("\n" + "=" * 80)
print("2. TIME-RELATED COLUMN INVENTORY")
print("=" * 80)


TIME_KEYWORDS = [
    "season",
    "date",
    "game",
    "inning",
    "plate",
    "at_bat",
    "atbat",
    "pitch",
    "order",
    "sequence",
    "seq",
]


time_related_cols = [
    col
    for col in train.columns
    if any(
        keyword in col.lower()
        for keyword in TIME_KEYWORDS
    )
]


print("Time-related candidate columns:")

for col in time_related_cols:
    print(f"- {col} ({train[col].dtype})")


time_column_inventory = pd.DataFrame(
    {
        "column": time_related_cols,
        "dtype": [
            str(train[col].dtype)
            for col in time_related_cols
        ],
        "nunique": [
            train[col].nunique(dropna=True)
            for col in time_related_cols
        ],
        "missing_count": [
            train[col].isna().sum()
            for col in time_related_cols
        ],
        "missing_rate": [
            train[col].isna().mean()
            for col in time_related_cols
        ],
    }
)


time_column_inventory.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "time_column_inventory.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


# =============================================================================
# 4. Chronological Level별 실제 컬럼 탐색
# =============================================================================

def resolve_column(df, candidates):
    """
    candidate 목록 중 실제 DataFrame에 존재하는 첫 번째 컬럼을 반환한다.
    없으면 None.
    """

    for col in candidates:
        if col in df.columns:
            return col

    return None


resolved_columns = {
    level: resolve_column(
        train,
        candidates,
    )
    for level, candidates
    in CHRONOLOGICAL_LEVELS.items()
}


print("\n" + "=" * 80)
print("3. CHRONOLOGICAL COLUMN RESOLUTION")
print("=" * 80)


resolution_rows = []

for level, resolved_col in resolved_columns.items():

    exists = (
        resolved_col is not None
    )

    resolution_rows.append(
        {
            "level": level,
            "resolved_column": resolved_col,
            "exists": exists,
        }
    )

    print(
        f"{level:<20}: "
        f"{resolved_col if exists else 'NOT FOUND'}"
    )


resolution_df = pd.DataFrame(
    resolution_rows
)


resolution_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "chronological_column_resolution.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


# =============================================================================
# 5. Chronological Key 생성 가능 여부
# =============================================================================

REQUIRED_LEVELS = [
    "season",
    "game_date",
    "game_id",
    "inning",
    "plate_appearance",
    "pitch_order",
]


missing_levels = [
    level
    for level in REQUIRED_LEVELS
    if resolved_columns[level] is None
]


chronological_key_available = (
    len(missing_levels) == 0
)


print("\n" + "=" * 80)
print("4. CHRONOLOGICAL KEY AVAILABILITY")
print("=" * 80)

print(
    f"Complete chronological key: "
    f"{chronological_key_available}"
)

print(
    f"Missing levels             : "
    f"{missing_levels}"
)


if chronological_key_available:

    CHRONOLOGICAL_KEY = [
        resolved_columns[level]
        for level in REQUIRED_LEVELS
    ]

else:

    CHRONOLOGICAL_KEY = []


print(
    f"Resolved sorting key       : "
    f"{CHRONOLOGICAL_KEY}"
)


# =============================================================================
# 6. game_date dtype 검증
# =============================================================================

print("\n" + "=" * 80)
print("5. GAME DATE VALIDATION")
print("=" * 80)


game_date_col = resolved_columns[
    "game_date"
]


if game_date_col is None:

    game_date_valid = False

    print(
        "game_date 역할을 하는 컬럼이 없습니다."
    )

else:

    parsed_game_date = pd.to_datetime(
        train[game_date_col],
        errors="coerce",
    )

    original_missing = (
        train[game_date_col]
        .isna()
    )

    parse_failed = (
        parsed_game_date.isna()
        & ~original_missing
    )

    parse_failure_count = int(
        parse_failed.sum()
    )

    game_date_valid = (
        parse_failure_count == 0
    )


    print(
        f"Column              : "
        f"{game_date_col}"
    )

    print(
        f"Original dtype      : "
        f"{train[game_date_col].dtype}"
    )

    print(
        f"Missing count       : "
        f"{original_missing.sum():,}"
    )

    print(
        f"Parse failure count : "
        f"{parse_failure_count:,}"
    )

    print(
        f"Datetime convertible: "
        f"{game_date_valid}"
    )


# =============================================================================
# 7. game_id 시즌 내 순서 검증
#
# game_date가 존재할 경우 동일 시즌에서
# game_id와 game_date의 순서가 일관적인지 확인한다.
#
# game_id가 단순 식별자라면 monotonic하지 않아도 오류는 아니다.
# 따라서 chronology 결정은 game_date를 우선한다.
# =============================================================================

print("\n" + "=" * 80)
print("6. GAME ID ORDER VALIDATION")
print("=" * 80)


game_id_col = resolved_columns[
    "game_id"
]


game_id_order_rows = []


if (
    game_id_col is None
    or game_date_col is None
):

    print(
        "game_id 또는 game_date가 없어 "
        "시즌 내 game_id 순서를 검증할 수 없습니다."
    )

else:

    game_order_df = train[
        [
            SEASON_COL,
            game_date_col,
            game_id_col,
        ]
    ].copy()

    game_order_df[
        "_game_date"
    ] = pd.to_datetime(
        game_order_df[
            game_date_col
        ],
        errors="coerce",
    )

    game_order_df = (
        game_order_df[
            [
                SEASON_COL,
                "_game_date",
                game_id_col,
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                SEASON_COL,
                "_game_date",
                game_id_col,
            ]
        )
    )


    for season, season_df in game_order_df.groupby(
        SEASON_COL
    ):

        if pd.api.types.is_numeric_dtype(
            season_df[game_id_col]
        ):

            monotonic = (
                season_df[
                    game_id_col
                ]
                .is_monotonic_increasing
            )

        else:

            monotonic = np.nan


        game_id_order_rows.append(
            {
                "season": season,
                "game_count":
                    season_df[
                        game_id_col
                    ]
                    .nunique(),

                "game_id_monotonic":
                    monotonic,
            }
        )


    game_id_order_summary = pd.DataFrame(
        game_id_order_rows
    )

    print(
        game_id_order_summary
        .to_string(
            index=False,
        )
    )


    game_id_order_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "game_id_order_validation.csv",
        ),
        encoding="utf-8-sig",
        index=False,
    )


# =============================================================================
# 8. Inning 순서 검증
# =============================================================================

print("\n" + "=" * 80)
print("7. INNING VALIDATION")
print("=" * 80)


inning_col = resolved_columns[
    "inning"
]


if inning_col is None:

    inning_valid = False

    print(
        "inning 컬럼이 없습니다."
    )

else:

    inning_missing_count = int(
        train[inning_col]
        .isna()
        .sum()
    )

    inning_nonpositive_count = int(
        (
            train[inning_col]
            .dropna()
            < 1
        )
        .sum()
    )

    inning_valid = (
        inning_missing_count == 0
        and inning_nonpositive_count == 0
    )


    print(
        f"Column         : "
        f"{inning_col}"
    )

    print(
        f"Min / Max      : "
        f"{train[inning_col].min()} / "
        f"{train[inning_col].max()}"
    )

    print(
        f"Missing        : "
        f"{inning_missing_count:,}"
    )

    print(
        f"Value < 1      : "
        f"{inning_nonpositive_count:,}"
    )

    print(
        f"Inning valid   : "
        f"{inning_valid}"
    )


# =============================================================================
# 9. Plate Appearance 순서 컬럼 검증
# =============================================================================

print("\n" + "=" * 80)
print("8. PLATE APPEARANCE VALIDATION")
print("=" * 80)


pa_col = resolved_columns[
    "plate_appearance"
]


if pa_col is None:

    pa_valid = False

    print(
        "Plate Appearance 순서를 표현하는 "
        "컬럼이 없습니다."
    )

else:

    pa_missing_count = int(
        train[pa_col]
        .isna()
        .sum()
    )

    pa_valid = (
        pa_missing_count == 0
    )

    print(
        f"Column       : "
        f"{pa_col}"
    )

    print(
        f"Missing      : "
        f"{pa_missing_count:,}"
    )

    print(
        f"Unique count : "
        f"{train[pa_col].nunique():,}"
    )

    print(
        f"PA valid     : "
        f"{pa_valid}"
    )


# =============================================================================
# 10. 한 타석 내 Pitch Order 검증
# =============================================================================

print("\n" + "=" * 80)
print("9. PITCH ORDER VALIDATION")
print("=" * 80)


pitch_order_col = resolved_columns[
    "pitch_order"
]


if pitch_order_col is None:

    pitch_order_valid = False

    print(
        "한 타석 내 pitch order를 표현하는 "
        "컬럼이 없습니다."
    )

else:

    pitch_order_missing_count = int(
        train[pitch_order_col]
        .isna()
        .sum()
    )

    pitch_order_nonpositive_count = int(
        (
            train[pitch_order_col]
            .dropna()
            < 1
        )
        .sum()
    )

    pitch_order_valid = (
        pitch_order_missing_count == 0
        and pitch_order_nonpositive_count == 0
    )


    print(
        f"Column         : "
        f"{pitch_order_col}"
    )

    print(
        f"Missing        : "
        f"{pitch_order_missing_count:,}"
    )

    print(
        f"Value < 1      : "
        f"{pitch_order_nonpositive_count:,}"
    )

    print(
        f"Pitch order OK : "
        f"{pitch_order_valid}"
    )


# =============================================================================
# 11. 동일 Chronological Key 중복 여부
# =============================================================================

print("\n" + "=" * 80)
print("10. DUPLICATE ORDERING KEY")
print("=" * 80)


if not chronological_key_available:

    duplicate_ordering_rows = np.nan
    duplicate_ordering_keys = np.nan

    print(
        "완전한 chronological key가 없어 "
        "중복 key 검증을 수행할 수 없습니다."
    )

else:

    duplicate_key_mask = (
        train
        .duplicated(
            subset=CHRONOLOGICAL_KEY,
            keep=False,
        )
    )

    duplicate_ordering_rows = int(
        duplicate_key_mask.sum()
    )

    duplicate_ordering_keys = int(
        train.loc[
            duplicate_key_mask,
            CHRONOLOGICAL_KEY,
        ]
        .drop_duplicates()
        .shape[0]
    )


    print(
        f"Duplicate rows : "
        f"{duplicate_ordering_rows:,}"
    )

    print(
        f"Duplicate keys : "
        f"{duplicate_ordering_keys:,}"
    )


    if duplicate_ordering_rows > 0:

        duplicate_ordering_df = (
            train.loc[
                duplicate_key_mask,
                CHRONOLOGICAL_KEY
                + (
                    [ID]
                    if ID in train.columns
                    else []
                ),
            ]
            .sort_values(
                CHRONOLOGICAL_KEY
            )
        )


        duplicate_ordering_df.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "duplicate_ordering_keys.csv",
            ),
            encoding="utf-8-sig",
            index=False,
        )


# =============================================================================
# 12. 정렬 전 원본 Mapping 컬럼 생성
#
# 원본 row 순서를 chronological order로 사용하지 않는다.
# 오직 mapping 복원 목적으로만 저장한다.
# =============================================================================

def add_original_mapping(df):
    """
    정렬 전 원본 위치를 보존한다.

    _original_position은 chronological feature가 아니며,
    정렬 후 원본 row 복구 목적으로만 사용한다.
    """

    result = df.copy()

    result[
        "_original_position"
    ] = np.arange(
        len(result),
        dtype=np.int64,
    )

    return result


# =============================================================================
# 13. 공통 Sorting Function
#
# 핵심:
# - 완전한 chronological key가 없으면 정렬하지 않는다.
# - chronological key 중복이 있으면 strict mode에서 실패한다.
# - row_id / 원본 row 순서를 시간 순서 대용으로 사용하지 않는다.
# =============================================================================

def sort_chronologically(
    df,
    chronological_key,
    strict=True,
):
    """
    명시적 chronological key 기준으로만 정렬한다.

    Parameters
    ----------
    df : pd.DataFrame

    chronological_key : list[str]
        예:
        [
            "season",
            "game_date",
            "game_id",
            "inning",
            "plate_appearance",
            "pitch_order",
        ]

    strict : bool
        True일 경우 ordering key 중복이 있으면 ValueError 발생.

    Returns
    -------
    sorted_df : pd.DataFrame

    Notes
    -----
    row_id와 원본 row index는 chronological ordering을
    추론하는 용도로 사용하지 않는다.

    _original_position은 정렬 후 원본 mapping 복원용이다.
    """

    if not chronological_key:
        raise ValueError(
            "완전한 chronological key가 정의되지 않았습니다. "
            "원본 row 순서 또는 row_id를 시간 순서로 대체하지 않습니다."
        )


    missing_cols = [
        col
        for col in chronological_key
        if col not in df.columns
    ]


    if missing_cols:
        raise KeyError(
            f"Chronological key 컬럼이 없습니다: "
            f"{missing_cols}"
        )


    result = add_original_mapping(
        df
    )


    # -------------------------------------------------------------------------
    # game_date datetime 변환
    # -------------------------------------------------------------------------

    date_col = resolved_columns[
        "game_date"
    ]

    if (
        date_col is not None
        and date_col in result.columns
    ):

        result[
            date_col
        ] = pd.to_datetime(
            result[
                date_col
            ],
            errors="raise",
        )


    # -------------------------------------------------------------------------
    # Key 결측 검증
    # -------------------------------------------------------------------------

    key_missing = (
        result[
            chronological_key
        ]
        .isna()
        .any(
            axis=1,
        )
    )


    key_missing_count = int(
        key_missing.sum()
    )


    if key_missing_count > 0:

        raise ValueError(
            f"Chronological key에 결측값이 있는 row가 "
            f"{key_missing_count:,}개 존재합니다."
        )


    # -------------------------------------------------------------------------
    # Key 중복 검증
    # -------------------------------------------------------------------------

    duplicate_mask = (
        result
        .duplicated(
            subset=chronological_key,
            keep=False,
        )
    )


    duplicate_count = int(
        duplicate_mask.sum()
    )


    if (
        strict
        and duplicate_count > 0
    ):

        raise ValueError(
            f"Chronological key가 중복되는 row가 "
            f"{duplicate_count:,}개 존재합니다. "
            "실제 pitch order를 구분할 추가 컬럼이 필요합니다."
        )


    # -------------------------------------------------------------------------
    # 정렬
    #
    # mergesort:
    # stable sort를 사용하지만 원본 순서를 chronology로 간주하지 않는다.
    # -------------------------------------------------------------------------

    result = (
        result
        .sort_values(
            chronological_key,
            kind="mergesort",
        )
        .reset_index(
            drop=True,
        )
    )


    result[
        "_chronological_position"
    ] = np.arange(
        len(result),
        dtype=np.int64,
    )


    return result


# =============================================================================
# 14. 실제 정렬 수행
# =============================================================================

print("\n" + "=" * 80)
print("11. CHRONOLOGICAL SORT")
print("=" * 80)


sorting_status = "BLOCKED"
sorted_train = None


if not chronological_key_available:

    print(
        "BLOCKED: chronological key 구성에 필요한 컬럼이 부족합니다."
    )

    print(
        f"Missing levels: "
        f"{missing_levels}"
    )

else:

    try:

        sorted_train = sort_chronologically(
            train,
            CHRONOLOGICAL_KEY,
            strict=True,
        )

        sorting_status = "SUCCESS"

        print(
            "Chronological sorting 성공"
        )

        print(
            f"Sorting key: "
            f"{CHRONOLOGICAL_KEY}"
        )

    except (
        ValueError,
        KeyError,
    ) as error:

        sorting_status = "BLOCKED"

        print(
            f"BLOCKED: {error}"
        )


# =============================================================================
# 15. 원본 Row Mapping 보존 검증
# =============================================================================

print("\n" + "=" * 80)
print("12. ORIGINAL ROW MAPPING VALIDATION")
print("=" * 80)


mapping_valid = False


if sorted_train is None:

    print(
        "정렬이 수행되지 않아 mapping 검증을 생략합니다."
    )

else:

    original_positions = (
        sorted_train[
            "_original_position"
        ]
        .sort_values()
        .to_numpy()
    )

    expected_positions = np.arange(
        len(train)
    )


    mapping_valid = np.array_equal(
        original_positions,
        expected_positions,
    )


    print(
        f"Original mapping preserved : "
        f"{mapping_valid}"
    )


    if ID in train.columns:

        row_id_mapping_valid = (
            sorted_train[ID].nunique()
            == train[ID].nunique()
            == len(train)
        )

        print(
            f"row_id mapping preserved   : "
            f"{row_id_mapping_valid}"
        )


    mapping_cols = [
        "_chronological_position",
        "_original_position",
    ]

    if ID in sorted_train.columns:
        mapping_cols.append(ID)

    mapping_cols += CHRONOLOGICAL_KEY


    sorted_train[
        mapping_cols
    ].to_csv(
        os.path.join(
            OUTPUT_DIR,
            "chronological_mapping.csv",
        ),
        encoding="utf-8-sig",
        index=False,
    )


# =============================================================================
# 16. 2019 -> 2024 시즌 순서 검증
# =============================================================================

print("\n" + "=" * 80)
print("13. SEASON ORDER VALIDATION")
print("=" * 80)


season_order_valid = False


if sorted_train is None:

    print(
        "정렬이 수행되지 않아 시즌 순서 검증을 생략합니다."
    )

else:

    season_sequence = (
        sorted_train[
            SEASON_COL
        ]
        .drop_duplicates()
        .tolist()
    )

    expected_season_sequence = sorted(
        train[
            SEASON_COL
        ]
        .dropna()
        .unique()
        .tolist()
    )


    season_order_valid = (
        season_sequence
        == expected_season_sequence
    )


    print(
        f"Observed season order : "
        f"{season_sequence}"
    )

    print(
        f"Expected season order : "
        f"{expected_season_sequence}"
    )

    print(
        f"Season order valid    : "
        f"{season_order_valid}"
    )


# =============================================================================
# 17. Pitcher별 Chronological Ordering 검증
#
# 전체 chronological key로 정렬한 이후
# 각 pitcher의 key가 단조 증가하는지 확인한다.
# =============================================================================

print("\n" + "=" * 80)
print("14. PITCHER CHRONOLOGICAL ORDER VALIDATION")
print("=" * 80)


pitcher_order_violations = []


if sorted_train is None:

    print(
        "정렬이 수행되지 않아 pitcher별 검증을 생략합니다."
    )

else:

    for pitcher_id, pitcher_df in sorted_train.groupby(
        PITCHER_ID_COL,
        sort=False,
    ):

        key_tuples = list(
            pitcher_df[
                CHRONOLOGICAL_KEY
            ]
            .itertuples(
                index=False,
                name=None,
            )
        )

        is_ordered = all(
            key_tuples[i]
            <= key_tuples[i + 1]
            for i in range(
                len(key_tuples) - 1
            )
        )


        if not is_ordered:

            pitcher_order_violations.append(
                {
                    "pitcher_id":
                        pitcher_id,

                    "row_count":
                        len(pitcher_df),
                }
            )


    pitcher_order_violation_df = pd.DataFrame(
        pitcher_order_violations
    )


    print(
        f"Pitchers checked   : "
        f"{sorted_train[PITCHER_ID_COL].nunique():,}"
    )

    print(
        f"Order violations   : "
        f"{len(pitcher_order_violations):,}"
    )


    pitcher_order_violation_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "pitcher_order_violations.csv",
        ),
        encoding="utf-8-sig",
        index=False,
    )


# =============================================================================
# 18. 과거 / 미래 구분 가능성 검증
#
# 정렬이 정상 완료된 경우 동일 pitcher 안에서
# _chronological_position 기준으로 이전 row와 이후 row를
# 명확히 구분할 수 있다.
# =============================================================================

print("\n" + "=" * 80)
print("15. PAST / FUTURE SEPARATION")
print("=" * 80)


past_future_separable = False


if sorted_train is None:

    print(
        "BLOCKED: chronological ordering이 확정되지 않아 "
        "pitcher별 과거/미래를 안전하게 구분할 수 없습니다."
    )

else:

    pitcher_order_violation_count = len(
        pitcher_order_violations
    )

    past_future_separable = (
        pitcher_order_violation_count == 0
        and mapping_valid
        and season_order_valid
    )


    print(
        f"Past/Future separable : "
        f"{past_future_separable}"
    )


# =============================================================================
# 19. 최종 Validation Summary
# =============================================================================

print("\n" + "=" * 80)
print("16. FINAL ORDERING VALIDATION SUMMARY")
print("=" * 80)


summary = {
    "complete_chronological_key":
        chronological_key_available,

    "missing_key_levels":
        ", ".join(missing_levels),

    "resolved_sorting_key":
        ", ".join(CHRONOLOGICAL_KEY),

    "game_date_valid":
        game_date_valid,

    "duplicate_ordering_rows":
        duplicate_ordering_rows,

    "sorting_status":
        sorting_status,

    "mapping_preserved":
        mapping_valid,

    "season_order_valid":
        season_order_valid,

    "past_future_separable":
        past_future_separable,
}


summary_df = pd.DataFrame(
    list(
        summary.items()
    ),
    columns=[
        "item",
        "value",
    ],
)


print(
    summary_df
    .to_string(
        index=False,
    )
)


summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ordering_validation_summary.csv",
    ),
    encoding="utf-8-sig",
    index=False,
)


# =============================================================================
# 20. 최종 판정
# =============================================================================

print("\n" + "=" * 80)
print("17. LR-03 FINAL STATUS")
print("=" * 80)


if (
    chronological_key_available
    and sorting_status == "SUCCESS"
    and past_future_separable
):

    print(
        "SUCCESS"
    )

    print(
        "모든 투구에 대해 deterministic chronological order를 "
        "정의할 수 있습니다."
    )

    print(
        f"Final sorting key: "
        f"{CHRONOLOGICAL_KEY}"
    )

else:

    print(
        "BLOCKED"
    )

    print(
        "현재 데이터만으로 정확한 chronological order를 "
        "확정할 수 없습니다."
    )

    print(
        "원본 row 순서 또는 row_id를 시간 순서로 대신 사용하지 않습니다."
    )

    if missing_levels:

        print(
            f"추가로 필요한 ordering 정보: "
            f"{missing_levels}"
        )


print("\n" + "=" * 80)
print("LR-03 VALIDATION COMPLETE")
print("=" * 80)

print(
    f"산출물 저장 위치: "
    f"{OUTPUT_DIR}"
)