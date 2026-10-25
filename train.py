"""Обучение ансамбля CatBoost для прогноза дефицита курьеров.

Скрипт подходит для локального запуска, Kaggle и Jupyter. Он самостоятельно
находит исходные CSV, строит временные признаки, проводит валидацию и сохраняет
готовый ``submission.csv`` вместе с диагностическими таблицами.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

TEST_DATE = pd.Timestamp("2025-11-24")
VALIDATION_DATES = [pd.Timestamp("2025-11-10"), pd.Timestamp("2025-11-17")]
SEEDS = (13, 42, 73, 101, 202)
REQUIRED_DATA_FILES = {
    "train.csv",
    "test.csv",
    "facts.csv",
    "shifts_prediction.csv",
}
PREDICTION_COLUMNS = (
    "predicted_staff_value",
    "predicted_num_orders",
    "predicted_load_factor",
)
FACT_HISTORY_COLUMNS = (
    "fact_staff_value_lag_1",
    "fact_load_factor_lag_1",
    "num_available_couriers_lag_1",
    "fact_num_orders_lag_1",
    "fact_percent_lateness_lag_1",
    "fact_staff_churn",
    "flag_high_load_lag_1",
    "fact_couriers_with_shifts_lag_1",
)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Делит два ряда, заменяя нулевой знаменатель на пропуск."""
    return numerator / denominator.replace(0, np.nan)


def add_exact_lags(
    frame: pd.DataFrame,
    value_columns: list[str],
    lags: tuple[int, ...],
    prefix: str,
) -> pd.DataFrame:
    """Добавляет точные календарные лаги для каждого даркстора."""
    result = frame.copy()
    keys = ["calendar_dt", "store_id"]
    base = frame[keys + value_columns]
    for lag in lags:
        shifted = base.copy()
        shifted["calendar_dt"] = shifted["calendar_dt"] + pd.Timedelta(weeks=lag)
        shifted = shifted.rename(
            columns={column: f"{prefix}{column}_lag_{lag}" for column in value_columns}
        )
        result = result.merge(shifted, on=keys, how="left", validate="one_to_one")
    return result


def build_prediction_features(predictions: pd.DataFrame) -> pd.DataFrame:
    """Строит лаги и динамические признаки прогнозов внешней системы."""
    values = list(PREDICTION_COLUMNS)
    result = add_exact_lags(predictions, values, (1, 2, 3, 4), "history_")

    for column in values:
        lag_1 = f"history_{column}_lag_1"
        result[f"{column}_diff_1"] = result[column] - result[lag_1]
        result[f"{column}_ratio_1"] = safe_divide(result[column], result[lag_1])
        history = [result[column]] + [
            result[f"history_{column}_lag_{lag}"] for lag in (1, 2, 3)
        ]
        result[f"{column}_mean_4"] = pd.concat(history, axis=1).mean(axis=1)
        result[f"{column}_std_4"] = pd.concat(history, axis=1).std(axis=1)
    return result


def build_fact_features(facts: pd.DataFrame) -> pd.DataFrame:
    """Строит историю и производные признаки по фактическим данным."""
    laggable = list(FACT_HISTORY_COLUMNS)
    result = add_exact_lags(facts, laggable, (1,), "history_")
    result["predicted_staff_gap_available"] = np.nan
    result["fact_staff_gap_available"] = (
        result["fact_staff_value_lag_1"] - result["num_available_couriers_lag_1"]
    )
    result["fact_staff_gap_with_shifts"] = (
        result["fact_staff_value_lag_1"]
        - result["fact_couriers_with_shifts_lag_1"]
    )
    result["shift_participation_rate"] = safe_divide(
        result["fact_couriers_with_shifts_lag_1"],
        result["num_available_couriers_lag_1"],
    )
    result["orders_per_available_courier"] = safe_divide(
        result["fact_num_orders_lag_1"], result["num_available_couriers_lag_1"]
    )
    for column in laggable:
        history_column = f"history_{column}_lag_1"
        result[f"{column}_diff_1"] = result[column] - result[history_column]
    return result


def attach_target_lags(rows: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Добавляет значения target за одну и две предыдущие недели."""
    result = rows.copy()
    keys = ["calendar_dt", "store_id"]
    target_history = train[keys + ["target"]]
    for lag in (1, 2):
        shifted = target_history.copy()
        shifted["calendar_dt"] = shifted["calendar_dt"] + pd.Timedelta(weeks=lag)
        shifted = shifted.rename(columns={"target": f"target_lag_{lag}"})
        result = result.merge(shifted, on=keys, how="left", validate="one_to_one")
    result["target_lag_mean_2"] = result[["target_lag_1", "target_lag_2"]].mean(
        axis=1
    )
    result["target_lag_diff_1"] = result["target_lag_1"] - result["target_lag_2"]
    return result


def contains_required_files(data_dir: Path) -> bool:
    """Проверяет наличие всех исходных CSV в указанном каталоге."""
    return all((data_dir / filename).is_file() for filename in REQUIRED_DATA_FILES)


def resolve_data_dir(requested_dir: Path | None) -> Path:
    """Находит данные локально или внутри подключённого Kaggle Dataset."""
    if requested_dir is not None:
        data_dir = requested_dir.expanduser().resolve()
        if not contains_required_files(data_dir):
            missing = sorted(
                filename
                for filename in REQUIRED_DATA_FILES
                if not (data_dir / filename).is_file()
            )
            raise FileNotFoundError(
                f"В каталоге {data_dir} не найдены файлы: {missing}"
            )
        return data_dir

    current_dir = Path.cwd().resolve()
    if contains_required_files(current_dir):
        return current_dir

    kaggle_input = Path("/kaggle/input")
    candidates: set[Path] = set()
    if kaggle_input.is_dir():
        for train_path in kaggle_input.rglob("train.csv"):
            if contains_required_files(train_path.parent):
                candidates.add(train_path.parent.resolve())

    if len(candidates) == 1:
        return candidates.pop()
    if len(candidates) > 1:
        options = "\n".join(f"- {path}" for path in sorted(candidates))
        raise RuntimeError(
            "Найдено несколько наборов данных. Укажите нужный через "
            f"--data-dir:\n{options}"
        )
    raise FileNotFoundError(
        "Не найдены train.csv, test.csv, facts.csv и shifts_prediction.csv. "
        "Добавьте Dataset в Kaggle или укажите каталог через --data-dir."
    )


def default_output_dir() -> Path:
    """Выбирает стандартный каталог результатов для Kaggle или локального запуска."""
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.is_dir():
        return kaggle_working
    return Path("outputs") / "seed_ensemble_with_flag"


def load_dataset(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Загружает исходные CSV и формирует строки train и test с признаками."""
    train = pd.read_csv(data_dir / "train.csv", parse_dates=["calendar_dt"])
    facts = pd.read_csv(data_dir / "facts.csv", parse_dates=["calendar_dt"])
    predictions = pd.read_csv(
        data_dir / "shifts_prediction.csv", parse_dates=["calendar_dt"]
    )
    test = pd.read_csv(data_dir / "test.csv")

    for name, frame in {
        "train": train,
        "facts": facts,
        "shifts_prediction": predictions,
    }.items():
        if frame.duplicated(["calendar_dt", "store_id"]).any():
            raise ValueError(f"В {name} найдены повторяющиеся calendar_dt/store_id")
    if test["store_id"].duplicated().any():
        raise ValueError("В test найдены повторяющиеся store_id")

    prediction_features = build_prediction_features(predictions)
    fact_features = build_fact_features(facts)
    feature_table = fact_features.merge(
        prediction_features,
        on=["calendar_dt", "store_id"],
        how="left",
        validate="one_to_one",
    )
    feature_table["predicted_staff_gap_available"] = (
        feature_table["predicted_staff_value"]
        - feature_table["num_available_couriers_lag_1"]
    )
    feature_table["predicted_staff_gap_with_shifts"] = (
        feature_table["predicted_staff_value"]
        - feature_table["fact_couriers_with_shifts_lag_1"]
    )
    feature_table["predicted_orders_growth_vs_fact"] = safe_divide(
        feature_table["predicted_num_orders"], feature_table["fact_num_orders_lag_1"]
    )
    feature_table["calendar_week"] = (
        feature_table["calendar_dt"].dt.isocalendar().week.astype(int)
    )
    feature_table["calendar_month"] = feature_table["calendar_dt"].dt.month

    train_rows = train.merge(
        feature_table,
        on=["calendar_dt", "store_id"],
        how="left",
        validate="one_to_one",
    )
    test_rows = test.assign(calendar_dt=TEST_DATE).merge(
        feature_table,
        on=["calendar_dt", "store_id"],
        how="left",
        validate="one_to_one",
    )
    train_rows = attach_target_lags(train_rows, train)
    test_rows = attach_target_lags(test_rows, train)

    if train_rows["city_nm"].isna().any() or test_rows["city_nm"].isna().any():
        raise ValueError("В facts отсутствует часть строк из train или test")
    if test_rows["predicted_staff_value"].isna().any():
        raise ValueError("В shifts_prediction отсутствует часть строк из test")
    return train_rows, test_rows


def prepare_features(
    train_rows: pd.DataFrame, test_rows: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Отделяет признаки от ключей и подготавливает категориальные колонки."""
    excluded = {"target", "calendar_dt", "store_id"}
    feature_columns = [column for column in train_rows.columns if column not in excluded]
    categorical_columns = ["city_nm"]
    train_features = train_rows[feature_columns].copy()
    test_features = test_rows[feature_columns].copy()
    for column in categorical_columns:
        train_features[column] = train_features[column].fillna("missing").astype(str)
        test_features[column] = test_features[column].fillna("missing").astype(str)
    return train_features, test_features, feature_columns, categorical_columns


def wape_parts(
    y_true: np.ndarray, prediction: np.ndarray
) -> tuple[float, float, float]:
    """Возвращает WAPE, сумму абсолютных ошибок и сумму целевой переменной."""
    absolute_error_sum = float(np.abs(y_true - prediction).sum())
    target_sum = float(y_true.sum())
    return absolute_error_sum / target_sum, absolute_error_sum, target_sum


def new_model(seed: int) -> CatBoostRegressor:
    """Создаёт CatBoost с зафиксированными параметрами эксперимента."""
    return CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=300,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=6.0,
        random_seed=seed,
        allow_writing_files=False,
        verbose=False,
        thread_count=-1,
    )


def fit_seed_models(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    prediction_features: pd.DataFrame,
    categorical_columns: list[str],
) -> tuple[np.ndarray, list[CatBoostRegressor]]:
    """Обучает модели для всех seed и возвращает матрицу прогнозов."""
    categorical_indices = [
        train_features.columns.get_loc(column) for column in categorical_columns
    ]
    predictions: list[np.ndarray] = []
    models: list[CatBoostRegressor] = []
    for seed in SEEDS:
        print(
            f"Обучаем seed={seed} на {len(train_features):,} строках",
            flush=True,
        )
        model = new_model(seed)
        model.fit(
            train_features,
            train_target,
            cat_features=categorical_indices,
        )
        predictions.append(model.predict(prediction_features).clip(0))
        models.append(model)
    return np.column_stack(predictions), models


def prediction_candidates(
    seed_predictions: np.ndarray,
    flag_high_load_lag_1: np.ndarray,
) -> dict[str, np.ndarray]:
    """Создаёт одиночные и ансамблевые варианты постобработки прогноза."""
    candidates: dict[str, np.ndarray] = {}

    def add_candidate(name: str, prediction: np.ndarray) -> None:
        """Добавляет исходную, округлённую и gated-версии прогноза."""
        rounded_prediction = np.rint(prediction).clip(0)
        flag_is_zero = flag_high_load_lag_1 == 0
        candidates[name] = prediction
        candidates[f"{name}_rounded"] = rounded_prediction
        candidates[f"{name}_gated"] = np.where(flag_is_zero, 0, prediction)
        candidates[f"{name}_gated_rounded"] = np.where(
            flag_is_zero, 0, rounded_prediction
        )

    for position, seed in enumerate(SEEDS):
        prediction = seed_predictions[:, position]
        add_candidate(f"seed_{seed}", prediction)

    mean_prediction = seed_predictions.mean(axis=1)
    median_prediction = np.median(seed_predictions, axis=1)
    add_candidate("ensemble_mean", mean_prediction)
    add_candidate("ensemble_median", median_prediction)
    return candidates


def baseline_candidates(features: pd.DataFrame) -> dict[str, np.ndarray]:
    """Формирует простые ориентиры для сравнения на валидации."""
    churn = features["fact_staff_churn"].fillna(0).to_numpy().clip(0)
    last_target = features["target_lag_1"].fillna(
        features["fact_staff_churn"].fillna(0)
    ).to_numpy().clip(0)
    return {
        "baseline_churn": churn,
        "baseline_last_target": last_target,
    }


def validate(
    rows: pd.DataFrame,
    features: pd.DataFrame,
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, str]:
    """Проводит expanding-window валидацию и выбирает лучший ансамбль."""
    records: list[dict[str, object]] = []
    ensemble_names = {
        "ensemble_mean",
        "ensemble_mean_rounded",
        "ensemble_mean_gated",
        "ensemble_mean_gated_rounded",
        "ensemble_median",
        "ensemble_median_rounded",
        "ensemble_median_gated",
        "ensemble_median_gated_rounded",
    }

    for validation_date in VALIDATION_DATES:
        train_mask = rows["calendar_dt"] < validation_date
        validation_mask = rows["calendar_dt"] == validation_date
        x_train = features.loc[train_mask]
        y_train = rows.loc[train_mask, "target"]
        x_validation = features.loc[validation_mask]
        y_validation = rows.loc[validation_mask, "target"].to_numpy()

        print(f"Дата валидации: {validation_date.date()}", flush=True)
        seed_predictions, _ = fit_seed_models(
            x_train,
            y_train,
            x_validation,
            categorical_columns,
        )
        candidates = {
            **baseline_candidates(x_validation),
            **prediction_candidates(
                seed_predictions,
                x_validation["flag_high_load_lag_1"].to_numpy(),
            ),
        }
        for model_name, prediction in candidates.items():
            wape, absolute_error_sum, target_sum = wape_parts(
                y_validation, prediction
            )
            records.append(
                {
                    "validation_date": validation_date.date().isoformat(),
                    "model": model_name,
                    "wape": wape,
                    "absolute_error_sum": absolute_error_sum,
                    "target_sum": target_sum,
                    "n_rows": int(validation_mask.sum()),
                }
            )

    results = pd.DataFrame(records)
    pooled = (
        results.groupby("model", as_index=False)[
            ["absolute_error_sum", "target_sum"]
        ]
        .sum()
        .assign(pooled_wape=lambda frame: frame.absolute_error_sum / frame.target_sum)
    )
    results = results.merge(
        pooled[["model", "pooled_wape"]], on="model", how="left"
    ).sort_values(["pooled_wape", "validation_date"])
    ensemble_results = pooled[pooled["model"].isin(ensemble_names)].copy()
    ungated_results = ensemble_results[
        ~ensemble_results["model"].str.contains("_gated")
    ].sort_values(["pooled_wape", "model"])
    gated_results = ensemble_results[
        ensemble_results["model"].str.contains("_gated")
    ].sort_values(["pooled_wape", "model"])
    best_ungated = ungated_results.iloc[0]
    best_gated = gated_results.iloc[0]
    if best_gated["pooled_wape"] < best_ungated["pooled_wape"] - 1e-12:
        best_ensemble = str(best_gated["model"])
    else:
        best_ensemble = str(best_ungated["model"])
    return results, best_ensemble


def save_feature_importance(
    models: list[CatBoostRegressor],
    feature_names: pd.Index,
    output_path: Path,
) -> None:
    """Сохраняет среднюю важность признаков и её разброс по seed."""
    matrix = np.column_stack([model.get_feature_importance() for model in models])
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": matrix.mean(axis=1),
            "importance_std": matrix.std(axis=1),
            **{
                f"importance_seed_{seed}": matrix[:, position]
                for position, seed in enumerate(SEEDS)
            },
        }
    ).sort_values("importance_mean", ascending=False)
    importance.to_csv(output_path, index=False)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Читает пользовательские аргументы и игнорирует служебные аргументы Jupyter."""
    parser = argparse.ArgumentParser(
        description="Обучение ансамбля для прогноза дефицита курьеров."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Каталог с исходными CSV; в Kaggle определяется автоматически.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Каталог результатов; по умолчанию /kaggle/working в Kaggle.",
    )
    args, unknown_args = parser.parse_known_args(argv)
    if unknown_args:
        print(
            f"Игнорируем служебные аргументы Jupyter: {unknown_args}",
            flush=True,
        )
    args.data_dir = resolve_data_dir(args.data_dir)
    args.output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else default_output_dir().resolve()
    )
    return args


def main(argv: list[str] | None = None) -> None:
    """Запускает валидацию, финальное обучение и сохранение результатов."""
    args = parse_arguments(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Каталог данных: {args.data_dir}", flush=True)
    print(f"Каталог результатов: {args.output_dir}", flush=True)

    train_rows, test_rows = load_dataset(args.data_dir)
    train_features, test_features, _, categorical_columns = prepare_features(
        train_rows, test_rows
    )
    required_flag_features = {
        "flag_high_load_lag_1",
        "history_flag_high_load_lag_1_lag_1",
        "flag_high_load_lag_1_diff_1",
    }
    missing_flag_features = required_flag_features.difference(train_features.columns)
    if missing_flag_features:
        raise ValueError(
            f"Не найдены признаки флага: {sorted(missing_flag_features)}"
        )
    print(
        f"Признаки флага включены в модель: {sorted(required_flag_features)}",
        flush=True,
    )
    print(
        "Сравниваем обычный прогноз с правилом flag=0 -> prediction=0",
        flush=True,
    )

    validation_results, selected_ensemble = validate(
        train_rows, train_features, categorical_columns
    )
    validation_results.to_csv(
        args.output_dir / "validation_results.csv", index=False
    )

    print("Обучаем финальный ансамбль", flush=True)
    test_seed_predictions, final_models = fit_seed_models(
        train_features,
        train_rows["target"],
        test_features,
        categorical_columns,
    )
    test_flag = test_features["flag_high_load_lag_1"].to_numpy()
    test_candidates = prediction_candidates(test_seed_predictions, test_flag)
    selected_prediction = test_candidates[selected_ensemble].clip(0)

    seed_output = test_rows[["store_id"]].copy()
    for position, seed in enumerate(SEEDS):
        seed_output[f"prediction_seed_{seed}"] = test_seed_predictions[:, position]
    seed_output["ensemble_mean"] = test_seed_predictions.mean(axis=1)
    seed_output["ensemble_median"] = np.median(test_seed_predictions, axis=1)
    seed_output["ensemble_mean_gated"] = test_candidates["ensemble_mean_gated"]
    seed_output["ensemble_median_gated"] = test_candidates[
        "ensemble_median_gated"
    ]
    seed_output["selected_prediction"] = selected_prediction
    seed_output = seed_output.sort_values("store_id").reset_index(drop=True)
    seed_output.to_csv(args.output_dir / "test_seed_predictions.csv", index=False)

    submission = seed_output[["store_id"]].copy()
    submission["target"] = seed_output["selected_prediction"]
    if selected_ensemble.endswith("_rounded"):
        submission["target"] = submission["target"].astype(int)
    submission.to_csv(args.output_dir / "submission.csv", index=False)

    save_feature_importance(
        final_models,
        train_features.columns,
        args.output_dir / "feature_importance.csv",
    )

    summary = (
        validation_results[["model", "pooled_wape"]]
        .drop_duplicates()
        .sort_values("pooled_wape")
    )
    print(summary.to_string(index=False), flush=True)
    print(f"Выбранный ансамбль: {selected_ensemble}", flush=True)
    print(f"Submission: {args.output_dir / 'submission.csv'}", flush=True)


if __name__ == "__main__":
    main()
