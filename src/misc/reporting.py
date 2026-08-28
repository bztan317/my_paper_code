import csv
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_scalar(value: Any):
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return _is_number(value) and math.isfinite(float(value))


def flatten_metrics(data: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat = {}
    for key, value in data.items():
        name = f"{prefix }_{key }" if prefix else str(key)
        value = _to_scalar(value)

        if isinstance(value, Mapping):
            flat.update(flatten_metrics(value, prefix=name))
        elif isinstance(value, (list, tuple)):
            if all(not isinstance(item, (Mapping, list, tuple)) for item in value):
                for idx, item in enumerate(value):
                    flat[f"{name }_{idx }"] = _to_scalar(item)
            else:
                flat[name] = json.dumps(value, ensure_ascii=False)
        else:
            flat[name] = value
    return flat


def load_epoch_history(epoch_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(epoch_dir.glob("epoch_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append(flatten_metrics(data))
    rows.sort(key=lambda row: row.get("epoch", 0))
    return rows


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if _is_number(value):
        value = float(value)
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        abs_value = abs(value)
        if abs_value >= 1e4 or (0 < abs_value < 1e-4):
            return f"{value :.4e}"
        return f"{value :.6f}"
    return str(value)


def _write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def build_metric_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []

    columns = []
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(key)

    summaries = []
    for column in columns:
        if column == "epoch" or not _should_keep_column(column):
            continue

        series = []
        for row in rows:
            value = row.get(column, None)
            if _is_finite_number(value):
                series.append((int(row.get("epoch", len(series))), float(value)))

        if not series:
            continue

        epochs = [epoch for epoch, _ in series]
        values = [value for _, value in series]
        min_value = min(values)
        max_value = max(values)
        min_index = values.index(min_value)
        max_index = values.index(max_value)

        summaries.append(
            {
                "metric": column,
                "count": len(values),
                "last": values[-1],
                "mean": sum(values) / len(values),
                "min": min_value,
                "min_epoch": epochs[min_index],
                "max": max_value,
                "max_epoch": epochs[max_index],
            }
        )

    summaries.sort(key=lambda row: row["metric"])
    return summaries


def _metric_group(column: str) -> str:
    if column in {"epoch", "n_parameters"}:
        return "meta"
    if "time" in column or column.startswith("elapsed_total"):
        return "timing"
    if column.startswith("train_"):
        return "train"
    if column.startswith("test_"):
        return "test"
    return "other"


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("._") or "metric"


def _should_keep_column(column: str) -> bool:
    if column in {
        "epoch",
        "epoch_index",
        "epoch_total",
        "n_parameters",
        "epoch_started_at",
        "epoch_finished_at",
        "train_time",
        "eval_time",
        "epoch_time",
        "elapsed_total",
        "train_time_seconds",
        "eval_time_seconds",
        "epoch_time_seconds",
        "elapsed_total_seconds",
    }:
        return True

    if re.search(r"_\d+$", column):
        return False

    if column.startswith("train_"):
        return True

    if column.startswith("test_"):
        return any(
            token in column for token in ["ap", "acc", "loss", "precision", "recall"]
        )

    return False


def _should_plot_metric(column: str) -> bool:
    if column in {
        "train_time_seconds",
        "eval_time_seconds",
        "epoch_time_seconds",
        "elapsed_total_seconds",
    }:
        return True

    if re.search(r"_\d+$", column):
        return False

    if column.startswith("train_"):
        return column == "train_lr" or "loss" in column

    if column.startswith("test_"):
        return column in {
            "test_bbox_ap",
            "test_bbox_ap50",
            "test_bbox_ap75",
            "test_bbox_ap50_small",
            "test_bbox_ap50_medium",
            "test_bbox_ap50_large",
            "test_acc",
            "test_loss",
        } or any(token in column for token in ["precision", "recall"])

    return False


def _plot_single_metric(
    epochs: List[int], values: List[float], metric: str, path: Path
):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(epochs, values, marker="o", linewidth=2.0, markersize=4)
    ax.set_title(metric)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_group_metrics(
    epochs: List[int], series_map: Dict[str, List[float]], title: str, path: Path
):
    if not series_map:
        return

    fig, ax = plt.subplots(figsize=(11, 5.6))
    for name, values in series_map.items():
        ax.plot(epochs, values, marker="o", linewidth=1.8, markersize=3.5, label=name)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def generate_metric_charts(rows: List[Dict[str, Any]], charts_dir: Path):
    if not rows:
        return

    charts_dir.mkdir(parents=True, exist_ok=True)
    overview_dir = charts_dir / "overview"
    overview_dir.mkdir(exist_ok=True)

    epochs = [int(row.get("epoch", index)) for index, row in enumerate(rows)]
    columns = []
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(key)

    group_series: Dict[str, Dict[str, List[float]]] = {}
    for column in columns:
        if column == "epoch":
            continue

        values = [row.get(column, None) for row in rows]
        if not all(value is None or _is_number(value) for value in values):
            continue
        if not _should_plot_metric(column):
            continue

        numeric_points = [
            (epoch, float(value))
            for epoch, value in zip(epochs, values)
            if _is_finite_number(value)
        ]
        if not numeric_points:
            continue

        metric_epochs = [epoch for epoch, _ in numeric_points]
        metric_values = [value for _, value in numeric_points]

        group = _metric_group(column)
        metric_dir = charts_dir / group
        metric_dir.mkdir(exist_ok=True)
        _plot_single_metric(
            metric_epochs,
            metric_values,
            column,
            metric_dir / f"{_sanitize_filename (column )}.png",
        )

        if len(metric_values) == len(epochs):
            group_series.setdefault(group, {})[column] = metric_values

    for group, series_map in group_series.items():
        _plot_group_metrics(
            epochs,
            series_map,
            f"{group .title ()} Metrics Overview",
            overview_dir / f"{group }_overview.png",
        )


def write_metric_tables(run_dir: Path, rows: List[Dict[str, Any]]):
    reports_dir = run_dir / "reports"
    tables_dir = reports_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    seen_columns = set()
    columns = ["epoch"]
    for row in rows:
        for key in sorted(row.keys()):
            if key == "epoch" or key in seen_columns or not _should_keep_column(key):
                continue
            seen_columns.add(key)
            columns.append(key)
    metric_rows = [{column: row.get(column, "") for column in columns} for row in rows]
    _write_csv(tables_dir / "epoch_metrics.csv", metric_rows, columns)

    timing_columns = [
        column
        for column in [
            "epoch",
            "epoch_index",
            "epoch_total",
            "epoch_started_at",
            "epoch_finished_at",
            "train_time",
            "eval_time",
            "epoch_time",
            "elapsed_total",
            "train_time_seconds",
            "eval_time_seconds",
            "epoch_time_seconds",
            "elapsed_total_seconds",
        ]
        if column in columns
    ]
    timing_rows = [
        {column: row.get(column, "") for column in timing_columns} for row in rows
    ]
    _write_csv(tables_dir / "epoch_timing.csv", timing_rows, timing_columns)

    summary_rows = build_metric_summary(rows)
    summary_columns = [
        "metric",
        "count",
        "last",
        "mean",
        "min",
        "min_epoch",
        "max",
        "max_epoch",
    ]
    _write_csv(tables_dir / "metric_summary.csv", summary_rows, summary_columns)


def write_report_index(run_dir: Path):
    reports_dir = run_dir / "reports"
    tables_dir = reports_dir / "tables"
    charts_dir = reports_dir / "charts"
    summary_path = reports_dir / "summary.md"

    table_files = (
        sorted(p.name for p in tables_dir.glob("*") if p.is_file())
        if tables_dir.exists()
        else []
    )
    chart_files = (
        sorted(
            str(p.relative_to(reports_dir)).replace("\\", "/")
            for p in charts_dir.rglob("*.png")
        )
        if charts_dir.exists()
        else []
    )

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# Run Report\n\n")
        f.write("## Tables\n\n")
        if table_files:
            for name in table_files:
                f.write(f"- tables/{name }\n")
        else:
            f.write("- No tables generated.\n")

        f.write("\n## Charts\n\n")
        if chart_files:
            for name in chart_files:
                f.write(f"- {name }\n")
        else:
            f.write("- No charts generated.\n")


def update_run_report(run_dir: Path):
    run_dir = Path(run_dir)
    rows = load_epoch_history(run_dir / "epochs")
    if not rows:
        return

    reports_dir = run_dir / "reports"
    if reports_dir.exists():
        shutil.rmtree(reports_dir)

    write_metric_tables(run_dir, rows)
    generate_metric_charts(rows, run_dir / "reports" / "charts")
    write_report_index(run_dir)
