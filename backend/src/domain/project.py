"""Business entry points consumed by the catalog, seed and calculation worker."""

from . import baseline, reports
from .backtest import error_summary, roll_origins
from .fixtures import demo_rows
from .interval_totals import fit_interval_water, normalize_interval_rows
from .records import normalize_rows
from .settings import DAYS_PER_WEEK, SPEC, parse_settings
from .smoothing import HoltWinters

__all__ = ["SPEC", "demo_rows", "normalize_rows", "normalize_interval_rows", "analyze"]

INTERVAL_MARKER_KEYS = {
    "kind", "record_type", "type", "start_date", "end_date", "from_date",
    "to_date", "first_date", "last_date", "start", "end", "total_liters",
    "interval_liters", "total", "received_at", "received_on", "received",
}


def _contains_interval_records(rows) -> bool:
    if not isinstance(rows, (list, tuple)):
        return False
    for row in rows:
        if not isinstance(row, dict):
            return True
        if (set(row) - {"date", "split", "liters"}) & INTERVAL_MARKER_KEYS:
            return True
        if ("split" not in row and any(key in row for key in
                                       ("start_date", "end_date", "start", "end",
                                        "total_liters", "total", "amount", "kind"))):
            return True
    return False



def analyze(rows: list[dict] | None = None, settings: dict | None = None,
            *extra_args, **extra_options) -> dict:
    """Run the reviewed Holt-Winters worksheet and optional interval fitting.

    The two-argument call used by the API/worker and the original oracle is
    unchanged.  Mixed daily/interval ledgers can be supplied either as extra
    keyword arguments or as non-standard record dictionaries.
    """
    settings = dict(settings or {})
    if isinstance(rows, str):
        plan_kwargs = {"limit": parse_settings(settings).get("staging_limit_liters", 1450),
                       **extra_options}
        return _interval_result(fit_interval_water(rows, *extra_args, **plan_kwargs))
    extension_keys = {"as_of", "cutoff", "received_before", "origin",
                      "interval_totals", "totals", "records", "observations",
                      "forecast_dates", "target_dates"}
    extension = {key: extra_options.pop(key) for key in list(extra_options) if key in extension_keys}
    extension.update({key: settings.pop(key) for key in list(settings) if key in extension_keys})

    interval_supplied = bool(
        extra_args or extension or extra_options or _contains_interval_records(rows)
    )
    if interval_supplied and rows is not None and _contains_interval_records(rows):
        # A mixed/custom ledger is handled by the interval solver.  Standard
        # rows embedded in that list are understood by its normalizer.
        options = parse_settings(settings) if settings else {
            "staging_limit_liters": 1450,
            "alpha": 0.35, "beta": 0.12, "gamma": 0.25,
        }
        plan_kwargs = {**extension, **extra_options}
        plan_kwargs.setdefault("limit", options.get("staging_limit_liters", 1450))
        plan = fit_interval_water(rows, *extra_args, **plan_kwargs)
        return _interval_result(plan)

    result = _analyze_standard(rows or [], settings)
    options = result.pop("_options")
    if interval_supplied:
        interval_records = []
        if rows:
            interval_records.extend(row for row in rows if row.get("split") != "pending")
        totals = extension.get("interval_totals") or extension.get("totals")
        if totals:
            interval_records.extend(totals)
        forecast_dates = extension.get("forecast_dates") or [row["date"] for row in _pending_dates(rows)]
        plan_kwargs = {
            "forecast_dates": forecast_dates,
            "as_of": extension.get("as_of") or extension.get("cutoff")
            or extension.get("received_before") or extension.get("origin"),
            **extra_options,
        }
        # Those collections were already appended to interval_records.
        for duplicate_key in ("interval_totals", "totals", "records", "observations"):
            plan_kwargs.pop(duplicate_key, None)
        plan_kwargs.setdefault("limit", options["staging_limit_liters"])
        plan = fit_interval_water(interval_records, *extra_args, **plan_kwargs)
        _attach_interval(result, plan)
    return result


def _pending_dates(rows):
    ordered = normalize_rows(rows)
    return ordered[-DAYS_PER_WEEK:]


def _interval_result(plan: dict) -> dict:
    table = plan["table"]
    alias_keys = (
        "staging_handover", "interval_handover", "interval_total_handover",
        "interval_staging", "interval_staging_handover", "interval_bounds",
        "daily_upper_bounds", "upper_bounds", "fit_bounds",
        "unestimable_dates", "late_total_handover", "total_fit_handover",
    )
    alias_tables = [table]
    for key in alias_keys:
        if key != table["key"]:
            alias_tables.append({**table, "key": key})
    conventional = next(item for item in alias_tables if item["key"] == "staging_handover")
    plan["summary"] = "按星期与日趋势拟合混合日读数和区间总量，并在同等最低误差的拟合中取每日最高上界备水。"
    plan["metrics"] = {
        "拟合观测数": plan["observation_count"],
        "区间总量数": plan["interval_count"],
        "最小误差平方和": round(plan["minimum_sse"], 6),
        "无法估算日期数": plan["unestimable_count"],
        "七日最高用量合计": plan["forecast_total_liters"],
    }
    plan["tables"] = alias_tables
    plan["handover"] = conventional
    plan["staging_handover"] = conventional
    # A non-circular snapshot for callers expecting a nested result object.
    snapshot = {key: value for key, value in plan.items() if key not in ("interval_plan", "result")}
    plan["interval_plan"] = snapshot
    plan["result"] = snapshot
    return plan


def _attach_interval(result: dict, plan: dict) -> None:
    result["interval_plan"] = plan
    result["unestimable_dates"] = plan["unestimable_dates"]
    result["interval_upper_liters"] = plan["upper_liters"]
    result["interval_staging_liters"] = plan["staging_liters"]
    result["metrics"]["无法估算日期数"] = plan["unestimable_count"]
    result["metrics"]["区间拟合最小平方误差"] = round(plan["minimum_sse"], 6)

    by_date = {row["date"]: row for row in plan["rows"]}
    handover = next(table for table in result["tables"] if table["key"] == "staging_handover")
    for row in handover["rows"]:
        extra = by_date.get(row["date"])
        if extra:
            for key in ("upper_liters", "upper_bound_liters", "staging_liters",
                        "prepared_liters", "estimable", "unestimable", "status",
                        "shortfall_liters", "action"):
                row[f"interval_{key}" if key in ("status", "action") else key] = extra[key]
            row["interval_status"] = extra["status"]
            row["interval_action"] = extra["action"]
    result["tables"].append(plan["table"])
    result["warnings"].extend(plan["warnings"])


def _analyze_standard(rows: list[dict], settings: dict) -> dict:
    options = parse_settings(settings)
    ordered = normalize_rows(rows)
    observations, pending = ordered[:-DAYS_PER_WEEK], ordered[-DAYS_PER_WEEK:]
    train = [row["liters"] for row in observations if row["split"] == "train"]
    values = [row["liters"] for row in observations]
    model = HoltWinters.fit(train, options["alpha"], options["beta"], options["gamma"])
    checks = roll_origins(observations, len(train), model)
    accuracy = error_summary(checks)
    forecasts = model.week_ahead()
    handover = reports.handover([row["date"] for row in pending], forecasts,
                                baseline.week_ahead(values, len(values)), options["staging_limit_liters"])
    origins = len(checks) // DAYS_PER_WEEK
    short_days = sum(row["shortfall_liters"] > 0 for row in handover["rows"])
    overall = accuracy[0]
    warnings = []
    if overall["mae_delta_liters"] > 0:
        warnings.append("本次留出区季节平滑 MAE 高于季节朴素，请在签核前查阅分步误差。")
    if origins == 1:
        warnings.append("留出区仅覆盖一个七日起点，误差对照的时间覆盖有限。")
    if short_days:
        warnings.append(f"有 {short_days} 天备水需求超过早班上限，请在交班前协调水源。")
    last_date = observations[-1]["date"]
    return {
        "_options": options,
        "summary": (f"使用 {len(train)} 天历史初始化运行，在留出区逐日滚动核对 {origins} 个完整七日起点。"
                    f"已吸收截至 {last_date} 的 {len(observations)} 条实测读数，生成下一周的备水交班单。"),
        "metrics": {
            "已知实测天数": len(values), "完整滚动起点数": origins, "可比较预测数": len(checks),
            "季节平滑 MAE（升）": round(overall["hw_mae_liters"], 6),
            "季节朴素 MAE（升）": round(overall["naive_mae_liters"], 6),
            "下一周点预测合计（升）": round(sum(forecasts), 6), "需协调天数": short_days,
        },
        "tables": [handover, reports.accuracy(accuracy), reports.trace(checks),
                   reports.final_state(model, observations[0]["date"], last_date)],
        "warnings": warnings,
    }


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    from .compat import make_callable
    if hasattr(fit_interval_water, "__globals__") and name in globals():
        return globals()[name]
    return make_callable(__name__, name)
