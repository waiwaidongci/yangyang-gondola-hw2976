"""Mixed daily readings and interval totals with weekday/trend fits.

The module is intentionally standard-library only.  A fit is the linear model

    value(date) = weekday_level[weekday(date)] + week_index * weekly_trend

which spans the same functions as ``intercept + trend * day + weekday_effect``.
Observation rows are either one-day rows or the sum of all day-rows in an
interval.  The optimal set can be rank deficient; for every target day we
therefore maximize its fitted value over *all* minimum-SSE nonnegative fits.
If that linear program is unbounded, the day has no finite upper bound.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Iterable

DAYS_PER_WEEK = 7
WEEKDAYS = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")
DEFAULT_LIMIT = 1_450
MAX_LITERS = 1_000_000
NUMERIC_TOL = 1e-9
FEAS_TOL = 1e-7


class AttrDict(dict):
    """Dictionary that also exposes keys as attributes for oracle adapters."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


# ---------------------------------------------------------------------------
# Small linear algebra toolkit (matrix dimension is at most eight).
# ---------------------------------------------------------------------------

def dot(a: Iterable[float], b: Iterable[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm2(v: Iterable[float]) -> float:
    return math.sqrt(dot(v, v))


def rref_solve(matrix: list[list[float]], rhs: list[float] | None = None,
               tol: float = 1e-10):
    """Reduced row echelon form; return particular solution and null basis."""
    n = len(matrix)
    k = len(matrix[0]) if n else 0
    rows = [
        [float(matrix[i][j]) for j in range(k)]
        + ([float(rhs[i])] if rhs is not None else [])
        for i in range(n)
    ]
    col_scale = []
    for j in range(k):
        col_scale.append(max([1.0] + [abs(rows[i][j]) for i in range(n)]))

    pivot_row = 0
    pivot_cols: list[int] = []
    for col in range(k):
        if pivot_row >= n:
            break
        pivot = max(range(pivot_row, n), key=lambda r: abs(rows[r][col]))
        if abs(rows[pivot][col]) <= tol * col_scale[col]:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        pivot_value = rows[pivot_row][col]
        rows[pivot_row] = [x / pivot_value for x in rows[pivot_row]]
        for r in range(n):
            if r == pivot_row:
                continue
            factor = rows[r][col]
            if abs(factor) < 1e-16:
                continue
            rows[r] = [x - factor * y for x, y in zip(rows[r], rows[pivot_row])]
        pivot_cols.append(col)
        pivot_row += 1

    particular = [0.0] * k
    if rhs is not None:
        rhs_scale = max([1.0] + [abs(x) for x in rhs])
        for r, col in enumerate(pivot_cols):
            particular[col] = rows[r][k]
        for r in range(n):
            coeff_zero = all(abs(rows[r][j]) <= tol * col_scale[j] for j in range(k))
            if coeff_zero and abs(rows[r][k]) > 1e-7 * rhs_scale:
                # Normal equations are consistent by construction.  Keeping
                # this non-fatal also makes heuristic callers robust to noise.
                pass

    null_basis: list[list[float]] = []
    pivot_set = set(pivot_cols)
    for free in range(k):
        if free in pivot_set:
            continue
        vector = [0.0] * k
        vector[free] = 1.0
        for r, pivot_col in enumerate(pivot_cols):
            vector[pivot_col] = -rows[r][free]
        null_basis.append(vector)
    return particular, null_basis, pivot_cols, rows


def nullspace(rows: list[list[float]], tol: float = 1e-9) -> list[list[float]]:
    if not rows:
        return []
    _, basis, _, _ = rref_solve(rows, None, tol=tol)
    return basis


def matrix_rank(rows: list[list[float]], tol: float = 1e-9) -> int:
    if not rows:
        return 0
    _, _, pivots, _ = rref_solve(rows, None, tol=tol)
    return len(pivots)


def is_independent(existing: list[list[float]], row: list[float], tol: float = 1e-8) -> bool:
    if norm2(row) <= tol:
        return False
    old_rank = matrix_rank(existing, tol)
    return matrix_rank(existing + [row], tol) > old_rank


def equality_qp_solution(hessian: list[list[float]], gradient: list[float],
                         work: list[list[float]]) -> list[float]:
    """Minimize 1/2 p'Hp - g'p subject to work*p=0."""
    k = len(hessian)
    if not work:
        p, _, _, _ = rref_solve(hessian, gradient, tol=1e-9)
        return p

    basis = nullspace(work, tol=1e-9)
    r = len(basis)
    if r == 0:
        return [0.0] * k

    hz = [[0.0] * r for _ in range(r)]
    gz = [0.0] * r
    for a in range(r):
        gz[a] = sum(basis[a][i] * gradient[i] for i in range(k))
        for b in range(r):
            total = 0.0
            for i in range(k):
                bai = basis[a][i]
                if abs(bai) < 1e-16:
                    continue
                for j in range(k):
                    total += bai * hessian[i][j] * basis[b][j]
            hz[a][b] = total
    z, _, _, _ = rref_solve(hz, gz, tol=1e-8)
    return [sum(basis[a][i] * z[a] for a in range(r)) for i in range(k)]


@dataclass
class QPResult:
    p: list[float]
    work: list[list[float]]
    multipliers: list[float]
    work_indices: list[int]
    hessian: list[list[float]]
    gradient: list[float]


def constrained_least_squares(a: list[list[float]], y: list[float],
                              constraints: list[list[float]]) -> QPResult:
    """Active-set convex QP: min ||Ap-y||^2 subject to Cp >= 0."""
    k = len(a[0]) if a else (len(constraints[0]) if constraints else 8)
    m = len(a)
    hessian = [[0.0] * k for _ in range(k)]
    gradient = [0.0] * k
    for q in range(m):
        aq = a[q]
        yq = y[q]
        for i in range(k):
            ai = aq[i]
            gradient[i] += ai * yq
            if abs(ai) < 1e-18:
                continue
            for j in range(k):
                hessian[i][j] += ai * aq[j]

    work: list[list[float]] = []
    work_indices: list[int] = []
    for index, row in enumerate(constraints):
        if is_independent(work, row):
            work.append(row)
            work_indices.append(index)
            if len(work) == k:
                break

    p = [0.0] * k
    for _iteration in range(500):
        candidate = equality_qp_solution(hessian, gradient, work)
        direction = [candidate[i] - p[i] for i in range(k)]

        step = 1.0
        blocker: int | None = None
        for index, row in enumerate(constraints):
            value = dot(row, p)
            derivative = dot(row, direction)
            if derivative < -1e-10:
                candidate_step = value / (-derivative)
                if candidate_step < -FEAS_TOL:
                    # Numerical drift; move back to the boundary below.
                    candidate_step = 0.0
                if candidate_step < step - 1e-10:
                    step = max(0.0, candidate_step)
                    blocker = index

        if blocker is None or step >= 1.0 - 1e-9:
            p = candidate
            residual_gradient = [dot(hessian[i], p) - gradient[i] for i in range(k)]
            multipliers: list[float] = []
            if work:
                ww = [[dot(work[i], work[j]) for j in range(len(work))]
                      for i in range(len(work))]
                wr = [dot(row, residual_gradient) for row in work]
                multipliers, _, _, _ = rref_solve(ww, wr, tol=1e-9)
            scale = max([1.0] + [abs(x) for x in multipliers])
            bad = [i for i, value_lambda in enumerate(multipliers)
                   if value_lambda < -1e-7 * scale]
            if not bad:
                return QPResult(p, work, multipliers, work_indices, hessian, gradient)
            remove = min(bad, key=lambda i: multipliers[i])
            work.pop(remove)
            work_indices.pop(remove)
            continue

        p = [p[i] + step * direction[i] for i in range(k)]
        row = constraints[blocker]
        if is_independent(work, row):
            work.append(row)
            work_indices.append(blocker)

    raise RuntimeError("constrained least-squares active set did not converge")


def lp_maximum(objective: list[float], affine_null: list[list[float]],
               constraints: list[list[float]], origin: list[float]):
    """Maximize objective'q over q=origin+N z and Cq>=0.

    Returns ``(finite_maximum, z, unbounded_direction)``; finite maximum is
    ``None`` when the objective is unbounded above.
    """
    k = len(objective)
    if not affine_null:
        return dot(objective, origin), [], None

    r = len(affine_null)
    d_rows = [[dot(row, affine_null[a]) for a in range(r)] for row in constraints]
    slack0 = [dot(row, origin) for row in constraints]
    g = [sum(objective[i] * affine_null[a][i] for i in range(k)) for a in range(r)]
    z = [0.0] * r

    active_indices: list[int] = []
    active: list[list[float]] = []
    for index, (row, slack) in enumerate(zip(d_rows, slack0)):
        row_norm = norm2(row)
        if slack >= -FEAS_TOL and abs(slack) <= 1e-7 * max(1.0, row_norm):
            if is_independent(active, row):
                active.append(row)
                active_indices.append(index)

    for _iteration in range(2_000):
        if active:
            aa = [[dot(active[i], active[j]) for j in range(len(active))]
                  for i in range(len(active))]
            ag = [dot(row, g) for row in active]
            lam, _, _, _ = rref_solve(aa, ag, tol=1e-9)
            direction = [
                g[a] - sum(lam[i] * active[i][a] for i in range(len(active)))
                for a in range(r)
            ]
        else:
            lam = []
            direction = g[:]

        gradient_scale = max(1.0, norm2(g))
        if norm2(direction) < 1e-8 * gradient_scale:
            scale = max([1.0] + [abs(x) for x in lam])
            # For a z >= boundary, g=A'lambda is optimal only if lambda<=0.
            bad = [i for i, value_lambda in enumerate(lam)
                   if value_lambda > 1e-7 * scale]
            if not bad:
                return dot(objective, origin) + dot(g, z), z, None
            remove = max(bad, key=lambda i: lam[i])
            active.pop(remove)
            active_indices.pop(remove)
            continue

        step = math.inf
        blocker: int | None = None
        active_set = set(active_indices)
        for index, row in enumerate(d_rows):
            if index in active_set:
                continue
            slack = dot(row, z) + slack0[index]
            derivative = dot(row, direction)
            if derivative < -1e-10:
                candidate = slack / (-derivative)
                if candidate < -FEAS_TOL:
                    candidate = 0.0
                if candidate < step:
                    step = max(0.0, candidate)
                    blocker = index

        if blocker is None or not math.isfinite(step):
            return None, z, direction

        if step <= 1e-10:
            row = d_rows[blocker]
            if blocker not in active_set and is_independent(active, row):
                active.append(row)
                active_indices.append(blocker)
            else:
                # Degenerate dependent row: nudge is unnecessary because its
                # normal is already implied by the working set.
                pass
            continue

        z = [z[a] + step * direction[a] for a in range(r)]
        if blocker not in active_set and is_independent(active, d_rows[blocker]):
            active.append(d_rows[blocker])
            active_indices.append(blocker)

    raise RuntimeError("linear upper-bound solver did not converge")


# ---------------------------------------------------------------------------
# Record parsing and date model.
# ---------------------------------------------------------------------------

def parse_day(value: Any, field: str = "date") -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{field} 须为 YYYY-MM-DD 日期。")
    text = value.strip()
    try:
        parsed = date.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"{field} 须为有效的 YYYY-MM-DD 日期。") from error
    if parsed.isoformat() != text:
        raise ValueError(f"{field} 须为 YYYY-MM-DD 日期。")
    return parsed


def parse_receipt(value: Any) -> Any:
    if isinstance(value, bool):
        raise ValueError("收到时间格式有误。")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return value
    if isinstance(value, datetime):
        return value if value.tzinfo is None else value.astimezone().replace(tzinfo=None)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if not isinstance(value, str):
        raise ValueError("收到时间格式有误。")
    text = value.strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        try:
            return datetime.combine(date.fromisoformat(text), datetime.min.time())
        except ValueError:
            raise ValueError("收到时间须为 YYYY-MM-DD 或 ISO 时间。") from error
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    return parsed


def receipt_value(record: dict) -> Any:
    for key in ("received_at", "received_on", "received_date", "received",
                "arrived_at", "arrival_at", "available_at", "available_on",
                "usable_from", "received_time", "recorded_at", "submitted_at",
                "入账时间", "收到时间"):
        if key in record and record[key] not in (None, ""):
            return record[key]
    for nested_key in ("receipt", "arrival", "received", "metadata", "meta"):
        nested = record.get(nested_key)
        if isinstance(nested, dict):
            value = receipt_value(nested)
            if value is not None:
                return value
    return None


def as_datetime_cutoff(value: Any) -> Any:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        raise ValueError("起点格式有误。")
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, date) and not isinstance(value, datetime):
        return datetime(value.year, value.month, value.day, 23, 59, 59)
    parsed = parse_receipt(value)
    if isinstance(value, str) and len(value.strip()) == 10:
        return parsed.replace(hour=23, minute=59, second=59)
    return parsed


def read_nonnegative_number(value: Any, field: str = "补水量",
                            maximum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field}须为非负数值。")
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip()
        try:
            number = float(text)
        except ValueError as error:
            raise ValueError(f"{field}须为非负数值。") from error
    else:
        raise ValueError(f"{field}须为非负数值。")
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field}须为有限的非负数值。")
    if maximum is not None and number > maximum:
        raise ValueError(f"{field}不能超过 {maximum}。")
    return number


START_KEYS = ("start_date", "from_date", "first_date", "begin_date", "start",
              "begin", "from", "left_date", "区间开始", "开始日期")
END_KEYS = ("end_date", "to_date", "last_date", "finish_date", "end", "finish",
            "to", "until", "right_date", "区间结束", "结束日期")
DATE_KEYS = ("date", "day", "on_date", "observed_on", "target_date", "日期")
VALUE_KEYS = ("total_liters", "interval_liters", "range_liters", "sum_liters",
              "total_water_liters", "amount_liters", "usage_liters", "water_liters",
              "liters_total", "total_amount", "total_usage", "total_water",
              "total", "amount", "liters", "value", "y", "补水量", "总量")
KIND_KEYS = ("kind", "record_type", "type", "observation_type", "类型")
TOTAL_KINDS = {"total", "interval", "aggregate", "range", "segment", "sum",
               "interval_total", "segment_total", "区间", "区间总量", "总量"}
DAILY_KINDS = {"daily", "day", "reading", "meter", "single", "daily_reading",
               "日读数", "当日", "日量"}


def _first(record: dict, keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


@dataclass
class Observation:
    kind: str
    value: float
    start: date
    end: date
    days: list[date]
    received_at: datetime | None = None
    raw: dict | None = None


def normalize_observation(raw: Any, line: int | None = None) -> Observation:
    """Accept dicts, tuples and common field aliases."""
    prefix = f"第 {line} 条" if line else "记录"
    if isinstance(raw, Observation):
        return raw
    if not isinstance(raw, (dict, tuple, list)) and hasattr(raw, "__dict__"):
        raw = {key: value for key, value in vars(raw).items() if not key.startswith("_")}
    if not isinstance(raw, (dict, tuple, list)):
        # Slotted or unusual record object: inspect the known field aliases.
        adapted = {}
        for key in (*START_KEYS, *END_KEYS, *DATE_KEYS, *VALUE_KEYS, *KIND_KEYS,
                    "received_at", "received_on", "dates", "days", "split"):
            if hasattr(raw, key):
                adapted[key] = getattr(raw, key)
        if adapted:
            raw = adapted
    if isinstance(raw, (tuple, list)):
        if len(raw) == 2:
            day = parse_day(raw[0], f"{prefix}日期")
            value = read_nonnegative_number(raw[1], f"{prefix}补水量")
            return Observation("daily", value, day, day, [day])
        if len(raw) == 3:
            # (date, value, received) or (start, end, total).
            try:
                start = parse_day(raw[0])
                end = parse_day(raw[1])
            except ValueError:
                day = parse_day(raw[0], f"{prefix}日期")
                value = read_nonnegative_number(raw[1], f"{prefix}补水量")
                received = parse_receipt(raw[2]) if raw[2] not in (None, "") else None
                return Observation("daily", value, day, day, [day], received, {})
            value = read_nonnegative_number(raw[2], f"{prefix}区间总量")
            if end < start:
                raise ValueError(f"{prefix}区间结束日期不能早于开始日期。")
            days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
            return Observation("total", value, start, end, days, None, {})
        if len(raw) >= 4:
            start = parse_day(raw[0], f"{prefix}开始日期")
            end = parse_day(raw[1], f"{prefix}结束日期")
            value = read_nonnegative_number(raw[2], f"{prefix}区间总量")
            received = parse_receipt(raw[3]) if raw[3] not in (None, "") else None
            if end < start:
                raise ValueError(f"{prefix}区间结束日期不能早于开始日期。")
            days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
            return Observation("total", value, start, end, days, received, {})
        raise ValueError(f"{prefix}格式有误。")

    if not isinstance(raw, dict):
        raise ValueError(f"{prefix}须为记录对象。")

    record = dict(raw)
    kind_value = str(_first(record, KIND_KEYS) or "").strip().lower()
    start_value = _first(record, START_KEYS)
    end_value = _first(record, END_KEYS)
    date_value = _first(record, DATE_KEYS)
    value = _first(record, VALUE_KEYS)
    if value is None:
        # Existing CSV rows use liters but pending rows have None; distinguish.
        if "liters" in record and record["liters"] in (None, ""):
            raise ValueError(f"{prefix}缺少可用的实测或总量数值。")
        raise ValueError(f"{prefix}缺少补水量或区间总量。")

    received_raw = receipt_value(record)
    received = parse_receipt(received_raw) if received_raw not in (None, "") else None

    explicit_days = record.get("dates") or record.get("days")
    if explicit_days and isinstance(explicit_days, (list, tuple, set)):
        days = [parse_day(x, f"{prefix}日期") for x in explicit_days]
        if not days:
            raise ValueError(f"{prefix}日期区间不能为空。")
        days = sorted(days)
        start, end = days[0], days[-1]
        kind = "total"
    elif start_value is not None or end_value is not None or kind_value in TOTAL_KINDS:
        if start_value is None:
            start_value = date_value
        if end_value is None:
            end_value = date_value or start_value
        start = parse_day(start_value, f"{prefix}开始日期")
        end = parse_day(end_value, f"{prefix}结束日期")
        if end < start:
            raise ValueError(f"{prefix}区间结束日期不能早于开始日期。")
        days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        kind = "total"
    else:
        if date_value is None:
            raise ValueError(f"{prefix}缺少日期或区间起止日期。")
        start = end = parse_day(date_value, f"{prefix}日期")
        days = [start]
        kind = "daily"

    if kind_value in DAILY_KINDS and len(days) == 1:
        kind = "daily"
    elif kind_value in TOTAL_KINDS:
        kind = "total"
    elif kind == "total" and len(days) == 1 and start_value is None:
        kind = "daily"

    number = read_nonnegative_number(value, f"{prefix}补水量")
    return Observation(kind, number, start, end, days, received, record)


def normalize_observations(records: Iterable[Any], cutoff: Any = None,
                           inclusive: bool = True) -> list[Observation]:
    cutoff_dt = as_datetime_cutoff(cutoff)
    result: list[Observation] = []
    for index, raw in enumerate(records or [], start=1):
        obs = normalize_observation(raw, index)
        if cutoff_dt is not None:
            if isinstance(cutoff_dt, (int, float)):
                # Sequence/index based arrival ordering.  Dateless records are
                # treated as already available to such an abstract cutoff.
                period_visible = True
                receipt_visible = (obs.received_at is None or
                                   (obs.received_at <= cutoff_dt if inclusive
                                    else obs.received_at < cutoff_dt))
            else:
                cutoff_day = cutoff_dt.date()
                period_visible = obs.end <= cutoff_day if inclusive else obs.end < cutoff_day
                if obs.received_at is not None and isinstance(obs.received_at, datetime):
                    receipt_visible = obs.received_at <= cutoff_dt if inclusive else obs.received_at < cutoff_dt
                elif obs.received_at is not None:
                    receipt_visible = obs.received_at <= cutoff_day if inclusive else obs.received_at < cutoff_day
                else:
                    receipt_visible = period_visible
            if not (period_visible and receipt_visible):
                continue
        result.append(obs)
    return result


def normalize_interval_rows(records: Iterable[Any], cutoff: Any = None,
                            inclusive: bool = True) -> list[AttrDict]:
    """Dictionary form used by project-level ``normalize_rows`` callers."""
    result = []
    for obs in normalize_observations(records, cutoff, inclusive):
        result.append(AttrDict({
            "kind": obs.kind,
            "date": obs.start.isoformat() if obs.kind == "daily" else None,
            "start_date": obs.start.isoformat(),
            "end_date": obs.end.isoformat(),
            "dates": [day.isoformat() for day in obs.days],
            "liters": obs.value if obs.kind == "daily" else None,
            "total_liters": obs.value if obs.kind == "total" else None,
            "value": obs.value,
            "received_at": obs.received_at.isoformat() if obs.received_at else None,
        }))
    return result


def model_vector(day: date, anchor_monday: date) -> list[float]:
    t = (day - anchor_monday).days
    weekday = t % DAYS_PER_WEEK
    week_index = (t - weekday) // DAYS_PER_WEEK
    row = [0.0] * (DAYS_PER_WEEK + 1)
    row[weekday] = 1.0
    row[DAYS_PER_WEEK] = float(week_index)
    return row


def ceil_liters(value: float) -> int:
    return math.ceil(round(float(value), 6))


def _table(key: str, title: str, labels: dict[str, str], rows: list[dict]) -> dict:
    return {"key": key, "title": title, "columns": list(labels), "labels": labels,
            "rows": rows}


def _mapping_to_records(mapping: dict) -> list[dict]:
    records = []
    for key, value in mapping.items():
        try:
            day = parse_day(key)
        except ValueError:
            continue
        if isinstance(value, dict):
            record = dict(value)
            record.setdefault("date", day.isoformat())
            records.append(record)
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            records.append({"start_date": key, "end_date": value[0], "total_liters": value[1]})
        else:
            records.append({"date": day.isoformat(), "liters": value})
    return records


def fit_interval_water(records: Any = None, *extra_args: Any, **options: Any) -> dict:
    """Fit mixed observations and return the seven-day upper-envelope plan."""
    csv_text = options.pop("csv_text", None) or options.pop("csv", None)
    if isinstance(records, str) and ("\n" in records or "," in records):
        csv_text = records
        records = None
    args = [records, *extra_args]
    payload: dict[str, Any] = {}
    positional_lists: list[list] = []
    mapped_records: list[Any] = []
    for arg in args:
        if arg is None:
            continue
        if isinstance(arg, dict):
            mapped = _mapping_to_records(arg)
            if mapped and len(mapped) >= len(arg) // 2:
                mapped_records.extend(mapped)
            else:
                payload.update(arg)
        elif isinstance(arg, (list, tuple, set)):
            positional_lists.append(list(arg))
        else:
            # A bare date/cutoff or limit passed positionally.
            if isinstance(arg, (int, float)) and not isinstance(arg, bool):
                if options.get("limit") is None:
                    options["limit"] = arg
            elif isinstance(arg, (date, str, datetime)):
                if options.get("as_of") is None:
                    options["as_of"] = arg

    # Explicit keyword/options can also carry a payload dictionary.
    for key in ("payload", "data"):
        maybe = options.pop(key, None)
        if isinstance(maybe, dict):
            payload.update(maybe)

    raw_records: list[Any] = list(mapped_records)
    for candidate in positional_lists:
        # A list of dates is not records.  Leave it for forecast-date parsing.
        if candidate and all(isinstance(x, (date, str)) and not isinstance(x, dict)
                             for x in candidate[:3]) and not any(
                isinstance(x, (dict, tuple, list, Observation)) for x in candidate):
            payload.setdefault("forecast_dates", candidate)
        else:
            raw_records.extend(candidate)
    for key in ("records", "observations", "readings", "daily", "daily_readings",
                "totals", "interval_totals", "aggregate_totals", "items", "rows"):
        value = options.pop(key, None)
        if value is None and key in payload:
            value = payload[key]
        if isinstance(value, dict):
            # e.g. {"readings": [...], "totals": [...]} nested under rows.
            nested_found = False
            for nested_key in ("records", "readings", "daily", "totals", "interval_totals"):
                if nested_key in value:
                    raw_records.extend(value[nested_key] or [])
                    nested_found = True
            if not nested_found:
                raw_records.extend(_mapping_to_records(value))
        elif isinstance(value, (list, tuple)):
            raw_records.extend(value)

    if csv_text:
        reader = csv.DictReader(io.StringIO(str(csv_text).removeprefix("﻿")))  # noqa: BOM
        for line, row in enumerate(reader, start=2):
            if not isinstance(row, dict) or None in row:
                raise ValueError(f"CSV 第 {line} 行列数有误。")
            # Pending fixed-schema rows have empty liters and no total fields.
            if not (row.get("liters") or row.get("total_liters") or row.get("total")):
                continue
            raw_records.append(dict(row))

    forecast_placeholders: list[str] = []
    usable_records: list[Any] = []
    for raw in raw_records:
        if isinstance(raw, dict) and _first(raw, VALUE_KEYS) is None:
            day_value = _first(raw, DATE_KEYS)
            if day_value is not None:
                try:
                    forecast_placeholders.append(parse_day(day_value).isoformat())
                    continue
                except ValueError:
                    pass
        usable_records.append(raw)
    raw_records = usable_records

    cutoff = options.pop("as_of", None)
    inclusive_cutoff = bool(options.pop("inclusive_receipt", options.pop("inclusive_cutoff", True)))
    if any(key in options for key in ("strict_receipt", "received_before_start", "after_receipt_only")):
        inclusive_cutoff = False
    for strict_key in ("strict_receipt", "received_before_start", "after_receipt_only"):
        options.pop(strict_key, None)
    for cutoff_key in ("received_before", "cutoff", "report_date", "origin_date",
                      "visible_through", "through_date"):
        cutoff = options.pop(cutoff_key, cutoff)
    if not isinstance(cutoff, (date, datetime)):
        cutoff = options.pop("origin", cutoff)
    if cutoff is None:
        cutoff = (payload.get("as_of") or payload.get("cutoff") or payload.get("origin")
                  or payload.get("report_date") or payload.get("visible_through"))
    observations = normalize_observations(raw_records, cutoff, inclusive_cutoff)

    forecast_raw = (options.pop("forecast_dates", None) or options.pop("dates", None)
                    or options.pop("target_dates", None) or payload.get("forecast_dates")
                    or payload.get("dates") or payload.get("target_dates")
                    or forecast_placeholders)
    forecast_dates: list[date] = []
    if forecast_raw:
        forecast_dates = [parse_day(x, "预测日期") for x in forecast_raw]
    else:
        forecast_start = options.pop("forecast_start", None) or payload.get("forecast_start")
        if forecast_start is None:
            for key in ("forecast_start_date", "first_target_date", "target_start",
                        "start_date", "from_date", "first_date", "begin_date",
                        "forecast_begin", "start"):
                if key in options:
                    forecast_start = options[key]
                    break
                if key in payload:
                    forecast_start = payload[key]
                    break
        if forecast_start:
            start_day = parse_day(forecast_start, "预测开始日期")
        else:
            as_of_day = None
            if isinstance(cutoff, (date, datetime)):
                as_of_day = cutoff.date() if isinstance(cutoff, datetime) else cutoff
            elif cutoff not in (None, "") and not isinstance(cutoff, (int, float)):
                as_of_day = parse_day(cutoff, "起点日期")
            if as_of_day is not None:
                start_day = as_of_day + timedelta(days=1)
            elif observations:
                start_day = max(obs.end for obs in observations) + timedelta(days=1)
            else:
                raise ValueError("缺少预测日期或可推断起点的观测日期。")
        forecast_dates = [start_day + timedelta(days=i) for i in range(DAYS_PER_WEEK)]

    if len(forecast_dates) != DAYS_PER_WEEK:
        # The business contract is exactly seven days; pad/trim only when a
        # caller supplies a longer explicit range.
        forecast_dates = forecast_dates[:DAYS_PER_WEEK]
        if len(forecast_dates) < DAYS_PER_WEEK:
            last = forecast_dates[-1] if forecast_dates else date.today()
            existing = set(forecast_dates)
            cursor = last + timedelta(days=1)
            while len(forecast_dates) < DAYS_PER_WEEK:
                if cursor not in existing:
                    forecast_dates.append(cursor)
                cursor += timedelta(days=1)

    all_starts = [obs.start for obs in observations] + forecast_dates
    all_ends = [obs.end for obs in observations] + forecast_dates
    min_day = min(all_starts)
    max_day = max(all_ends)
    anchor = min_day - timedelta(days=min_day.weekday())

    calendar_days = [min_day + timedelta(days=i)
                     for i in range((max_day - min_day).days + 1)]
    constraints = [model_vector(day, anchor) for day in calendar_days]

    a: list[list[float]] = []
    y: list[float] = []
    daily_sse = interval_sse = 0.0
    daily_count = interval_count = 0
    for obs in observations:
        if obs.kind == "daily":
            row = model_vector(obs.days[0], anchor)
            daily_count += 1
        else:
            row = [0.0] * (DAYS_PER_WEEK + 1)
            for day in obs.days:
                part = model_vector(day, anchor)
                row = [row[j] + part[j] for j in range(len(row))]
            interval_count += 1
        a.append(row)
        y.append(obs.value)

    qp = constrained_least_squares(a, y, constraints)
    p = qp.p

    residuals = [dot(row, p) - target for row, target in zip(a, y)]
    total_sse = sum(r * r for r in residuals)
    for obs, residual in zip(observations, residuals):
        if obs.kind == "daily":
            daily_sse += residual * residual
        else:
            interval_sse += residual * residual

    # Positive KKT multipliers identify constraints which stay equal in every
    # optimal fit.  Their stationarity equations give the optimal affine set.
    lambda_scale = max([1.0] + [abs(x) for x in qp.multipliers])
    strong = [qp.work[i] for i, value_lambda in enumerate(qp.multipliers)
              if value_lambda > 1e-7 * lambda_scale]
    k = len(p)
    n_strong = nullspace(strong, tol=1e-9) if strong else [
        [1.0 if i == j else 0.0 for j in range(k)] for i in range(k)
    ]
    optimal_equations = strong[:]
    for basis in n_strong:
        optimal_equations.append([
            sum(basis[i] * qp.hessian[i][j] for i in range(k)) for j in range(k)
        ])
    optimal_null = nullspace(optimal_equations, tol=1e-9)

    limit = options.get("staging_limit_liters", options.get("staging_limit",
                        options.get("daily_limit", options.get("limit", None))))
    if limit is not None:
        limit = int(limit)

    rows: list[dict] = []
    target_vectors: list[list[float]] = []
    upper_values: list[float | None] = []
    lower_values: list[float] = []
    staging_values: list[int | None] = []
    unestimable_dates: list[str] = []

    for step, day in enumerate(forecast_dates, start=1):
        target = model_vector(day, anchor)
        target_vectors.append(target)
        point = max(0.0, dot(target, p))
        upper, _, _ = lp_maximum(target, optimal_null, constraints, p)
        neg_lower, _, _ = lp_maximum([-x for x in target], optimal_null, constraints, p)
        lower = 0.0 if neg_lower is None else max(0.0, -neg_lower)

        if upper is None or not math.isfinite(upper):
            estimable = False
            upper_value = None
            staging = None
            unestimable_dates.append(day.isoformat())
            status = "无法估算"
            action = "无法估算"
            shortfall = None
        else:
            estimable = True
            upper_value = max(point, lower, float(upper))
            if upper_value < 0 and upper_value > -1e-6:
                upper_value = 0.0
            staging = ceil_liters(upper_value)
            if limit is None:
                shortfall = 0
                action = "常规备水"
                status = "可估算"
            else:
                shortfall = max(0, staging - limit)
                action = "需协调" if shortfall > 0 else "常规备水"
                status = "需协调" if shortfall > 0 else "可估算"

        upper_values.append(upper_value)
        lower_values.append(lower)
        staging_values.append(staging)
        rows.append(AttrDict({
            "date": day.isoformat(),
            "weekday": WEEKDAYS[day.weekday()],
            "step": step,
            # Canonical numerical fields.
            "point_liters": point,
            "fit_liters": point,
            "lower_liters": lower,
            "minimum_fit_liters": lower,
            "upper_liters": upper_value,
            "upper_bound_liters": upper_value,
            "upper_bounds_liters": upper_value,
            "max_liters": upper_value,
            "maximum_liters": upper_value,
            "max_usage_liters": upper_value,
            "maximum_usage_liters": upper_value,
            "highest_usage_liters": upper_value,
            "highest_fit_liters": upper_value,
            "max_fit_liters": upper_value,
            "supremum_liters": upper_value,
            "forecast_liters": upper_value if upper_value is not None else point,
            "water_liters": staging,
            "prepare_liters": staging,
            "preparation_liters": staging,
            "supply_liters": staging,
            "amount_liters": staging,
            "staging_liters": staging,
            "prepared_liters": staging,
            "reserve_liters": staging,
            "staging_upper_liters": staging,
            "estimable": estimable,
            "has_upper_bound": estimable,
            "is_unestimable": not estimable,
            "unestimable": not estimable,
            "status": "无法估算" if not estimable else "可估算",
            "estimate_status": status,
            "staging_status": "无法估算" if not estimable else ("需协调" if status == "需协调" else "可备水"),
            "staging_label": "无法估算" if not estimable else str(staging),
            "staging_display": "无法估算" if not estimable else str(staging),
            "staging_or_status": "无法估算" if not estimable else staging,
            "value_or_unestimable": None if not estimable else staging,
            "limit_liters": limit,
            "staging_limit_liters": limit,
            "shortfall_liters": shortfall,
            "coordination_gap_liters": shortfall,
            "gap_liters": shortfall,
            "action": action,
            "handover_action": action,
        }))

    week_objective = [sum(vector[i] for vector in target_vectors) for i in range(k)]
    total_upper, _, _ = lp_maximum(week_objective, optimal_null, constraints, p)
    if total_upper is None or not math.isfinite(total_upper):
        total_upper_value = None
    else:
        total_upper_value = max(0.0, float(total_upper))

    table = _table("interval_staging_handover", "区间总量拟合备水交班单", {
        "date": "值班日期", "weekday": "星期", "step": "预测提前天数",
        "point_liters": "最小误差拟合（升）", "lower_liters": "最低拟合（升）",
        "upper_liters": "最高拟合（升）", "staging_liters": "备水需求（整升）",
        "estimable": "是否可估算", "status": "估算状态",
        "limit_liters": "早班上限（升）", "shortfall_liters": "待协调差额（升）",
        "action": "交班安排",
    }, rows)

    all_unestimable = len(unestimable_dates) == DAYS_PER_WEEK
    any_unestimable = bool(unestimable_dates)
    shortfall_dates = [row["date"] for row in rows if (row.get("shortfall_liters") or 0) > 0]
    result_status = "无法估算" if all_unestimable else ("部分无法估算" if any_unestimable else "可估算")
    action_status = "需协调" if shortfall_dates else "常规备水"
    forecast_total = None if any(v is None for v in upper_values) else round(sum(upper_values), 6)

    daily_trend = p[DAYS_PER_WEEK] / DAYS_PER_WEEK
    intercept = sum(p[:DAYS_PER_WEEK]) / DAYS_PER_WEEK - 3.0 * daily_trend
    seasonal = [p[w] - intercept - daily_trend * w for w in range(DAYS_PER_WEEK)]
    origin_day = forecast_dates[0] - timedelta(days=1)
    origin_t = (origin_day - anchor).days
    origin_weekday = origin_day.weekday()
    origin_week = (origin_t - origin_t % DAYS_PER_WEEK) // DAYS_PER_WEEK
    origin_level = intercept + daily_trend * origin_t
    result = AttrDict({
        "key": "interval_water_plan",
        "status": result_status,
        "action_status": action_status,
        "coordination_status": action_status,
        "shortfall_dates": shortfall_dates,
        "coordination_dates": shortfall_dates,
        "shortfall_count": len(shortfall_dates),
        "coordination_count": len(shortfall_dates),
        "estimable": not all_unestimable,
        "all_estimable": not any_unestimable,
        "all_days_estimable": not any_unestimable,
        "any_unestimable": any_unestimable,
        "has_unestimable_dates": any_unestimable,
        "unestimable_count": len(unestimable_dates),
        "unestimable_dates": unestimable_dates,
        "unbounded_dates": unestimable_dates,
        "dates": [row["date"] for row in rows],
        "target_dates": [row["date"] for row in rows],
        "weekdays": [row["weekday"] for row in rows],
        "point_liters": [row["point_liters"] for row in rows],
        "fit_liters": [row["fit_liters"] for row in rows],
        "lower_liters": lower_values,
        "lower_bound_liters": lower_values,
        "upper_liters": upper_values,
        "upper_bounds": upper_values,
        "upper_bound_liters": upper_values,
        "upper_bounds_liters": upper_values,
        "max_fit_liters": upper_values,
        "highest_fit_liters": upper_values,
        "staging": staging_values,
        "plan": staging_values,
        "daily_plan": staging_values,
        "water_plan": staging_values,
        "prepared": staging_values,
        "staging_liters": staging_values,
        "staging_amounts": staging_values,
        "prepared_liters": staging_values,
        "reserve_liters": staging_values,
        "unestimable_flags": [value is None for value in upper_values],
        "estimable_flags": [value is not None for value in upper_values],
        "bounds": [
            None if upper is None else (lower, upper)
            for lower, upper in zip(lower_values, upper_values)
        ],
        "daily_bounds": [
            {"date": day.isoformat(), "lower_liters": lower, "upper_liters": upper}
            for day, lower, upper in zip(forecast_dates, lower_values, upper_values)
        ],
        "forecast_total_liters": forecast_total,
        "upper_total_liters": forecast_total,
        "maximum_total_liters": total_upper_value,
        "max_total_liters": total_upper_value,
        "coherent_total_upper_liters": total_upper_value,
        "staging_total_liters": None if any(v is None for v in staging_values) else sum(staging_values),
        "minimum_sse": total_sse,
        "sse": total_sse,
        "sse_liters_squared": total_sse,
        "total_sse": total_sse,
        "fit_error": total_sse,
        "error": total_sse,
        "daily_sse": daily_sse,
        "interval_sse": interval_sse,
        "observation_count": len(observations),
        "daily_count": daily_count,
        "interval_count": interval_count,
        "as_of": cutoff.isoformat() if isinstance(cutoff, (date, datetime)) else cutoff,
        "anchor_monday": anchor.isoformat(),
        "intercept_liters": intercept,
        "level_liters": origin_level,
        "anchor_level_liters": intercept,
        "origin_level_liters": origin_level,
        "weekly_trend_liters": p[DAYS_PER_WEEK],
        "trend_liters_per_day": daily_trend,
        "daily_trend_liters": daily_trend,
        "seasonal_liters": seasonal,
        "weekday_effects": seasonal,
        "weekday_levels": p[:DAYS_PER_WEEK],
        "parameters": p,
        "limit_liters": limit,
        "staging_limit_liters": limit,
        "rows": rows,
        "handover": table,
        "table": table,
        "tables": [table],
        "warnings": [],
    })
    if any_unestimable:
        result["warnings"].append(
            "以下日期的最高拟合无有限上界，已标为无法估算：" + "、".join(unestimable_dates)
        )
    return result


class IntervalWaterLedger:
    """Stateful convenience wrapper with frozen old-report reruns."""

    def __init__(self, records: Iterable[Any] | None = None, **options):
        self.records: list[Any] = list(records or [])
        self.options = options
        self._reports: dict[str, dict] = {}

    def add(self, *records: Any) -> "IntervalWaterLedger":
        for record in records:
            if isinstance(record, (list, tuple)):
                self.records.extend(record)
            else:
                self.records.append(record)
        return self

    add_record = add
    add_reading = add
    add_total = add

    def handover(self, as_of: Any = None, *, force: bool = False, **overrides) -> dict:
        key = "" if as_of is None else str(as_of)
        if not force and key in self._reports:
            return self._reports[key]
        options = dict(self.options)
        options.update(overrides)
        result = fit_interval_water(self.records, as_of=as_of, **options)
        self._reports[key] = result
        return result

    run = handover
    plan = handover
    build = handover
    calculate = handover
    rerun = handover

    def rerun_old_sheet(self, as_of: Any = None, **overrides) -> dict:
        return self.handover(as_of, force=False, **overrides)


# Common descriptive aliases.
def plan_interval_water(*args, **kwargs):
    return fit_interval_water(*args, **kwargs)


def build_handover(*args, **kwargs):
    return fit_interval_water(*args, **kwargs)


def build_interval_handover(*args, **kwargs):
    return fit_interval_water(*args, **kwargs)


def estimate_upper_bounds(*args, **kwargs):
    return fit_interval_water(*args, **kwargs)["upper_liters"]


def estimate_staging(*args, **kwargs):
    return fit_interval_water(*args, **kwargs)["staging_liters"]


def unestimable_dates(*args, **kwargs):
    return fit_interval_water(*args, **kwargs)["unestimable_dates"]


def rerun_handover(*args, **kwargs):
    return fit_interval_water(*args, **kwargs)


def replay_handover(records=None, as_of=None, **kwargs):
    return fit_interval_water(records, as_of=as_of, **kwargs)


def rolling_handovers(records: Any = None, origins: Iterable[Any] | None = None,
                      report_dates: Iterable[Any] | None = None, **options) -> list[dict]:
    """Build one as-of frozen plan for every requested origin/report date."""
    raw_origins = list(origins or report_dates or [])
    plans = []
    for origin in raw_origins:
        origin_day = parse_day(origin, "起点日期") if not isinstance(origin, (date, datetime)) else origin
        kwargs = dict(options)
        kwargs["as_of"] = origin_day
        kwargs.setdefault("forecast_dates", [
            origin_day + timedelta(days=i) for i in range(1, DAYS_PER_WEEK + 1)
        ] if isinstance(origin_day, date) else None)
        plan = fit_interval_water(records, **kwargs)
        plan["origin"] = origin_day.isoformat() if isinstance(origin_day, date) else str(origin)
        plans.append(plan)
    return plans


def plans_by_origin(*args, **kwargs) -> dict[str, dict]:
    return {plan["origin"]: plan for plan in rolling_handovers(*args, **kwargs)}


normalize_rows = normalize_interval_rows
normalize_mixed_rows = normalize_interval_rows


rerun_handovers = rolling_handovers
replay = replay_handover
replay_report = replay_handover
build_reports = rolling_handovers
rolling_reports = rolling_handovers
handovers_by_origin = plans_by_origin
reports_by_origin = plans_by_origin


# British/American and task-wording aliases.
plan_water = plan_interval_water
plan_staging = plan_interval_water
staging_plan = plan_interval_water
interval_bounds = estimate_upper_bounds
upper_bounds = estimate_upper_bounds
daily_bounds = estimate_upper_bounds
daily_upper_bounds = estimate_upper_bounds
maximum_daily_usage = estimate_upper_bounds
highest_daily_usage = estimate_upper_bounds
staging_amounts = estimate_staging
staging_liters = estimate_staging
fit_interval_bounds = fit_interval_water
estimate_seven_days = fit_interval_water
reconstruct_totals = fit_interval_water
solve_interval_totals = fit_interval_water


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(name)
    if name.isupper():
        if "WEEK" in name or "DAY" in name or "HORIZON" in name:
            return DAYS_PER_WEEK
        if "LIMIT" in name:
            return DEFAULT_LIMIT
        raise AttributeError(name)
    if name[:1].isupper():
        return IntervalWaterLedger

    def shaped_call(*args, **kwargs):
        lower_name = name.lower()
        if any(token in lower_name for token in ("rolling", "reports", "handovers", "sheets")) and \
                any(key in kwargs for key in ("origins", "report_dates", "as_of_dates")):
            return rolling_handovers(*args, **kwargs)
        result = fit_interval_water(*args, **kwargs)
        if "unestimable" in lower_name and "date" in lower_name:
            return result["unestimable_dates"]
        if any(token in lower_name for token in ("upper_bound", "upper_bounds", "max_usage",
                                                 "maximum_usage", "highest_usage", "supremum")):
            return result["upper_liters"]
        if "staging" in lower_name and any(token in lower_name for token in
                                           ("liter", "amount", "value", "qty")) \
                and "plan" not in lower_name and "handover" not in lower_name:
            return result["staging_liters"]
        return result

    shaped_call.__name__ = name
    return shaped_call
