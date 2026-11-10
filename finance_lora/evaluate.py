"""Score raw financial extraction outputs against a labeled JSONL split."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from decimal import Decimal
from pathlib import Path

AMOUNT_FIELDS = ("revenue", "net_profit")
SCALAR_FIELDS = ("company_name", *AMOUNT_FIELDS)
LIST_FIELDS = ("risks", "operating_changes", "key_metrics")
FIELDS = (*SCALAR_FIELDS, *LIST_FIELDS)
_GOLD_FIELDS = {"id", "company_id", "template_id", "text", "target"}
_UNIT_EXPONENTS = {"元": 0, "万元": 4, "亿元": 8}
_AMOUNT = re.compile(
    r"(?P<number>[+-]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?)"
    r"\s*(?P<unit>亿元|万元|元)"
)


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate JSON object key")
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-JSON numeric constant: {value}")


def _parse_json(raw: str):
    return json.loads(
        raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant
    )


def _nonblank(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        value.encode("utf-8")
    except UnicodeError:
        return False
    return True


def normalize_amount(value: str) -> Decimal:
    """Convert a signed yuan-denominated amount to yuan without rounding."""
    cleaned = unicodedata.normalize("NFKC", value).strip().replace("−", "-")
    match = _AMOUNT.fullmatch(cleaned)
    if match is None:
        raise ValueError(
            "Expected a signed or unsigned decimal amount in 元, 万元, or 亿元"
        )
    sign, digits, exponent = Decimal(match["number"].replace(",", "")).as_tuple()
    return Decimal((sign, digits, exponent + _UNIT_EXPONENTS[match["unit"]]))


def validate_target(target: object) -> None:
    """Require the six extraction fields and their declared types and amount units."""
    if not isinstance(target, dict) or target.keys() != set(FIELDS):
        raise ValueError("Expected exactly these fields: " + ", ".join(FIELDS))
    for field in SCALAR_FIELDS:
        value = target[field]
        if value is not None and not _nonblank(value):
            raise ValueError(f"{field} must be null or a nonempty string")
        if value is not None and field in AMOUNT_FIELDS:
            try:
                normalize_amount(value)
            except ValueError as error:
                raise ValueError(f"{field}: {error}") from error
    for field in ("risks", "operating_changes"):
        if not isinstance(target[field], list) or any(
            not _nonblank(item) for item in target[field]
        ):
            raise ValueError(f"{field} must be an array of nonempty strings")
    if not isinstance(target["key_metrics"], list):
        raise ValueError("key_metrics must be an array")
    for metric in target["key_metrics"]:
        if (
            not isinstance(metric, dict)
            or metric.keys() != {"name", "value"}
            or not all(_nonblank(metric[key]) for key in ("name", "value"))
        ):
            raise ValueError(
                "Each key_metrics item needs nonempty name and value strings"
            )


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = _parse_json(line)
                if not isinstance(row, dict):
                    raise ValueError("Expected a JSON object")
            except (ValueError, RecursionError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            rows.append(row)
    return rows


def _index_rows(rows: list[dict], label: str) -> dict[str, dict]:
    index = {}
    for line_number, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or not _nonblank(row.get("id")):
            raise ValueError(f"{label} row {line_number} requires a nonempty string id")
        row_id = row["id"]
        if row_id in index:
            raise ValueError(f"Duplicate {label} ID: {row_id!r}")
        index[row_id] = row
    return index


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _set_items(field: str, values: list) -> set:
    if field == "key_metrics":
        return {(item["name"], item["value"]) for item in values}
    return set(values)


def _list_metrics(counts: Counter) -> dict:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
    }


def _amount_metrics(counts: Counter) -> dict:
    correct, eligible = counts["correct"], counts["eligible"]
    return {
        "correct": correct,
        "eligible": eligible,
        "accuracy": _ratio(correct, eligible),
    }


def _decode_prediction(raw: object) -> tuple[dict | None, dict | None]:
    if not isinstance(raw, str):
        return None, {
            "error": "invalid_json",
            "message": "prediction must contain a raw JSON string",
        }
    try:
        value = _parse_json(raw)
    except (ValueError, RecursionError) as error:
        return None, {"error": "invalid_json", "message": str(error)}
    try:
        validate_target(value)
    except ValueError as error:
        return None, {"error": "invalid_schema", "message": str(error)}
    return value, None


def evaluate(
    gold_rows: list[dict], prediction_rows: list[dict]
) -> tuple[dict, list[dict]]:
    """Return aggregate metrics and errors ordered by gold ID."""
    gold = _index_rows(gold_rows, "gold")
    predictions = _index_rows(prediction_rows, "prediction")
    if not gold:
        raise ValueError("Gold dataset must contain at least one sample")
    absent = sorted(gold.keys() - predictions.keys())
    extra = sorted(predictions.keys() - gold.keys())
    if absent or extra:
        raise ValueError(
            f"Prediction IDs differ from gold ({len(absent)} missing, {len(extra)} extra). "
            f"Missing examples: {absent[:5]!r}. Extra examples: {extra[:5]!r}."
        )

    sample_count = len(gold)
    json_count = schema_count = exact_count = 0
    field_matches: Counter = Counter()
    list_counts = {field: Counter() for field in LIST_FIELDS}
    amount_counts = {field: Counter() for field in AMOUNT_FIELDS}
    presence: Counter = Counter()
    errors = []
    for row_id in sorted(gold):
        row = gold[row_id]
        if row.keys() != _GOLD_FIELDS or not all(
            _nonblank(row[key]) for key in ("company_id", "template_id", "text")
        ):
            raise ValueError(
                f"Gold ID {row_id!r} requires company_id, template_id, text, and target"
            )
        target = row["target"]
        try:
            validate_target(target)
        except ValueError as error:
            raise ValueError(
                f"Invalid target for gold ID {row_id!r}: {error}"
            ) from error

        prediction_row = predictions[row_id]
        if prediction_row.keys() - {"id", "prediction"}:
            raise ValueError(f"Unexpected prediction envelope fields for ID {row_id!r}")
        prediction, error = _decode_prediction(prediction_row.get("prediction"))
        mismatches = [
            field
            for field in FIELDS
            if prediction is None or prediction[field] != target[field]
        ]
        json_count += error is None or error["error"] != "invalid_json"
        schema_count += prediction is not None
        exact_count += not mismatches
        field_matches.update(field for field in FIELDS if field not in mismatches)
        if mismatches:
            errors.append(
                {
                    "id": row_id,
                    **(error or {"error": "field_mismatch"}),
                    "fields": mismatches,
                }
            )

        for field in FIELDS:
            expected = target[field]
            observed = prediction[field] if prediction is not None else None
            if expected is not None and expected != []:
                presence["present"] += 1
                presence["missing"] += observed is None or observed == []
            elif prediction is None:
                presence["invalid_absent"] += 1
            else:
                presence["absent"] += 1
                presence["hallucinated"] += observed is not None and observed != []

        for field in LIST_FIELDS:
            expected = _set_items(field, target[field])
            observed = (
                _set_items(field, prediction[field])
                if prediction is not None
                else set()
            )
            list_counts[field].update(
                tp=len(expected & observed),
                fp=len(observed - expected),
                fn=len(expected - observed),
            )
        for field in AMOUNT_FIELDS:
            if target[field] is not None:
                amount_counts[field]["eligible"] += 1
                if prediction is not None and prediction[field] is not None:
                    amount_counts[field]["correct"] += normalize_amount(
                        target[field]
                    ) == normalize_amount(prediction[field])

    metrics = {
        "sample_count": sample_count,
        "valid_json_rate": json_count / sample_count,
        "valid_schema_rate": schema_count / sample_count,
        "exact_match": exact_count / sample_count,
        "field_accuracy": {
            field: field_matches[field] / sample_count for field in FIELDS
        },
        "list_micro_f1": {
            "overall": _list_metrics(sum(list_counts.values(), Counter())),
            **{field: _list_metrics(counts) for field, counts in list_counts.items()},
        },
        "numeric_amount_accuracy": {
            "overall": _amount_metrics(sum(amount_counts.values(), Counter())),
            **{
                field: _amount_metrics(counts)
                for field, counts in amount_counts.items()
            },
        },
        "hallucination": {
            "rate": _ratio(presence["hallucinated"], presence["absent"]),
            "hallucinated": presence["hallucinated"],
            "eligible": presence["absent"],
            "invalid_schema_excluded": presence["invalid_absent"],
        },
        "missing": {
            "rate": _ratio(presence["missing"], presence["present"]),
            "missing": presence["missing"],
            "eligible": presence["present"],
        },
    }
    return metrics, errors


def _write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="Metrics JSON path")
    parser.add_argument(
        "--errors", type=Path, help="Optional per-sample errors JSONL path"
    )
    args = parser.parse_args(argv)
    try:
        inputs = {args.gold.resolve(), args.predictions.resolve()}
        outputs = [
            path.resolve() for path in (args.out, args.errors) if path is not None
        ]
        if inputs.intersection(outputs) or len(set(outputs)) != len(outputs):
            raise ValueError(
                "Output paths must be distinct from each other and from input paths"
            )
        metrics, errors = evaluate(read_jsonl(args.gold), read_jsonl(args.predictions))
        _write(
            args.out,
            json.dumps(
                metrics, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
            )
            + "\n",
        )
        if args.errors is not None:
            lines = (
                json.dumps(error, sort_keys=True, ensure_ascii=False) + "\n"
                for error in errors
            )
            _write(args.errors, "".join(lines))
    except (OSError, ValueError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
