import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from finance_lora.evaluate import evaluate, main, normalize_amount, read_jsonl


def target(**overrides):
    return {
        "company_name": None,
        "revenue": None,
        "net_profit": None,
        "risks": [],
        "operating_changes": [],
        "key_metrics": [],
        **overrides,
    }


def gold(row_id, expected):
    return {
        "id": row_id,
        "company_id": "c1",
        "template_id": "t1",
        "text": "合成财报片段。",
        "target": expected,
    }


def prediction(row_id, output):
    return {"id": row_id, "prediction": json.dumps(output, ensure_ascii=False)}


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class EvaluateTests(unittest.TestCase):
    def test_metrics_have_explicit_denominators(self):
        first = target(
            company_name="甲公司",
            revenue="+1.20亿元",
            net_profit="-300万元",
            risks=["供应链风险", "汇率风险"],
            operating_changes=["扩产"],
            key_metrics=[{"name": "毛利率", "value": "15%"}],
        )
        first_prediction = {
            **first,
            "revenue": "12000万元",
            "net_profit": "300万元",
            "risks": ["汇率风险", "供应链风险", "新风险", "供应链风险"],
            "operating_changes": [],
            "key_metrics": [{"name": "毛利率", "value": "16%"}],
        }
        metrics, errors = evaluate(
            [
                gold("3", target(company_name="乙公司")),
                gold("1", first),
                gold("2", target()),
            ],
            [
                prediction("1", first_prediction),
                {"id": "2", "prediction": None},
                prediction(
                    "3",
                    target(company_name="乙公司", revenue="1元", risks=["假设风险"]),
                ),
            ],
        )
        self.assertEqual(metrics["sample_count"], 3)
        self.assertEqual(metrics["valid_json_rate"], 2 / 3)
        self.assertEqual(metrics["valid_schema_rate"], 2 / 3)
        self.assertEqual(metrics["exact_match"], 0)
        self.assertEqual(metrics["field_accuracy"]["company_name"], 2 / 3)
        self.assertEqual(metrics["field_accuracy"]["revenue"], 0)
        self.assertEqual(metrics["field_accuracy"]["net_profit"], 1 / 3)
        amounts = metrics["numeric_amount_accuracy"]
        self.assertEqual(
            amounts["overall"], {"correct": 1, "eligible": 2, "accuracy": 0.5}
        )
        self.assertEqual(amounts["revenue"]["accuracy"], 1)
        self.assertEqual(amounts["net_profit"]["accuracy"], 0)
        self.assertEqual(
            metrics["list_micro_f1"]["overall"],
            {
                "tp": 2,
                "fp": 3,
                "fn": 2,
                "f1": 4 / 9,
                "precision": 2 / 5,
                "recall": 1 / 2,
            },
        )
        self.assertEqual(
            metrics["hallucination"],
            {
                "hallucinated": 2,
                "eligible": 5,
                "rate": 2 / 5,
                "invalid_schema_excluded": 6,
            },
        )
        self.assertEqual(
            metrics["missing"], {"missing": 1, "eligible": 7, "rate": 1 / 7}
        )
        self.assertEqual([error["id"] for error in errors], ["1", "2", "3"])
        self.assertEqual(errors[1]["error"], "invalid_json")
        self.assertEqual(len(errors[1]["fields"]), 6)

    def test_exact_match_and_list_set_match_are_distinct(self):
        expected = target(risks=["供应链风险", "汇率风险"])
        metrics, errors = evaluate(
            [gold("1", expected), gold("2", expected)],
            [
                prediction("1", expected),
                prediction(
                    "2", {**expected, "risks": list(reversed(expected["risks"]))}
                ),
            ],
        )
        self.assertEqual(metrics["exact_match"], 0.5)
        self.assertEqual(metrics["field_accuracy"]["risks"], 0.5)
        self.assertEqual(metrics["list_micro_f1"]["risks"]["f1"], 1)
        self.assertIsNone(metrics["numeric_amount_accuracy"]["overall"]["accuracy"])
        self.assertEqual(
            errors, [{"id": "2", "error": "field_mismatch", "fields": ["risks"]}]
        )

    def test_json_and_schema_failures_receive_no_field_credit(self):
        empty = target()
        raw = json.dumps(empty)
        malformed = (
            (f"```json\n{raw}\n```", "invalid_json"),
            (raw[:-1] + ',"company_name":null}', "invalid_json"),
            (raw.replace("null", "NaN", 1), "invalid_json"),
            ("null", "invalid_schema"),
            ('{"risks": []}', "invalid_schema"),
            (json.dumps({**empty, "risks": [1]}), "invalid_schema"),
            (
                json.dumps({**empty, "key_metrics": [{"name": "ROE", "value": 0.1}]}),
                "invalid_schema",
            ),
            (json.dumps({**empty, "revenue": "10"}), "invalid_schema"),
            (json.dumps({**empty, "unexpected": None}), "invalid_schema"),
        )
        for candidate, expected_error in malformed:
            with self.subTest(candidate=candidate):
                metrics, errors = evaluate(
                    [gold("1", empty)], [{"id": "1", "prediction": candidate}]
                )
                self.assertEqual(
                    metrics["valid_json_rate"], int(expected_error == "invalid_schema")
                )
                self.assertEqual(metrics["valid_schema_rate"], 0)
                self.assertEqual(metrics["exact_match"], 0)
                self.assertEqual(set(metrics["field_accuracy"].values()), {0})
                self.assertEqual(errors[0]["error"], expected_error)

    def test_invalid_output_misses_gold_items_and_amount(self):
        expected = target(revenue="1元", risks=["供应链风险"])
        metrics, _ = evaluate([gold("1", expected)], [{"id": "1"}])
        self.assertEqual(metrics["numeric_amount_accuracy"]["overall"]["accuracy"], 0)
        self.assertEqual(metrics["list_micro_f1"]["overall"]["fn"], 1)
        self.assertEqual(metrics["list_micro_f1"]["overall"]["f1"], 0)
        self.assertEqual(metrics["missing"], {"missing": 2, "eligible": 2, "rate": 1})

    def test_amount_conversion_preserves_sign_units_and_decimal_precision(self):
        self.assertEqual(normalize_amount("−１２３．５０ 万元"), Decimal("-1235000"))
        self.assertEqual(normalize_amount("1,234.56元"), Decimal("1234.56"))
        self.assertEqual(normalize_amount("+0.00000001亿元"), Decimal("1"))
        value = "123456789012345678901234567890.001"
        self.assertEqual(normalize_amount(value + "元"), Decimal(value))
        for value in ("123", "12,34元", "100美元", "1e4元", "未披露"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_amount(value)

    def test_dataset_ids_fail_closed(self):
        expected = target()
        g = gold("1", expected)
        p = prediction("1", expected)
        for gold_rows, predictions, message in (
            ([g, g], [p], "Duplicate gold ID"),
            ([g], [p, p], "Duplicate prediction ID"),
            ([g], [], "1 missing, 0 extra"),
            ([g], [p, prediction("2", expected)], "0 missing, 1 extra"),
        ):
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                evaluate(gold_rows, predictions)

    def test_cli_writes_deterministic_metrics_and_ordered_error_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold_path = root / "gold.jsonl"
            predictions_path = root / "predictions.jsonl"
            metrics_path = root / "run" / "metrics.json"
            errors_path = root / "run" / "errors.jsonl"
            write_jsonl(gold_path, [gold("2", target()), gold("1", target())])
            write_jsonl(
                predictions_path,
                [{"id": "1", "prediction": "null"}, {"id": "2", "prediction": "bad"}],
            )
            args = [
                "--gold",
                str(gold_path),
                "--predictions",
                str(predictions_path),
                "--out",
                str(metrics_path),
                "--errors",
                str(errors_path),
            ]
            self.assertEqual(main(args), 0)
            first_write = metrics_path.read_bytes(), errors_path.read_bytes()
            self.assertEqual(main(args), 0)
            self.assertEqual(
                first_write, (metrics_path.read_bytes(), errors_path.read_bytes())
            )
            self.assertEqual(json.loads(first_write[0])["valid_json_rate"], 0.5)
            self.assertEqual([row["id"] for row in read_jsonl(errors_path)], ["1", "2"])
            predictions_path.write_text('{"id":"1","id":"2"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate JSON object key"):
                read_jsonl(predictions_path)


if __name__ == "__main__":
    unittest.main()
