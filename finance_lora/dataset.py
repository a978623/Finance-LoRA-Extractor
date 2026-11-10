"""Generate a reproducible, entirely fictional Chinese financial IE benchmark."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile


DEFAULT_SEED = 2026
TRAIN_PREFIX_SIZES = (128, 512, 1024)
SPLIT_SIZES = {"train": 1024, "validation": 64, "test": 128, "stress": 48}
STANDARD_TEMPLATES = ("brief", "table", "numbered", "narrative")
STRESS_TEMPLATES = (
    "meeting_dialogue",
    "corrected_notice",
    "peer_comparison",
    "prior_values_only",
    "anonymous_note",
    "loss_written_in_words",
)

_NAME_STEMS = ("岚栈", "苇汀", "青岫", "砂鹭", "榆帆", "云砾", "梧岬", "竹垣")
_RISKS = (
    "核心原料采购价持续波动",
    "前五大客户收入占比较高",
    "主要外币结算汇率波动",
    "部分应收账款账期延长",
    "关键元件交付周期拉长",
    "新增产线良品率仍不稳定",
    "替代供应商验证尚未完成",
    "部分存货周转放缓",
    "新增服务业务合规成本上升",
    "设备维护支出可能增加",
    "部分订单对单一路线依赖较大",
    "淡季产能利用率可能下滑",
)
_CHANGES = (
    "新增两条检测产线",
    "区域配送中心投入使用",
    "停止承接低毛利代工订单",
    "完成生产设备数字化改造",
    "直销渠道覆盖范围扩大",
    "将包装工序转为内部生产",
    "维修网点增至九个",
    "新增区域仓储合作伙伴",
    "跨境订单改为分仓发货",
    "新版售后管理系统上线",
    "将夜班产能调配至白班",
    "完成第一批材料替换试产",
)
_STRESS_RISKS = (
    "第二供货渠道尚未完成耐久验证",
    "零部件到港延迟可能压缩备货时间",
    "关键订单的验收条款仍待确认",
    "区域运费继续上行或抬高履约成本",
    "回款集中在季度末增加了资金周转压力",
    "新版标签切换可能造成短期退货",
    "老化测试产能暂时无法覆盖全部订单",
    "仓库租约续签价格尚不确定",
)
_STRESS_CHANGES = (
    "将零散工单合并为每周两批处理",
    "在样品入库环节增设条码复核",
    "合并三处备品库并调整补货频次",
    "暂停外箱定制业务并集中现有订单",
    "首次启用售后配件循环回收流程",
    "将质检安排前移至每批发货之前",
    "自营渠道改为分区核算",
    "把试制车间迁到原有厂区旁",
)
# name, inclusive fixed-point bounds, decimal places, unit
_METRICS = (
    ("毛利率", 126, 634, 1, "%"),
    ("资产负债率", 236, 698, 1, "%"),
    ("存货周转率", 21, 96, 1, "次"),
    ("经营活动现金净额", 260, 3280, 0, "万元"),
    ("流动比率", 113, 238, 2, "倍"),
    ("净资产收益率", 26, 219, 1, "%"),
    ("应收账款周转天数", 27, 112, 0, "天"),
    ("研发投入", 48, 1260, 0, "万元"),
)


@dataclass(frozen=True)
class _FinancialExample:
    company_name: str | None
    revenue: str | None
    net_profit: str | None
    risks: list[str]
    operating_changes: list[str]
    key_metrics: list[dict[str, str]]
    old_revenue: str
    old_profit: str
    old_metric: str

    def target(self) -> dict:
        return {
            "company_name": self.company_name,
            "revenue": self.revenue,
            "net_profit": self.net_profit,
            "risks": self.risks,
            "operating_changes": self.operating_changes,
            "key_metrics": self.key_metrics,
        }


def _fixed_point(value: int, decimals: int, unit: str) -> str:
    if decimals == 0:
        return f"{value}{unit}"
    integer, fraction = divmod(value, 10**decimals)
    return f"{integer}.{fraction:0{decimals}d}{unit}"


def _money(rng: random.Random, is_profit: bool, unit: str) -> str:
    if unit == "元":
        low, high = (18_000, 852_000) if is_profit else (2_110_000, 93_370_000)
        decimals = 0
    elif unit == "万元":
        low, high = (185, 8230) if is_profit else (18_750, 93_260)
        decimals = 1
    else:
        low, high = (12, 2208) if is_profit else (2280, 9864)
        decimals = 2
    return _fixed_point(rng.randint(low, high), decimals, unit)


def _example(
    rng: random.Random, company_index: int, template: str, row: int, stress: bool
) -> _FinancialExample:
    # The multiplier is coprime with 10,000, so the visible suffix stays unique.
    suffix = (company_index * 7919 + 4729) % 10_000
    company = f"虚构{rng.choice(_NAME_STEMS)}{suffix:04d}公司"
    unit = rng.choice(("万元", "万元", "万元", "亿元", "元"))
    profit_unit = "万元" if unit == "亿元" and rng.random() < 0.25 else unit
    revenue = _money(rng, False, unit)
    profit = _money(rng, True, profit_unit)
    old_revenue = _money(rng, False, unit)
    while old_revenue == revenue:
        old_revenue = _money(rng, False, unit)
    old_profit = _money(rng, True, profit_unit)
    while old_profit == profit:
        old_profit = _money(rng, True, profit_unit)

    if rng.random() < 0.28 or template == "loss_written_in_words":
        profit = f"-{profit}"
    risks = rng.sample(_STRESS_RISKS if stress else _RISKS, rng.choice((0, 1, 1, 2, 2)))
    changes = rng.sample(
        _STRESS_CHANGES if stress else _CHANGES, rng.choice((0, 1, 1, 2, 2))
    )
    metric_specs = rng.sample(_METRICS, rng.choice((0, 1, 2, 2, 3)))
    metrics = [
        {
            "name": name,
            "value": _fixed_point(rng.randint(low, high), decimals, metric_unit),
        }
        for name, low, high, decimals, metric_unit in metric_specs
    ]
    old_metric = ""
    if metric_specs:
        name, low, high, decimals, metric_unit = metric_specs[0]
        old_value = _fixed_point(rng.randint(low, high), decimals, metric_unit)
        while old_value == metrics[0]["value"]:
            old_value = _fixed_point(rng.randint(low, high), decimals, metric_unit)
        old_metric = f"、{name}{old_value}"

    if template == "prior_values_only":
        revenue = None
        if row % 2 == 0:
            profit = None
    elif template == "anonymous_note":
        company = None
        if row % 2 == 0:
            revenue = None
            profit = None
        if row % 4 == 0:
            risks, changes, metrics = [], [], []
    elif not stress:
        if rng.random() < 0.09:
            revenue = None
        if rng.random() < 0.10:
            profit = None

    return _FinancialExample(
        company,
        revenue,
        profit,
        risks,
        changes,
        metrics,
        old_revenue,
        old_profit,
        old_metric,
    )


def _render(template: str, example: _FinancialExample, row: int) -> str:
    name = example.company_name
    revenue = example.revenue or "未披露"
    profit = example.net_profit or "未披露"
    risks = "；".join(example.risks) or "未披露具体事项"
    changes = "；".join(example.operating_changes) or "未披露具体事项"
    metrics = (
        "；".join(f"{item['name']}{item['value']}" for item in example.key_metrics)
        or "未披露"
    )
    current = f"本期营业收入{revenue}，净利润{profit}"
    prior = (
        f"营业收入{example.old_revenue}、净利润{example.old_profit}{example.old_metric}"
    )
    if template == "brief":
        comparison = f"上期对比：{prior}。\n" if row % 3 == 0 else ""
        return (
            f"{name}财报摘要\n{current}。\n{comparison}"
            f"风险提示：{risks}。\n经营变化：{changes}。\n本期关键指标：{metrics}。"
        )
    if template == "table":
        return (
            f"企业：{name}\n项目｜本期｜上期\n营业收入｜{revenue}｜{example.old_revenue}\n"
            f"净利润｜{profit}｜{example.old_profit}\n指标（本期）｜{metrics}\n"
            f"经营事项｜{changes}\n风险事项｜{risks}"
        )
    if template == "numbered":
        return (
            f"经营简报（{name}）\n（一）业务调整：{changes}。\n"
            f"（二）本期净利润为{profit}，营业收入为{revenue}。\n"
            f"（三）风险因素：{risks}。\n（四）本期指标：{metrics}。"
        )
    if template == "narrative":
        return (
            f"{name}在本期说明中披露，营业收入为{revenue}，净利润为{profit}。"
            f"指标方面，本期为{metrics}。业务运行的变化包括{changes}。"
            f"报告提醒需留意以下风险：{risks}。"
        )
    if template == "meeting_dialogue":
        return (
            f"经营会速记\n主持：今天只看{name}。\n财务：按这次结账口径，{current}；补充读数是{metrics}。\n"
            f"运营：这段时间实际做了这些调整——{changes}。\n风控：请把以下不确定因素记入——{risks}。"
        )
    if template == "corrected_notice":
        return (
            f"{name}发布补记。草稿曾列{prior}，该稿已撤回。"
            f"请以这组数为准：{current}，本期指标{metrics}。"
            f"补记的业务变动为{changes}，提示留意{risks}。"
        )
    if template == "peer_comparison":
        return (
            f"横向复盘小条：对标样本另有一组{prior}，仅供比较。"
            f"本条记录对象为{name}，其{current}，本期测得{metrics}。"
            f"对象在经营端记录了{changes}；在风险端记录了{risks}。"
        )
    if template == "prior_values_only":
        profit_note = (
            f"本期可引用的净利润为{profit}。"
            if example.net_profit
            else "本期净利润暂不对外发布。"
        )
        return (
            f"给审稿人的页边便签｜{name}。附表数字是去年存档：{prior}。"
            f"本期营业收入尚未公开。{profit_note}能引用的本期指标为{metrics}。"
            f"业务端确认{changes}，尚须关注{risks}。"
        )
    if template == "anonymous_note":
        if not any(example.target().values()):
            return f"匿名备忘第{row + 1}条。收到的只有通用问卷和空白附件。所有本期财务数及经营、风险细节均未公布。"
        return (
            f"匿名现场记录：文件没有落款或企业信息。{current}；本期另有读数{metrics}。"
            f"现场确认的调整包括{changes}；仍需留意{risks}。"
        )
    if template == "loss_written_in_words":
        return (
            f"报表页脚译注｜报送方{name}。这一期的营业收入记为{revenue}。"
            f"净利润口径录为亏损，亏损金额{profit.removeprefix('-')}。"
            f"其余本期量化读数：{metrics}。经营侧更新：{changes}。风险附注：{risks}。"
        )
    raise ValueError(f"Unknown template: {template}")


def _generate_rows(seed: int, split: str, company_offset: int) -> list[dict]:
    rng = random.Random(f"{seed}:{split}")
    stress = split == "stress"
    templates = STRESS_TEMPLATES if stress else STANDARD_TEMPLATES
    rows = []
    for index in range(SPLIT_SIZES[split]):
        company_index = company_offset + index + 1
        template = templates[index % len(templates)]
        row = index // len(templates)
        example = _example(rng, company_index, template, row, stress)
        rows.append(
            {
                "id": f"{split}-{index + 1:04d}",
                "company_id": f"synthetic-{company_index:04d}",
                "template_id": template,
                "text": _render(template, example, row),
                "target": example.target(),
            }
        )
    return rows


def _write_bytes(path: Path, contents: bytes) -> None:
    temporary_path = None
    try:
        with NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(contents)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def generate_dataset(out: str | Path, seed: int = DEFAULT_SEED) -> dict:
    """Write the four splits and a manifest, then return the manifest contents."""
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "synthetic": True,
        "train_prefix_sizes": list(TRAIN_PREFIX_SIZES),
        "splits": {},
    }
    company_offset = 0
    for split in SPLIT_SIZES:
        rows = _generate_rows(seed, split, company_offset)
        company_offset += len(rows)
        filename = f"{split}.jsonl"
        contents = "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in rows
        ).encode("utf-8")
        _write_bytes(out_path / filename, contents)
        manifest["splits"][split] = {
            "file": filename,
            "count": len(rows),
            "sha256": sha256(contents).hexdigest(),
            "template_ids": sorted({row["template_id"] for row in rows}),
        }
    _write_bytes(
        out_path / "manifest.json",
        (json.dumps(manifest, indent=2) + "\n").encode("utf-8"),
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("data/generated"), help="Output directory"
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="Synthetic data seed"
    )
    args = parser.parse_args()
    manifest = generate_dataset(args.out, args.seed)
    counts = ", ".join(
        f"{split}={details['count']}" for split, details in manifest["splits"].items()
    )
    print(f"Wrote {counts} to {args.out}")


if __name__ == "__main__":
    main()
