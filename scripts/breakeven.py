"""Commercial versus self-hosted, computed rather than argued.

    python scripts/breakeven.py --tokens-per-sec 950

Use *your* measured throughput from Module 2, not a marketing number, and price the
ops overhead honestly — patching, upgrades, capacity, on-call.

The comparison is run against **two** hosted tiers on purpose, because "is
self-hosting cheaper?" has two different answers and quoting one of them is how
this arithmetic gets used dishonestly:

* against the **cheap** hosted model, a single-GPU 7B usually loses at every
  utilisation, and that is the honest answer for the 70% of traffic that is FAQ;
* against the **flagship** tier — what residency-bound traffic would otherwise
  have to use — the crossover arrives at a utilisation a real service can reach.

Which is the strategic framing: residency and control usually decide *whether*
self-hosting happens at all. The break-even decides *what traffic* it should carry
to be worth the GPUs.
"""

from __future__ import annotations

import argparse

from _common import bootstrap, rule, write_json

bootstrap()

from retail_support.config import get_settings  # noqa: E402

UTILISATIONS = (0.05, 0.10, 0.20, 0.25, 0.40, 0.50, 0.60, 0.80, 1.00)


def selfhost_sar_per_mtok(
    utilisation: float, gpu_hour_sar: float, tokens_per_sec: float, ops_overhead: float
) -> float:
    tokens_per_hour = tokens_per_sec * 3600 * utilisation
    if tokens_per_hour <= 0:
        return float("inf")
    return (gpu_hour_sar * ops_overhead) / (tokens_per_hour / 1e6)


def blended(row, input_share: float = 0.8) -> float:
    return round(row.input_per_mtok * input_share + row.output_per_mtok * (1 - input_share), 2)


def main() -> int:
    settings = get_settings()
    cheap = blended(settings.prices.for_model("retail_support-small"))
    flagship = blended(settings.prices.for_model("retail_support-flagship"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-hour-sar", type=float, default=12.0, help="your quoted GPU price")
    parser.add_argument(
        "--tokens-per-sec",
        type=float,
        default=950.0,
        help="measured in Module 2 at realistic concurrency, not at concurrency 1",
    )
    parser.add_argument(
        "--ops-overhead",
        type=float,
        default=1.35,
        help="patching, upgrades, capacity, on-call — never free",
    )
    parser.add_argument("--cheap-sar-per-mtok", type=float, default=cheap)
    parser.add_argument("--flagship-sar-per-mtok", type=float, default=flagship)
    args = parser.parse_args()

    rule("breakeven | self-host vs commercial API")
    print(
        f"  GPU {args.gpu_hour_sar} SAR/hour x {args.ops_overhead} ops overhead | "
        f"{args.tokens_per_sec:.0f} tok/s measured\n"
        f"  hosted, 80/20 input/output blend: cheap {args.cheap_sar_per_mtok} SAR/Mtok | "
        f"flagship {args.flagship_sar_per_mtok} SAR/Mtok\n"
    )

    rows = []
    crossovers: dict[str, float | None] = {"cheap": None, "flagship": None}
    for utilisation in UTILISATIONS:
        cost = selfhost_sar_per_mtok(
            utilisation, args.gpu_hour_sar, args.tokens_per_sec, args.ops_overhead
        )
        beats_cheap = cost < args.cheap_sar_per_mtok
        beats_flagship = cost < args.flagship_sar_per_mtok
        if beats_cheap and crossovers["cheap"] is None:
            crossovers["cheap"] = utilisation
        if beats_flagship and crossovers["flagship"] is None:
            crossovers["flagship"] = utilisation
        rows.append(
            {
                "utilisation": utilisation,
                "sar_per_mtok": round(cost, 2),
                "beats_cheap": beats_cheap,
                "beats_flagship": beats_flagship,
            }
        )
        verdict = (
            "beats both"
            if beats_cheap
            else "beats the flagship tier"
            if beats_flagship
            else ""
        )
        print(f"  utilisation {utilisation:>5.0%}: self-host {cost:8.2f} SAR/Mtok   {verdict}")

    print()
    for tier in ("flagship", "cheap"):
        point = crossovers[tier]
        print(
            f"  vs the {tier} tier: crossover at roughly {point:.0%} sustained utilisation"
            if point
            else f"  vs the {tier} tier: self-hosting never wins on these numbers"
        )
    print(
        "\n  Read both lines before quoting either. The recommendation this supports is\n"
        "  a routing table, not a migration: residency-bound and flagship-tier traffic\n"
        "  onto the GPU, routine FAQ traffic left on the cheap hosted model — and the\n"
        "  utilisation you assume has to be the one your traffic curve actually sustains,\n"
        "  not the one the GPU is capable of."
    )
    write_json(
        "breakeven.json",
        {
            "gpu_hour_sar": args.gpu_hour_sar,
            "tokens_per_sec": args.tokens_per_sec,
            "ops_overhead": args.ops_overhead,
            "cheap_sar_per_mtok": args.cheap_sar_per_mtok,
            "flagship_sar_per_mtok": args.flagship_sar_per_mtok,
            "crossovers": crossovers,
            "rows": rows,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
