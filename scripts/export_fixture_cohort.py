#!/usr/bin/env python3
"""Build a team catalogue from the differential fixture corpus.

The two-actor execution topology is validated against the fixture cohort
because those battles are provably inside the currently ported native subset:
every fixture is a complete legal reference battle replayed by the Rust corpus.
Training-pool teams still hit unported mechanics, so using them would report
operational failures as if they were engine throughput.

This is a development/benchmark instrument. It never runs Pokemon Showdown and
never touches the training pool.

Usage:
    python3 scripts/export_fixture_cohort.py [--fixtures PATH] [--out PATH]
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixtures",
        default=os.path.join(ROOT, "engine/data/turn-fixtures.json"),
    )
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "engine/benchmarks/fixture_teams.json"),
    )
    args = parser.parse_args()

    with open(args.fixtures) as handle:
        corpus = json.load(handle)
    fixtures = corpus["fixtures"] if isinstance(corpus, dict) else corpus
    teams = []
    seen = set()
    for fixture in fixtures:
        for team in fixture["teams"]:
            key = json.dumps(team["members"], sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            teams.append(team)
    with open(args.out, "w") as handle:
        json.dump(teams, handle)
    print(f"{len(fixtures)} fixtures -> {len(teams)} unique teams -> {args.out}")


if __name__ == "__main__":
    main()
