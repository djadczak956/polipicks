"""Load House committee assignments for the 117th-119th Congresses.

The congress-legislators repo only ships CURRENT committee membership, so
past Congresses are recovered from its git history: for each Congress we
take the last commit of committee-membership-current.yaml before a cutoff
date early in that Congress.

Writes data/interim/committee_assignments.parquet with the same columns as
load_committee_assignments.py: memberId, congress, committee_code,
committee_name. Full committees only; subcommittee codes (e.g. HSBA01) are
dropped.

Requires a FULL clone (not --depth 1) at data/external/congress-legislators:
    git clone https://github.com/unitedstates/congress-legislators \
        data/external/congress-legislators
"""

from pathlib import Path
import re
import subprocess

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent.parent

LEGISLATORS_REPO = ROOT / "data/external/congress-legislators"
MEMBERSHIP_FILE = "committee-membership-current.yaml"
SNAPSHOT_DIR = ROOT / "data/external"

OUT = ROOT / "data/interim/committee_assignments.parquet"

# Cutoff = last commit before this date. The repo added 117th Congress House
# assignments on 2021-03-01, so that cutoff is pushed to 2021-03-15. The
# 119th China select committee (HSZS) was added 2025-03-13, hence 04-01.
CONGRESS_CUTOFFS = {
    117: "2021-03-15",
    118: "2023-03-01",
    119: "2025-04-01",
}

# House full committees are 4 chars: HS?? standing/select, HL?? = Intelligence.
HOUSE_COMMITTEE = re.compile(r"H[SL][A-Z]{2}")


def git(*args):
    return subprocess.run(
        ["git", "-C", str(LEGISLATORS_REPO), *args],
        capture_output=True, text=True, check=True,
    ).stdout


def load_snapshot(congress, cutoff):
    commit = git(
        "log", f"--before={cutoff}", "-1", "--format=%H %cs",
        "--", MEMBERSHIP_FILE,
    ).strip()

    if not commit:
        raise RuntimeError(
            f"{congress}th: no commit of {MEMBERSHIP_FILE} before {cutoff}. "
            "Is the clone shallow? Re-clone without --depth."
        )

    sha, commit_date = commit.split()
    text = git("show", f"{sha}:{MEMBERSHIP_FILE}")

    # Keep a copy on disk so the exact snapshot used is inspectable.
    (SNAPSHOT_DIR / f"committees_{congress}.yaml").write_text(text)

    membership = yaml.safe_load(text)

    rows = [
        {"memberId": m["bioguide"], "congress": congress, "committee_code": code}
        for code, members in membership.items()
        if HOUSE_COMMITTEE.fullmatch(code)
        for m in members
        if m.get("bioguide")
    ]

    n_committees = len({r["committee_code"] for r in rows})
    print(
        f"{congress}th: commit {sha[:10]} ({commit_date}) -> "
        f"{n_committees} House committees, {len(rows)} assignments"
    )

    # An empty House snapshot is how the original 2021-03-01 cutoff failed
    # silently. Fail loudly instead.
    if n_committees < 15:
        raise RuntimeError(
            f"{congress}th: only {n_committees} House committees in snapshot. "
            "Move the cutoff later."
        )

    return rows


def load_committee_names():
    names = {}
    for f in ["committees-historical.yaml", "committees-current.yaml"]:
        for c in yaml.safe_load((LEGISLATORS_REPO / f).read_text()):
            if "thomas_id" in c:
                names[c["thomas_id"]] = c["name"]
    return names


def main():
    rows = []
    for congress, cutoff in CONGRESS_CUTOFFS.items():
        rows.extend(load_snapshot(congress, cutoff))

    names = load_committee_names()

    assignments = (
        pd.DataFrame(rows)
        .drop_duplicates()
        .assign(committee_name=lambda d: d.committee_code.map(names))
        .sort_values(["congress", "memberId", "committee_code"])
        .reset_index(drop=True)
    )

    print()
    print(assignments.groupby("congress").agg(
        members=("memberId", "nunique"),
        assignments=("memberId", "size"),
        committees=("committee_code", "nunique"),
    ).to_string())

    unnamed = assignments.loc[assignments.committee_name.isna(), "committee_code"]
    if not unnamed.empty:
        print(f"\nWARNING: no name for codes {sorted(unnamed.unique())}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
