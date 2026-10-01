from pathlib import Path
import os

import pandas as pd
import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/interim/legislators.parquet"

BASE_URL = "https://api.congress.gov/v3"

CONGRESSES = [117, 118, 119]

load_dotenv(ROOT / ".env")

API_KEY = os.getenv("CONGRESS_API_KEY")


def get_first_house_start_year(member):
    """
    Find the earliest year the member began serving
    in the House of Representatives.
    """

    terms = member.get(
        "terms",
        {}
    ).get(
        "item",
        []
    )

    house_start_years = []

    for term in terms:

        chamber = term.get("chamber")
        start_year = term.get("startYear")

        if (
            chamber == "House of Representatives"
            and start_year is not None
        ):
            house_start_years.append(
                int(start_year)
            )

    if not house_start_years:
        return None

    return min(house_start_years)


def get_members_for_congress(congress):

    rows = []

    offset = 0
    limit = 250

    while True:

        url = (
            f"{BASE_URL}/member/"
            f"congress/{congress}"
        )

        params = {
            "api_key": API_KEY,
            "format": "json",
            "currentMember": "false",
            "limit": limit,
            "offset": offset,
        }

        response = requests.get(
            url,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        members = data.get(
            "members",
            []
        )

        if not members:
            break

        for member in members:

            terms = member.get(
                "terms",
                {}
            ).get(
                "item",
                []
            )

            # Check whether this person served
            # in the House.
            house_terms = [
                term
                for term in terms
                if term.get("chamber")
                == "House of Representatives"
            ]

            if not house_terms:
                continue

            first_house_start_year = (
                get_first_house_start_year(
                    member
                )
            )

            if first_house_start_year is not None:

                # Congress normally begins January 3.
                # This gives us a usable tenure feature.
                term_start = pd.Timestamp(
                    year=first_house_start_year,
                    month=1,
                    day=3
                )

            else:
                term_start = pd.NaT

            rows.append({
                "memberId":
                    member.get("bioguideId"),

                "congress":
                    congress,

                "name":
                    member.get("name"),

                "state":
                    member.get("state"),

                "district":
                    member.get("district"),

                "party":
                    member.get("partyName"),

                "chamber":
                    "House of Representatives",

                "first_house_start_year":
                    first_house_start_year,

                "term_start":
                    term_start,
            })

        if len(members) < limit:
            break

        offset += limit

    return rows


def main():

    if not API_KEY:
        raise ValueError(
            "CONGRESS_API_KEY was not "
            "found in .env"
        )

    all_rows = []

    for congress in CONGRESSES:

        print(
            f"Loading Congress "
            f"{congress}..."
        )

        rows = get_members_for_congress(
            congress
        )

        print(
            f"  House members loaded: "
            f"{len(rows)}"
        )

        all_rows.extend(
            rows
        )

    df = pd.DataFrame(
        all_rows
    )

    df = df.drop_duplicates(
        subset=[
            "memberId",
            "congress"
        ]
    )

    df["term_start"] = pd.to_datetime(
        df["term_start"]
    )

    print()
    print(
        f"total rows       = "
        f"{len(df)}"
    )

    print(
        f"unique members   = "
        f"{df.memberId.nunique()}"
    )

    print(
        f"memberId nulls   = "
        f"{df.memberId.isna().sum()}"
    )

    print(
        f"term_start nulls = "
        f"{df.term_start.isna().sum()}"
    )

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_parquet(
        OUT,
        index=False
    )

    print(
        f"\nwrote "
        f"{OUT.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()