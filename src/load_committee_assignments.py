from io import BytesIO
from pathlib import Path
import re
import unicodedata
from difflib import SequenceMatcher

import pandas as pd
import pdfplumber
import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parent.parent

LEGISLATORS_FILE = ROOT / "data/interim/legislators.parquet"
COMMITTEE_SECTORS_FILE = ROOT / "config/committee_sectors.csv"

OUT = ROOT / "data/interim/committee_assignments.parquet"
UNMATCHED_OUT = ROOT / "data/interim/committee_assignment_unmatched.csv"


# Official House History PDFs for historical committee assignments.
HISTORICAL_PDF_URLS = {
    117: (
        "https://historycms.house.gov/"
        "WorkArea/DownloadAsset.aspx?id=25769822883"
    ),
    118: (
        "https://historycms.house.gov/"
        "WorkArea/DownloadAsset.aspx?id=36507226670"
    ),
}


# Committees that existed historically but do not exist
# in the current 119th Congress.
HISTORICAL_ONLY_CODES = {
    "HSCN",
}


# Committee names sometimes change between Congresses.
COMMITTEE_ALIASES = {
    "HSED": [
        "Education and Workforce",
        "Education and Labor",
    ],

    "HSGO": [
        "Oversight and Government Reform",
        "Oversight and Reform",
        "Oversight and Accountability",
    ],

    "HSCN": [
        "Select Committee on the Climate Crisis",
    ],
}


STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE",
    "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS",
    "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY",
    "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY", "DC", "PR", "VI", "GU", "AS", "MP",
}


def normalize_text(value):
    if pd.isna(value):
        return ""

    value = unicodedata.normalize(
        "NFKD",
        str(value)
    )

    value = (
        value
        .encode("ascii", "ignore")
        .decode()
    )

    value = value.lower()

    value = re.sub(
        r"\b(jr|sr|ii|iii|iv)\b",
        "",
        value
    )

    value = re.sub(
        r"[^a-z0-9 ]",
        " ",
        value
    )

    return " ".join(
        value.split()
    )


def normalize_name(name):
    """
    Converts names into a comparable form.

    Examples:
        Adams, Alma S.
        Alma S. Adams
    """

    name = str(name).strip()

    # Remove nicknames in quotation marks.
    name = re.sub(
        r'["“”][^"“”]+["“”]',
        "",
        name
    )

    if "," in name:
        parts = [
            part.strip()
            for part in name.split(",")
        ]

        if len(parts) >= 2:
            last = parts[0]
            first = parts[1]

            name = f"{first} {last}"

    return normalize_text(name)


def clerk_code(committee_code):
    """
    Converts House committee codes into the
    format used by the Clerk website.

    Examples:
        HSBA -> BA00
        HSIF -> IF00
        HSAG -> AG00
    """

    return committee_code[-2:] + "00"


def read_committee_codes():
    df = pd.read_csv(
        COMMITTEE_SECTORS_FILE
    )

    df.columns = [
        column.strip()
        for column in df.columns
    ]

    if "committee_code" not in df.columns:
        raise ValueError(
            "committee_sectors.csv must contain "
            "a committee_code column"
        )

    codes = (
        df["committee_code"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    return sorted(codes)


def load_legislators():
    if not LEGISLATORS_FILE.exists():
        raise FileNotFoundError(
            "Run load_legislators.py first. "
            f"Missing: {LEGISLATORS_FILE}"
        )

    df = pd.read_parquet(
        LEGISLATORS_FILE
    )

    required = {
        "memberId",
        "congress",
        "name",
        "state",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"legislators.parquet is missing: "
            f"{missing}"
        )

    return df


def load_119_assignments(
    committee_codes
):
    """
    Loads current 119th Congress committee
    membership from the House Clerk website.

    Clerk member links contain the Bioguide ID,
    so no fuzzy name matching is required here.
    """

    rows = []
    committee_names = {}

    headers = {
        "User-Agent": (
            "PoliPick academic research project"
        )
    }

    for committee_code in committee_codes:

        if (
            committee_code
            in HISTORICAL_ONLY_CODES
        ):
            print(
                f"119th: skipping historical-only "
                f"{committee_code}"
            )
            continue

        code = clerk_code(
            committee_code
        )

        url = (
            "https://clerk.house.gov/"
            f"committees/{code}"
        )

        print(
            f"119th: loading "
            f"{committee_code}..."
        )

        response = requests.get(
            url,
            headers=headers,
            timeout=30
        )

        if response.status_code != 200:
            print(
                f"  WARNING: could not load "
                f"{url}"
            )
            continue

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        title = soup.find("h1")

        if title:
            committee_name = (
                title.get_text(
                    " ",
                    strip=True
                )
                .replace(
                    "Committee Profile",
                    ""
                )
                .strip()
            )
        else:
            committee_name = (
                committee_code
            )

        committee_name = re.sub(
            r"^Committee on\s+",
            "",
            committee_name,
            flags=re.IGNORECASE
        )

        committee_names[
            committee_code
        ] = committee_name

        member_ids = set()

        for link in soup.find_all(
            "a",
            href=True
        ):

            match = re.search(
                r"/Members/([A-Z]\d{6})",
                link["href"]
            )

            if match:
                member_ids.add(
                    match.group(1)
                )

        print(
            f"  members: "
            f"{len(member_ids)}"
        )

        for member_id in member_ids:

            rows.append({
                "memberId": member_id,
                "congress": 119,
                "committee_code":
                    committee_code,
                "committee_name":
                    committee_name,
            })

    return rows, committee_names


def find_historical_pdf(
    congress
):
    if (
        congress
        not in HISTORICAL_PDF_URLS
    ):
        raise ValueError(
            "No historical committee PDF "
            f"configured for Congress "
            f"{congress}"
        )

    return HISTORICAL_PDF_URLS[
        congress
    ]


def group_words_into_lines(words):
    """
    Groups PDF words that appear at roughly
    the same vertical position.
    """

    lines = []

    for word in sorted(
        words,
        key=lambda w: (
            w["top"],
            w["x0"]
        )
    ):

        placed = False

        for line in lines:

            if abs(
                line["top"]
                - word["top"]
            ) <= 2.5:

                line["words"].append(
                    word
                )

                placed = True
                break

        if not placed:
            lines.append({
                "top": word["top"],
                "words": [word],
            })

    return lines


def parse_member_text(text):
    """
    Attempts to identify member lines
    in the House History PDFs.

    Example:
        Adams, Alma S., 12th NC
    """

    text = text.strip()

    # Remove page number if attached
    # directly to the first word.
    text = re.sub(
        r"^\d+(?=[A-Za-z])",
        "",
        text
    )

    state_match = re.search(
        r"\b([A-Z]{2})\s*$",
        text
    )

    if not state_match:
        return None

    state = state_match.group(1)

    if state not in STATES:
        return None

    name = re.sub(
        r",?\s+"
        r"(?:"
        r"\d+(?:st|nd|rd|th|d)?"
        r"|At Large"
        r")"
        r",?\s+"
        r"[A-Z]{2}\s*$",
        "",
        text,
        flags=re.IGNORECASE
    )

    if name == text:
        return None

    if "," not in name:
        return None

    return {
        "name": name.strip(),
        "state": state,
    }


def build_aliases(
    committee_codes,
    current_names
):
    aliases = {}

    for code in committee_codes:

        names = []

        current_name = (
            current_names.get(
                code
            )
        )

        if current_name:
            names.append(
                current_name
            )

        names.extend(
            COMMITTEE_ALIASES.get(
                code,
                []
            )
        )

        aliases[code] = [
            normalize_text(name)
            for name in names
        ]

    return aliases


def match_legislator(
    member_name,
    state,
    congress,
    legislators
):
    """
    Match historical PDF names to
    Congress.gov Bioguide IDs.
    """

    candidates = legislators[
        (
            legislators["congress"]
            == congress
        )
        &
        (
            legislators["state"]
            .astype(str)
            .str.upper()
            == state
        )
    ]

    if candidates.empty:
        return None, 0.0

    source_name = normalize_name(
        member_name
    )

    best_id = None
    best_score = 0.0

    for _, row in candidates.iterrows():

        candidate_name = normalize_name(
            row["name"]
        )

        score = SequenceMatcher(
            None,
            source_name,
            candidate_name
        ).ratio()

        if score > best_score:
            best_score = score
            best_id = row["memberId"]

    # Avoid silently matching
    # questionable names.
    if best_score < 0.72:
        return None, best_score

    return best_id, best_score


def committees_from_text(
    text,
    aliases
):
    text = normalize_text(
        text
    )

    found = []

    for code, names in (
        aliases.items()
    ):

        for name in names:

            if (
                name
                and name in text
            ):
                found.append(
                    code
                )
                break

    return found


def parse_historical_pdf(
    congress,
    pdf_bytes,
    legislators,
    aliases,
    committee_names
):
    rows = []
    unmatched = []

    current_member = None
    committee_text = []

    def flush_member():
        nonlocal current_member
        nonlocal committee_text

        if current_member is None:
            return

        member_id, score = (
            match_legislator(
                current_member[
                    "name"
                ],
                current_member[
                    "state"
                ],
                congress,
                legislators
            )
        )

        committee_codes = (
            committees_from_text(
                " ".join(
                    committee_text
                ),
                aliases
            )
        )

        if member_id is None:

            unmatched.append({
                "congress": congress,
                "name":
                    current_member[
                        "name"
                    ],
                "state":
                    current_member[
                        "state"
                    ],
                "match_score":
                    score,
            })

        else:

            for code in (
                committee_codes
            ):

                rows.append({
                    "memberId":
                        member_id,
                    "congress":
                        congress,
                    "committee_code":
                        code,
                    "committee_name":
                        committee_names.get(
                            code,
                            code
                        ),
                })

        current_member = None
        committee_text = []

    with pdfplumber.open(
        BytesIO(pdf_bytes)
    ) as pdf:

        for page in pdf.pages:

            words = (
                page.extract_words(
                    keep_blank_chars=False,
                    use_text_flow=True
                )
            )

            lines = (
                group_words_into_lines(
                    words
                )
            )

            # PDF layout is roughly:
            #
            # Member name | committees
            #
            # Split page around center.
            split_x = (
                page.width * 0.53
            )

            for line in lines:

                left_words = [
                    word
                    for word
                    in line["words"]
                    if word["x0"]
                    < split_x
                ]

                right_words = [
                    word
                    for word
                    in line["words"]
                    if word["x0"]
                    >= split_x
                ]

                left_text = " ".join(
                    word["text"]
                    for word
                    in left_words
                ).strip()

                right_text = " ".join(
                    word["text"]
                    for word
                    in right_words
                ).strip()

                parsed_member = (
                    parse_member_text(
                        left_text
                    )
                )

                if parsed_member:

                    flush_member()

                    current_member = (
                        parsed_member
                    )

                    if right_text:
                        committee_text.append(
                            right_text
                        )

                elif (
                    current_member
                    and right_text
                ):

                    committee_text.append(
                        right_text
                    )

    flush_member()

    return rows, unmatched


def load_historical_assignments(
    congress,
    legislators,
    aliases,
    committee_names
):
    print(
        f"{congress}th: "
        f"finding official PDF..."
    )

    pdf_url = (
        find_historical_pdf(
            congress
        )
    )

    print(
        f"  downloading {pdf_url}"
    )

    response = requests.get(
        pdf_url,
        timeout=60
    )

    response.raise_for_status()

    rows, unmatched = (
        parse_historical_pdf(
            congress,
            response.content,
            legislators,
            aliases,
            committee_names
        )
    )

    print(
        f"  assignments: "
        f"{len(rows)}"
    )

    print(
        f"  unmatched members: "
        f"{len(unmatched)}"
    )

    return rows, unmatched


def main():

    committee_codes = (
        read_committee_codes()
    )

    legislators = (
        load_legislators()
    )

    print(
        "Committee codes being loaded: "
        f"{len(committee_codes)}"
    )

    # ---------------------------------
    # 119th Congress
    # ---------------------------------

    current_rows, committee_names = (
        load_119_assignments(
            committee_codes
        )
    )

    aliases = build_aliases(
        committee_codes,
        committee_names
    )

    # ---------------------------------
    # 117th and 118th Congresses
    # ---------------------------------

    all_rows = list(
        current_rows
    )

    all_unmatched = []

    for congress in [
        117,
        118
    ]:

        rows, unmatched = (
            load_historical_assignments(
                congress,
                legislators,
                aliases,
                committee_names
            )
        )

        all_rows.extend(
            rows
        )

        all_unmatched.extend(
            unmatched
        )

    # ---------------------------------
    # Save committee assignments
    # ---------------------------------

    assignments = pd.DataFrame(
        all_rows
    )

    if assignments.empty:
        raise RuntimeError(
            "No committee assignments "
            "were collected."
        )

    assignments = (
        assignments
        .drop_duplicates(
            subset=[
                "memberId",
                "congress",
                "committee_code",
            ]
        )
        .sort_values(
            [
                "congress",
                "memberId",
                "committee_code",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    assignments.to_parquet(
        OUT,
        index=False
    )

    print()
    print(
        "total committee assignments = "
        f"{len(assignments)}"
    )

    print(
        "unique members = "
        f"{assignments.memberId.nunique()}"
    )

    print(
        f"wrote "
        f"{OUT.relative_to(ROOT)}"
    )

    # ---------------------------------
    # Save unmatched historical names
    # ---------------------------------

    if all_unmatched:

        unmatched_df = (
            pd.DataFrame(
                all_unmatched
            )
        )

        unmatched_df.to_csv(
            UNMATCHED_OUT,
            index=False
        )

        print()
        print(
            "WARNING: "
            f"{len(unmatched_df)} "
            "historical members could not "
            "be confidently matched."
        )

        print(
            "Review "
            f"{UNMATCHED_OUT.relative_to(ROOT)}"
        )

    else:
        print()
        print(
            "All historical members matched."
        )


if __name__ == "__main__":
    main()