"""
Benchmarking Part A — Golden Dataset validation
================================================
Makes golden_dataset.json self-consistent with the real data and the real policy document:

  * database / hybrid questions : runs every expected_sql on credit_risk.db and writes the rows
                                  into expected_result (fails loudly if a gold query is invalid).
  * calculator questions        : recomputes expected_values with the same formulas the agent's
                                  tools use (loan amortisation, compound interest, DTI).
  * policy / hybrid questions   : verifies that every expected_sections id really exists in
                                  credit_policy.txt and that each expected keyword appears in it.
  * prints a summary table + the category mix (15 core / 3 poison).

Run:  python validate_golden.py            (no API key needed — nothing calls OpenAI)
      python validate_golden.py --check    (validate only, do not rewrite the JSON)
"""

import os
import re
import sys
import json
import sqlite3
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "golden_dataset.json")
DB_PATH = os.path.join(BASE_DIR, "credit_risk.db")

sys.path.insert(0, BASE_DIR)
from build_vector_db import load_policy_text, chunk_policy  # noqa: E402


# ─────────────────────────────────────────────
# Reference formulas (identical to the @tool functions in app.py)
# ─────────────────────────────────────────────

def monthly_payment(principal, annual_rate_percent, term_months):
    if annual_rate_percent <= 0:
        return {"monthly_payment": round(principal / term_months, 2), "total_paid": round(principal, 2), "total_interest": 0.0}
    r = annual_rate_percent / 100 / 12
    m = principal * (r * (1 + r) ** term_months) / ((1 + r) ** term_months - 1)
    return {"monthly_payment": round(m, 2), "total_paid": round(m * term_months, 2), "total_interest": round(m * term_months - principal, 2)}


def compound_interest(principal, annual_rate_percent, years):
    fv = principal * (1 + annual_rate_percent / 100) ** years
    return {"future_value": round(fv, 2), "interest_earned": round(fv - principal, 2)}


def debt_to_income(monthly_debt_payment, monthly_gross_income):
    dti = monthly_debt_payment / monthly_gross_income * 100
    band = "Low Risk" if dti < 28 else "Moderate Risk" if dti < 36 else "High Risk" if dti < 43 else "Very High Risk"
    return {"dti_ratio_percent": round(dti, 1), "risk_level": band}


CALCULATORS = {
    "calculate_monthly_payment": monthly_payment,
    "calculate_compound_interest": compound_interest,
    "calculate_debt_to_income": debt_to_income,
}


# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────

def run_sql(con, sql):
    cur = con.execute(sql)
    cols = [c[0] for c in cur.description]
    rows = [list(r) for r in cur.fetchall()]
    return {"columns": cols, "rows": rows}


def policy_index():
    """{section_id: full chunk text} from the real policy document."""
    return {d.metadata["subsection"]: d.page_content for d in chunk_policy(load_policy_text())}


def normalize_number_text(text):
    return text.replace(",", "")


def validate(dataset, write_back=True):
    con = sqlite3.connect(DB_PATH)
    sections = policy_index()
    problems = []
    summary = []

    for q in dataset["questions"]:
        qid, qtype = q["id"], q["type"]
        line = f"{qid:4} {qtype:11} "

        if "expected_sql" in q:
            try:
                res = run_sql(con, q["expected_sql"])
                q["expected_result"] = res
                line += f"SQL ok → {res['rows'] if len(res['rows']) <= 4 else str(res['rows'][:4]) + ' …'}"
            except Exception as e:
                problems.append(f"{qid}: expected_sql failed: {e}")
                line += f"SQL ERROR: {e}"

        if qtype == "calculator":
            fn = CALCULATORS.get(q.get("expected_calculator"))
            if fn is None:
                problems.append(f"{qid}: unknown calculator {q.get('expected_calculator')}")
            else:
                q["expected_values"] = fn(**q["inputs"])
                line += f"calc → {q['expected_values']}"
                for k in q.get("required_values", []):
                    if k not in q["expected_values"]:
                        problems.append(f"{qid}: required value {k!r} is not produced by {q['expected_calculator']}")

        if qtype in ("policy", "hybrid"):
            for sid in q.get("expected_sections", []):
                if sid not in sections:
                    problems.append(f"{qid}: expected section {sid} not found in credit_policy.txt")
            for sid in q.get("excluded_sections", []):
                if sid not in sections:
                    problems.append(f"{qid}: excluded section {sid} not found in credit_policy.txt")
            covered = " ".join(normalize_number_text(sections[s]) for s in q.get("expected_sections", []) if s in sections)
            for kw in q.get("expected_keywords", []):
                if normalize_number_text(kw) not in covered:
                    problems.append(f"{qid}: keyword {kw!r} not found in sections {q.get('expected_sections')}")
            line += f" | sections {q.get('expected_sections')} ok"

        if qtype == "hybrid" and q.get("expected_table") and q.get("expected_result"):
            gold_values = [row["ערך"] for row in q["expected_table"]]
            sql_values = q["expected_result"]["rows"][0]
            if [float(v) for v in gold_values] != [float(v) for v in sql_values]:
                problems.append(f"{qid}: expected_table values {gold_values} != SQL result {sql_values}")
            else:
                line += " | table matches SQL"

        summary.append(line)

    print("\n".join(summary))

    core = [q for q in dataset["questions"] if q["type"] != "poison"]
    poison = [q for q in dataset["questions"] if q["type"] == "poison"]
    mix = {}
    for q in core:
        mix[q["type"]] = mix.get(q["type"], 0) + 1
    print(f"\nCore questions: {len(core)}  {mix}")
    print(f"Poison questions: {len(poison)}")
    if len(core) != 15 or len(poison) != 3:
        problems.append(f"expected 15 core + 3 poison, found {len(core)} + {len(poison)}")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  -", p)
        return False

    if write_back:
        with open(DATASET_PATH, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"\nexpected_result / expected_values written back to {os.path.basename(DATASET_PATH)}")
    print("Golden dataset is valid ✅")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate only, do not rewrite golden_dataset.json")
    args = parser.parse_args()
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    ok = validate(data, write_back=not args.check)
    sys.exit(0 if ok else 1)
