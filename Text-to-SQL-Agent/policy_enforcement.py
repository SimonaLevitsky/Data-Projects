"""
Policy-category enforcement (Python-level, model-independent)
=============================================================
The system prompt asks the agent to cover EVERY policy section that defines a
category (e.g. "High Risk" appears in Sections 2.1, 4.1 and 8.1) and to check every
checkable criterion in SQL. A small model tends to ignore that and answer from a
single section, so the code enforces it in three steps:

  1. detect_policy_categories(question)  → which policy-defined categories the
     question mentions, with an EXHAUSTIVE scan of all their sections and the
     DB columns that can verify the criteria found there.
  2. build_policy_context(categories)    → text injected into the agent's input
     so it sees all sections BEFORE answering.
  3. check_completeness(answer, ...)     → validates the JSON answer: every
     section cited, every checkable column present in the SQL. The app retries
     once with a correction message if something is missing.

No Streamlit / OpenAI dependency — fully testable offline.
"""

import re
import json

# Columns that exist in the database (loans ⨝ demographics)
CHECKABLE_COLUMNS = ["credit_score", "loan_amount", "age", "default_status",
                     "employment_years", "annual_income", "marital_status"]

# (label, trigger regex over the NORMALIZED question, scan terms for the exhaustive policy scan)
# Normalization = lowercase, dashes → spaces, collapsed whitespace (so "High-Risk" == "high risk").
POLICY_CATEGORIES = [
    ("High Risk / Tier 4",        r"high risk|tier 4|סיכון גבוה|מסוכנ",                  "high risk|tier 4"),
    ("Low Risk / Tier 1",         r"low risk|tier 1|סיכון נמוך",                          "low risk|tier 1"),
    ("Moderate Risk",             r"moderate risk|סיכון בינוני",                          "moderate risk"),
    ("Prime",                     r"(?<!sub )(?<!near )\bprime\b|פריים",                  "prime"),
    ("Near-Prime / Tier 2",       r"near prime|tier 2",                                   "near prime|tier 2"),
    ("Sub-Prime / Tier 3",        r"sub prime|tier 3",                                    "sub prime|tier 3"),
    ("Guarantor",                 r"guarantor|ערב",                                       "guarantor"),
    ("Collateral",                r"collateral|בטוח",                                     "collateral"),
    ("Credit Committee",          r"credit committee|ועדת אשראי",                         "credit committee"),
    ("Eligibility",               r"eligib|זכאות|זכאי",                                   "eligib"),
    ("Minimum income",            r"minimum (annual )?income|הכנסה מינימלית",             "minimum annual income|minimum income"),
    ("Preferred applicant",       r"preferred applicant|מועמד מועדף",                     "preferred applicant"),
    ("Watch list",                r"\bwatch\b",                                           "watch"),
    ("Substandard",               r"substandard",                                         "substandard"),
]

# Keyword → DB column, applied to the policy lines found for a category.
# Used to decide which criteria the agent MUST check in SQL.
COLUMN_HINTS = [
    (r"score",                                      "credit_score"),
    (r"employment",                                 "employment_years"),
    (r"loan amount|loan amounts|\bils\b",           "loan_amount"),
    (r"income",                                     "annual_income"),
    (r"\bage\b|years old|\baged\b",                 "age"),
    (r"marital|married|divorced|widowed|single",    "marital_status"),
    (r"default",                                    "default_status"),
]

# Does the question ask for portfolio DATA (count / list / share), not only for the rule?
DATA_INTENT_RE = re.compile(
    r"how many|count|which|list|share|percent|portfolio|clients|loans|borrowers|"
    r"כמה|אילו|איזה|רשימ|אחוז|שיעור|בתיק|לקוחות|הלוואות|לווים"
)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def normalize_for_match(text: str) -> str:
    """Lowercase, unify dashes/hyphens to spaces, collapse whitespace."""
    text = str(text).lower().replace("-", " ").replace("–", " ").replace("—", " ")
    return re.sub(r"\s+", " ", text)


def wants_data(question: str) -> bool:
    return bool(DATA_INTENT_RE.search(normalize_for_match(question)))


def scan_policy(chunks, terms: str) -> list[dict]:
    """
    EXHAUSTIVE scan: every policy sub-section whose text mentions any of the terms
    ('|' or ',' separated). Returns [{section, parent_section, matching_lines, full_text}, ...]
    in document order. `chunks` = output of build_vector_db.chunk_policy().
    """
    term_list = [t.strip() for t in re.split(r"[|,]", terms) if t.strip()]
    norm_terms = [normalize_for_match(t) for t in term_list]
    matches = []
    for d in chunks:
        m = d.metadata
        if m.get("section") == "0":
            continue
        lines = d.page_content.split("\n")[1:]  # drop the "Section X — Y" header line
        matching_lines = [ln.strip().lstrip("-").strip() for ln in lines
                          if any(t in normalize_for_match(ln) for t in norm_terms)]
        if matching_lines:
            matches.append({
                "section": f"{m.get('subsection', '')} {m.get('subsection_title', '')}".strip(),
                "parent_section": f"Section {m.get('section', '')}: {m.get('section_title', '')}",
                "matching_lines": matching_lines,
                "full_text": d.page_content,
            })
    return matches


# ─────────────────────────────────────────────
# 1. Detection
# ─────────────────────────────────────────────

def detect_policy_categories(question: str, chunks) -> list[dict]:
    """
    Every policy-defined category mentioned in the question, each with ALL its
    policy sections and the DB columns that can verify criteria from those sections.
    """
    q = normalize_for_match(question)
    found = []
    for label, trigger, terms in POLICY_CATEGORIES:
        if not re.search(trigger, q):
            continue
        sections = scan_policy(chunks, terms)
        if not sections:
            continue
        columns = []
        for sec in sections:
            text = normalize_for_match(" ".join(sec["matching_lines"]))
            for pat, col in COLUMN_HINTS:
                if re.search(pat, text) and col not in columns:
                    columns.append(col)
        found.append({
            "label": label,
            "terms": terms,
            "sections": sections,
            "section_ids": [s["section"].split(" ")[0] for s in sections],
            "columns": columns,
        })
    return found


# ─────────────────────────────────────────────
# 2. Context injection
# ─────────────────────────────────────────────

def build_policy_context(categories: list[dict], enforce_sql: bool) -> str:
    parts = []
    for c in categories:
        lines = []
        for s in c["sections"]:
            lines.append(f"  Section {s['section']}:")
            lines += [f"    - {ln}" for ln in s["matching_lines"]]
        cols = ", ".join(c["columns"]) or "none"
        parts.append(
            f"Category '{c['label']}' (exhaustive scan for: {c['terms']}) appears in "
            f"{len(c['sections'])} policy sections — ALL of them must be covered:\n"
            + "\n".join(lines)
            + f"\n  DB columns that can verify criteria from these sections: {cols}."
        )

    sql_req = (
        "2. Separate CLASSIFICATION criteria (score / employment / income / age thresholds that make a client "
        "belong to the category) from LIMITS (max loan amount, guarantors, committee review — rules for clients "
        "already in the category). Run ONE SQL query with: a count per classification criterion; ONE combined "
        "count of clients meeting AT LEAST ONE classification criterion (OR over the classification criteria "
        "ONLY — never put a loan-amount limit inside this OR); for each checkable limit, the number of "
        "violations counted ONLY among clients who meet the matching classification criterion "
        "(e.g. credit_score < 550 AND loan_amount > 50000); and the total.\n"
        "   Label the combined line EXACTLY: Hebrew \"לקוחות שעומדים בלפחות קריטריון אחד: <n>\" / "
        "English \"Clients meeting at least one criterion: <n>\".\n"
        if enforce_sql else
        "2. If you query the database, check EVERY criterion that maps to the DB columns listed above.\n"
    )
    return (
        "\n\n[POLICY CONTEXT — injected automatically by the system. This is the COMPLETE list of policy "
        "sections for the category in the question; you do not need to call find_all_policy_mentions again.]\n"
        + "\n\n".join(parts)
        + "\n\nREQUIREMENTS FOR YOUR ANSWER (MANDATORY WORKFLOW):\n"
        "1. List EVERY criterion from EVERY section above, each with its exact section number. "
        "Use ONLY the section numbers given here. NEVER write the '§' sign — write the word: "
        "Hebrew 'סעיף 2.1', English 'Section 2.1'.\n"
        + sql_req +
        "3. Under 'Data limitations', list every criterion that cannot be checked in the data and why "
        "(guarantors, Credit Committee review, approval rates, collateral, DTI, missed payments... are NOT in the DB).\n"
        "4. Set policy_sources to ALL the sections above and fill data_coverage (checked / missing).\n"
        "5. EVERY section number listed above must appear EXPLICITLY in the answer text (not only in "
        "policy_sources). If two sections state the same rule (e.g. 2.1 and 8.1 both give the 50,000 ILS "
        "limit for Tier 4), cite BOTH, e.g. 'סעיפים 2.1 ו-8.1' / 'Sections 2.1 and 8.1'. Do not merge or "
        "drop a section because it looks redundant.\n"
        "An answer that omits any of these sections, or that is based on a single threshold, is INCOMPLETE "
        "and will be rejected."
    )


# ─────────────────────────────────────────────
# 3. Validation
# ─────────────────────────────────────────────

def check_completeness(data: dict, categories: list[dict], enforce_sql: bool) -> dict:
    """Which expected sections / SQL columns are missing from the agent's JSON answer."""
    # (a) sections listed in policy_sources
    in_sources = set()
    for s in data.get("policy_sources", []) or []:
        m = re.match(r"\s*(?:§|section|סעיף)?\s*(\d+\.\d+)", str(s), flags=re.IGNORECASE)
        if m:
            in_sources.add(m.group(1))
    # (b) sections cited IN THE ANSWER TEXT — this is what the user actually reads, so it is the real gate.
    #     policy_sources alone is NOT enough (the model used to list a section there and omit it from the text).
    answer_text = str(data.get("answer", "") or "")
    in_answer = set(re.findall(r"(?:§|section|סעיף)\s*(\d+\.\d+)", answer_text, flags=re.IGNORECASE))
    # plural lists: "סעיפים 2.1, 4.1 ו-8.1" / "Sections 2.1, 4.1 and 8.1" / "Sections 2.1/8.1"
    for lst in re.findall(r"(?:sections|סעיפים)\s*((?:\d+\.\d+[\s,/]*(?:and|ו-|ו)?\s*)+)", answer_text, flags=re.IGNORECASE):
        in_answer.update(re.findall(r"\d+\.\d+", lst))

    sql = (data.get("sql_query") or "").lower()
    expected_sections, expected_columns = [], []
    for c in categories:
        expected_sections += [s for s in c["section_ids"] if s not in expected_sections]
        expected_columns += [col for col in c["columns"] if col not in expected_columns]

    missing_sections = [s for s in expected_sections if s not in in_answer]
    missing_in_sources = [s for s in expected_sections if s not in in_sources]
    missing_columns = [c for c in expected_columns if c not in sql] if (enforce_sql or sql) else []
    return {
        "expected_sections": expected_sections,
        "expected_columns": expected_columns,
        "missing_sections": missing_sections,          # not mentioned in the answer text
        "missing_in_sources": missing_in_sources,      # not listed in policy_sources
        "missing_columns": missing_columns,
        "ok": not missing_sections and not missing_in_sources and not missing_columns,
    }


def build_correction(report: dict) -> str:
    msg = "Your previous answer is INCOMPLETE and was REJECTED by an automatic completeness check.\n"
    if report["missing_sections"]:
        msg += ("- Policy sections NOT mentioned in your answer text: "
                + ", ".join("Section " + s for s in report["missing_sections"])
                + ". Each of them must appear explicitly in the answer (its number AND what it says), "
                  "even if it repeats a rule already stated in another section.\n")
    if report.get("missing_in_sources"):
        msg += "- Sections missing from policy_sources: " + ", ".join(report["missing_in_sources"]) + "\n"
    if report["missing_columns"]:
        msg += "- DB columns your SQL did not check: " + ", ".join(report["missing_columns"]) + "\n"
    msg += (
        "Rewrite the COMPLETE answer now, following the MANDATORY WORKFLOW: cover EVERY section listed in the "
        "POLICY CONTEXT with its exact section number, run ONE SQL query that checks EVERY checkable criterion "
        "(including limit violations), list the unverifiable criteria under data limitations, and fill "
        "policy_sources and data_coverage. Return ONLY the JSON object."
    )
    return msg


if __name__ == "__main__":
    # Offline self-test against the real policy document
    from build_vector_db import load_policy_text, chunk_policy
    chunks = chunk_policy(load_policy_text())
    for q in ["כמה לקוחות בתיק נחשבים High Risk לפי המדיניות?",
              "How many clients are Prime according to the policy?",
              "What does the policy say about guarantors?",
              "מה אחוז ה-Default בתיק?"]:
        cats = detect_policy_categories(q, chunks)
        print(f"\nQ: {q}\n   data intent: {wants_data(q)} | categories: "
              + (", ".join(f"{c['label']} → Sections {'/'.join(c['section_ids'])} | cols={c['columns']}" for c in cats) or "none"))
