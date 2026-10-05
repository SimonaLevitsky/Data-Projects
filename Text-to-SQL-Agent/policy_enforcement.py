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

# A policy line is a CLASSIFICATION CRITERION when it states a threshold on one of these attributes
# (what makes a client belong to the category). Lines without any of them only set LIMITS / consequences
# for clients already in the category (max loan amount, guarantors, collateral, committee review) and
# are EXCLUDED from "how many clients are <category>" questions.
CRITERION_RE = re.compile(
    r"score|employment|income|\bage\b|\baged\b|years old|\bdti\b|missed payment|default|"
    r"marital|married|divorced|widowed|single"
)

# Criterion keyword → DB column (loan_amount is deliberately absent: amounts are limits, not criteria)
CRITERION_HINTS = [
    (r"score",                                      "credit_score"),
    (r"employment",                                 "employment_years"),
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
    Every policy-defined category mentioned in the question. For each one:
      sections        — CRITERION sections (lines with a classification threshold) → must be covered
      limit_sections  — sections that only set limits/consequences for members → must be EXCLUDED
      columns         — DB columns that can verify the criteria
    """
    q = normalize_for_match(question)
    found = []
    for label, trigger, terms in POLICY_CATEGORIES:
        if not re.search(trigger, q):
            continue
        all_sections = scan_policy(chunks, terms)
        if not all_sections:
            continue
        criteria, limits, columns = [], [], []
        for sec in all_sections:
            text = normalize_for_match(" ".join(sec["matching_lines"]))
            if CRITERION_RE.search(text):
                criteria.append(sec)
                for pat, col in CRITERION_HINTS:
                    if re.search(pat, text) and col not in columns:
                        columns.append(col)
            else:
                limits.append(sec)
        if not criteria:            # category defined only by limits → nothing to count; keep for context
            criteria, limits = all_sections, []
        found.append({
            "label": label,
            "terms": terms,
            "sections": criteria,
            "limit_sections": limits,
            "section_ids": [s["section"].split(" ")[0] for s in criteria],
            "limit_section_ids": [s["section"].split(" ")[0] for s in limits],
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
        block = (
            f"Category '{c['label']}' (exhaustive scan for: {c['terms']}).\n"
            f"CLASSIFICATION CRITERIA — {len(c['sections'])} section(s) that define WHO belongs to the category. "
            f"ALL of them must be covered:\n" + "\n".join(lines)
            + f"\n  DB columns that can verify these criteria: {cols}."
        )
        if c["limit_sections"]:
            lim = "; ".join(f"Section {s['section']}" for s in c["limit_sections"])
            block += (
                f"\nLIMITS ONLY — {lim}: these sections set restrictions for clients who are ALREADY in the "
                "category (maximum loan amount, guarantors, Credit Committee...). They are NOT criteria. "
                "Do NOT mention them in the answer, do NOT add rows for them, do NOT list them in policy_sources."
            )
        parts.append(block)

    if enforce_sql:
        data_req = (
            "2. Run ONE SQL query over the WHOLE portfolio with: one count per criterion above (each checked "
            "standalone); ONE combined count = clients meeting AT LEAST ONE criterion (OR over the criteria); "
            "and the total. No limit-related columns, no AND-combinations.\n"
            "3. Put the numbers in \"table_data\" ONLY (not in \"answer\"), rows {\"מדד\": label, \"ערך\": n} "
            "(English: Metric / Value), in THIS order: one row per criterion in bullet order (e.g. "
            "\"לקוחות עם ציון מתחת ל-550\", \"לקוחות עם 0–1 שנות תעסוקה\"), then EXACTLY "
            "\"לקוחות שעומדים בלפחות קריטריון אחד\" / \"Clients meeting at least one criterion\", then "
            "\"סך כל הלקוחות\" / \"Total clients\". No other rows. "
            "Set output_format to \"table+text\" (+sql if the user asked for the query).\n"
        )
    else:
        data_req = "2. If you query the database, check EVERY criterion that maps to the DB columns listed above.\n3. (no table needed)\n"

    return (
        "\n\n[POLICY CONTEXT — injected automatically by the system. This is the COMPLETE list of policy "
        "sections for the category in the question; you do not need to call find_all_policy_mentions again.]\n"
        + "\n\n".join(parts)
        + "\n\nREQUIREMENTS FOR YOUR ANSWER (MANDATORY WORKFLOW):\n"
        "1. \"answer\" = the policy definition ONLY: the header line (\"לפי המדיניות, לקוחות בקטגוריית <label> "
        "מוגדרים על פי:\" / \"According to the policy, <label> clients are defined by:\") followed by ONE bullet "
        "PER CRITERION SECTION listed above, in section order, each ending with its section number — written as "
        "the word, NEVER the '§' sign: Hebrew '(סעיף 2.1)', English '(Section 2.1)'. Use ONLY the criterion "
        "section numbers given here. No numbers/counts inside \"answer\".\n"
        + data_req +
        "4. \"data_coverage.checked\" = the criteria you verified, with sections. \"data_coverage.missing\" = "
        "ONLY criteria that cannot be checked in the data (e.g. DTI, missed payments) — leave it EMPTY when every "
        "criterion was verified. Never list limits (guarantors, committee, amounts) there.\n"
        "5. \"policy_sources\" = exactly the criterion sections above. EVERY one of them must appear EXPLICITLY "
        "in the \"answer\" text as its own bullet, and NO limit-only section may appear anywhere.\n"
        "An answer that omits a criterion section, includes a limit-only section, or is based on a single "
        "threshold is INCOMPLETE and will be rejected."
    )


# ─────────────────────────────────────────────
# 3. Validation
# ─────────────────────────────────────────────

def check_completeness(data: dict, categories: list[dict], enforce_sql: bool) -> dict:
    """Which criterion sections / SQL columns are missing, and which limit-only sections were wrongly cited."""
    in_sources = set()
    for s in data.get("policy_sources", []) or []:
        m = re.match(r"\s*(?:§|section|סעיף)?\s*(\d+\.\d+)", str(s), flags=re.IGNORECASE)
        if m:
            in_sources.add(m.group(1))
    answer_text = str(data.get("answer", "") or "")
    in_answer = set(re.findall(r"(?:§|section|סעיף)\s*(\d+\.\d+)", answer_text, flags=re.IGNORECASE))
    for lst in re.findall(r"(?:sections|סעיפים)\s*((?:\d+\.\d+[\s,/]*(?:and|ו-|ו)?\s*)+)", answer_text, flags=re.IGNORECASE):
        in_answer.update(re.findall(r"\d+\.\d+", lst))

    sql = (data.get("sql_query") or "").lower()
    expected_sections, expected_columns, excluded_sections = [], [], []
    for c in categories:
        expected_sections += [s for s in c["section_ids"] if s not in expected_sections]
        expected_columns += [col for col in c["columns"] if col not in expected_columns]
        excluded_sections += [s for s in c.get("limit_section_ids", []) if s not in excluded_sections]

    missing_sections = [s for s in expected_sections if s not in in_answer]
    missing_in_sources = [s for s in expected_sections if s not in in_sources]
    unexpected_sections = [s for s in excluded_sections if s in in_answer or s in in_sources]
    missing_columns = [c for c in expected_columns if c not in sql] if (enforce_sql or sql) else []
    return {
        "expected_sections": expected_sections,
        "excluded_sections": excluded_sections,
        "expected_columns": expected_columns,
        "missing_sections": missing_sections,          # criterion sections not mentioned in the answer text
        "missing_in_sources": missing_in_sources,      # criterion sections not listed in policy_sources
        "unexpected_sections": unexpected_sections,    # limit-only sections that were wrongly included
        "missing_columns": missing_columns,
        "ok": not missing_sections and not missing_in_sources and not unexpected_sections and not missing_columns,
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
    if report.get("unexpected_sections"):
        msg += ("- Sections you included that are NOT criteria (they only set limits for clients already in the "
                "category) and must be REMOVED from the answer, the table and policy_sources: "
                + ", ".join("Section " + s for s in report["unexpected_sections"]) + "\n")
    if report["missing_columns"]:
        msg += "- DB columns your SQL did not check: " + ", ".join(report["missing_columns"]) + "\n"
    msg += (
        "Rewrite the COMPLETE answer now, following the MANDATORY WORKFLOW: cover EVERY section listed in the "
        "POLICY CONTEXT with its exact section number, run ONE SQL query that checks EVERY checkable criterion "
        "(each condition standalone over the whole portfolio, then at least one condition, then total), list the unverifiable requirements in data_coverage.missing, and fill "
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
              + (", ".join(f"{c['label']} → criteria {'/'.join(c['section_ids'])} | excluded limits {'/'.join(c['limit_section_ids']) or '-'} | cols={c['columns']}" for c in cats) or "none"))
