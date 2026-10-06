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

# Bumped whenever app.py starts relying on a new function/signature here. app.py checks it at startup so a
# half-updated deployment (new app.py + old policy_enforcement.py) fails with a clear message, not a TypeError.
ENFORCEMENT_VERSION = 3

# Does the question refer to the policy / a rule / a policy-defined concept? If not, the policy tools are
# unnecessary for it (plain portfolio statistics) and a "hybrid" routing is normalised back to "database".
POLICY_INTENT_RE = re.compile(
    r"מדיניות|policy|\bכלל|rule|תקר|מגבל|דריש|requir|limit|eligib|זכא|tier|prime|guarantor|ערב|collateral|בטוח|"
    r"committee|ועד|risk|סיכון|מסוכן|underwrit|חיתום",
    re.IGNORECASE)

# Vague qualifiers with no numeric threshold in the question and no policy definition → ask, don't guess.
AMBIGUOUS_TERMS_RE = re.compile(
    r"גדול[הים]*|קטנ[הים]*|גבוה[הים]*|נמוכ[הים]*|צעיר[הים]*|מבוגר[הים]*|ותיק[הים]*|הרבה|מעט|טוב[הים]*|חזק[הים]*|בעייתי[הים]*|"
    r"\b(large|big|small|high|low|young|old|experienced|many|few|good|strong|problematic)\b",
    re.IGNORECASE)
SUPERLATIVE_RE = re.compile(r"ביותר|הכי|\b(most|highest|lowest|largest|smallest|biggest|oldest|youngest|top)\b", re.IGNORECASE)

def ambiguous_term(question: str, categories: list | None = None) -> str | None:
    """
    The vague term that makes the question unanswerable without a threshold, or None.
    Not ambiguous when: the question carries a number, uses a superlative ("הגבוה ביותר"),
    or the term is a policy-defined category (handled by the policy pipeline).
    """
    if categories:
        return None
    if re.search(r"\d", question) or SUPERLATIVE_RE.search(question):
        return None
    m = AMBIGUOUS_TERMS_RE.search(question)
    return m.group(0) if m else None

def clarification_for(term: str, question: str) -> str:
    hebrew = bool(re.search(r"[֐-׿]", question))
    if hebrew:
        return (f"❓ המונח \"{term}\" אינו מוגדר במדיניות ואין לו סף מספרי בשאלה. "
                f"אנא ציינו סף מדויק (למשל: \"{term} = מעל 100,000 ₪\" או \"מעל 700\") כדי שאוכל להריץ את הניתוח הנכון.")
    return (f"❓ The term \"{term}\" has no numeric threshold in the question and no definition in the policy. "
            f"Please specify a threshold (e.g. \"{term} = above 100,000\") so I can run the right analysis.")


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
    ("Minimum income",            r"minimum (annual )?income|הכנסה[^.]{0,20}מינימלית|מינימום הכנסה",             "minimum annual income|minimum income"),
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

# Criteria the policy may use but the DB cannot verify (DTI, missed payments). Only when such a criterion
# is present does the answer show a "criteria that cannot be verified" block.
UNVERIFIABLE_RE = re.compile(r"\bdti\b|debt.to.income|missed payment")

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


# Columns/rows the user did not ask for: percentages, shares, ratios.
RATIO_Q_RE      = re.compile(r"אחוז|שיעור|percent|share|ratio|\brate\b", re.IGNORECASE)
RATIO_EXPR_RE   = re.compile(
    r"/|\*\s*100(?:\.0+)?\b|\b100(?:\.0+)?\s*\*|(?:^|[\s_(])(?:percent|percentage|share|ratio|rate|pct)(?:$|[\s_),])",
    re.IGNORECASE)
COMBINED_LABEL_RE = re.compile(r"לפחות קריטריון|at least one", re.IGNORECASE)

def asks_ratio(question: str) -> bool:
    return bool(RATIO_Q_RE.search(question or ""))

def _split_top_level(s: str, sep: str = ",") -> list[str]:
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur)); cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts

def ratio_columns(sql: str) -> list[str]:
    """SELECT-list expressions that compute a percentage / share / ratio."""
    m = re.match(r"(?is)^\s*select\s+(.*?)\s+from\s+", (sql or "").strip())
    if not m:
        return []
    return [e.strip() for e in _split_top_level(m.group(1)) if RATIO_EXPR_RE.search(e)]

def strip_ratio_columns(sql: str) -> tuple[str, int]:
    """Remove ratio expressions from the SELECT list. Returns (new_sql, removed_count)."""
    m = re.match(r"(?is)^\s*select\s+(.*?)\s+from\s+(.*)$", (sql or "").strip().rstrip(";"))
    if not m:
        return sql, 0
    exprs = _split_top_level(m.group(1))
    keep = [e.strip() for e in exprs if not RATIO_EXPR_RE.search(e)]
    removed = len(exprs) - len(keep)
    if removed == 0 or not keep:
        return sql, 0
    return "SELECT " + ", ".join(keep) + " FROM " + m.group(2).strip() + ";", removed


def wants_data(question: str) -> bool:
    return bool(DATA_INTENT_RE.search(normalize_for_match(question)))


# MEMBERSHIP question = count/list clients of a category ("כמה לקוחות הם High Risk", "which clients are Prime").
# Anything else that mentions a category is a RULE question ("מהי תקרת ההלוואה ל-Tier 2", "what does the policy
# say about guarantors") — there the LIMIT sections are exactly what is asked, and no SQL must run.
MEMBERSHIP_RE = re.compile(
    r"כמה|how many|\bcount\b|מספר ה|אחוז|שיעור|percent|share|"
    r"(?:אילו|איזה|which|list|רשימת|הצג)\W+(?:\w+\W+){0,1}?(?:לקוח|הלווא|client|loan|borrower)",
    re.IGNORECASE)

def question_mode(question: str) -> str:
    return "membership" if MEMBERSHIP_RE.search(normalize_for_match(question)) else "rule"


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
    mode = question_mode(question)
    found = []
    for label, trigger, terms in POLICY_CATEGORIES:
        if not re.search(trigger, q):
            continue
        all_sections = scan_policy(chunks, terms)
        if not all_sections:
            continue
        if mode == "rule":
            # A section is REQUIRED when the category is the subject of one of its lines
            # ("Tier 2 (Near-Prime): Up to 200,000 ILS..."), OPTIONAL when it is only mentioned in passing
            # ("...approved automatically if credit score is Tier 1 or Tier 2"). No subject lines → all required.
            norm_terms = [normalize_for_match(t) for t in re.split(r"[|,]", terms) if t.strip()]
            subject = [s for s in all_sections
                       if any(normalize_for_match(ln).startswith(t) for ln in s["matching_lines"] for t in norm_terms)]
            required = subject or all_sections
            optional = [s for s in all_sections if s not in required]
            found.append({
                "label": label, "terms": terms, "mode": "rule",
                "sections": required, "optional_sections": optional, "limit_sections": [],
                "section_ids": [s["section"].split(" ")[0] for s in required],
                "limit_section_ids": [], "columns": [], "unverifiable_sections": [],
            })
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
        unverifiable = [s["section"].split(" ")[0] for s in criteria
                        if UNVERIFIABLE_RE.search(normalize_for_match(" ".join(s["matching_lines"])))]
        found.append({
            "label": label,
            "terms": terms,
            "mode": "membership",
            "sections": criteria,
            "limit_sections": limits,
            "section_ids": [s["section"].split(" ")[0] for s in criteria],
            "limit_section_ids": [s["section"].split(" ")[0] for s in limits],
            "columns": columns,
            "unverifiable_sections": unverifiable,
        })
    return found


# ─────────────────────────────────────────────
# 2. Context injection
# ─────────────────────────────────────────────

def build_rule_context(categories: list[dict]) -> str:
    parts = []
    for c in categories:
        lines = []
        for s in c["sections"]:
            lines.append(f"  Section {s['section']}:")
            lines += [f"    - {ln}" for ln in s["matching_lines"]]
        block = f"Category '{c['label']}' — policy sections about it (MUST all be cited):\n" + "\n".join(lines)
        if c.get("optional_sections"):
            opt = []
            for s in c["optional_sections"]:
                opt.append(f"  Section {s['section']}:")
                opt += [f"    - {ln}" for ln in s["matching_lines"]]
            block += "\nSections that only mention it in passing (cite if relevant to the question):\n" + "\n".join(opt)
        parts.append(block)
    return (
        "\n\n[POLICY CONTEXT — injected automatically by the system. This question asks what the POLICY SAYS about "
        "the category (a rule, a limit, a requirement), NOT about the portfolio data.]\n"
        + "\n\n".join(parts)
        + "\n\nREQUIREMENTS FOR YOUR ANSWER:\n"
        "1. Answer ONLY from the sections above. Do NOT query the database — no sql_db_query call; "
        "sql_query = \"\", table_data = [], tool_used = \"policy\", output_format = \"text\".\n"
        "2. State the specific figures the question asks for EXACTLY as written in the policy "
        "(amounts such as \"200,000 ILS\", score ranges, percentages, durations).\n"
        "3. Cite EVERY section listed above in the answer text, written as the word — Hebrew \"סעיף 8.1\", "
        "English \"Section 8.1\" — never the '§' sign. policy_sources = all of these sections.\n"
        "4. confidence_score 90-100: the answer is grounded in the policy text.\n"
        "An answer that omits one of these sections, or that queries the database, will be rejected."
    )


def build_policy_context(categories: list[dict], enforce_sql: bool) -> str:
    if categories and categories[0].get("mode") == "rule":
        return build_rule_context(categories)
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

    n_criteria = sum(len(c["sections"]) for c in categories)
    if enforce_sql and n_criteria == 1:
        data_req = (
            "2. There is exactly ONE criterion. Run ONE SQL query over the WHOLE portfolio with: the count of "
            "clients meeting it, and the total — nothing else. No percentage/share/ratio column unless the user "
            "asked for one, no limit-related columns.\n"
            "3. Put the numbers in \"table_data\" ONLY (not in \"answer\"), rows {\"מדד\": label, \"ערך\": n} "
            "(English: Metric / Value), in THIS order: the criterion row, then \"סך כל הלקוחות\" / \"Total clients\". "
            "Do NOT add a \"לקוחות שעומדים בלפחות קריטריון אחד\" row — it is meaningless with a single criterion. "
            "No other rows. Set output_format to \"table+text\" (+sql if the user asked for the query).\n"
        )
    elif enforce_sql:
        data_req = (
            "2. Run ONE SQL query over the WHOLE portfolio with: one count per criterion above (each checked "
            "standalone); ONE combined count = clients meeting AT LEAST ONE criterion (OR over the criteria); "
            "and the total. No limit-related columns, no AND-combinations, no percentage/share/ratio columns "
            "unless the user asked for them.\n"
            "3. Put the numbers in \"table_data\" ONLY (not in \"answer\"), rows {\"מדד\": label, \"ערך\": n} "
            "(English: Metric / Value), in THIS order: one row per criterion in bullet order (e.g. "
            "\"לקוחות עם ציון מתחת ל-550\", \"לקוחות עם 0–1 שנות תעסוקה\"), then EXACTLY "
            "\"לקוחות שעומדים בלפחות קריטריון אחד\" / \"Clients meeting at least one criterion\", then "
            "\"סך כל הלקוחות\" / \"Total clients\". No other rows, no percentage/share/ratio rows. "
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

def check_completeness(data: dict, categories: list[dict], enforce_sql: bool, question: str = "") -> dict:
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

    if categories and categories[0].get("mode") == "rule":
        unexpected_sql = bool(sql.strip())
        return {
            "mode": "rule",
            "expected_sections": expected_sections, "excluded_sections": [], "expected_columns": [],
            "has_unverifiable": False,
            "missing_sections": missing_sections, "missing_in_sources": missing_in_sources,
            "unexpected_sections": [], "missing_columns": [], "multi_statement": False,
            "forbidden_columns_used": [], "unexpected_sql": unexpected_sql,
            "ok": not (missing_sections or missing_in_sources or unexpected_sql),
        }
    unexpected_sections = [s for s in excluded_sections if s in in_answer or s in in_sources]
    missing_columns = [c for c in expected_columns if c not in sql] if (enforce_sql or sql) else []
    # the SQL must be ONE statement over the criteria only — no limit columns, no extra queries
    statements = [st for st in sql.split(";") if st.strip()]
    multi_statement = len(statements) > 1
    forbidden = [c for c in CHECKABLE_COLUMNS if c not in expected_columns]
    forbidden_columns_used = [c for c in forbidden if re.search(r"\b" + c + r"\b", sql)] if sql else []
    has_unverifiable = any(c.get("unverifiable_sections") for c in categories)
    # nothing the user did not ask for
    unrequested_ratio = ratio_columns(sql) if (sql and not asks_ratio(question)) else []
    single = len(expected_sections) == 1
    rows = [r for r in (data.get("table_data") or []) if isinstance(r, dict) and r]
    labels = [str(next(iter(r.values()), "")) for r in rows]
    redundant_combined_row = single and any(COMBINED_LABEL_RE.search(l) for l in labels)
    unrequested_ratio_row = (not asks_ratio(question)) and any(RATIO_Q_RE.search(l) for l in labels)
    return {
        "mode": "membership",
        "single_criterion": single,
        "unrequested_ratio": unrequested_ratio,
        "redundant_combined_row": redundant_combined_row,
        "unrequested_ratio_row": unrequested_ratio_row,
        "expected_sections": expected_sections,
        "excluded_sections": excluded_sections,
        "expected_columns": expected_columns,
        "has_unverifiable": has_unverifiable,
        "unexpected_sql": False,
        "missing_sections": missing_sections,          # criterion sections not mentioned in the answer text
        "missing_in_sources": missing_in_sources,      # criterion sections not listed in policy_sources
        "unexpected_sections": unexpected_sections,    # limit-only sections that were wrongly included
        "missing_columns": missing_columns,
        "multi_statement": multi_statement,            # more than one SQL statement
        "forbidden_columns_used": forbidden_columns_used,  # limit columns (e.g. loan_amount) in the SQL
        "ok": not (missing_sections or missing_in_sources or unexpected_sections or missing_columns
                   or multi_statement or forbidden_columns_used or unrequested_ratio
                   or redundant_combined_row or unrequested_ratio_row),
    }


def sanitize_policy_answer(data: dict, categories: list[dict], report: dict, question: str = "") -> list[str]:
    """
    Deterministic clean-up applied AFTER the retry round, so what the user sees is right even if the
    model ignored a correction. Returns a list of human-readable notes describing what was changed.
    """
    notes = []
    excluded = report.get("excluded_sections", [])

    if report.get("mode") == "rule":
        # a rule question: the policy text is the answer; any SQL/table the model ran is noise
        if (data.get("sql_query") or "").strip() or data.get("table_data"):
            data["sql_query"] = ""
            data["table_data"] = []
            data["output_format"] = "text"
            notes.append("dropped the database query/table (the question asks about the policy rule, not the data)")
        if data.get("tool_used") in ("database", "hybrid"):
            data["tool_used"] = "policy"
        cov = data.get("data_coverage")
        if isinstance(cov, dict):
            cov["checked"], cov["missing"] = [], []
        return notes

    # 1. SQL: keep only the first statement (the per-criterion count query)
    sql = data.get("sql_query") or ""
    statements = [st.strip() for st in sql.split(";") if st.strip()]
    if len(statements) > 1:
        data["sql_query"] = statements[0] + ";"
        notes.append(f"dropped {len(statements) - 1} extra SQL statement(s)")

    # 1b. columns / rows the user did not ask for (percentages, shares) and a meaningless combined row
    if not asks_ratio(question):
        new_sql, n = strip_ratio_columns(data.get("sql_query") or "")
        if n:
            data["sql_query"] = new_sql
            notes.append(f"removed {n} unrequested ratio column(s) from the SQL")
    rows = [r for r in (data.get("table_data") or []) if isinstance(r, dict) and r]
    kept = []
    for r in rows:
        label = str(next(iter(r.values()), ""))
        if not asks_ratio(question) and RATIO_Q_RE.search(label):
            notes.append("removed an unrequested ratio row from the table"); continue
        if report.get("single_criterion") and COMBINED_LABEL_RE.search(label):
            notes.append("removed the 'at least one criterion' row (single criterion)"); continue
        kept.append(r)
    if len(kept) != len(rows):
        data["table_data"] = kept

    # 2. answer text: remove bullet lines that cite an excluded (limit-only) section
    if excluded:
        pat = re.compile(r"(?:§|section|סעיף)\s*(" + "|".join(re.escape(s) for s in excluded) + r")\b", re.IGNORECASE)
        kept, removed = [], []
        for line in str(data.get("answer", "") or "").split("\n"):
            (removed if pat.search(line) else kept).append(line)
        if removed:
            data["answer"] = "\n".join(kept).strip()
            notes.append("removed " + ", ".join("Section " + s for s in sorted({m.group(1) for ln in removed for m in [pat.search(ln)] if m})) + " from the answer")

        # 3. policy_sources: drop excluded sections
        srcs = data.get("policy_sources", []) or []
        clean = [s for s in srcs if not re.match(r"\s*(?:§|section|סעיף)?\s*(" + "|".join(re.escape(x) for x in excluded) + r")\b", str(s), re.IGNORECASE)]
        if len(clean) != len(srcs):
            data["policy_sources"] = clean
            notes.append("removed limit-only section(s) from policy sources")

        # 4. table rows that mention an excluded section or loan-amount limits
        rows = data.get("table_data", []) or []
        clean_rows = [r for r in rows if not (isinstance(r, dict) and r and
                      (pat.search(str(next(iter(r.values())))) or re.search(r"חריג|violation|תקרת|limit", str(next(iter(r.values()))), re.IGNORECASE)))]
        if len(clean_rows) != len(rows):
            data["table_data"] = clean_rows
            notes.append(f"removed {len(rows) - len(clean_rows)} limit-related table row(s)")

    # 5. "cannot be verified" block only when a criterion truly has no DB column (DTI, missed payments)
    cov = data.get("data_coverage")
    if isinstance(cov, dict) and cov.get("missing") and not report.get("has_unverifiable"):
        cov["missing"] = []
        notes.append("cleared the 'cannot be verified' list (all criteria are verifiable in the DB)")

    # 6. routing label is a fact, not a self-report: policy context was injected AND SQL ran → hybrid
    if (data.get("sql_query") or "").strip() and data.get("tool_used") != "hybrid":
        data["tool_used"] = "hybrid"
        notes.append("routing label set to 'hybrid' (policy definition + database query)")

    return notes


def final_confidence(data: dict, report: dict) -> int:
    """
    Confidence policy for policy-category answers: when the completeness check passed and every criterion
    is verifiable in the DB, the answer is fully grounded → high confidence (≥ 95), whatever the model said.
    """
    try:
        conf = int(data.get("confidence_score", 0))
    except (TypeError, ValueError):
        conf = 0
    if report.get("ok") and not report.get("has_unverifiable"):
        return max(conf, 95)
    return conf





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
    if report.get("unexpected_sql"):
        msg += ("- This question asks what the POLICY says (a rule / limit), not about the portfolio: do NOT query "
                "the database. Set sql_query to \"\", table_data to [], tool_used to \"policy\", and answer from the "
                "policy sections with their exact figures.\n")
    if report.get("multi_statement"):
        msg += "- Your sql_query contains more than one statement. Return EXACTLY ONE SELECT statement.\n"
    if report.get("unrequested_ratio"):
        msg += ("- Your SQL computes a percentage/share the user did not ask for — remove: "
                + "; ".join(report["unrequested_ratio"]) + "\n")
    if report.get("unrequested_ratio_row"):
        msg += "- Remove the percentage/share row from table_data — the user did not ask for it.\n"
    if report.get("redundant_combined_row"):
        msg += "- There is only ONE criterion: remove the 'לקוחות שעומדים בלפחות קריטריון אחד' row.\n"
    if report.get("forbidden_columns_used"):
        msg += ("- Your SQL uses columns that belong to LIMITS, not criteria, and must be removed: "
                + ", ".join(report["forbidden_columns_used"]) + "\n")
    msg += (
        "Rewrite the COMPLETE answer now, following the MANDATORY WORKFLOW: cover EVERY section listed in the "
        "POLICY CONTEXT with its exact section number, run ONE SQL query that checks EVERY checkable criterion "
        "(ONE statement: each criterion standalone over the whole portfolio, then at least one criterion, then total; "
        "no limit columns, no extra queries), leave data_coverage.missing EMPTY unless a criterion has no DB column, "
        "and fill policy_sources with the criterion sections only. Return ONLY the JSON object."
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
