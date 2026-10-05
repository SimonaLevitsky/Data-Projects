import os
import re
import json
import math
import time
import pandas as pd
import streamlit as st
from langchain.agents import create_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from build_vector_db import load_or_build_index, load_policy_text, chunk_policy, POLICY_PATH, INDEX_DIR
from policy_enforcement import (scan_policy, detect_policy_categories, build_policy_context,
                                check_completeness, build_correction, wants_data,
                                sanitize_policy_answer, final_confidence)

APP_START = time.time()

def startup_log(msg: str):
    """Terminal-side trace of the startup phases (visible where `streamlit run` was launched)."""
    print(f"[startup +{time.time() - APP_START:5.1f}s] {msg}", flush=True)

startup_log("imports done")

st.set_page_config(
    page_title="AI Credit Risk Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Credit Risk Assistant")
st.markdown("Ask questions about the loan portfolio, request financial calculations, or consult the credit underwriting policy — the agent routes each question to the right tool (SQL, calculator, or policy retriever).")

# --- API Key ---
openai_api_key = None
try:
    openai_api_key = st.secrets.get("OPENAI_API_KEY")
except Exception:
    pass
openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("שגיאה: מפתח ה-OPENAI_API_KEY לא נמצא.")
    st.stop()

# --- Database ---
current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_dir, "credit_risk.db")

if not os.path.exists(db_path):
    st.error(f"שגיאה: קובץ מסד הנתונים '{db_path}' לא נמצא.")
    st.stop()

startup_log("API key found; opening SQLite DB")
db = SQLDatabase.from_uri(
    f"sqlite:///{db_path}",
    include_tables=["loans", "demographics"],
    sample_rows_in_table_info=2
)
startup_log("DB ready; parsing policy document (local, no network)")

# --- Policy Vector Store (RAG) ---
if not os.path.exists(POLICY_PATH):
    st.error(f"שגיאה: מסמך המדיניות '{POLICY_PATH}' לא נמצא.")
    st.stop()

@st.cache_resource(show_spinner="טוען את מאגר המדיניות (Vector DB)...")
def get_policy_retriever(_api_key: str):
    """Load the FAISS index once per server process (built on first run if missing)."""
    t0 = time.time()
    existed = os.path.exists(os.path.join(INDEX_DIR, "index.faiss"))
    store = load_or_build_index(_api_key)
    info = {
        "source": "loaded from disk" if existed else "BUILT now (OpenAI embeddings call)",
        "vectors": store.index.ntotal,
        "seconds": round(time.time() - t0, 1),
    }
    return store.as_retriever(search_kwargs={"k": 5}), info

@st.cache_resource
def get_policy_chunks():
    """All policy sub-sections in document order — used for exhaustive keyword scans."""
    return chunk_policy(load_policy_text())

def get_retriever():
    """Lazy accessor: the FAISS index is loaded/built on the FIRST policy question, never at startup."""
    retriever, info = get_policy_retriever(openai_api_key)
    st.session_state["index_info"] = info
    return retriever

policy_chunks = get_policy_chunks()   # local text parsing only — no network
startup_log("policy chunks ready (vector DB deferred to first policy question); building agent")

# ── Sidebar diagnostics: where does the time go? ──
with st.sidebar:
    st.markdown("### 🔧 Diagnostics")
    _info = st.session_state.get("index_info")
    if _info:
        st.caption(f"Vector DB: {_info['source']} · {_info['vectors']} vectors · {_info['seconds']}s")
    elif os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        st.caption("Vector DB: on disk — loads on the first policy question")
    else:
        st.caption("Vector DB: not built yet — will be built on the first policy question (one OpenAI call)")
    st.caption(f"Policy chunks: {len(policy_chunks)} · Startup: {round(time.time() - APP_START, 1)}s")
    if st.button("Load / build vector DB now", help="Runs the OpenAI embeddings call explicitly and shows the time or the error."):
        try:
            _t = time.time()
            get_retriever()
            st.success(f"OK in {time.time() - _t:.1f}s — {st.session_state['index_info']['source']}")
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")
    _lt = st.session_state.get("last_timings")
    if _lt:
        st.markdown("**Last question**")
        for k, v in _lt.items():
            st.caption(f"{k}: {v}")

# ─────────────────────────────────────────────
# FINANCIAL CALCULATOR TOOLS
# ─────────────────────────────────────────────

@tool
def calculate_monthly_payment(principal: float, annual_rate_percent: float, term_months: int) -> str:
    """
    Calculate the fixed monthly payment for a loan.
    Use when the user asks: 'what is the monthly payment for a loan of X at Y% for Z months/years?'
    or any variant involving monthly installments, repayment amount, or loan cost.
    principal: loan amount in currency units.
    annual_rate_percent: annual interest rate as a percentage (e.g. 5.5 for 5.5%).
    term_months: total number of monthly payments.
    """
    if annual_rate_percent <= 0:
        monthly = principal / term_months
        total = principal
        interest = 0.0
    else:
        r = annual_rate_percent / 100 / 12
        monthly = principal * (r * (1 + r) ** term_months) / ((1 + r) ** term_months - 1)
        total = monthly * term_months
        interest = total - principal

    return json.dumps({
        "tool": "monthly_payment_calculator",
        "principal": round(principal, 2),
        "annual_rate_percent": annual_rate_percent,
        "term_months": term_months,
        "monthly_payment": round(monthly, 2),
        "total_paid": round(total, 2),
        "total_interest": round(interest, 2)
    })


@tool
def calculate_compound_interest(principal: float, annual_rate_percent: float, years: int) -> str:
    """
    Calculate compound interest and the future value of a principal amount.
    Use when the user asks: 'if I invest/deposit X at Y% for Z years, how much will I have?'
    or any question about future value, compound growth, or interest earned over time.
    principal: starting amount.
    annual_rate_percent: annual interest rate as a percentage.
    years: number of years.
    """
    r = annual_rate_percent / 100
    future_value = principal * (1 + r) ** years
    interest_earned = future_value - principal

    return json.dumps({
        "tool": "compound_interest_calculator",
        "principal": round(principal, 2),
        "annual_rate_percent": annual_rate_percent,
        "years": years,
        "future_value": round(future_value, 2),
        "interest_earned": round(interest_earned, 2)
    })


@tool
def calculate_debt_to_income(monthly_debt_payment: float, monthly_gross_income: float) -> str:
    """
    Calculate the Debt-to-Income (DTI) ratio and assess loan affordability.
    Use when the user asks: 'can a client afford this loan?', 'what is the DTI ratio?',
    or 'is this client eligible for a loan given their income and payment?'
    monthly_debt_payment: total monthly debt obligations in currency units.
    monthly_gross_income: gross monthly income before taxes.
    """
    if monthly_gross_income <= 0:
        return json.dumps({"error": "Monthly income must be greater than zero."})

    dti = (monthly_debt_payment / monthly_gross_income) * 100

    if dti < 28:
        risk_level = "Low Risk"
        eligibility = "Typically eligible for most loan products."
    elif dti < 36:
        risk_level = "Moderate Risk"
        eligibility = "Eligible for most conventional loans; some lenders may require a higher credit score."
    elif dti < 43:
        risk_level = "High Risk"
        eligibility = "At the upper limit for conventional loans (43% is the common threshold)."
    else:
        risk_level = "Very High Risk"
        eligibility = "Generally ineligible for conventional loans. High probability of default."

    return json.dumps({
        "tool": "dti_calculator",
        "monthly_debt_payment": round(monthly_debt_payment, 2),
        "monthly_gross_income": round(monthly_gross_income, 2),
        "dti_ratio_percent": round(dti, 1),
        "risk_level": risk_level,
        "eligibility_note": eligibility
    })


# ─────────────────────────────────────────────
# POLICY RETRIEVER TOOL (RAG)
# ─────────────────────────────────────────────

@tool
def search_credit_policy(query: str) -> str:
    """
    Semantic search over the institution's Credit Underwriting Policy (internal document).
    Use for ANY question about rules, thresholds, definitions, eligibility, tiers, limits,
    or procedures — e.g. 'what credit score is considered high risk?', 'what is the maximum DTI?',
    'what are the loan limits for Tier 2?', 'how are defaulted loans handled?',
    'what does the policy say about marital status / age / employment years?'.
    Also use it to look up the official definition of a vague term (risky, young, experienced,
    large loan) BEFORE deciding whether to query the database.
    query: a short natural-language description of the rule you are looking for (English works best).
    Returns the most relevant policy passages with their section references.
    NOTE: this is a similarity search — it returns the BEST matches, not necessarily ALL of them.
    When you need every section that mentions a specific category/term, use find_all_policy_mentions.
    """
    docs = get_retriever().invoke(query)
    passages = []
    for d in docs:
        m = d.metadata
        passages.append({
            "section": f"{m.get('subsection', '')} {m.get('subsection_title', '')}".strip(),
            "parent_section": f"Section {m.get('section', '')}: {m.get('section_title', '')}",
            "text": d.page_content
        })
    return json.dumps({
        "tool": "credit_policy_retriever",
        "query": query,
        "num_passages": len(passages),
        "passages": passages
    }, ensure_ascii=False)


@tool
def find_all_policy_mentions(terms: str) -> str:
    """
    EXHAUSTIVE scan of the ENTIRE Credit Underwriting Policy for a specific term or category.
    Returns EVERY sub-section that mentions it (not just the top matches), with the exact matching lines.
    Use this whenever a question is about a policy-defined category or label, e.g.
    'High Risk', 'Tier 4', 'Prime', 'Sub-Prime', 'Low Risk', 'Watch', 'Default', 'guarantor',
    'collateral', 'Credit Committee', 'self-employed', 'minimum income' — so that the answer covers
    ALL the criteria the policy attaches to that category across all sections.
    terms: one or more search terms separated by '|' (synonyms are matched independently),
           e.g. 'high risk|tier 4'  or  'low risk|tier 1|prime'. Case- and hyphen-insensitive.
    Typical flow: call this first → collect every criterion → then decide which criteria can be
    checked against the database and which cannot.
    """
    matches = scan_policy(policy_chunks, terms)
    term_list = [t.strip() for t in re.split(r"[|,]", terms) if t.strip()]
    return json.dumps({
        "tool": "policy_exhaustive_scan",
        "terms": term_list,
        "num_sections_found": len(matches),
        "sections": matches,
        "note": "This list is complete: no other sub-section of the policy mentions these terms."
    }, ensure_ascii=False)


# ─────────────────────────────────────────────
# SYSTEM PROMPT — MULTI-TOOL ROUTER
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Senior Credit Risk Manager AI assistant at a financial institution.
You have access to THREE types of tools and must route each question to the correct one(s).

═══════════════════════════════════════════════
TOOL ROUTING — DECIDE BEFORE EVERY ANSWER:
═══════════════════════════════════════════════

1. SQL DATABASE TOOLS → use for questions about the loan portfolio data:
   - Aggregate statistics (averages, counts, distributions)
   - Client lookups, comparisons, rankings
   - Default rates, credit scores, incomes from the database

2. FINANCIAL CALCULATOR TOOLS → use for hypothetical computations:
   - calculate_monthly_payment: "What would the monthly payment be for a $50,000 loan at 6% for 5 years?"
   - calculate_compound_interest: "If I invest $10,000 at 4% for 10 years, what do I get?"
   - calculate_debt_to_income: "Is a client with $3,000/month income and $900/month payments eligible?"

3. POLICY TOOLS → use for questions about RULES, not data:
   - search_credit_policy(query): semantic search — best for "what does the policy say about X?"
     "What is the maximum DTI?", "How does the policy treat divorced applicants?", "When is a loan written off?"
   - find_all_policy_mentions(terms): EXHAUSTIVE scan — returns EVERY section mentioning a category/label.
     MANDATORY whenever the question names a policy-defined category (High Risk, Low Risk, Tier 1-4,
     Prime, Sub-Prime, Watch, Substandard, guarantor, collateral, Credit Committee, minimum income...).
     A category is often defined in SEVERAL sections (e.g. "High Risk" appears in Sections 2.1, 4.1 and 8.1);
     a single semantic search would miss some of them, which makes the answer INCOMPLETE and WRONG.
   Always cite the section numbers you relied on.
   SECTION CITATION FORMAT — NEVER use the "§" sign. Write the word:
     Hebrew answer → "סעיף 2.1" (plural: "סעיפים 2.1, 4.1 ו-8.1");  English answer → "Section 2.1".
   NEVER answer a policy question from memory — if you did not retrieve it, you do not know it.

4. HYBRID → use when the question combines two or more tool types:
   a) DB + Calculator: "Average monthly payment for the top 10 loans at 5% for 10 years?"
      → query the DB for the loan amounts, then call calculate_monthly_payment.
   b) POLICY + DB (the most valuable case) — see the mandatory workflow below.
      Examples: "How many clients are High Risk per the policy?", "Which clients violate the minimum
      income rule?", "Which clients are Prime / Tier 1?", "What share of the portfolio fails the
      eligibility criteria?", "Is our default rate within the policy benchmark?"
   c) POLICY + Calculator: "Client earns 8,000/month and pays 3,600 — is that allowed?"
      → compute DTI, then retrieve the DTI thresholds and apply them.

═══════════════════════════════════════════════
MANDATORY WORKFLOW — POLICY-DEFINED CATEGORY APPLIED TO THE DATABASE
═══════════════════════════════════════════════
Trigger: the question asks to COUNT / LIST / MEASURE clients (or loans) using a label or category that
the POLICY defines (High Risk, Low Risk, Tier N, Prime, Sub-Prime, eligible/ineligible, preferred
applicant, policy violation, within/above a policy limit, etc.).

STEP 1 — COLLECT EVERY SECTION (never rely on a single section):
   Call find_all_policy_mentions with the label and its synonyms (e.g. "high risk|tier 4").
   A category is usually mentioned in SEVERAL sections.

STEP 2 — SPLIT THE SECTIONS INTO CRITERIA vs LIMITS:
   ✔ CRITERION section = states WHO belongs to the category: a threshold on credit score, employment
     years, income, age, DTI, missed payments, default status, marital status.
     e.g. Section 2.1 "Tier 4 — High Risk (Score below 550)", Section 4.1 "0–1 years of employment: High risk".
   ✘ LIMIT section = states what applies to clients who are ALREADY in the category: maximum loan amount,
     guarantors, collateral, Credit Committee review, approval rate, max term.
     e.g. Section 8.1 "Tier 4 (High Risk): Up to 50,000 ILS with two guarantors".
   A question like "how many clients are <category>?" asks for the CRITERIA. LIMIT sections are NOT part
   of the answer: do not mention them, do not count them, do not list them in policy_sources.
   (A line that contains both a threshold and limits — like Section 2.1 — is a CRITERION section; use its
   threshold and ignore its limits.)

STEP 3 — ONE CONDITION PER CRITERION SECTION, EACH COUNTED OVER THE WHOLE PORTFOLIO:
   Boundary rules: "below 550" → credit_score < 550; "0–1 years" → employment_years < 1 (state it).
   Run EXACTLY ONE SQL statement (never several queries separated by ';') with: one count per criterion,
   ONE combined count = clients meeting AT LEAST ONE
   criterion (OR over the criteria), and the total:
   SELECT
     SUM(CASE WHEN l.credit_score < 550 THEN 1 ELSE 0 END)                           AS score_below_550,
     SUM(CASE WHEN d.employment_years < 1 THEN 1 ELSE 0 END)                         AS employment_under_1_year,
     SUM(CASE WHEN l.credit_score < 550 OR d.employment_years < 1 THEN 1 ELSE 0 END) AS at_least_one_criterion,
     COUNT(*) AS total_clients
   FROM loans l JOIN demographics d ON l.client_id = d.client_id;
   This rule is GENERAL: whenever a question involves several criteria, report each criterion separately,
   then "at least one criterion", then the total. Never add AND-combinations or limit columns.

STEP 4 — ANSWER STRUCTURE (STRICT — the UI renders these fields in a fixed layout):

   "answer" = PART 1 ONLY — the policy definition:
      Hebrew:  "לפי המדיניות, לקוחות בקטגוריית <label> מוגדרים על פי:"
      English: "According to the policy, <label> clients are defined by:"
      followed by ONE bullet PER CRITERION SECTION, in section order, ending with the section number:
         * ציון אשראי מתחת ל-550 (סעיף 2.1).
         * 0–1 שנות תעסוקה (סעיף 4.1).
      NOTHING ELSE in "answer": no numbers, no counts, no limits. (If you made a boundary assumption,
      add ONE short final line: "הנחה: ..." / "Assumption: ...".)

   "table_data" = PART 2 — "what was checked in the data", rows {"מדד": "<label>", "ערך": <number>}
      (English: {"Metric": ..., "Value": ...}), in THIS order:
      1. one row per criterion, same order as the bullets:   "לקוחות עם ציון מתחת ל-550" → 28
                                                              "לקוחות עם 0–1 שנות תעסוקה" → 4
      2. "לקוחות שעומדים בלפחות קריטריון אחד" / "Clients meeting at least one criterion" → OR over the criteria
      3. "סך כל הלקוחות" / "Total clients".
      Exactly these rows — no extra rows.
   output_format = "table+text"  (append "+sql" when the user asked for the query).

   "data_coverage.checked" = the criteria you verified, each with its section.
   "data_coverage.missing" = ONLY criteria that cannot be verified in the data (e.g. DTI, missed payments).
      Leave it EMPTY when every criterion was verified. Never put limits (guarantors, amounts, committee) there.
   "policy_sources" = exactly the criterion sections.  tool_used: "hybrid".
   confidence_score: 90-100 when every criterion is defined by the policy AND verified in the DB (the normal
   case — the answer is fully grounded); 70-89 only if a criterion could not be verified in the data.
   NEVER present a single-criterion count as "the" answer when the policy lists several criteria.

WORKED EXAMPLE — "How many clients in the portfolio are High Risk according to the policy?"
   → find_all_policy_mentions("high risk|tier 4") returns Section 2.1 (Tier 4: score below 550 + limits),
     Section 4.1 (0–1 years employment: high risk + a 30,000 ILS restriction), Section 8.1 (Tier 4 limits only).
   → Criteria: credit_score < 550 (Section 2.1); employment_years < 1 (Section 4.1).
     Section 8.1 is a LIMIT section → excluded from the answer entirely.
   → Run the SQL from STEP 3; "answer" = 2 bullets; "table_data" = 2 criterion rows + "at least one
     criterion" + total; data_coverage.missing = [] (both criteria verified).

POLICY LOOKUP FOR VAGUE TERMS:
When the user uses a vague qualifier (risky, high risk, young, experienced, large loan, good score, etc.),
FIRST call find_all_policy_mentions / search_credit_policy to check whether the policy defines it.
- If the policy defines it → use that definition (ALL its sections), state it explicitly in the answer,
  set confidence_score 70-89 (interpretation based on policy), and follow the MANDATORY WORKFLOW above.
- If the policy does NOT define it → fall back to the AMBIGUITY DETECTION rule below and ask.

DATABASE SCHEMA — EXACT, COMPLETE:
- loans: client_id (INTEGER), age (INTEGER), loan_amount (INTEGER), credit_score (INTEGER), default_status (INTEGER — 0=performing, 1=default)
- demographics: client_id (INTEGER), employment_years (INTEGER), annual_income (INTEGER), marital_status (TEXT — Single/Married/Divorced/Widowed)
The two tables are joined on client_id. There are NO other tables, columns, or data fields.

LANGUAGE RULE:
Detect the user's language and respond in the same language in the "answer" field.
Hebrew → answer in Hebrew. English → answer in English.

OUTPUT FORMAT — STRICT JSON, NO PROSE OUTSIDE IT:
Always return a valid JSON object with exactly these fields:

{
  "answer": "Natural language finding in the user's language.",
  "sql_query": "SQL executed, or empty string if no database query was run.",
  "tool_used": "<one of: database | calculator | policy | hybrid | none>",
  "confidence_score": <integer 0-100>,
  "output_format": "<one of: text | table | sql | text+sql | table+sql | table+text+sql>",
  "table_data": [ {"column": value, ...}, ... ],
  "policy_sources": [ "2.1 Credit Score Tiers", ... ],
  "data_coverage": {
    "checked": [ "credit_score < 550 (Section 2.1)", ... ],
    "missing": [ "Two guarantors required (Section 2.1) — guarantor data not in DB", ... ]
  }
}

policy_sources: list the policy section labels (as returned by the policy tools in the "section" field)
that you actually used in the answer. Empty list [] if no policy tool was used.
data_coverage: ONLY for policy-applied-to-data questions — "checked" = criteria verified by SQL,
"missing" = criteria that could not be verified and why. Otherwise {"checked": [], "missing": []}.

OUTPUT FORMAT SELECTION RULES:
- User asks for a table / טבלה → output_format "table", populate table_data
- User asks for SQL / query / שאילתה / קוד → include "sql" in output_format
- Combination → use "+": "table+sql", "table+text+sql"
- Default → output_format "text", table_data = []

CONFIDENCE SCORE RULES:
- 90-100: Unambiguous question, exact schema match or clear calculation. No interpretation needed.
- 70-89: Minor interpretation required. MUST state the assumption in the answer.
- 50-69: Ambiguous question. Do NOT answer — ask for clarification.
- 0-49: Missing data or cannot answer reliably. Ask for clarification.

AMBIGUITY DETECTION — MANDATORY BEFORE EVERY RESPONSE:
If ANY of these ambiguous qualifiers appear AND the policy does not define them (see POLICY LOOKUP FOR VAGUE TERMS),
set confidence_score ≤ 50 and ask for clarification:
- "מסוכן" / "risky" / "high risk" → what criterion? (credit score threshold? default status? formula?)
- "בעייתי" / "problematic" → what definition?
- "קשר" / "relationship" / "connection" → what analysis type? (group average? correlation? distribution?)
- "גדול/ה" / "large" / "big" / "high" (without a number) → what minimum threshold?
- "קטן/ה" / "small" / "low" (without a number) → what maximum threshold?
- "צעיר/ה" / "young" → maximum age?
- "מבוגר" / "older" → minimum age?
- "ותיק" / "experienced" → minimum employment_years?
- "הרבה" / "many" / "רב" → quantity threshold?
- "מעט" / "few" → quantity threshold?
- "טוב" / "good" / "strong" (re: score or income) → numeric threshold?

Clarification format: "❓ [specific question about the ambiguous term]. Please clarify so I can run the right analysis."

HALLUCINATION PREVENTION:
Concepts not in the database: mortgage/משכנתא, interest rate/ריבית, loan type/סוג הלוואה, balance/יתרה, missed payments, guarantors, collateral.
If asked for DATA about these → confidence_score=0, sql_query="", explain what IS available.
For calculator questions, these concepts ARE valid inputs (the user provides them directly).
For POLICY questions, these concepts may be answered from the retrieved policy text — but ONLY from what was retrieved.
If search_credit_policy returns nothing relevant → say the policy does not cover it (confidence_score=0). Do not invent rules.

OUT-OF-DOMAIN (recipes, weather, stocks, coding, personal questions, DB modifications):
{"answer": "⚠️ I'm a Credit Risk Assistant. I can only answer questions about the loan portfolio data, the credit underwriting policy, or perform financial calculations.", "sql_query": "", "tool_used": "none", "confidence_score": 0, "output_format": "text", "table_data": [], "policy_sources": [], "data_coverage": {"checked": [], "missing": []}}"""

# ─────────────────────────────────────────────
# LLM, TOOLS & AGENT
# ─────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key, timeout=90, max_retries=2)
AGENT_RECURSION_LIMIT = 30  # max agent steps (model calls + tool calls) per attempt — prevents endless tool loops

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_tools = toolkit.get_tools()
calculator_tools = [calculate_monthly_payment, calculate_compound_interest, calculate_debt_to_income]
policy_tools = [search_credit_policy, find_all_policy_mentions]

agent = create_agent(
    model=llm,
    tools=sql_tools + calculator_tools + policy_tools,
    system_prompt=SYSTEM_PROMPT
)
startup_log("agent ready; rendering UI")

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

TOOL_BADGES = {
    "database":   ("#1f77b4", "🗄️ Database"),
    "calculator": ("#6f42c1", "🧮 Calculator"),
    "policy":     ("#20a39e", "📜 Policy (RAG)"),
    "hybrid":     ("#e67e22", "🔀 Hybrid"),
    "none":       ("#6c757d", "⚠️ N/A"),
}

HEBREW_RE = re.compile(r"[֐-׿]")

def is_hebrew(text: str) -> bool:
    """True if the text contains at least one Hebrew letter."""
    return bool(text) and bool(HEBREW_RE.search(str(text)))

def rtl_markdown(text: str, style: str = ""):
    """
    Render markdown; if the text contains Hebrew, wrap it in a right-to-left block.
    The blank lines around the content keep Markdown (bold, lists, line breaks)
    rendering normally inside the HTML wrapper.
    """
    if not text:
        return
    if is_hebrew(text):
        st.markdown(
            f'<div dir="rtl" style="direction: rtl; text-align: right; {style}">\n\n{text}\n\n</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(text)

def tool_badge(tool_used: str) -> str:
    color, label = TOOL_BADGES.get(tool_used, ("#6c757d", tool_used))
    return (
        f'<span style="background:{color};color:white;padding:2px 10px;'
        f'border-radius:12px;font-size:0.8em;font-weight:600;">{label}</span>'
    )

def confidence_badge(score: int) -> str:
    if score >= 90:
        color, label = "#28a745", "High"
    elif score >= 70:
        color, label = "#ffc107", "Medium"
    elif score >= 50:
        color, label = "#fd7e14", "Low"
    else:
        color, label = "#dc3545", "Very Low"
    return (
        f'<span style="background:{color};color:white;padding:2px 10px;'
        f'border-radius:12px;font-size:0.8em;font-weight:600;">'
        f'{label} confidence ({score}%)</span>'
    )

def parse_agent_output(result: dict) -> dict:
    messages = result.get("messages", [])
    raw = messages[-1].content if messages else ""
    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(cleaned)
        data.setdefault("answer", "")
        data.setdefault("sql_query", "")
        data.setdefault("tool_used", "database")
        data.setdefault("confidence_score", 50)
        data.setdefault("output_format", "text")
        data.setdefault("table_data", [])
        data.setdefault("policy_sources", [])
        data.setdefault("data_coverage", {"checked": [], "missing": []})
        return strip_section_sign(data)
    except (json.JSONDecodeError, AttributeError):
        return {
            "answer": raw or "לא התקבלה תשובה מהסוכן.",
            "sql_query": "", "tool_used": "database",
            "confidence_score": 50, "output_format": "text", "table_data": [],
            "policy_sources": [], "data_coverage": {"checked": [], "missing": []}
        }

MAX_COMPLETENESS_RETRIES = 1  # one automatic correction round if the answer is incomplete

def _invoke_agent(messages: list) -> dict:
    """One bounded agent run. Raises on timeout / step-limit so the UI can show it instead of hanging."""
    return agent.invoke({"messages": messages}, config={"recursion_limit": AGENT_RECURSION_LIMIT})


def run_agent(query: str, status=None) -> dict:
    """
    Run the agent. If the question mentions a policy-defined category (High Risk, Tier 4, Prime...):
      1. inject ALL the policy sections for that category into the agent's input,
      2. validate the answer (every section cited, every checkable column in the SQL),
      3. retry once with a correction message if something is missing.
    The verdict is attached under data["_completeness"] and surfaced in the UI.
    `status` is an optional st.status container used to show progress and timings.
    """
    def say(msg):
        if status is not None:
            status.update(label=msg)
            status.write(f"{time.strftime('%H:%M:%S')} — {msg}")

    timings = {}
    t0 = time.time()
    categories = detect_policy_categories(query, policy_chunks)
    enforce_sql = wants_data(query)
    if categories:
        say(f"זוהתה קטגוריית מדיניות: {', '.join(c['label'] for c in categories)} — מזריק {sum(len(c['sections']) for c in categories)} סעיפים")

    agent_input = query + build_policy_context(categories, enforce_sql) if categories else query
    say("מריץ את הסוכן (ניסיון 1)...")
    t1 = time.time()
    result = _invoke_agent([("human", agent_input)])
    timings["attempt 1"] = f"{time.time() - t1:.1f}s · {len(result.get('messages', []))} messages"
    data = parse_agent_output(result)

    if not categories:
        timings["total"] = f"{time.time() - t0:.1f}s"
        st.session_state["last_timings"] = timings
        return data

    retries = 0
    report = check_completeness(data, categories, enforce_sql)
    while not report["ok"] and retries < MAX_COMPLETENESS_RETRIES:
        retries += 1
        say(f"בדיקת שלמות נכשלה (חסר: {report['missing_sections'] + report['missing_columns']}) — סבב תיקון {retries}...")
        t1 = time.time()
        result = _invoke_agent(result["messages"] + [("human", build_correction(report))])
        timings[f"correction {retries}"] = f"{time.time() - t1:.1f}s · {len(result.get('messages', []))} messages"
        data = parse_agent_output(result)
        report = check_completeness(data, categories, enforce_sql)

    # Deterministic clean-up of whatever the model still got wrong, then re-check the cleaned answer
    cleaned = sanitize_policy_answer(data, categories, report)
    if cleaned:
        say("ניקוי אוטומטי: " + "; ".join(cleaned))
        report = check_completeness(data, categories, enforce_sql)
    data["confidence_score"] = final_confidence(data, report)

    say("בדיקת שלמות עברה ✅" if report["ok"] else "בדיקת שלמות עדיין נכשלת — מציג תשובה חלקית ⚠️")
    data["_completeness"] = {
        "categories": [c["label"] for c in categories],
        "retries": retries,
        "cleaned": cleaned,
        **report,
    }
    timings["total"] = f"{time.time() - t0:.1f}s"
    st.session_state["last_timings"] = timings
    return data

CONFIDENCE_THRESHOLD = 70  # מתחת לסף זה → לא מציגים תשובה, רק בקשת הבהרה

def section_label(source: str, hebrew: bool) -> str:
    """'2.1 Credit Score Tiers' → 'סעיף 2.1 Credit Score Tiers' / 'Section 2.1 Credit Score Tiers'."""
    s = re.sub(r"^\s*(§|section|סעיף)?\s*", "", str(source), flags=re.IGNORECASE)
    return ("סעיף " if hebrew else "Section ") + s

def strip_section_sign(data: dict) -> dict:
    """Safety net: the model must not emit '§' — replace it with the word in the answer's language."""
    word = "סעיף " if is_hebrew(data.get("answer", "")) else "Section "
    data["answer"] = re.sub(r"§\s*", word, data.get("answer", "") or "")
    data["policy_sources"] = [re.sub(r"^\s*§\s*", "", str(s)) for s in data.get("policy_sources", []) or []]
    cov = data.get("data_coverage")
    if isinstance(cov, dict):
        for k in ("checked", "missing"):
            cov[k] = [re.sub(r"§\s*", word, str(x)) for x in cov.get(k, []) or []]
    return data

COMBINED_ROW_RE = re.compile(r"לפחות קריטריון|at least one", re.IGNORECASE)
TOTAL_ROW_RE    = re.compile(r"סך|סה\"כ|total", re.IGNORECASE)

def order_policy_rows(rows: list) -> list:
    """Enforce the fixed order: per-section rows → 'at least one criterion' → total (stable sort)."""
    def rank(r):
        label = str(next(iter(r.values()), "")) if isinstance(r, dict) and r else ""
        if TOTAL_ROW_RE.search(label):
            return 2
        if COMBINED_ROW_RE.search(label):
            return 1
        return 0
    return sorted(rows, key=rank)

def markdown_table(rows: list) -> str:
    cols = list(rows[0].keys())
    esc = lambda v: str(v).replace("|", "/")
    lines = ["| " + " | ".join(esc(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(esc(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(lines)

def render_policy_answer(answer: str, table_data: list, missing: list):
    """Fixed layout for policy-category answers: definition → 'checked in the data' table → limitations."""
    hebrew = is_hebrew(answer) or any(is_hebrew(str(v)) for r in table_data if isinstance(r, dict) for v in r.values())
    rtl_markdown(answer)

    rows = order_policy_rows([r for r in table_data if isinstance(r, dict) and r])
    if rows:
        header = "**מה שנבדק בנתונים:**" if hebrew else "**What was checked in the data:**"
        if hebrew:
            rtl_markdown(header + "\n\n" + markdown_table(rows))
        else:
            st.markdown(header)
            try:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            except Exception:
                st.write(rows)

    if missing:
        title = "**⚠️ קריטריונים שלא ניתן לבדוק בנתונים:**" if hebrew else "**⚠️ Criteria that cannot be verified in the data:**"
        rtl_markdown(title + "\n\n" + "\n".join(f"- {m}" for m in missing))

def render_completeness(comp):
    """Show the verdict of the Python-level policy completeness check (if the question triggered it)."""
    if not comp:
        return
    cats = ", ".join(comp.get("categories", []))
    secs = "Sections " + ", ".join(comp.get("expected_sections", []))
    if comp.get("cleaned"):
        st.caption("🧹 Auto-cleaned: " + "; ".join(comp["cleaned"]) + ".")
    if not comp.get("ok"):
        parts = []
        if comp.get("missing_sections"):
            parts.append("policy sections not covered: " + ", ".join(comp["missing_sections"]))
        if comp.get("missing_columns"):
            parts.append("criteria not checked in SQL: " + ", ".join(comp["missing_columns"]))
        if comp.get("missing_in_sources") and not comp.get("missing_sections"):
            parts.append("sections missing from the sources list: " + ", ".join(comp["missing_in_sources"]))
        if comp.get("unexpected_sections"):
            parts.append("limit-only sections wrongly included: " + ", ".join(comp["unexpected_sections"]))
        if comp.get("multi_statement"):
            parts.append("SQL contains more than one statement")
        if comp.get("forbidden_columns_used"):
            parts.append("SQL uses limit columns: " + ", ".join(comp["forbidden_columns_used"]))
        st.warning(
            f"⚠️ **Completeness check failed — treat this answer as PARTIAL.** "
            f"The policy defines *{cats}* in {secs}. " + "; ".join(parts) + "."
        )
    elif comp.get("retries"):
        st.caption(f"🔁 Answer regenerated after an automatic completeness check — now covers {secs} for *{cats}*.")
    else:
        st.caption(f"✅ Completeness check passed — covers {secs} for *{cats}*.")

def render_response(data: dict):
    fmt          = data.get("output_format", "text")
    answer       = data.get("answer", "")
    sql_query    = data.get("sql_query", "")
    confidence   = data.get("confidence_score", 0)
    table_data   = data.get("table_data", [])
    tool_used    = data.get("tool_used", "database")
    sources      = data.get("policy_sources", []) or []
    coverage     = data.get("data_coverage") or {}
    checked      = coverage.get("checked", []) or [] if isinstance(coverage, dict) else []
    missing      = coverage.get("missing", []) or [] if isinstance(coverage, dict) else []
    comp         = data.get("_completeness")

    # ── PYTHON-LEVEL ENFORCEMENT: confidence < threshold ──────────────────
    # גם אם הסוכן "שכח" לבקש הבהרה בפרומפט — הקוד מאכף
    if 0 < confidence < CONFIDENCE_THRESHOLD:
        body = answer if answer else "The question is too ambiguous to answer reliably. Please rephrase or provide more detail."
        if is_hebrew(body):
            # Custom warning box so the Hebrew clarification request is rendered RTL
            rtl_markdown(
                f"⚠️ **רמת הביטחון נמוכה מדי כדי לענות ({confidence}%).**\n\n{body}",
                style="background-color: #fff3cd; color: #664d03; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px;",
            )
        else:
            st.warning(f"⚠️ **Confidence too low to answer ({confidence}%).**\n\n{body}")
        render_completeness(comp)
        # Confidence badge בלבד — ללא תוצאה, ללא SQL
        st.markdown(confidence_badge(confidence), unsafe_allow_html=True)
        st.progress(confidence / 100)
        return  # עוצרים כאן — לא מציגים תשובה

    # ── confidence = 0: שאלה מחוץ לתחום / נתון חסר ─────────────────────
    if confidence == 0:
        rtl_markdown(answer)
        return

    # ── confidence ≥ threshold: מציגים תשובה מלאה ───────────────────────

    is_policy_answer = bool(comp) and bool(table_data)

    if is_policy_answer:
        # Fixed layout: 1) policy definition  2) "checked in the data" table  3) data limitations
        render_policy_answer(answer, table_data, missing)
    else:
        # תשובה מילולית
        if answer and ("text" in fmt or "table" not in fmt):
            rtl_markdown(answer)

        # טבלה
        if "table" in fmt:
            if table_data:
                try:
                    st.dataframe(pd.DataFrame(table_data), use_container_width=True)
                except Exception:
                    st.write(table_data)
            elif answer:
                rtl_markdown(answer)

    render_completeness(comp)

    # Badges row
    badges_html = tool_badge(tool_used) + "&nbsp;&nbsp;" + confidence_badge(confidence)
    st.markdown(badges_html, unsafe_allow_html=True)
    st.progress(confidence / 100)

    # SQL expander
    if "sql" in fmt and sql_query:
        with st.expander("🔍 SQL Query"):
            st.code(sql_query, language="sql")

    # Policy sources expander (RAG transparency)
    if sources:
        with st.expander(f"📜 Policy sources ({len(sources)})"):
            for s in sources:
                rtl_markdown(f"- {section_label(s, hebrew=is_hebrew(answer))}")

    # Data coverage expander — what was verified in the DB vs. what the data cannot answer
    # (for policy answers the "missing" part is already shown inline under the table)
    show_missing = [] if is_policy_answer else missing
    if checked or show_missing:
        with st.expander(f"📊 Data coverage — checked: {len(checked)} · missing: {len(missing)}", expanded=bool(show_missing)):
            if checked:
                st.markdown("**✔ Checked against the database:**")
                rtl_markdown("\n".join(f"- {c}" for c in checked))
            if show_missing:
                st.markdown("**✘ Not available in the database (partial answer):**")
                rtl_markdown("\n".join(f"- {m}" for m in show_missing))

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("""
<div style="direction: rtl; text-align: right; background-color: #e8f4fd; border-left: 4px solid #1f77b4; border-radius: 6px; padding: 16px; margin-bottom: 16px;">
<strong>💡 דוגמאות לשאלות שתוכלו לשאול:</strong>
<ul style="margin-top: 8px; margin-bottom: 0;">
<li>מה אחוז ה-Default בתיק? הצג טבלה ושאילתה</li>
<li>מה ממוצע ה-Credit Score של לקוחות ב-Default לעומת תקינים?</li>
<li>מה ההחזר החודשי על הלוואה של 50,000 ₪ בריבית 6% ל-5 שנים?</li>
<li>לקוח מרוויח 8,000 ₪ בחודש ומשלם 2,400 ₪. מה יחס ה-DTI שלו?</li>
<li>מה שיעור הכשל הממוצע לפי מצב משפחתי? תציג שאילתה</li>
<li>אם אשקיע 100,000 ₪ בריבית 4% ל-10 שנים, כמה אקבל?</li>
<li>📜 מה המדיניות אומרת לגבי יחס DTI מקסימלי?</li>
<li>📜 כמה לקוחות בתיק נחשבים High Risk לפי המדיניות? הצג שאילתה</li>
</ul>
</div>
""", unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and isinstance(message.get("data"), dict):
            render_response(message["data"])
        else:
            rtl_markdown(message["content"])

user_query = st.chat_input("שאלו שאלה על תיק האשראי או בקשו חישוב פיננסי...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        rtl_markdown(user_query)

    with st.chat_message("assistant"):
        with st.status("מנתח את הנתונים...", expanded=False) as status:
            try:
                data = run_agent(user_query, status=status)
                status.update(label="הניתוח הושלם", state="complete")
            except Exception as e:
                status.update(label=f"שגיאה: {type(e).__name__}", state="error")
                data = {
                    "answer": f"אופס, אירעה שגיאה ({type(e).__name__}): {e}",
                    "sql_query": "", "tool_used": "none",
                    "confidence_score": 0, "output_format": "text", "table_data": []
                }

        render_response(data)
        st.session_state.messages.append({
            "role": "assistant",
            "content": data.get("answer", ""),
            "data": data
        })
