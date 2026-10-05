import os
import re
import json
import math
import pandas as pd
import streamlit as st
from langchain.agents import create_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from build_vector_db import load_or_build_index, POLICY_PATH

st.set_page_config(
    page_title="AI Credit Risk Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Credit Risk Assistant")
st.markdown("Ask questions about the loan portfolio, request financial calculations, or consult the credit underwriting policy — the agent routes each question to the right tool (SQL, calculator, or policy retriever).")

# --- API Key ---
try:
    openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
except Exception:
    openai_api_key = None

if not openai_api_key:
    st.error("שגיאה: מפתח ה-OPENAI_API_KEY לא נמצא.")
    st.stop()

# --- Database ---
current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_dir, "credit_risk.db")

if not os.path.exists(db_path):
    st.error(f"שגיאה: קובץ מסד הנתונים '{db_path}' לא נמצא.")
    st.stop()

db = SQLDatabase.from_uri(
    f"sqlite:///{db_path}",
    include_tables=["loans", "demographics"],
    sample_rows_in_table_info=2
)

# --- Policy Vector Store (RAG) ---
if not os.path.exists(POLICY_PATH):
    st.error(f"שגיאה: מסמך המדיניות '{POLICY_PATH}' לא נמצא.")
    st.stop()

@st.cache_resource(show_spinner="טוען את מאגר המדיניות (Vector DB)...")
def get_policy_retriever(_api_key: str):
    """Load the FAISS index once per server process (built on first run if missing)."""
    store = load_or_build_index(_api_key)
    return store.as_retriever(search_kwargs={"k": 4})

policy_retriever = get_policy_retriever(openai_api_key)

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
    """
    docs = policy_retriever.invoke(query)
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

3. POLICY RETRIEVER TOOL (search_credit_policy) → use for questions about RULES, not data:
   - Thresholds, tiers, limits, eligibility criteria, definitions, procedures
   - "What credit score counts as high risk?", "What is the maximum DTI?", "What are the loan limits for Tier 3?"
   - "How does the policy treat divorced applicants?", "When is a loan written off?"
   Always cite the section numbers you relied on (e.g. "per policy §2.1").
   NEVER answer a policy question from memory — if you did not retrieve it, you do not know it.

4. HYBRID → use when the question combines two or more tool types:
   a) DB + Calculator: "Average monthly payment for the top 10 loans at 5% for 10 years?"
      → query the DB for the loan amounts, then call calculate_monthly_payment.
   b) POLICY + DB (the most valuable case): "How many clients fall into the policy's High Risk tier?"
      → FIRST call search_credit_policy to get the official definition (Tier 4 = credit_score < 550),
        THEN write the SQL using that exact threshold, and cite the section in the answer.
      Other examples: "Which clients violate the minimum income rule?", "What share of the portfolio
      is below the minimum employment duration?", "Is our default rate within the policy benchmark?"
   c) POLICY + Calculator: "Client earns 8,000/month and pays 3,600 — is that allowed?"
      → compute DTI, then retrieve the DTI thresholds and apply them.

POLICY LOOKUP FOR VAGUE TERMS:
When the user uses a vague qualifier (risky, high risk, young, experienced, large loan, good score, etc.),
FIRST call search_credit_policy to check whether the policy defines it.
- If the policy defines it → use that definition, state it explicitly in the answer with its section,
  set confidence_score 70-89 (interpretation based on policy), and proceed.
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
  "policy_sources": [ "2.1 Credit Score Tiers", ... ]
}

policy_sources: list the policy section labels (as returned by search_credit_policy in the "section" field)
that you actually used in the answer. Empty list [] if the policy tool was not used.

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
{"answer": "⚠️ I'm a Credit Risk Assistant. I can only answer questions about the loan portfolio data, the credit underwriting policy, or perform financial calculations.", "sql_query": "", "tool_used": "none", "confidence_score": 0, "output_format": "text", "table_data": [], "policy_sources": []}"""

# ─────────────────────────────────────────────
# LLM, TOOLS & AGENT
# ─────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_tools = toolkit.get_tools()
calculator_tools = [calculate_monthly_payment, calculate_compound_interest, calculate_debt_to_income]
policy_tools = [search_credit_policy]

agent = create_agent(
    model=llm,
    tools=sql_tools + calculator_tools + policy_tools,
    system_prompt=SYSTEM_PROMPT
)

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

def run_agent(query: str) -> dict:
    result = agent.invoke({"messages": [("human", query)]})
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
        return data
    except (json.JSONDecodeError, AttributeError):
        return {
            "answer": raw or "לא התקבלה תשובה מהסוכן.",
            "sql_query": "", "tool_used": "database",
            "confidence_score": 50, "output_format": "text", "table_data": [],
            "policy_sources": []
        }

CONFIDENCE_THRESHOLD = 70  # מתחת לסף זה → לא מציגים תשובה, רק בקשת הבהרה

def render_response(data: dict):
    fmt          = data.get("output_format", "text")
    answer       = data.get("answer", "")
    sql_query    = data.get("sql_query", "")
    confidence   = data.get("confidence_score", 0)
    table_data   = data.get("table_data", [])
    tool_used    = data.get("tool_used", "database")
    sources      = data.get("policy_sources", []) or []

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
        # Confidence badge בלבד — ללא תוצאה, ללא SQL
        st.markdown(confidence_badge(confidence), unsafe_allow_html=True)
        st.progress(confidence / 100)
        return  # עוצרים כאן — לא מציגים תשובה

    # ── confidence = 0: שאלה מחוץ לתחום / נתון חסר ─────────────────────
    if confidence == 0:
        rtl_markdown(answer)
        return

    # ── confidence ≥ threshold: מציגים תשובה מלאה ───────────────────────

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
                st.markdown(f"- §{s}")

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
        with st.spinner("מנתח את הנתונים..."):
            try:
                data = run_agent(user_query)
            except Exception as e:
                data = {
                    "answer": f"אופס, אירעה שגיאה: {e}",
                    "sql_query": "", "tool_used": "none",
                    "confidence_score": 0, "output_format": "text", "table_data": []
                }

        render_response(data)
        st.session_state.messages.append({
            "role": "assistant",
            "content": data.get("answer", ""),
            "data": data
        })
