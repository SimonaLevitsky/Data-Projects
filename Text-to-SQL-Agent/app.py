import os
import json
import pandas as pd
import streamlit as st
from langchain.agents import create_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI

st.set_page_config(
    page_title="AI Credit Risk Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Credit Risk Assistant")
st.markdown("Ask questions about the loan portfolio in natural language — the agent translates them into SQL queries and returns the results.")

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

# --- System Prompt ---
SYSTEM_PROMPT = """You are a Senior Credit Risk Manager AI assistant at a financial institution.
Your ONLY role is to analyze credit risk data from the internal SQLite database.

DATABASE SCHEMA — EXACT, COMPLETE:
- loans: client_id (INTEGER), age (INTEGER), loan_amount (INTEGER), credit_score (INTEGER), default_status (INTEGER — 0=performing, 1=default)
- demographics: client_id (INTEGER), employment_years (INTEGER), annual_income (INTEGER), marital_status (TEXT — Single/Married/Divorced/Widowed)
The two tables are joined on client_id. There are NO other tables, columns, or data fields.

LANGUAGE RULE:
Detect the language of the user's question and respond in that exact language.
- Hebrew question → answer in Hebrew.
- English question → answer in English.
This applies to the "answer" field only. SQL stays in SQL syntax regardless.

OUTPUT FORMAT — STRICT JSON, NO PROSE OUTSIDE IT:
Always return a valid JSON object with exactly these fields:

{
  "answer": "Natural language finding in the user's language. Empty string if output_format is 'table' only.",
  "sql_query": "The exact SQL executed. Empty string if no query was run.",
  "confidence_score": <integer 0-100>,
  "output_format": "<one of: text | table | sql | text+sql | table+sql | table+text+sql>",
  "table_data": [ {"column": value, ...}, ... ]
}

OUTPUT FORMAT SELECTION RULES — read the user's request carefully:
- User asks for a table / טבלה → set output_format to "table", populate table_data as a list of row objects
- User asks for a verbal / text answer only → set output_format to "text", table_data = []
- User asks to show the SQL / query / שאילתה / קוד → include "sql" in output_format
- User asks for a combination (e.g. "table and query") → combine with "+": "table+sql", "table+text+sql", etc.
- Default (no explicit format requested) → output_format = "text", table_data = []

table_data rules:
- Must be a JSON array of objects, one object per result row
- Keys are column names (English), values are the raw query results
- Only populate when output_format contains "table"
- For aggregation results (e.g. AVG by group), each group is one object

CONFIDENCE SCORE RULES:
- 90-100: Clean query, schema matched perfectly, result unambiguous. No interpretation needed whatsoever.
- 70-89: Minor interpretation required, result likely correct. MUST note the assumption made.
- 50-69: Question was ambiguous or sparse data. Do NOT answer — ask for clarification instead.
- 0-49: Cannot reliably answer — ask for clarification or explain missing data.

AMBIGUITY DETECTION — MANDATORY CHECK BEFORE EVERY ANSWER:
Before forming a query, scan the user's question for ANY ambiguous qualifier. If found, you MUST set confidence_score ≤ 50 and return a clarification question in "answer" instead of a result. NEVER assume a threshold or definition on the user's behalf.

Ambiguous qualifiers that ALWAYS require clarification:
- "מסוכן" / "risky" / "high risk" → Ask: what is the risk criterion? (low credit score? default status? both? a formula?)
- "בעייתי" / "problematic" → Ask: what defines problematic? (in default? credit score below X? other?)
- "קשר" / "relationship" / "connection" → Ask: what type of analysis? (average comparison by group? correlation coefficient? distribution breakdown?)
- "גדול" / "גדולה" / "large" / "big" / "high loan" → Ask: what is the minimum threshold?
- "קטן" / "קטנה" / "small" / "low loan" → Ask: what is the maximum threshold?
- "צעיר" / "צעירה" / "young" → Ask: what is the maximum age?
- "מבוגר" / "older" / "senior client" → Ask: what is the minimum age?
- "ותיק" / "experienced" / "long tenure" → Ask: minimum employment_years threshold?
- "הרבה" / "many" / "a lot" / "רב" → Ask: what quantity defines "many"?
- "מעט" / "few" / "little" → Ask: what quantity defines "few"?
- "טוב" / "good" / "strong" / "healthy" (re: credit score, income) → Ask: what value defines "good"?
- "גבוה" / "high" (without a number) → Ask: what threshold defines "high"?
- "נמוך" / "low" (without a number) → Ask: what threshold defines "low"?

When asking for clarification, the answer field must be a direct clarification question — not a guess with a caveat. Format:
"❓ [Question about the ambiguous term]. Please clarify so I can run the right query."

HALLUCINATION PREVENTION — CRITICAL:
Before answering, verify every concept maps to an actual column.
- "mortgage" / "משכנתא" → NOT in database
- "interest rate" / "ריבית" → NOT in database
- "loan type" / "סוג הלוואה" → NOT in database
- "balance" / "יתרה" → NOT in database
If a concept has no matching column, return confidence_score=0, sql_query="", table_data=[], and explain in the answer field (in the user's language) what IS available.

OUT-OF-DOMAIN: For recipes, weather, stocks, coding help, personal questions, or DB modifications:
{"answer": "⚠️ I'm a Credit Risk Assistant. I can only answer questions about the loan portfolio data.", "sql_query": "", "confidence_score": 0, "output_format": "text", "table_data": []}"""

# --- LLM & Agent ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)

# --- Confidence badge ---
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

# --- Agent runner ---
def run_agent(query: str) -> dict:
    result = agent.invoke({"messages": [("human", query)]})
    messages = result.get("messages", [])
    raw = messages[-1].content if messages else ""
    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(cleaned)
        # ודא שכל השדות קיימים
        data.setdefault("answer", "")
        data.setdefault("sql_query", "")
        data.setdefault("confidence_score", 50)
        data.setdefault("output_format", "text")
        data.setdefault("table_data", [])
        return data
    except (json.JSONDecodeError, AttributeError):
        return {
            "answer": raw or "לא התקבלה תשובה מהסוכן.",
            "sql_query": "",
            "confidence_score": 50,
            "output_format": "text",
            "table_data": []
        }

# --- Render response ---
def render_response(data: dict):
    fmt = data.get("output_format", "text")
    answer = data.get("answer", "")
    sql_query = data.get("sql_query", "")
    confidence = data.get("confidence_score", 0)
    table_data = data.get("table_data", [])

    # טקסט מילולי
    if answer and "table" not in fmt or (fmt == "text") or ("text" in fmt):
        if answer:
            st.markdown(answer)

    # טבלה
    if "table" in fmt and table_data:
        try:
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)
        except Exception:
            st.write(table_data)
    elif "table" in fmt and not table_data and answer:
        # אם ביקשו טבלה אבל אין נתונים — הצג טקסט
        st.markdown(answer)

    # Confidence
    if confidence > 0:
        st.markdown(confidence_badge(confidence), unsafe_allow_html=True)
        st.progress(confidence / 100)

    # SQL
    if "sql" in fmt and sql_query:
        with st.expander("🔍 SQL Query"):
            st.code(sql_query, language="sql")

# --- UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("""
<div style="direction: rtl; text-align: right; background-color: #e8f4fd; border-left: 4px solid #1f77b4; border-radius: 6px; padding: 16px; margin-bottom: 16px;">
<strong>💡 דוגמאות לשאלות שתוכלו לשאול:</strong>
<ul style="margin-top: 8px; margin-bottom: 0;">
<li>מה אחוז ה-Default בתיק?</li>
<li>מה ההכנסה הממוצעת של לקוחות נשואים?</li>
<li>מהם 5 הלקוחות עם הלוואות הגדולות ביותר? הצג טבלה</li>
<li>מה ממוצע ה-Credit Score של לקוחות ב-Default לעומת תקינים? הצג טבלה ושאילתה</li>
<li>מה הקשר בין שנות ותק בעבודה לבין Default?</li>
<li>מה שיעור הכשל הממוצע לפי מצב משפחתי? תציג שאילתה</li>
</ul>
</div>
""", unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and isinstance(message.get("data"), dict):
            render_response(message["data"])
        else:
            st.markdown(message["content"])

user_query = st.chat_input("שאלו שאלה על תיק האשראי...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("מנתח את הנתונים ומייצר שאילתה..."):
            try:
                data = run_agent(user_query)
            except Exception as e:
                data = {
                    "answer": f"אופס, אירעה שגיאה: {e}",
                    "sql_query": "",
                    "confidence_score": 0,
                    "output_format": "text",
                    "table_data": []
                }

        render_response(data)
        st.session_state.messages.append({
            "role": "assistant",
            "content": data.get("answer", ""),
            "data": data
        })
