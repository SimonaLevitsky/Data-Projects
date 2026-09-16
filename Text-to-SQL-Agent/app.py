import os
import json
import streamlit as st
from langchain.agents import create_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI

# הגדרת עיצוב הדף ב-Streamlit
st.set_page_config(
    page_title="AI Credit Risk Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Credit Risk Assistant")
st.markdown("Ask questions about the loan portfolio in natural language — the agent translates them into SQL queries and returns the results.")

# --- חיבור מאובטח למפתח ה-API ---
try:
    openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
except Exception:
    openai_api_key = None

if not openai_api_key:
    st.error("שגיאה: מפתח ה-OPENAI_API_KEY לא נמצא. יש להגדיר אותו ב-Secrets של Streamlit.")
    st.stop()

# --- חיבור לדאטהבייס ---
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

# --- סיסטם פרומפט: JSON Structured Output + Confidence Score ---
SYSTEM_PROMPT = """You are a Senior Credit Risk Manager AI assistant at a financial institution.
Your ONLY role is to analyze credit risk data from the internal SQLite database and answer questions related to loan portfolios, default rates, borrower profiles, and credit metrics.

DATABASE SCHEMA — EXACT, COMPLETE:
- loans: client_id (INTEGER), age (INTEGER), loan_amount (INTEGER), credit_score (INTEGER), default_status (INTEGER — 0=performing, 1=default)
- demographics: client_id (INTEGER), employment_years (INTEGER), annual_income (INTEGER), marital_status (TEXT — Single/Married/Divorced/Widowed)
The two tables are joined on client_id. There are NO other tables, columns, or data fields.

LANGUAGE RULE:
Detect the language of the user's question and respond in that exact language.
- If the question is in Hebrew → answer in Hebrew.
- If the question is in English → answer in English.
This applies to the "answer" field only. SQL stays in SQL syntax regardless.

OUTPUT FORMAT — STRICT:
You MUST always respond with a valid JSON object and nothing else. No prose before or after. No markdown fences.
The JSON must contain exactly these three fields:

{
  "answer": "Your natural language finding, in the same language as the user's question.",
  "sql_query": "The exact SQL you executed against the database.",
  "confidence_score": <integer 0-100>
}

CONFIDENCE SCORE RULES:
- 90-100: Query ran cleanly, schema matched perfectly, result is unambiguous.
- 70-89: Minor interpretation required, result is likely correct.
- 50-69: Question was ambiguous or data is sparse — answer may be incomplete.
- 0-49: Could not find a reliable answer. Set answer to a clarification request.

If confidence_score < 70, set "answer" to a clarification question — do NOT guess or approximate.

HALLUCINATION PREVENTION — CRITICAL:
Before answering, verify that EVERY concept in the user's question maps to an actual column in the schema above.
- "mortgage" / "משכנתא" → does NOT exist in the database. Do not proxy loan_amount as a substitute.
- "interest rate" / "ריבית" → does NOT exist.
- "loan type" / "סוג הלוואה" → does NOT exist.
- "balance" / "יתרה" → does NOT exist.
If the user asks about a concept that has NO matching column, return:
{"answer": "The data you asked about ([concept]) does not exist in our database. Available fields are: age, loan_amount, credit_score, default_status, employment_years, annual_income, marital_status.", "sql_query": "", "confidence_score": 0}
Translate the answer field to the user's language. NEVER run a query and present results as if they answer a question they do not.

RULES:
1. ALWAYS query the database before answering. Never answer from memory.
2. When a question involves both tables, ALWAYS JOIN on client_id.
3. For out-of-domain questions (recipes, weather, stocks, coding help, personal questions, database modifications), return:
   {"answer": "⚠️ I'm a Credit Risk Assistant. I can only answer questions about the loan portfolio and borrower data in our database.", "sql_query": "", "confidence_score": 0}"""

# --- אתחול מודל השפה ---
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=openai_api_key
)

# --- יצירת הסוכן ---
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
)

# --- Confidence badge helper ---
def confidence_badge(score: int) -> str:
    if score >= 90:
        color, label = "#28a745", "High"
    elif score >= 70:
        color, label = "#ffc107", "Medium"
    elif score >= 50:
        color, label = "#fd7e14", "Low"
    else:
        color, label = "#dc3545", "Very Low"
    return f'<span style="background:{color};color:white;padding:2px 10px;border-radius:12px;font-size:0.8em;font-weight:600;">{label} confidence ({score}%)</span>'

def run_agent(query: str) -> dict:
    """מפעיל את הסוכן ומחזיר dict מפורסר."""
    result = agent.invoke({"messages": [("human", query)]})
    messages = result.get("messages", [])
    raw = messages[-1].content if messages else ""

    # נסה לפרסר JSON
    try:
        # נקה markdown fences אם יש
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(cleaned)
    except (json.JSONDecodeError, AttributeError):
        # Fallback: החזר כתשובה רגילה עם confidence נמוך
        return {
            "answer": raw or "לא התקבלה תשובה מהסוכן.",
            "sql_query": "",
            "confidence_score": 50
        }

def render_response(data: dict, show_sql: bool = False):
    """מציג את התשובה המובנית ב-Streamlit."""
    answer = data.get("answer", "")
    sql_query = data.get("sql_query", "")
    confidence = data.get("confidence_score", 0)

    # תשובה ראשית
    st.markdown(answer)

    # Confidence indicator
    if confidence > 0:
        st.markdown(
            confidence_badge(confidence),
            unsafe_allow_html=True
        )
        st.progress(confidence / 100)

    # SQL — רק אם המשתמש ביקש
    if show_sql and sql_query:
        with st.expander("🔍 SQL Query"):
            st.code(sql_query, language="sql")

# --- בדיקה אם המשתמש ביקש לראות SQL ---
SQL_KEYWORDS = ["show sql", "show the sql", "show query", "sql used", "תציג קוד", "תציג שאילתה", "קוד sql", "מקור נתונים"]

def user_wants_sql(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in SQL_KEYWORDS)

# --- ממשק משתמש (Chat) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# שאלות לדוגמה בפעם הראשונה
if not st.session_state.messages:
    st.markdown("""
<div style="direction: rtl; text-align: right; background-color: #e8f4fd; border-left: 4px solid #1f77b4; border-radius: 6px; padding: 16px; margin-bottom: 16px;">
<strong>💡 דוגמאות לשאלות שתוכלו לשאול:</strong>
<ul style="margin-top: 8px; margin-bottom: 0;">
<li>מה אחוז ה-Default בתיק?</li>
<li>מה ההכנסה הממוצעת של לקוחות נשואים?</li>
<li>מה ממוצע ה-Credit Score של לקוחות ב-Default לעומת לקוחות תקינים?</li>
<li>מהם 5 הלקוחות עם הלוואות הגדולות ביותר?</li>
<li>מה הקשר בין שנות ותק בעבודה לבין Default?</li>
<li>מה שיעור הכשל הממוצע לפי מצב משפחתי? תציג שאילתה</li>
</ul>
</div>
""", unsafe_allow_html=True)

# הצגת היסטוריה
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and isinstance(message.get("data"), dict):
            render_response(message["data"], show_sql=message.get("show_sql", False))
        else:
            st.markdown(message["content"])

# קלט משתמש
user_query = st.chat_input("שאלו שאלה על תיק האשראי...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    show_sql = user_wants_sql(user_query)

    with st.chat_message("assistant"):
        with st.spinner("מנתח את הנתונים ומייצר שאילתה..."):
            try:
                data = run_agent(user_query)
            except Exception as e:
                data = {
                    "answer": f"אופס, אירעה שגיאה: {e}",
                    "sql_query": "",
                    "confidence_score": 0
                }

        render_response(data, show_sql=show_sql)
        st.session_state.messages.append({
            "role": "assistant",
            "content": data.get("answer", ""),
            "data": data,
            "show_sql": show_sql
        })
