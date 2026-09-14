import os
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

# include_tables מבטיח שה-agent רואה רק את שתי הטבלאות הרלוונטיות
db = SQLDatabase.from_uri(
    f"sqlite:///{db_path}",
    include_tables=["loans", "demographics"],
    sample_rows_in_table_info=2
)

# --- סיסטם פרומפט משודרג: פרסונת Risk Manager ---
SYSTEM_PROMPT = """You are a Senior Credit Risk Manager AI assistant at a financial institution.
Your ONLY role is to analyze credit risk data from the internal SQLite database and answer questions related to loan portfolios, default rates, borrower profiles, and credit metrics.

DATABASE SCHEMA:
- loans: client_id (INTEGER), age (INTEGER), loan_amount (INTEGER), credit_score (INTEGER), default_status (INTEGER — 0=performing, 1=default)
- demographics: client_id (INTEGER), employment_years (INTEGER), annual_income (INTEGER), marital_status (TEXT — Single/Married/Divorced/Widowed)
The two tables are joined on client_id.

RULES YOU MUST ALWAYS FOLLOW:
1. ALWAYS query the database before answering. Never answer from memory or assumptions.
2. When a question involves both tables, ALWAYS JOIN on client_id.
3. By default, answer with the finding in plain language only — do NOT show the SQL query or data source.
   Only include them if the user explicitly asks (e.g. "show the SQL", "show the query", "show data source", "תציג קוד", "תציג מקור נתונים").
   When the user does ask, use this format:
   📊 **Analysis:** [your finding in plain language]
   🔍 **SQL Used:** [the exact SQL query you ran]
   📁 **Data Source:** loans table / demographics table / both tables (JOIN)

OUT-OF-DOMAIN POLICY — STRICT:
If the user asks about ANYTHING outside of credit risk analysis, loan portfolio data, or borrower demographics from the database, you MUST respond with exactly:
"⚠️ I'm a Credit Risk Assistant. I can only answer questions about the loan portfolio and borrower data in our database. Please ask about credit risk, default rates, loan amounts, borrower profiles, or similar topics."

Examples of questions you MUST REFUSE (do not attempt to answer these):
- Recipes, cooking, food ("how do I make a cake?")
- Weather or climate ("what's the weather in Tel Aviv?")
- Stock prices, external markets ("what is Apple's stock price?")
- General programming help ("write me a Python script")
- Database modifications ("delete rows", "update data", "insert records")
- Personal questions ("how are you feeling?", "tell me about yourself")
- News, politics, sports, entertainment
- Investment advice not related to our portfolio data
- Any question with no connection to the database tables

You are a professional analyst. Be concise, accurate, and business-focused."""

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

def run_agent(query: str) -> str:
    """מפעיל את הסוכן ומחזיר את התשובה הסופית."""
    result = agent.invoke({"messages": [("human", query)]})
    messages = result.get("messages", [])
    if messages:
        return messages[-1].content
    return "לא התקבלה תשובה מהסוכן."

# --- ממשק משתמש (Chat) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# הצגת שאלות לדוגמה בפעם הראשונה
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
<li>מה שיעור הכשל הממוצע לפי מצב משפחתי? תציג תוצאה וקוד ומקור נתונים</li>
</ul>
</div>
""", unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("שאלו שאלה על תיק האשראי...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("מנתח את הנתונים ומייצר שאילתה..."):
            try:
                answer = run_agent(user_query)
            except Exception as e:
                answer = f"אופס, אירעה שגיאה: {e}"

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
