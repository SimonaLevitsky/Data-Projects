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
st.markdown("שאלו שאלות על תיק האשראי בשפה טבעית, והסוכן יתרגם אותן לשאילתות SQL ויציג את התשובות.")

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

# --- סיסטם פרומפט ---
# שמות העמודות מדויקים לפי הסכמה האמיתית של ה-DB
SYSTEM_PROMPT = """You are a Credit Risk SQL analyst working with a SQLite database.
You have access to two tables:
- loans: client_id, age, loan_amount, credit_score, default_status
- demographics: client_id, employment_years, annual_income, marital_status

The tables are linked by client_id. When a question involves both tables,
ALWAYS JOIN them on client_id. Always query the database before answering.
Never say you don't have enough information — query the database first."""

# --- אתחול מודל השפה ---
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=openai_api_key
)

# --- יצירת הסוכן ---
# create_agent מה-langchain החדש (1.4+) עם SQLDatabaseToolkit
# מחזיר LangGraph agent שמשתמש ב-function calling — אמין ויציב
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
    # LangGraph מחזיר רשימת הודעות — ההודעה האחרונה היא תשובת ה-AI
    messages = result.get("messages", [])
    if messages:
        return messages[-1].content
    return "לא התקבלה תשובה מהסוכן."

# --- ממשק משתמש (Chat) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("לדוגמה: מה ההכנסה הממוצעת של לקוחות עם הלוואה ב-Default?")

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
