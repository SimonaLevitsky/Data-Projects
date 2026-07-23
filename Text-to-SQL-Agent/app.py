import os
import sqlite3
import pandas as pd
import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent

# הגדרת עיצוב הדף ב-Streamlit
st.set_page_config(
    page_title="AI Credit Risk Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Credit Risk Assistant")
st.markdown("שאלו שאלות על תיק האשראי בשפה טבעית, והסוכן יתרגם אותן לשאילתות SQL ויציג את התשובות.")

# --- הגדרת חיבור למפתח ה-API בצורה מאובטחת ---
try:
    # ניסיון לקרוא את המפתח מה-Secrets של Streamlit Cloud או ממשתני סביבה
    openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
except Exception:
    openai_api_key = None

# אם אין מפתח בכלל, נציג הודעת שגיאה ברורה למשתמש
if not openai_api_key:
    st.error("שגיאה: מפתח ה-OPENAI_API_KEY לא נמצא. יש להגדיר אותו ב-Secrets של Streamlit או כמשתנה סביבה.")
    st.stop()

# --- הגדרת חיבור לדאטהבייס ---
# בניית נתיב דינמי ובטוח לקובץ ה-Database באותו המקום שבו נמצא app.py
current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_dir, "credit_risk.db")

if not os.path.exists(db_path):
    st.error(f"שגיאה: קובץ מסד הנתונים '{db_path}' לא נמצא בתיקייה.")
    st.stop()

# יצירת חיבור ל-LangChain SQLDatabase
db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

# --- אתחול מודל השפה והסוכן ---
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0, 
    api_key=openai_api_key
)

# יצירת סוכן ה-SQL
agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    verbose=True,
    handle_parsing_errors=True
)

# --- ממשק המשתמש (Chat Interface) ---
# שמירת היסטוריית השיחה
if "messages" not in st.session_state:
    st.session_state.messages = []

# הצגת ההיסטוריה על המסך
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# תיבת קלט לשאלה מהמשתמש
user_query = st.chat_input("לדוגמה: מה סכום ההלוואה הממוצע בתיק?")

if user_query:
    # הצגת הודעת המשתמש במסך
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # הרצת הסוכן ותשובה
    with st.chat_message("assistant"):
        with st.spinner("מנתח את הנתונים ומייצר שאילתה..."):
            try:
                # הפעלת הסוכן על השאלה של המשתמש
                response = agent_executor.invoke({"input": user_query})
                # חילוץ התשובה מתוך המבנה של LangChain
                answer = response.get("output", str(response))
            except Exception as e:
                answer = f"אופס, אירעה שגיאה בעיבוד השאילתה: {e}"
                
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})