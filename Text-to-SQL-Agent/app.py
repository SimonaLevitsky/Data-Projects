import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
import os


# כותרת האפליקציה בממשק המשתמש
st.title("עוזר AI לאנליזת סיכוני אשראי 🤖")

# התחברות למסד הנתונים שיצרנו
db = SQLDatabase.from_uri("sqlite:///credit_risk.db")

# הגדרת מודל השפה (נשתמש במודל מהיר וזול)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key="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")

# יצירת הסוכן החכם שמשלב בין ה-LLM לדאטהבייס
agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

# תיבת טקסט לקבלת שאלה מהמשתמש
user_question = st.text_input("איזו שאלה עסקית תרצי לשאול על הנתונים?")

# ברגע שיש שאלה, נפעיל את הסוכן
if user_question:
    with st.spinner("מנתח את הנתונים ומייצר שאילתה..."):
        try:
            # הפעלת הסוכן
            response = agent_executor.invoke({"input": user_question})
            # הצגת התשובה על המסך
            st.success(response["output"])
        except Exception as e:
            st.error(f"התרחשה שגיאה: {e}")