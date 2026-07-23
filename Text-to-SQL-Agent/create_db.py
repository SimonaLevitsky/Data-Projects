import sqlite3
import pandas as pd
import random

# יצירת נתוני דמי 
data = {
    'client_id': range(1, 101),
    'age': [random.randint(20, 70) for _ in range(100)],
    'loan_amount': [random.randint(10000, 500000) for _ in range(100)],
    'credit_score': [random.randint(400, 850) for _ in range(100)],
    'default_status': [random.choice([0, 1]) for _ in range(100)]
}

df = pd.DataFrame(data)

# יצירת חיבור למסד הנתונים ושמירת הטבלה
conn = sqlite3.connect('credit_risk.db')
df.to_sql('loans', conn, if_exists='replace', index=False)
conn.close()

print("Database created successfully!")