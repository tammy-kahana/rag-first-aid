import truststore
truststore.inject_into_ssl()  # לסמוך על תעודות Windows (נדרש בגלל נטפרי)

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()  # קורא את המפתח מקובץ .env
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# בדיקה אמיתית: האם כל מודל עונה במסלול החינמי
LLM_CANDIDATES = ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-2.5-flash"]
EMBED_MODEL = "gemini-embedding-2"

for name in LLM_CANDIDATES:
    try:
        reply = client.models.generate_content(model=name, contents="ענה במילה אחת: מה בירת ישראל?")
        print("OK  ", name, "->", reply.text.strip())
    except Exception as e:
        print("FAIL", name, "->", str(e)[:120])

try:
    result = client.models.embed_content(model=EMBED_MODEL, contents="כוויה ממים רותחים")
    print("OK  ", EMBED_MODEL, "-> vector length", len(result.embeddings[0].values))
except Exception as e:
    print("FAIL", EMBED_MODEL, "->", str(e)[:120])
