"""
run_tests.py - מריץ את שאלות הבדיקה על שתי אסטרטגיות ה-Chunking
ושומר את התוצאות בקובץ results.md להגשה (כל תשובה נשמרת מיד).

הרצה רגילה:              python run_tests.py
אחזור בלבד (בלי LLM):     python run_tests.py --retrieval-only
"""
import sys
import time

from rag_engine import (CHUNK_STRATEGIES, QuotaExceeded, ask, get_query_engine,
                        get_retriever, retrieve)

sys.stdout.reconfigure(encoding="utf-8")  # כדי שעברית תודפס בלי שגיאות

QUESTIONS = [
    "מה עושים מיד אחרי כוויה ממים רותחים?",                # שאלה ישירה
    "הילד שלי שתה חומר ניקוי, מה עושים?",                  # ניסוח שונה מהמסמך
    "למה אסור להטות את הראש אחורה כשיורד דם מהאף?",         # שאלת "למה"
    "באיזה קצב לוחצים על החזה בהחייאה?",                   # שאלה על מספר מדויק
    "איך מונעים מילד ליפול מהחלון?",                       # שאלת מניעה
    "מה עושים אחרי עקיצת דבורה?",                          # בנושא, אבל אין תשובה במאגר
    "איזה יום היום?",                                     # לא בנושא בכלל
]

RETRIEVAL_ONLY = "--retrieval-only" in sys.argv
OUTPUT_FILE = "results_retrieval.md" if RETRIEVAL_ONLY else "results.md"


def write_question(f, engine, i, question):
    """מריץ שאלה אחת וכותב את התוצאה לקובץ."""
    if RETRIEVAL_ONLY:  # רק ציוני דמיון - לא צורך מכסה של מודל השפה
        nodes = retrieve(engine, question)
        scores = ", ".join(f"{n.metadata.get('file_name')} {n.score:.3f}" for n in nodes)
        print(f"[{i}] {question}\n    {scores}")
        f.write(f"- **{question}** → {scores}\n")
        return

    start = time.time()
    answer, sources = ask(engine, question)
    seconds = time.time() - start

    print(f"\n[{i}] {question}\n{answer}")
    f.write(f"\n### {i}. {question}\n\n**תשובה:** {answer}\n\n")
    f.write(f"זמן: {seconds:.1f} שניות\n\n**מקורות שאוחזרו:** {len(sources)}\n")
    for src in sources:
        f.write(f"- {src['file']} (ציון {src['score']}): {src['text'][:150]}...\n")
    time.sleep(4)  # הפסקה קטנה כדי לא לחרוג ממגבלת הבקשות לדקה


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("# תוצאות בדיקות\n")
    try:
        for strategy in CHUNK_STRATEGIES:
            engine = get_retriever(strategy) if RETRIEVAL_ONLY else get_query_engine(strategy)
            f.write(f"\n## אסטרטגיה: {strategy} {CHUNK_STRATEGIES[strategy]}\n\n")
            print(f"\n===== {strategy} =====")
            for i, question in enumerate(QUESTIONS, 1):
                write_question(f, engine, i, question)
                f.flush()  # שומר מיד, כדי שעצירה באמצע לא תמחק את מה שכבר רץ
    except QuotaExceeded as error:
        print(f"\n⚠️ {error}")
        f.write(f"\n⚠️ ההרצה נעצרה: {error}\n")

print(f"\nהתוצאות נשמרו ב-{OUTPUT_FILE}")
