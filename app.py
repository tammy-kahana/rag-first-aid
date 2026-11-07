"""
app.py - ממשק צ'אט ב-Streamlit. הרצה: streamlit run app.py
"""
import streamlit as st

from rag_engine import (CHUNK_STRATEGIES, TOP_K, QuotaExceeded, ask, chunk_count,
                        get_query_engine)

ERROR_MESSAGE = "אירעה שגיאה בעת חיפוש המידע או יצירת התשובה. בדקי את החיבור ואת המפתח."

st.set_page_config(page_title="עזרה ראשונה - שאלות ותשובות", page_icon="🩹")

# יישור לימין עבור עברית
st.markdown(
    "<style>.stApp, .stChatMessage, .stMarkdown {direction: rtl; text-align: right;}</style>",
    unsafe_allow_html=True,
)

st.title("🩹 עזרה ראשונה ובטיחות בבית")
st.caption("מידע כללי בלבד, אינו מחליף גורם רפואי. במצב חירום: מד\"א 101")

if "messages" not in st.session_state:
    st.session_state.messages = []

# סרגל צד - בחירת אסטרטגיית Chunking ומספר מקטעים (לבונוס ולבדיקות)
with st.sidebar:
    st.header("הגדרות")
    strategy = st.radio("אסטרטגיית Chunking", list(CHUNK_STRATEGIES))
    top_k = st.slider("מספר מקטעים לאחזור (top_k)", 1, 6, TOP_K)
    if st.button("נקה שיחה", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


@st.cache_resource(show_spinner="טוען את המאגר...")
def load_engine(strategy, top_k):
    return get_query_engine(strategy, top_k)


try:
    engine = load_engine(strategy, top_k)
except QuotaExceeded as error:
    st.error(str(error))
    st.stop()
except Exception as error:
    st.error(f"שגיאה באתחול המערכת: {error}")
    st.stop()

st.sidebar.write(f"מספר מקטעים במאגר: {chunk_count(strategy)}")


def show_sources(sources):
    with st.expander(f"📚 מקורות ({len(sources)})"):
        for src in sources:
            st.markdown(f"**{src['file']}** · ציון דמיון: {src['score']}")
            st.text(src["text"][:400] + "...")


# הצגת היסטוריית השיחה
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"])

# שאלה חדשה
question = st.chat_input("שאלי שאלה על עזרה ראשונה...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("מחפש במאגר..."):
                answer, sources = ask(engine, question)
        except Exception as error:
            st.error(str(error) if isinstance(error, QuotaExceeded) else ERROR_MESSAGE)
            st.session_state.messages.pop()  # לא משאירים שאלה בלי תשובה בהיסטוריה
            st.stop()
        st.markdown(answer)
        if sources:
            show_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
