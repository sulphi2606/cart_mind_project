"""
CartMind AI — Streamlit Frontend
==================================
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import sys
import os

st.set_page_config(
    page_title="CartMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import agents

@st.cache_resource
def load_agent():
    agents.initialize()
    return True

with st.spinner("🧠 CartMind is loading... please wait"):
    load_agent()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #f8f7f4;
    color: #1a1a2e;
}
.stApp { background: #f8f7f4; }
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e8e4df;
}
.hero-wrap {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
    border-bottom: 1px solid #e8e4df;
    margin-bottom: 1.5rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: #1a1a2e;
    letter-spacing: -1px;
}
.hero-title span { color: #2d6a4f; }
.hero-tagline { font-size: 0.92rem; color: #999; margin-top: 0.5rem; }
.user-wrap { display:flex; justify-content:flex-end; margin:1rem 0; }
.user-bubble {
    background: #2d6a4f;
    color: white;
    padding: 13px 18px;
    border-radius: 18px 18px 4px 18px;
    max-width: 70%;
    font-size: 0.93rem;
    line-height: 1.6;
    box-shadow: 0 2px 8px rgba(45,106,79,0.25);
}
.bot-wrap { display:flex; justify-content:flex-start; margin:1rem 0; }
.bot-bubble {
    background: #ffffff;
    border: 1px solid #e8e4df;
    color: #1a1a2e;
    padding: 15px 20px;
    border-radius: 18px 18px 18px 4px;
    max-width: 78%;
    font-size: 0.93rem;
    line-height: 1.7;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.msg-label {
    font-size: 0.68rem; color: #aaa;
    margin-bottom: 5px;
    text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600;
}
.empty-state { text-align: center; padding: 4rem 2rem; }
.empty-icon  { font-size: 3rem; margin-bottom: 1rem; }
.empty-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem; color: #bbb; margin-bottom: 0.5rem;
}
.empty-sub   { font-size: 0.85rem; color: #ccc; }
.chip {
    display: inline-block;
    background: #f0f4f1; border: 1px solid #d4e6db;
    color: #2d6a4f; padding: 6px 14px;
    border-radius: 20px; font-size: 0.78rem; margin: 3px; font-weight: 500;
}
.stats-row  { display:flex; gap:12px; padding:10px 0; border-bottom:1px solid #f0ebe5; margin-bottom:12px; }
.stat-item  { font-size:0.75rem; color:#aaa; }
.stat-item b{ color:#2d6a4f; }
.section-label {
    font-size:0.7rem; font-weight:600; color:#aaa;
    text-transform:uppercase; letter-spacing:1px; margin:1rem 0 0.5rem 0;
}
.stack-item { font-size:0.78rem; color:#888; padding:4px 0; border-bottom:1px solid #f5f5f5; }
.stTextInput > div > div > input {
    background: #ffffff !important; border: 1.5px solid #e0dbd5 !important;
    border-radius: 14px !important; color: #1a1a2e !important;
    padding: 14px 18px !important; font-size: 0.95rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #2d6a4f !important;
    box-shadow: 0 0 0 3px rgba(45,106,79,0.1) !important;
}
.stTextInput > div > div > input::placeholder { color: #bbb !important; }
.stButton > button {
    background: #2d6a4f !important; color: white !important;
    border: none !important; border-radius: 12px !important;
    font-weight: 500 !important; width: 100% !important;
}
.stButton > button:hover { background: #1e4d38 !important; }
hr { border-color: #f0ebe5 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────
if "messages"      not in st.session_state: st.session_state.messages      = []
if "chat_history"  not in st.session_state: st.session_state.chat_history  = []
if "total_queries" not in st.session_state: st.session_state.total_queries = 0

# ── Ask with memory ────────────────────────────────────────────────
def ask_with_memory(question: str) -> str:
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    history_text = ""
    if st.session_state.chat_history:
        for human, ai in st.session_state.chat_history[-4:]:
            history_text += f"User: {human}\nAssistant: {ai}\n\n"

    return (
        {
            "context":  agents.retriever | agents.format_docs,
            "question": RunnablePassthrough(),
            "history":  lambda _: history_text
        }
        | agents.prompts
        | agents.model
        | StrOutputParser()
    ).invoke(question)

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:"Playfair Display",serif;font-size:1.6rem;
    font-weight:700;color:#1a1a2e;margin-bottom:2px'>🧠 CartMind</div>
    <div style='font-size:0.75rem;color:#aaa;margin-bottom:1.2rem'>
    Think Less. Shop Smarter.</div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='stats-row'>
        <div class='stat-item'>Queries <b>{st.session_state.total_queries}</b></div>
        <div class='stat-item'>Messages <b>{len(st.session_state.messages)}</b></div>
        <div class='stat-item'>Memory <b>{len(st.session_state.chat_history)} turns</b></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-label'>💡 Try asking</div>", unsafe_allow_html=True)

    for s in ["Best watch under $20","Gift necklace for mom",
              "Top rated fashion item","Affordable jewelry",
              "Something for travel","Bag for outdoor use"]:
        if st.button(s, key=f"s_{s}"):
            st.session_state["prefill"] = s
            st.rerun()

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.total_queries = 0
            st.rerun()
    with col2:
        if st.button("🧠 Memory"):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")
    st.markdown("<div class='section-label'>⚙️ Powered by</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='stack-item'>🤖 Gemini 2.5 Flash</div>
    <div class='stack-item'>🔍 BAAI/bge-small-en-v1.5</div>
    <div class='stack-item'>🎨 Chroma Vector DB</div>
    <div class='stack-item'>📦 Amazon 283K Dataset</div>
    <div class='stack-item'>🍃 MongoDB Atlas</div>
    <div class='stack-item'>🔗 LangChain LCEL</div>
    """, unsafe_allow_html=True)

# ── Main UI ────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-wrap'>
    <div style='font-size:2.2rem;margin-bottom:0.5rem'>🧠</div>
    <div class='hero-title'>Cart<span>Mind</span></div>
    <div class='hero-tagline'>Think Less. Shop Smarter. — AI-powered product recommendations</div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("""
    <div class='empty-state'>
        <div class='empty-icon'>🛍️</div>
        <div class='empty-title'>What are you looking for today?</div>
        <div class='empty-sub'>Ask me anything — I'll find the perfect product for you</div>
        <br>
        <span class='chip'>🕐 Best watch under $20</span>
        <span class='chip'>💍 Gift necklace</span>
        <span class='chip'>👜 Travel bag</span>
        <span class='chip'>⭐ Top rated items</span>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class='user-wrap'><div>
            <div class='msg-label' style='text-align:right'>You</div>
            <div class='user-bubble'>{msg['content']}</div>
        </div></div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='bot-wrap'><div>
            <div class='msg-label'>🧠 CartMind</div>
            <div class='bot-bubble'>{msg['content']}</div>
        </div></div>""", unsafe_allow_html=True)

# ── Input ──────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
prefill = st.session_state.pop("prefill", "")

col1, col2 = st.columns([6, 1])
with col1:
    user_input = st.text_input(
        "query", value=prefill,
        placeholder="✦  Ask me anything... e.g. 'Best affordable watch for daily use'",
        label_visibility="collapsed", key="user_query"
    )
with col2:
    send = st.button("Send →")

if send and user_input.strip():
    question = user_input.strip()
    if not st.session_state.messages or st.session_state.messages[-1].get("content") != question:
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.total_queries += 1

    with st.spinner("🔍 Finding the best products for you..."):
        try:
            response = ask_with_memory(question)
        except Exception as e:
            response = f"⚠️ Error: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.chat_history.append((question, response))

    if len(st.session_state.chat_history) > 10:
        st.session_state.chat_history = st.session_state.chat_history[-10:]

    st.rerun()