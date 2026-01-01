import html
import textwrap

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ================== GENEL AYARLAR ================== #
load_dotenv()  # .env içindeki GOOGLE_API_KEY vb.

PDF_PATH = "pharmacy.pdf"
EMBED_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash-lite"

st.set_page_config(
    page_title="Eczacı SUT Asistanı",
    page_icon="💊",
    layout="centered",
)
# =================================================== #


def apply_custom_css():
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

:root {
    --primary: #10b981;
    --primary-dark: #059669;
    --secondary: #6366f1;
    --accent: #f59e0b;
    --bg-dark: #0f172a;
    --bg-card: #1e293b;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
}

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Streamlit default gizle */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 900px;
}

/* ---------- HEADER ---------- */
.header-container {
    text-align: center;
    padding: 2.5rem 1rem;
    margin-bottom: 2rem;
    background: linear-gradient(135deg, rgba(16,185,129,0.1) 0%, rgba(99,102,241,0.1) 100%);
    border-radius: 24px;
    border: 1px solid rgba(16,185,129,0.2);
    backdrop-filter: blur(10px);
}

.logo-icon {
    font-size: 4rem;
    margin-bottom: 0.5rem;
    display: block;
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
}

.main-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #10b981 0%, #6366f1 50%, #f59e0b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -0.02em;
}

.subtitle {
    color: #94a3b8;
    font-size: 1.1rem;
    margin-top: 0.75rem;
    font-weight: 400;
}

/* ---------- WELCOME BOX ---------- */
.welcome-box {
    background: linear-gradient(135deg, rgba(99,102,241,0.1) 0%, rgba(16,185,129,0.1) 100%);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 2rem 0;
}

.welcome-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
}

.welcome-title {
    color: #e2e8f0;
    font-size: 1.3rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

.welcome-text {
    color: #94a3b8;
    font-size: 0.95rem;
    line-height: 1.6;
}

.feature-pills {
    display: flex;
    justify-content: center;
    gap: 0.75rem;
    margin-top: 1.5rem;
    flex-wrap: wrap;
}

.feature-pill {
    background: rgba(30,41,59,0.8);
    border: 1px solid rgba(51,65,85,0.6);
    border-radius: 50px;
    padding: 0.5rem 1rem;
    font-size: 0.8rem;
    color: #94a3b8;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
}

/* ---------- ÖZEL CHAT GRID ---------- */
.chat-wrapper {
    background: rgba(15,23,42,0.6);
    border-radius: 20px;
    border: 1px solid rgba(30,64,175,0.4);
    padding: 1.5rem 1.5rem 1.0rem 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(8px);
}

.chat-grid {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}

.chat-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    align-items: flex-start;
}

.assistant-cell,
.user-cell {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
}

.assistant-cell {
    justify-content: flex-start;
}

.user-cell {
    justify-content: flex-end;
}

.chat-avatar {
    width: 32px;
    height: 32px;
    border-radius: 999px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
}

.assistant-avatar {
    background: linear-gradient(135deg, #f97316, #ec4899);
    color: #fff;
}

.user-avatar {
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: #fff;
}

.chat-bubble {
    max-width: 100%;
    padding: 0.55rem 0.95rem;
    border-radius: 18px;
    font-size: 0.94rem;
    line-height: 1.6;
    word-wrap: break-word;
    box-shadow: 0 2px 4px rgba(0,0,0,0.25);
}

.assistant-bubble {
    background: #202c33;
    color: #e9edef;
    border-radius: 18px 10px 18px 18px;
}

.user-bubble {
    background: #005c4b;
    color: #e9edef;
    border-radius: 10px 18px 18px 18px;
}

/* ---------- INPUT ALANI ---------- */
[data-testid="stChatInput"] > div {
    background: #f8fafc !important;
    border: 2px solid #e2e8f0 !important;
    border-radius: 16px !important;
    padding: 0.3rem !important;
}

[data-testid="stChatInput"] > div:focus-within {
    border-color: #10b981 !important;
    box-shadow: 0 0 0 3px rgba(16,185,129,0.25) !important;
}

[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #0f172a !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.95rem !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #9ca3af !important;
}
</style>
""",
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown(
        """
<div class="header-container">
    <span class="logo-icon">💊</span>
    <h1 class="main-title">Eczacı SUT Asistanı</h1>
    <p class="subtitle">PDF tabanlı yapay zeka destekli soru-cevap sistemi</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_welcome_message():
    st.markdown(
        """
<div class="welcome-box">
    <div class="welcome-icon">🔬</div>
    <div class="welcome-title">Hoş Geldiniz!</div>
    <div class="welcome-text">
        Eczacılık SUT dokümanınız hakkında sorularınızı yanıtlamaya hazırım.<br>
        Solda asistan, sağda siz olacak şekilde soru-cevapları görebilirsiniz.
    </div>
    <div class="feature-pills">
        <span class="feature-pill">📄 PDF Tabanlı</span>
        <span class="feature-pill">🎯 Odaklı Yanıtlar</span>
        <span class="feature-pill">🔒 Güvenilir Kaynak</span>
        <span class="feature-pill">⚡ Hızlı Arama</span>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def escape_markdown(text: str) -> str:
    """Mesajı HTML'e uygun hale getir (satır sonlarını koru)."""
    return html.escape(text).replace("\n", "<br>")


def render_chat(messages):
    """Solda asistan, sağda kullanıcı olacak şekilde grid chat render et."""
    if not messages:
        return

    rows_html = ""

    for msg in messages:
        if msg["role"] == "assistant":
            # Sol sütunda asistan
            rows_html += (
                '<div class="chat-row">'
                '  <div class="assistant-cell">'
                '    <div class="chat-avatar assistant-avatar">💊</div>'
                f'    <div class="chat-bubble assistant-bubble">{escape_markdown(msg["content"])}</div>'
                '  </div>'
                '  <div class="user-cell"></div>'
                '</div>'
            )
        else:
            # Sağ sütunda kullanıcı
            rows_html += (
                '<div class="chat-row">'
                '  <div class="assistant-cell"></div>'
                '  <div class="user-cell">'
                f'    <div class="chat-bubble user-bubble">{escape_markdown(msg["content"])}</div>'
                '    <div class="chat-avatar user-avatar">👤</div>'
                '  </div>'
                '</div>'
            )

    full_html = (
        '<div class="chat-wrapper">'
        '  <div class="chat-grid">'
        f'    {rows_html}'
        '  </div>'
        '</div>'
    )

    st.markdown(full_html, unsafe_allow_html=True)



# ================== RAG ZİNCİRİ ================== #
@st.cache_resource
def build_rag_chain():
    loader = PyPDFLoader(PDF_PATH)
    data = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(data)

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0.3,
        max_output_tokens=512,
    )

    system_prompt = (
        "Sen eczacılık alanında uzman, SUT dokümanına göre cevap veren bir asistansın. "
        "Görevin, sadece sana verilen SUT PDF'inden elde edilen bilgilere dayanarak "
        "kullanıcıların sorularını yanıtlamaktır. "
        "Eğer cevap SUT dokümanında yoksa 'Bu konuda SUT dokümanında bilgi bulamıyorum.' de "
        "ve kesinlikle uydurma yapma. "
        "Cevapları Türkçe ver ve mümkün olduğunca kısa ve net yaz (en fazla 3-4 cümle).\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "{input}"),
        ]
    )

    output_parser = StrOutputParser()

    qa_chain = (
        {
            "context": retriever
            | (lambda x: "\n\n".join(doc.page_content for doc in x)),
            "input": RunnablePassthrough(),
        }
        | prompt
        | llm
        | output_parser
    )

    return qa_chain


# ================== UYGULAMA AKIŞI ================== #
apply_custom_css()
render_header()
qa_chain = build_rag_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    render_welcome_message()
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": "Merhaba! 👋 Eczacılık SUT dokümanınız hakkında nasıl yardımcı olabilirim?",
        }
    )

# Chat grid’i çiz
render_chat(st.session_state.messages)

# Yeni soru inputu
query = st.chat_input("SUT hakkında bir soru yazın...", key="sut_chat_input")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    q_lower = query.lower().strip()
    greeting_keywords = [
        "merhaba",
        "selam",
        "selamlar",
        "iyi günler",
        "günaydın",
        "iyi akşamlar",
    ]
    farewell_keywords = [
        "görüşürüz",
        "hoşça kal",
        "bay bay",
        "iyi geceler",
        "teşekkürler",
        "teşekkür ederim",
    ]

    answer = None

    # Selamlama intenti
    if any(word in q_lower for word in greeting_keywords):
        answer = (
            "Merhaba! 👋 Eczacılık SUT dokümanınız hakkında nasıl yardımcı olabilirim?"
        )

    # Veda intenti
    elif any(word in q_lower for word in farewell_keywords):
        answer = (
            "Rica ederim, yardımcı olabildiysem ne mutlu. "
            "Yeni bir SUT sorunuz olursa her zaman yazabilirsiniz. Görüşmek üzere! 👋"
        )

    # Diğer durumlarda RAG çalışsın
    if answer is None:
        answer = qa_chain.invoke(query)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    st.rerun()
