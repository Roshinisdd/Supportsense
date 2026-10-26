import os, glob
import streamlit as st
import chromadb
from groq import Groq

st.set_page_config(page_title="Support Sense", layout="wide")
st.title("Support Sense: Agent-Assist Chatbot")


@st.cache_resource
def load_kb():
    col = chromadb.Client().get_or_create_collection("support_kb")
    if col.count() == 0:
        docs, ids, metas = [], [], []
        for path in glob.glob("data/*.txt"):
            name = os.path.basename(path)
            text = open(path, encoding="utf-8").read()
            for i, chunk in enumerate(text.split("\n\n")):
                if chunk.strip():
                    docs.append(chunk.strip())
                    ids.append(f"{name}-{i}")
                    metas.append({"source": name})
        if docs:
            col.add(documents=docs, ids=ids, metadatas=metas)
    return col


def pick_model(client):
    available = [m.id for m in client.models.list().data]
    preferred = [
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-20b",
        "llama-3.1-8b-instant",
        "openai/gpt-oss-120b",
    ]
    for m in preferred:
        if m in available:
            return m
    for m in available:
        if not any(x in m for x in ["whisper", "guard", "tts", "orpheus"]):
            return m
    raise RuntimeError("No usable Groq model found for this API key.")


try:
    col = load_kb()
except Exception as e:
    st.error(f"Could not load the knowledge base: {e}")
    st.stop()

if col.count() == 0:
    st.error("No documents found. Check that the 'data' folder has your .txt files.")
    st.stop()

left, right = st.columns(2)

with left:
    st.subheader("Customer message")
    question = st.text_area("Paste the customer's message here", height=150)
    go = st.button("Suggest reply")

if go:
    if not question.strip():
        with left:
            st.warning("Type a customer message first, then click Suggest reply.")
    else:
        try:
            res = col.query(query_texts=[question], n_results=2)
            chunks = res["documents"][0]
            sources = [m["source"] for m in res["metadatas"][0]]
            context = "\n\n".join(chunks)

            prompt = f"""You help a customer support agent.
Use ONLY the policy text below to write a short, polite reply to the customer.
If the answer is not in the policy, say the agent should escalate.

Policy:
{context}

Customer message:
{question}"""

            client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            chosen = pick_model(client)
            st.sidebar.caption(f"Model in use: {chosen}")

            with left:
                with st.spinner("Writing reply..."):
                    resp = client.chat.completions.create(
                        model=chosen,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=500,
                    )
                st.subheader("Suggested reply")
                st.write(resp.choices[0].message.content)

            with right:
                st.subheader("Sources used")
                for c, s in zip(chunks, sources):
                    st.info(f"**{s}**\n\n{c}")

        except Exception as e:
            with left:
                st.error(f"Something went wrong: {e}")
