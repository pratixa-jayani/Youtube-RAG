from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# EMBEDDING MODEL
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# ============================================================
# GET YOUTUBE TRANSCRIPT
# ============================================================

def get_transcript(video_id):

    ytt_api = YouTubeTranscriptApi()

    transcript = ytt_api.fetch(
        video_id,
        languages=["en"]
    )

    transcript_list = list(transcript)

    text = " ".join(
        snippet.text
        for snippet in transcript_list
    )

    return text


# ============================================================
# CREATE VECTOR STORE
# ============================================================

def create_vector_store(text, video_id):

    document = Document(
        page_content=text,
        metadata={
            "source": "YouTube",
            "video_id": video_id
        }
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(
        [document]
    )

    print("Number of chunks:", len(chunks))

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_store


# ============================================================
# ASK QUESTION
# ============================================================

def ask_question(vector_store, question):

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 4
        }
    )

    retrieved_docs = retriever.invoke(
        question
    )

    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    prompt = f"""
You are a helpful YouTube video assistant.

Answer the question ONLY using the transcript context below.

If the answer cannot be found in the transcript,
say:

"I don't know based on this video."

Do not use outside knowledge.

TRANSCRIPT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = llm.invoke(prompt)

    return response.content