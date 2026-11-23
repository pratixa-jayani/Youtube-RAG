import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from youtube_transcript_api import YouTubeTranscriptApi

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(title="YouTube RAG API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# REQUEST MODEL
# =========================================================

class QuestionRequest(BaseModel):
    video_id: str
    question: str


# =========================================================
# GROQ API KEY
# =========================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY is not set.")


# =========================================================
# EMBEDDING MODEL
# =========================================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

print("Embedding model loaded.")


# =========================================================
# LLM
# =========================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# =========================================================
# CACHE
# =========================================================

vectorstore_cache = {}


# =========================================================
# GET YOUTUBE TRANSCRIPT
# =========================================================

def get_transcript(video_id):

    ytt_api = YouTubeTranscriptApi()

    transcript_list = ytt_api.list(video_id)

    transcripts = list(transcript_list)

    if not transcripts:
        raise Exception("No transcripts available for this video.")

    # First prefer manually created transcript
    manual_transcripts = [
        t for t in transcripts
        if not t.is_generated
    ]

    if manual_transcripts:
        transcript = manual_transcripts[0]

    else:
        # Otherwise use auto-generated transcript
        generated_transcripts = [
            t for t in transcripts
            if t.is_generated
        ]

        if not generated_transcripts:
            raise Exception("No usable transcript found.")

        transcript = generated_transcripts[0]

    print("Selected language:", transcript.language)
    print("Language code:", transcript.language_code)
    print("Auto generated:", transcript.is_generated)

    fetched_transcript = transcript.fetch()

    transcript_list = list(fetched_transcript)

    text = " ".join(
        snippet.text
        for snippet in transcript_list
    )

    return text


# =========================================================
# CREATE VECTOR STORE
# =========================================================

def create_vectorstore(video_id):

    # Check cache first

    if video_id in vectorstore_cache:

        print("Using cached vector store.")

        return vectorstore_cache[video_id]


    # Get transcript

    transcript_text = get_transcript(video_id)


    # Create LangChain document

    document = Document(
        page_content=transcript_text,
        metadata={
            "source": "YouTube",
            "video_id": video_id
        }
    )


    # Split transcript

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(
        [document]
    )

    print(
        f"Created {len(chunks)} chunks."
    )


    # Create FAISS vector store

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )


    # Save in memory

    vectorstore_cache[video_id] = vectorstore


    return vectorstore


# =========================================================
# ASK QUESTION
# =========================================================

@app.post("/ask")
def ask_question(request: QuestionRequest):

    try:

        print("=" * 70)

        print(
            "Video ID:",
            request.video_id
        )

        print(
            "Question:",
            request.question
        )

        print("=" * 70)


        # -------------------------------------------------
        # CREATE / GET VECTOR STORE
        # -------------------------------------------------

        vectorstore = create_vectorstore(
            request.video_id
        )


        # -------------------------------------------------
        # RETRIEVE RELEVANT DOCUMENTS
        # -------------------------------------------------

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 5
            }
        )


        retrieved_docs = retriever.invoke(
            request.question
        )


        print(
            f"Retrieved {len(retrieved_docs)} chunks."
        )


        # -------------------------------------------------
        # CREATE CONTEXT
        # -------------------------------------------------

        context = "\n\n".join(
            doc.page_content
            for doc in retrieved_docs
        )


        # -------------------------------------------------
        # PROMPT
        # -------------------------------------------------

        prompt = f"""
You are a helpful YouTube video assistant.

Answer the user's question using ONLY the
provided transcript context.

If the answer is not present in the context,
say:

"I couldn't find that information in the video."

Give a clear and natural answer.

Transcript context:
-------------------

{context}

-------------------

Question:
{request.question}

Answer:
"""


        # -------------------------------------------------
        # CALL GROQ
        # -------------------------------------------------

        response = llm.invoke(prompt)


        answer = response.content


        print("ANSWER:")
        print(answer)


        # -------------------------------------------------
        # RETURN TO CHROME
        # -------------------------------------------------

        return {
            "answer": answer,
            "video_id": request.video_id
        }


    except Exception as e:

        print("=" * 70)

        print("ERROR:")
        print(type(e).__name__)
        print(str(e))

        print("=" * 70)


        return {
            "answer": f"Error: {str(e)}"
        }


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {
        "status": "running",
        "message": "YouTube RAG backend is running"
    }