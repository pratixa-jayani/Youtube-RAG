const askButton = document.getElementById("askButton");
const questionInput = document.getElementById("question");
const chatContainer = document.getElementById("chatContainer");

const buttonText = document.getElementById("buttonText");
const loading = document.getElementById("loading");
const status = document.getElementById("status");

const videoTitle = document.getElementById("videoTitle");


/* --------------------------------
   GET CURRENT YOUTUBE TAB
-------------------------------- */

async function getCurrentVideo() {

    const tabs = await chrome.tabs.query({
        active: true,
        currentWindow: true
    });

    if (!tabs.length) {
        return null;
    }

    const tab = tabs[0];

    if (!tab.url || !tab.url.includes("youtube.com/watch")) {
        return null;
    }

    const url = new URL(tab.url);

    return {
        video_id: url.searchParams.get("v"),
        title: tab.title || "YouTube Video"
    };
}


/* --------------------------------
   ADD MESSAGE TO CHAT
-------------------------------- */

function addMessage(text, type) {

    const message = document.createElement("div");

    message.className = `message ${type}`;

    if (type === "assistant") {

        message.innerHTML = `
            <div class="avatar">✦</div>

            <div class="bubble">
                ${text}
            </div>
        `;

    } else {

        message.innerHTML = `
            <div class="bubble">
                ${text}
            </div>
        `;
    }

    chatContainer.appendChild(message);

    chatContainer.scrollTop = chatContainer.scrollHeight;
}


/* --------------------------------
   ASK QUESTION
-------------------------------- */

async function askQuestion() {

    const question = questionInput.value.trim();

    if (!question) {
        return;
    }

    /* Get YouTube video */

    const video = await getCurrentVideo();

    if (!video) {

        addMessage(
            "Please open a YouTube video first.",
            "assistant"
        );

        return;
    }


    /* Show user question */

    addMessage(question, "user");

    questionInput.value = "";

    askButton.disabled = true;

    buttonText.classList.add("hidden");
    loading.classList.remove("hidden");

    status.textContent = "Searching the video...";


    try {

        /*
         IMPORTANT:

         This URL must match your FastAPI backend.
        */

        const response = await fetch(
            "http://127.0.0.1:8000/ask",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({

                    video_id: video.video_id,

                    question: question

                })
            }
        );


        if (!response.ok) {

            throw new Error(
                `Server returned ${response.status}`
            );
        }


        const data = await response.json();


        /* Display answer */

        addMessage(
            data.answer || "I couldn't find an answer in the video.",
            "assistant"
        );

        status.textContent = "";


    } catch (error) {

        console.error(error);

        addMessage(
            "⚠️ Cannot connect to the RAG backend. Make sure FastAPI is running on port 8000.",
            "assistant"
        );

        status.textContent = "Backend connection failed.";

    } finally {

        askButton.disabled = false;

        buttonText.classList.remove("hidden");
        loading.classList.add("hidden");
    }
}


/* --------------------------------
   BUTTON CLICK
-------------------------------- */

askButton.addEventListener(
    "click",
    askQuestion
);


/* --------------------------------
   ENTER TO ASK
-------------------------------- */

questionInput.addEventListener(
    "keydown",
    function(event) {

        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {

            event.preventDefault();

            askQuestion();
        }
    }
);


/* --------------------------------
   DETECT VIDEO ON OPEN
-------------------------------- */

async function loadVideo() {

    const video = await getCurrentVideo();

    if (!video) {

        videoTitle.textContent =
            "Open a YouTube video";

        return;
    }

    videoTitle.textContent =
        video.title;
}


loadVideo();