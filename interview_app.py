import os
import streamlit as st
import io
import time
import json 
import PyPDF2
import docx

# Voice Output Library
from gtts import gTTS 

# Microphone Recorder Component (FIXED) and STT Library
from st_audiorec import st_audiorec # <<< FIXED IMPORT
from google.cloud import speech 

# Gemini API Libraries
from google import genai
from google.genai import types 

# --- CONFIGURATION ---
APP_TITLE = "💅 Tarun's Interview Vibe Check" 
MODEL_NAME = 'gemini-2.5-flash'
SYSTEM_INSTRUCTION = (
    "You are a hyper-critical, professional, and uncompromising Senior Tech Lead. "
    "Your job is to gatekeep this role. Every question must be a deep technical drill-down "
    "based on the candidate's LAST ANSWER and the RESUME. Your tone must be formal and challenging. "
    "Your response MUST ONLY contain the structured JSON object. Do not output any prose, pleasantries, or introductions."
)

# Initialize Gemini Client (Omitted for brevity)
@st.cache_resource
def get_gemini_client():
    try:
        if "GEMINI_API_KEY" not in os.environ:
            st.error("GEMINI_API_KEY environment variable is not set.")
            return None
        return genai.Client()
    except Exception as e:
        st.error(f"Error initializing Gemini client: {e}")
        return None

client = get_gemINI_client()


# --- SPEECH-TO-TEXT (STT) FUNCTIONS ---

@st.cache_resource
def get_speech_client():
    """Initializes Google Speech-to-Text client. Requires GOOGLE_APPLICATION_CREDENTIALS."""
    try:
        if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
            st.warning("GOOGLE_APPLICATION_CREDENTIALS not set. Voice input will fail.")
            return None
        return speech.SpeechClient()
    except Exception as e:
        st.error(f"Error initializing Google Speech Client: {e}")
        return None

def transcribe_audio(audio_bytes):
    """Sends audio data to Google Speech-to-Text API for transcription."""
    speech_client = get_speech_client()
    if speech_client is None:
        return "ERROR: Speech-to-Text client not initialized."
        
    try:
        audio = speech.RecognitionAudio(content=audio_bytes)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.MP3, 
            sample_rate_hertz=16000, 
            language_code="en-US",
        )
        
        response = speech_client.recognize(config=config, audio=audio)
        
        if response.results:
            return response.results[0].alternatives[0].transcript
        else:
            return "Could not understand audio. Please speak clearly."
            
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        return f"ERROR: Transcription service failed ({e})."

# --- CORE UTILITY FUNCTIONS (TTS, File Parsing, AI Logic) Omitted for brevity ---

def speak_question(text):
    """Generates audio from text and returns the bytes for Streamlit audio player."""
    try:
        mp3_fp = io.BytesIO()
        tts = gTTS(text=text, lang='en')
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return mp3_fp.read()
    except Exception:
        return None

def extract_text_from_file(uploaded_file):
    """Extracts text from various file types (txt, pdf, docx)."""
    file_type = uploaded_file.name.split('.')[-1].lower()
    file_bytes = uploaded_file.read()
    
    if file_type == 'txt':
        return file_bytes.decode("utf-8")
    
    elif file_type == 'pdf':
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            text = "".join([page.extract_text() or "" for page in reader.pages])
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return None
            
    elif file_type in ['docx']:
        try:
            document = docx.Document(io.BytesIO(file_bytes))
            text = "\n".join([paragraph.text for paragraph in document.paragraphs])
            return text
        except Exception as e:
            st.error(f"Error reading DOCX: {e}")
            return None
            
    else:
        st.error(f"Unsupported file type: .{file_type}")
        return None

def generate_interview_question_and_score(resume_text, protocol_text, conversation_history):
    """Generates the next question and scores the last answer using Gemini and JSON schema."""
    if not client:
        return {"next_question": "AI system unavailable. What are your core strengths?", "score_out_of_5": 3, "feedback": "System error. Interview continuing."}

    # 1. Define the expected JSON output structure (Omitted for brevity)
    response_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "next_question": types.Schema(type=types.Type.STRING, description="The single, next, highly specific interview question."),
            "score_out_of_5": types.Schema(type=types.Type.INTEGER, description="A technical score for the candidate's immediately preceding answer, ranging from 1 (Needs Improvement) to 5 (Mastery)."),
            "feedback": types.Schema(type=types.Type.STRING, description="Specific, concise technical feedback (1-2 sentences) on the candidate's previous answer.")
        },
        required=["next_question", "score_out_of_5", "feedback"]
    )

    # 2. Build the Full Prompt (Omitted for brevity)
    full_prompt = (
        f"--- CANDIDATE RESUME ---\n{resume_text}\n\n"
        f"--- INTERVIEW PROTOCOL ---\n{protocol_text}\n\n"
        f"--- CONVERSATION HISTORY ---\n"
    )
    
    if conversation_history:
        for message in conversation_history:
            if message['role'] == 'user':
                full_prompt += f"Candidate Answer: {message['content']}\n"
            elif message['role'] == 'assistant':
                if not message['content'].startswith("**🔥 Score on Previous Answer:"):
                    full_prompt += f"Interviewer Question: {message['content']}\n"
    
    full_prompt += "\nBased on all the information above, generate the next interview question and provide a score and feedback for the candidate's last answer in the requested JSON format."

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=response_schema
            )
        )
        return json.loads(response.text.strip())
        
    except Exception as e:
        st.exception(e)
        return {"next_question": "I apologize, a system error occurred. Please answer the last question again.", "score_out_of_5": 3, "feedback": "System error: Could not process structured JSON response."}

# --- STREAMLIT INTERFACE ---

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
    
    # 🌟 NEW: HIDE MENUS AND FORK BUTTON
    hide_streamlit_elements() 

    # Aesthetic Title and Header (Gen Z Vibe Check)
    st.markdown(f"""
        <style>
        .big-font {{
            font-size: 40px !important;
            font-weight: 800;
            color: #FF00FF; 
            text-shadow: 2px 2px #00FFFF;
        }}
        .strict-protocol {{
            color: #FF4B4B;
            font-weight: 700;
        }}
        .stButton>button {{
            background-color: #00FFFF;
            color: black; 
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 20px;
        }}
        </style>
        <p class="big-font">✨ {APP_TITLE} ✨</p>
        <p class="strict-protocol">🚫 Protocol: No cap, we're testing your whole stack.</p>
        <hr style="border: 3px dashed #00FFFF;"/>
    """, unsafe_allow_html=True)

    # Initialize session state variables (Omitted for brevity)
    if "messages" not in st.session_state: st.session_state.messages = []
    if "resume_text" not in st.session_state: st.session_state.resume_text = None
    if "protocol_text" not in st.session_state: st.session_state.protocol_text = None
    if "interview_started" not in st.session_state: st.session_state.interview_started = False
    if "scores_history" not in st.session_state: st.session_state.scores_history = []
    if "interview_ended" not in st.session_state: st.session_state.interview_ended = False
        
    # --- Sidebar Setup (Omitted for brevity) ---
    with st.sidebar:
        st.header("1. 📁 Setup & Files")
        
        uploaded_resume = st.file_uploader("Upload Candidate Resume (TXT, PDF, DOCX)", type=["txt", "pdf", "docx"])
        if uploaded_resume is not None and st.session_state.resume_text is None:
            text = extract_text_from_file(uploaded_resume)
            if text: st.session_state.resume_text = text; st.success(f"Resume uploaded: {uploaded_resume.name}")

        uploaded_protocol = st.file_uploader("Upload Interview Protocol (TXT, PDF, DOCX)", type=["txt", "pdf", "docx"])
        if uploaded_protocol is not None and st.session_state.protocol_text is None:
            text = extract_text_from_file(uploaded_protocol)
            if text: st.session_state.protocol_text = text; st.success(f"Protocol uploaded: {uploaded_protocol.name}")
            
        st.markdown("---")
        st.header("2. 📸 Proctoring & Start")
        st.warning("Webcam check bypassed. Upload files to proceed.")

        can_start = (st.session_state.resume_text is not None and st.session_state.protocol_text is not None and not st.session_state.interview_started and not st.session_state.interview_ended)

        if can_start:
            if st.button("START TECHNICAL INTERVIEW"):
                st.session_state.interview_started = True
                response_data = generate_interview_question_and_score(st.session_state.resume_text, st.session_state.protocol_text, [])
                first_q = response_data.get('next_question', 'What are your core strengths?')
                st.session_state.messages.append({"role": "assistant", "content": first_q})
                st.rerun() 
        elif not st.session_state.interview_started and not st.session_state.interview_ended:
            st.info("Complete steps 1 & 2 to enable the Start button.")


    # --- Main Content Area ---
    
    if st.session_state.interview_started:
        st.subheader("🎤 Interview Vibe Check: Listen and Respond below.")
        
        if st.button("End Interview & View Scorecard", key="end_btn"):
            st.session_state.interview_started = False; st.session_state.interview_ended = True; st.rerun() 
            
        st.markdown("---")
        
        # Display chat history (omitted for brevity)
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and not message["content"].startswith("**🔥 Score on Previous Answer:"):
                    audio_bytes = speak_question(message["content"])
                    if audio_bytes: st.audio(audio_bytes, format='audio/mp3', autoplay=True)


        # --- AUDIO RECORDER AND STT LOGIC ---
        st.markdown("### 🗣️ Record Your Answer")
        
        # 1. Use the NEW audio recorder component
        audio_data = st_audiorec(
            'Answer_Recorder', 
            key='audio_input_key'
        )
        
        prompt = None

        if audio_data is not None and len(audio_data) > 1000:
            
            # Transcribe the audio
            with st.spinner("Processing audio... Transcribing speech to text..."):
                prompt = transcribe_audio(audio_data)
            
            st.info(f"🎤 Transcribed Answer: {prompt}")

        # Now, use the transcribed prompt to drive the interview
        if prompt and not prompt.startswith("ERROR"):
            
            # 1. Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)

            # 2. Get AI Response (Score/Feedback + Next Question)
            with st.spinner("Bot is analyzing your answer and generating the next question..."):
                response_data = generate_interview_question_and_score(st.session_state.resume_text, st.session_state.protocol_text, st.session_state.messages)
                next_q = response_data.get('next_question', 'What is your next answer?')
                score = response_data.get('score_out_of_5', 3)
                feedback = response_data.get('feedback', 'Could not retrieve feedback.')

            # 3. Store Score, Display Feedback, and Add Next Question
            score_message = f"**🔥 Score on Previous Answer: {score}/5**\n*Feedback: {feedback}*"
            st.session_state.scores_history.append({'score': score, 'feedback': feedback})

            st.session_state.messages.append({"role": "assistant", "content": score_message}) 
            with st.chat_message("assistant"): st.markdown(score_message)

            st.session_state.messages.append({"role": "assistant", "content": next_q})
            st.rerun() 
        elif prompt and prompt.startswith("ERROR"):
             st.error("Could not process voice input. Please check logs for STT key issues.")

    # --- Scorecard Display and Initial Screen (Omitted for brevity) ---
    elif st.session_state.interview_ended:
        st.balloons(); st.header("✅ Interview Complete: Final Scorecard")
        total_score = sum([s['score'] for s in st.session_state.scores_history]); max_score = len(st.session_state.scores_history) * 5
        final_percentage = (total_score / max_score) * 100 if max_score > 0 else 0
        rating = "👑 Slayed the interview. 10/10." if final_percentage >= 80 else "solid effort. No red flags." if final_percentage >= 60 else "😬 Needs work. Big yikes."
        st.markdown(f"## Overall Vibe: **{rating}**"); st.markdown(f"### Final Technical Score: <span style='color:#FF00FF;'>{total_score} / {max_score}</span>", unsafe_allow_html=True); st.progress(final_percentage / 100, text=f"Overall Performance: {final_percentage:.1f}%")
        st.subheader("Detailed Feedback Log (The Receipts)")
        for i, entry in enumerate(st.session_state.scores_history): st.markdown(f"#### Turn {i+1} Score: **{entry['score']}/5**"); st.info(f"**AI Feedback:** {entry['feedback']}")
        st.markdown("---"); st.markdown("Interview data recorded. Thanks for the vibe check!")
        
    else:
        st.info("Use the sidebar to prepare your session.")
        col1, col2 = st.columns(2)
        with col1: st.header("🚨 Instructions: Final Vibe Check"); st.markdown("""**The Rules (Read Carefully!):** 1. **Upload Files:** Drop your **Resume** and the **Protocol** (TXT, PDF, or DOCX). 2. **Start:** Click **'START TECHNICAL INTERVIEW'** to begin. 3. **Focus:** Speak clearly into the microphone after the question is asked. **We expect receipts.**""")
        with col2: st.header("💡 Sample Interview Flow"); st.code("""[BOT]: (Spoken) Describe a scenario where you used a decorator to solve a common cross-cutting concern.\n[YOU]: I used a retry decorator to wrap API calls.\n[FEEDBACK]: 🔥 Score: 4/5. Feedback: Good concept, but lacked specificity on concurrency controls."""); st.info("The conversation is strictly technical and fully voice-enabled.")


if __name__ == "__main__":
    main()
