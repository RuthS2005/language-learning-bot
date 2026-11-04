import os
import asyncio
import re
import gradio as gr
from dotenv import load_dotenv
from groq import AsyncGroq
import speech_recognition as sr
import traceback
import httpx
from gtts import gTTS

# 1. טעינת משתני סביבה והגדרת לקוח Groq
load_dotenv()

custom_http_client = httpx.AsyncClient(verify=False)

groq_client = AsyncGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    http_client=custom_http_client
)

# 2. פונקציית תמלול מקומית מהירה
def transcribe_audio_locally(audio_filepath):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_filepath) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language="en-US")
        return text

# 3. פונקציית TTS אסינכרונית עם gTTS (בטוחה מפני שגיאות SSL)
async def generate_speech(text, output_file="response.mp3"):
    loop = asyncio.get_running_loop()
    
    def _create_tts():
        # יצירת קובץ שמע באנגלית בלבד
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(output_file)
        
    await loop.run_in_executor(None, _create_tts)
    return output_file

# 4. פונקציית עיבוד אסינכרונית מול Groq
async def process_audio(audio_filepath, history):
    if history is None:
        history = []

    if not audio_filepath:
        return history, history, None, "⚠️ לא הוקלט שמע."

    target_language = "English"
    native_language = "Hebrew"

    # א. תמלול מקומי
    try:
        print("🎤 מתמלל שמע באופן מקומי...")
        loop = asyncio.get_running_loop()
        transcribed_text = await loop.run_in_executor(None, transcribe_audio_locally, audio_filepath)
        print(f"✅ תמלול התקבל: '{transcribed_text}'")
    except Exception as e:
        print(f"❌ שגיאה בתמלול: {e}")
        return history, history, None, "⚠️ לא הצלחתי לתמלל את השמע. נסה לדבר שוב."

    # ב. הרכבת ההיסטוריה
    history_context = ""
    for msg in history[-4:]:
        role = "User" if msg.get("role") == "user" else "Bot"
        history_context += f"{role}: {msg.get('content', '')}\n"

    # ג. Prompt מפורט ל-Groq
    prompt = f"""
    You are a friendly {target_language} tutor in an ongoing spoken conversation.
    The user is learning {target_language}. Explain grammar briefly in {native_language}.

    CRITICAL RULES FOR EVALUATION:
    - The user input comes from automated Speech-To-Text (STT), so it DOES NOT CONTAIN punctuation or capitalization.
    - DO NOT mark missing periods, commas, question marks, or capitalization as grammar errors!
    - Focus ONLY on actual spoken grammar errors (wrong tense, word order, incorrect verb/noun forms, missing articles).

    Conversation History:
    {history_context}

    User Spoken Input: "{transcribed_text}"

    Respond in this exact Markdown format:
    **מה שמעתי:** "{transcribed_text}"

    **ניתוח:** [Write '✅ דקדוק תקין!' or '❌ שגיאת דקדוק']
    [If spoken grammar is incorrect, write: **תיקון:** (corrected sentence) and **הסבר:** (brief explanation in Hebrew)]

    💬 **השאלה הבאה:** [Write the next brief conversational question in {target_language}]
    """

    # ד. פנייה אסינכרונית ל-Groq
    try:
        print("⚡ שולח ל-Groq...")
        response = await groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=600
        )
        
        bot_response_text = response.choices[0].message.content
        print("🚀 התקבלה תשובה מ-Groq במהירות שיא!")

        # ה. חילוץ שאלת ההמשך ויצירת קובץ קול (TTS)
        audio_output_path = None
        next_question_match = re.search(r"💬 \*\*השאלה הבאה:\*\*\s*(.*)", bot_response_text)
        
        if next_question_match:
            next_question_text = next_question_match.group(1).strip()
            print(f"🔊 מייצר קובץ שמע עבור השאלה: '{next_question_text}'")
            audio_output_path = await generate_speech(next_question_text)

        new_history = list(history)
        new_history.append({"role": "user", "content": transcribed_text})
        new_history.append({"role": "assistant", "content": bot_response_text})

        return new_history, new_history, audio_output_path, "⚡ התגובה עובדה והשמע מוכן!"

    except Exception as e:
        print("\n❌ --- פירוט השגיאה המלא ב-Groq ---")
        traceback.print_exc()
        print("-----------------------------------\n")
        return history, history, None, f"❌ שגיאה: {type(e).__name__} - {str(e)}"

# 5. הממשק ב-Gradio
with gr.Blocks(title="AI Language Tutor - Groq Powered") as demo:
    gr.Markdown("# 🎙️ AI Language Learning Tutor (Groq Speed)")
    
    conversation_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="הקלטת שמע")
            submit_btn = gr.Button("שלח לבדיקה 🚀", variant="primary")
            status_output = gr.Textbox(label="סטטוס", interactive=False)
            audio_output = gr.Audio(label="תגובה קולית של הבוט", autoplay=True)
        
        with gr.Column(scale=2):
            chatbot_ui = gr.Chatbot(label="היסטוריית השיחה")

    submit_btn.click(
        fn=process_audio,
        inputs=[audio_input, conversation_state],
        outputs=[chatbot_ui, conversation_state, audio_output, status_output]
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch()