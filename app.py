import os
import gradio as gr
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors
from pydantic import BaseModel

# 1. טעינת משתני סביבה ולקוח
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# 2. הגדרת Pydantic Schema
class ConversationAnalysis(BaseModel):
    transcribed_text: str      
    is_relevant: bool          
    grammar_correct: bool      
    corrected_text: str        
    grammar_explanation: str   
    next_question: str         

# 3. פונקציית העיבוד המוגנת
def process_audio(audio_filepath, history):
    # אתחול היסטוריה נקי
    if history is None:
        history = []

    if not audio_filepath:
        return history, history, "⚠️ לא הוקלט שמע."

    target_language = "English"
    native_language = "Hebrew"

    # הרכבת ה-Context
    history_context = ""
    for msg in history:
        role = "User" if msg.get("role") == "user" else "Bot"
        history_context += f"{role}: {msg.get('content', '')}\n"

    system_prompt = f"""
    You are a professional language tutor bot in an ongoing conversation.
    The user is learning {target_language}.
    Provide grammar explanations in {native_language}.

    Conversation History so far:
    {history_context}

    Analyze the audio input:
    1. Transcribe the spoken text accurately into 'transcribed_text'.
    2. Check if the user's response is relevant.
    3. Check grammar accuracy for {target_language}.
    4. If there are grammar errors, explain the rule in {native_language} simply.
    5. Prepare the next conversational question in {target_language}.
    """

    try:
        print("🎙️ מעבד שמע...")
        with open(audio_filepath, "rb") as f:
            audio_bytes = f.read()

        audio_part = types.Part.from_bytes(
            data=audio_bytes,
            mime_type="audio/wav"
        )

        print("📡 פונה ל-Gemini API...")
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=[audio_part, system_prompt],
            config={
                "response_mime_type": "application/json",
                "response_schema": ConversationAnalysis,
            }
        )
        print("✅ התקבלה תשובה מ-Gemini!")
        
        analysis = ConversationAnalysis.model_validate_json(response.text)

        grammar_status = "✅ דקדוק תקין!" if analysis.grammar_correct else "❌ שגיאת דקדוק"
        
        bot_response_text = f"**מה שמעתי:** \"{analysis.transcribed_text}\"\n\n"
        bot_response_text += f"**ניתוח:** {grammar_status}\n"
        if not analysis.grammar_correct:
            bot_response_text += f"**תיקון:** {analysis.corrected_text}\n"
            bot_response_text += f"**הסבר:** {analysis.grammar_explanation}\n\n"
        
        bot_response_text += f"💬 **השאלה הבאה:** {analysis.next_question}"

        # הוספת התור החדש להיסטוריה
        new_history = list(history)
        new_history.append({"role": "user", "content": analysis.transcribed_text})
        new_history.append({"role": "assistant", "content": bot_response_text})

        return new_history, new_history, "✅ הבדיקה הושלמה בהצלחה!"

    except errors.ServerError:
        print("❌ שגיאת שרת (503)")
        return history, history, "⚠️ השרת עמוס כרגע, נסה שוב בעוד כמה שניות."
    except Exception as e:
        print(f"❌ שגיאה: {str(e)}")
        return history, history, f"❌ שגיאה: {str(e)}"

# 4. הממשק הויזואלי
with gr.Blocks(title="AI Language Tutor") as demo:
    gr.Markdown("# 🎙️ AI Language Learning Tutor")
    
    conversation_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="הקלטת שמע")
            submit_btn = gr.Button("שלח לבדיקה 🚀", variant="primary")
            status_output = gr.Textbox(label="סטטוס / לוגים", interactive=False)
        
        with gr.Column(scale=2):
            chatbot_ui = gr.Chatbot(label="היסטוריית השיחה")

    submit_btn.click(
        fn=process_audio,
        inputs=[audio_input, conversation_state],
        outputs=[chatbot_ui, conversation_state, status_output],
        concurrency_limit=1  # מונע כפילות בקשות ותקיעות תורים
    )

if __name__ == "__main__":
    demo.queue()  # מפעיל תור בקשות יציב ב-Gradio
    demo.launch()