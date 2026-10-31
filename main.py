import os
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel
from audio_recorder import record_audio  # ייבוא פונקציית ההקלטה שכתבנו
import time
from google.genai import errors

# 1. טעינת משתני סביבה ולקוח
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# 2. הגדרת סכמת הפלט המובנה (JSON Schema)
class ConversationAnalysis(BaseModel):
    transcribed_text: str      # הטקסט שהמודל תמלל מהקול של המשתמש
    is_relevant: bool          # האם התשובה קשורה להקשר?
    grammar_correct: bool      # האם הדקדוק תקין?
    corrected_text: str        # המשפט המתוקן בשפת היעד
    grammar_explanation: str   # הסבר הכלל בדקדוק בשפת האם של המשתמש
    next_question: str         # שאלת ההמשך בשפת היעד

# 3. הגדרות שפה והקשר
target_language = "English"
native_language = "Hebrew"
context_question = "How are you doing today?"

system_prompt = f"""
You are a professional language tutor bot.
The user is learning {target_language}.
Provide grammar explanations in {native_language}.

Context question asked to user: '{context_question}'

Analyze the audio input:
1. Transcribe the spoken text accurately into 'transcribed_text'.
2. Check if the user's response is relevant to the question.
3. Check grammar accuracy for {target_language}.
4. If there are grammar errors, explain the rule in {native_language} clearly and simply.
5. Prepare the next conversational question in {target_language}.
"""

# 4. שלב ההקלטה - מקליטים 10 שניות מהמיקרופון
audio_filename = record_audio(duration=10)  # ניתן לשנות את משך ההקלטה לפי הצורך

# 5. העלאת קובץ השמע ל-API של Geminiprint("📤 מעלה את קובץ השמע לניתוח...")
audio_file = client.files.upload(file=audio_filename)

# ניסיון פנייה למודל עם מנגנון Retry
max_retries = 3
response = None

for attempt in range(1, max_retries + 1):
    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=[
                audio_file,
                system_prompt
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": ConversationAnalysis,
            }
        )
        break  # אם הקריאה הצליחה, יוצאים מהלולאה!
    except errors.ServerError as e:
        print(f"⚠️ השרת עמוס כרגע (ניסיון {attempt} מתוך {max_retries}). מנסה שוב בעוד 3 שניות...")
        time.sleep(3)
    except Exception as e:
        print(f"❌ אירעה שגיאה בלתי צפויה: {e}")
        break

if response:
    print("\n📊 תוצאת הניתוח:")
    print(response.text)
else:
    print("❌ לא ניתן היה לקבל מענה מהשרת עקב עומס. נסה שוב בעוד מספר רגעים.")