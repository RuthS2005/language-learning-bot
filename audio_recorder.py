import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

def record_audio(filename="user_input.wav", duration=20, fs=44100):
    """
    מקליטה שמע מהמיקרופון לשירות קובץ WAV.
    :param filename: שם הקובץ לשמירה
    :param duration: משך ההקלטה בשניות
    :param fs: קצב הדגימה (Sampling Rate)
    """
    print(f"\n🎤 מתחיל להקליט ל-{duration} שניות... דבר עכשיו!")
    
    # הקלטת השמע מהמיקרופון
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # ממתין עד שההקלטה תסתיים
    
    print("✅ ההקלטה הסתיימה. מעבד את השמע...")
    
    # שמירת ההקלטה לקובץ WAV
    write(filename, fs, recording)
    return filename

if __name__ == "__main__":
    # בדיקה יחידנית של רכיב ההקלטה
    record_audio(duration=3)