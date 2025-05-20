import speech_recognition as sr
import pyttsx3
import os
import time

recognizer = sr.Recognizer()
tts = pyttsx3.init()
tts.setProperty('rate', 150)  

def speak(text):
    tts.say(text)
    tts.runAndWait()

while True:
    with sr.Microphone() as micro:
        print('Скажите что-нибудь...')
        speak('Я слушаю')
        audio = recognizer.listen(micro)
    try:
        text = recognizer.recognize_google(audio, language='ru-RU')
        speak('Вы сказали: ' + text)
        print('Вы сказали: ' + text)
    except sr.UnknownValueError:
        speak('Повтори пожалуйста')
        continue
    except sr.RequestError as error:
        speak('Ничего не понял')
        continue

    if text.lower() == 'открой блокнот':
        os.startfile(r'C:\Windows\System32\notepad.exe')
        print('Пожалуйста, блокнот открыт')
        speak('Пожалуйста, блокнот открыт')
    elif 'привет' in text.lower():
        print('Привет, обращайся')
        speak('Привет, обращайся')
    elif 'как дела' in text.lower():
        print('Все своим чередом')
        speak('Все своим чередом')
    elif text.lower() in ['сколько времени', 'который час']:
        tm = time.strftime('%X')
        print('Сейчас ' + tm)
        speak('Сейчас ' + tm)
    elif text.lower() in ['пока', 'стоп']:
        print('Пока')
        speak('Пока')
        break


