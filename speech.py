import speech_recognition as sr
import threading
import time

paused = False
running = True
transcribed_text = ""

def pause_recognition():
    global paused
    paused = True
    print("Recognition paused.")

def resume_recognition():
    global paused
    paused = False
    print("Recognition resumed.")

def save_text_to_file(text, filename="transcription.txt"):
    try:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(text)
        print(f"Text successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving file: {str(e)}")

def transcribe_speech(api_choice, language):
    global paused, running, transcribed_text
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)
    
    print("Start speaking... (type 'pause', 'resume', or 'exit' in console to control)")
    
    while running:
        if paused:
            time.sleep(0.5)
            continue
        
        with mic as source:
            try:
                print("Listening...")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                print("Processing...")
                
                if api_choice == "google":
                    text = recognizer.recognize_google(audio, language=language)
                elif api_choice == "sphinx":
                    text = recognizer.recognize_sphinx(audio, language=language)
                else:
                    print("API not supported, defaulting to Google.")
                    text = recognizer.recognize_google(audio, language=language)
                
                print(f"Transcribed: {text}")
                transcribed_text += text + "\n"
                
            except sr.UnknownValueError:
                print("Sorry, could not understand the audio.")
            except sr.RequestError as e:
                print(f"Could not request results from API; {e}")
            except sr.WaitTimeoutError:
                print("Listening timed out while waiting for phrase to start.")
            except Exception as e:
                print(f"An unexpected error occurred: {str(e)}")

def main():
    global running
    
    print("Choose Speech Recognition API: ")
    print("1. Google")
    print("2. Sphinx (offline)")
    api_input = input("Enter choice (1/2): ").strip()
    api_choice = "google" if api_input == "1" else "sphinx"
    
    language = input("Enter language code (e.g., 'en-US', 'fr-FR'): ").strip()
    if not language:
        language = "en-US"
    
    recognition_thread = threading.Thread(target=transcribe_speech, args=(api_choice, language))
    recognition_thread.start()
    
    while True:
        command = input("Command (pause/resume/save/exit): ").strip().lower()
        if command == "pause":
            pause_recognition()
        elif command == "resume":
            resume_recognition()
        elif command == "save":
            save_text_to_file(transcribed_text)
        elif command == "exit":
            print("Exiting...")
            running = False
            recognition_thread.join()
            break
        else:
            print("Unknown command. Use pause, resume, save, or exit.")

if __name__ == "__main__":
    main()