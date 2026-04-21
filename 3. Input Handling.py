# 1. Input Handling (Offline Voice or Text)
def get_user_input():
    print("Choose input method:")
    print("1. Text")
    print("2. Voice (Local Whisper)")
    choice = input("Enter 1 or 2: ").strip()

    if choice == '2':
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("\nListening... Please speak your query.")
            recognizer.adjust_for_ambient_noise(source)
            try:
                # Capture the audio
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=15)

                # Save temporarily to process with local Whisper
                temp_wav = "temp_audio.wav"
                with open(temp_wav, "wb") as f:
                    f.write(audio.get_wav_data())

                print("Transcribing locally...")
                result = asr_model.transcribe(temp_wav)
                text = result["text"].strip()

                # Cleanup
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)

                print(f"Transcribed Text: '{text}'\n")
                return text

            except sr.WaitTimeoutError:
                print("Listening timed out. Falling back to text input.")
            except Exception as e:
                print(f"Voice input error: {e}. Falling back to text input.")

    # Default to text input
    return input("\nEnter your text query: ").strip()