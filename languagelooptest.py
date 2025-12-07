import curses
import threading
import queue
import time
import json
import pyaudio
from vosk import Model, KaldiRecognizer
from STT_EN import diarize_audio, get_cst_time, make_wav
from LanguageTranslationFuncs import CN_to_EN_Text, EN_to_CN_Text, EN_to_ES_Text, ES_to_EN_Text
from TextToSpeech import initialize_TTS, Generate_TTS_SpeakerOut


# Global queue for passing transcription results from the recognition thread to the TUI.
transcription_queue = queue.Queue()
client = initialize_TTS()


def get_model(language: str) -> Model:
    """Return the appropriate Vosk model based on the selected language."""
    if language == "english":
        return Model("vosk-model-en-US-0.22")  # Replace with the actual path for English
    elif language == "spanish":
        return Model("vosk-model-es-0.42")
    elif language == "chinese":
        return Model("vosk-model-cn-0.22")
    else:
        return Model("vosk-model-en-US.0.22")
    
def speech_recognition_loop(language: str, transcription_queue: queue.Queue, stop_event: threading.Event):
    """
    This function runs in a separate thread. It initializes the Vosk recognizer for the chosen
    language and continually reads from the microphone. When a segment is complete, it puts the
    transcribed text onto transcription_queue.
    """
    model = get_model(language)
    recognizer = KaldiRecognizer(model, 16000)
    mic = pyaudio.PyAudio()
    stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000,
                      input=True, frames_per_buffer=4096)
    stream.start_stream()
    raw_audio = b""

    while not stop_event.is_set():
        try:
            data = stream.read(4096, exception_on_overflow=False)
        except Exception as e:
            continue
        raw_audio += data

        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "").strip()

            #generates .wav for diarization
            wav_filename = make_wav(raw_audio)
            speaker_label = diarize_audio(raw_audio)

            #assigns current time
            timestamp = get_cst_time()        
        
            #prints formatted output.
            #Format: [time] [Speaker] audio text
            #EX: [02:15:23] [EMS1] audio text
            full_output = f"{timestamp} {speaker_label} {text}"
            
            if text:
            #adds the next line of the STT transcription to the output
                transcription_queue.put(full_output) #change back to full_output
            #    if language == "spanish":
             #       es_en = ES_to_EN_Text(text)
              #      Generate_TTS_SpeakerOut(es_en, client, True, "en_US")
               #     transcription_queue.put(es_en) #change back to full_output


            #clears audio that has already been diarized
            raw_audio = b""
    stream.stop_stream()
    stream.close()
    mic.terminate()

def curses_main(stdscr):
    # Set up curses
    curses.curs_set(1)  # show the cursor in the command area
    stdscr.nodelay(True)
    stdscr.clear()
    max_y, max_x = stdscr.getmaxyx()

    # Create two windows:
    #   1) output_win: occupies most of the terminal for transcription output
    #   2) input_win: a one-line window at the bottom for commands
    output_win = curses.newwin(max_y - 1, max_x, 0, 0)
    input_win = curses.newwin(1, max_x, max_y - 1, 0)

    transcript_lines = []  # to store lines of transcription output

    #initial display, shows that the device is working and beginning initilization of the model
    transcript_lines.append("Initializing English model...")
    output_win.erase()
    for idx, line in enumerate(transcript_lines):
        output_win.addstr(idx, 0, line[:max_x-1])
    output_win.refresh()

    # Start the speech recognition thread.
    language = "english"  # default language
    stop_event = threading.Event()
    recog_thread = threading.Thread(target=speech_recognition_loop, args=(language, transcription_queue, stop_event))
    recog_thread.daemon = True
    recog_thread.start()


    # Command input buffer (we read keys one by one).
    cmd_buffer = ""
    input_win.timeout(50)  # small timeout to allow loop to refresh frequently

    while True:
        # Retrieve any new transcription results and append them to transcript_lines.
        try:
            while True:  # get all available transcription lines
                new_line = transcription_queue.get_nowait()
                transcript_lines.append(new_line)
                # Keep only as many lines as will fit into the output window.
                if len(transcript_lines) > (max_y - 1):
                    transcript_lines = transcript_lines[-(max_y - 1):]
        except queue.Empty:
            pass

        # Refresh the output window with the transcript (redraw all lines).
        output_win.erase()
        for idx, line in enumerate(transcript_lines):
            # Make sure the line fits within the width.
            output_win.addnstr(idx, 0, line[:max_x-1])
        output_win.refresh()

        # Show the command prompt in the input window.
        input_win.erase()
        input_prompt = "Command (english/spanish/chinese, Esc to quit): "
        input_win.addstr(0, 0, input_prompt + cmd_buffer)
        input_win.refresh()

        # Process key input from the user.
        try:
            key = input_win.getch()
        except curses.error:
            key = -1  # No input

        if key != -1:
            # Handle Enter (10 or 13 are typical codes) to submit the command.
            if key in (curses.KEY_ENTER, 10, 13):
                command = cmd_buffer.strip().lower()
                if command in ("english", "spanish", "chinese"):
                    transcript_lines.append(f"Switching language to {command.upper()}")
                    # Signal recognition thread to stop.
                    stop_event.set()
                    recog_thread.join(timeout=1)
                    # Reset for the new language.
                    stop_event.clear()
                    transcription_queue.queue.clear()  # clear pending transcripts
                    language = command
                    recog_thread = threading.Thread(target=speech_recognition_loop, args=(language, transcription_queue, stop_event))
                    recog_thread.daemon = True
                    recog_thread.start()
                # Clear the command buffer after processing.
                cmd_buffer = ""
            # Escape key to exit the entire program.
            elif key == 27:
                break
            # Handle backspace (127 is common; curses.KEY_BACKSPACE may also be used).
            elif key in (curses.KEY_BACKSPACE, 127):
                cmd_buffer = cmd_buffer[:-1]
            # For regular characters, append to command buffer.
            else:
                try:
                    cmd_buffer += chr(key)
                except:
                    pass

        # Sleep briefly to avoid hogging CPU
        time.sleep(0.05)

    # Clean up: signal the recognition thread to stop and wait for it.
    stop_event.set()
    recog_thread.join(timeout=1)


if __name__ == "__main__":
    # Initialize curses (on Windows, ensure you have installed windows-curses)
    curses.wrapper(curses_main)
