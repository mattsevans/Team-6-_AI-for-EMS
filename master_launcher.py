#!/usr/bin/env python3
import os
import time
import json
import signal
import subprocess
from deploy_whispercpp_main import main


# ================== PATHS & CONSTANTS ======================

AI_WEAR_DIR = "/home/team6/NLP-System/AI_Wear_NLP"
AI_WEAR_VENV_PY = os.path.join(AI_WEAR_DIR, ".venv", "bin", "python")

WHISPER_CPP_DIR = "/home/team6/NLP-System/whisper.cpp"
WHISPER_SERVER_BIN = os.path.join(WHISPER_CPP_DIR, "build", "bin", "whisper-server")
WHISPER_MODEL_PATH = "/home/team6/NLP-System/whisper.cpp/models/ggml-tiny.bin"

MASTER_OUTPUT_JSONL = os.path.join(AI_WEAR_DIR, "MasterFileOutput.jsonl")
MASTER_INPUT_JSONL = os.path.join(AI_WEAR_DIR, "MasterFileInput.jsonl")

DOCKER_CONTAINER_NAME = "zora-ai-team6"

END_INCIDENT_LINE = "[16:53:18] EMS: Zora end incident"

# Button config
BUTTON_PIN = 18              # BCM 18 = physical pin 12
LONG_PRESS_THRESHOLD = 5.0   # seconds

# ================== GLOBAL STATE ===========================

procs = []                  # [0] = deploy_whispercpp_main.py, [1] = whisper-server
running = False             # are the 3 components running (Python + Docker)?
started = False             # has the pipeline been started at least once?
requested_incident_end = False  # long-press / Ctrl+C requested incident end
language_mode = 0           # 0 or 1 depending on short presses


def get_language_mode():
    return language_mode

# GPIO (button)
try:
    import RPi.GPIO as GPIO
    HAVE_GPIO = True
except ImportError:
    HAVE_GPIO = False
    print("[MASTER][WARN] RPi.GPIO not available; button will not work.")


# ================== PROCESS MANAGEMENT =====================

def start_pipeline():
    """Start deploy_whispercpp_main.py, whisper-server, and Docker container."""
    global procs, running

    if running:
        print("[MASTER] Pipeline already running, ignoring start request.")
        return

    procs = []
    print("[MASTER] Starting pipeline...")

    # 1) deploy_whispercpp_main.py via venv python
    env_ai = os.environ.copy()
    cmd_whisper_main = [AI_WEAR_VENV_PY, "deploy_whispercpp_main.py"]
    print(f"[MASTER]  -> starting deploy_whispercpp_main.py with: {cmd_whisper_main}")
    p1 = subprocess.Popen(
        cmd_whisper_main,
        cwd=AI_WEAR_DIR,
        env=env_ai,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    procs.append(p1)

    # 2) whisper-server
    env_server = os.environ.copy()
    ems_prompt = env_server.get("EMS_PROMPT", "")
    if not ems_prompt:
        print("[MASTER][WARN] EMS_PROMPT is empty; set it before running if needed.")

    cmd_server = [
        WHISPER_SERVER_BIN,
        "-m", WHISPER_MODEL_PATH,
        "-t", "4",
        "--host", "127.0.0.1",
        "--port", "8080",
        "--prompt", ems_prompt,
    ]
    print(f"[MASTER]  -> starting whisper-server with: {cmd_server}")
    p2 = subprocess.Popen(
        cmd_server,
        cwd=WHISPER_CPP_DIR,
        env=env_server,
        # Let logs go straight to the terminal:
        stdout=None,
        stderr=None,
        #stdout=subprocess.PIPE,
        #stderr=subprocess.PIPE,
    )
    procs.append(p2)

    # 3) Docker container (start if exists, else run)
    docker_cmd = (
        "docker start zora-ai-team6 2>/dev/null || "
        "docker run -d --name zora-ai-team6 --restart unless-stopped --user root "
        "--shm-size=256m "
        "--env-file /home/team6/AI_System/env/.env "
        "-e CHROME_PATH=/usr/bin/chromium "
        "-e GOOGLE_APPLICATION_CREDENTIALS=/app/creds/gcp.json "
        "--mount type=bind,src=/home/team6/AI_System/creds/gcp.json,dst=/app/creds/gcp.json,readonly "
        "--mount type=bind,src=/home/team6/AI_System/creds/gmail_client_secret.json,dst=/app/gmail_client_secret.json,readonly "
        "--mount type=bind,src=/home/team6/AI_System/creds/gmail_token.json,dst=/app/gmail_token.json "
        "--mount type=bind,src=/home/team6/AI_System/reports,dst=/app/reports "
        "--mount type=bind,src=/home/team6/NLP-System/AI_Wear_NLP/MasterFileInput.jsonl,dst=/app/MasterFileInput.jsonl,readonly "
        "--mount type=bind,src=/home/team6/NLP-System/AI_Wear_NLP/MasterFileOutput.jsonl,dst=/app/MasterFileOutput.jsonl "
        "zora-ai:py313-arm64-v2"
    )
    print(f"[MASTER]  -> ensuring Docker container {DOCKER_CONTAINER_NAME} is running...")
    subprocess.run(docker_cmd, shell=True, check=False)

    running = True
    print("[MASTER] All components started.")


def stop_python_processes():
    """Stop deploy_whispercpp_main.py and whisper-server only."""
    global procs

    if not procs:
        return

    print("[MASTER] Stopping Python-based processes (deploy_whisper + whisper-server)...")
    # 1) Try SIGINT
    for p in procs:
        if p.poll() is None:
            print(f"[MASTER]  -> sending SIGINT to PID {p.pid}")
            try:
                p.send_signal(signal.SIGINT)
            except Exception as e:
                print(f"[MASTER]     (error sending SIGINT: {e})")

    # Give them a bit to exit cleanly
    time.sleep(5.0)

    # 2) Force kill if still alive
    for p in procs:
        if p.poll() is None:
            print(f"[MASTER]  -> PID {p.pid} still alive, killing.")
            try:
                p.kill()
            except Exception as e:
                print(f"[MASTER]     (error killing process: {e})")

    procs.clear()


def stop_pipeline():
    """Stop Python processes and the Docker container immediately."""
    global running

    if not running:
        print("[MASTER] Pipeline not running, nothing to stop.")
        return

    print("[MASTER] Stopping entire pipeline (Python + Docker)...")

    # 1) Stop Python-based processes
    stop_python_processes()

    # 2) Stop Docker container
    print(f"[MASTER]  -> docker stop {DOCKER_CONTAINER_NAME}")
    subprocess.run(["docker", "stop", DOCKER_CONTAINER_NAME], check=False)

    running = False
    print("[MASTER] Pipeline stopped.")


def append_end_incident_to_input():
    """Append the special end-incident line to MasterFileInput.jsonl."""
    try:
        with open(MASTER_INPUT_JSONL, "a", encoding="utf-8") as f:
            f.write(END_INCIDENT_LINE + "\n")
        print(f"[MASTER] Appended end-incident line to {MASTER_INPUT_JSONL}")
    except Exception as e:
        print(f"[MASTER] Failed to append end-incident line: {e}")


# ================== JSONL TAILING (AUTO-SHUTDOWN) =========

def tail_output_step(fh):
    """
    Non-blocking 'tail -f' step:
      - Reads any new lines from MasterFileOutput.jsonl
      - Returns True if an incident-end phrase is seen, else False.
    """
    trigger_phrases = [
        "Incident closing due to timeout.",
        "Incident closed by EMS",
    ]

    while True:
        pos = fh.tell()
        line = fh.readline()
        if not line:
            fh.seek(pos)  # no new data, rewind to last position
            return False

        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except Exception:
            continue

        text = str(obj.get("text", ""))

        if any(phrase in text for phrase in trigger_phrases):
            print("[MASTER] Detected incident end in JSONL:")
            print(f"         {line}")
            return True


# ================== GPIO BUTTON HANDLING ===================

def setup_gpio():
    if not HAVE_GPIO:
        print("[MASTER] GPIO not available; button control disabled.")
        return

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    print(f"[MASTER] Monitoring button on GPIO {BUTTON_PIN} (physical pin 12, pull-up enabled).")
    print("Behavior:")
    print("  - First button press -> Start pipeline (all 3 components)")
    print("  - Short press (< 5s) after start -> Toggle language_mode 0/1")
    print("  - Long press (>= 5s) after start -> Stop Python, send end-incident line")
    print("                                      then wait for 'Incident closed by EMS' "
          "for full shutdown.\n")


# ================== MAIN LOOP =============================

def main():
    global started, running, requested_incident_end, language_mode

    setup_gpio()

    pressed = False
    press_start_time = None

    # For JSONL tailing
    output_fh = None
    output_ready = False

    print("[MASTER] Master launcher idle. Waiting for first button press to start pipeline...")
    print("Press Ctrl+C to request safe incident end (same as long press) while running.\n")

    try:
        while True:
            # --- 1) BUTTON HANDLING ------------------------------------
            if HAVE_GPIO:
                state = GPIO.input(BUTTON_PIN)  # HIGH = not pressed, LOW = pressed

                # Detect button press (edge: HIGH -> LOW)
                if not pressed and state == GPIO.LOW:
                    pressed = True
                    press_start_time = time.time()

                # Detect button release (edge: LOW -> HIGH)
                elif pressed and state == GPIO.HIGH:
                    press_duration = time.time() - press_start_time
                    pressed = False

                    # FIRST press: start pipeline regardless of duration
                    if not started:
                        started = True
                        print(f"[MASTER] Button pressed for {press_duration:.2f}s -> Starting pipeline")
                        start_pipeline()
                    else:
                        # After started, interpret duration
                        if press_duration >= LONG_PRESS_THRESHOLD:
                            # Long press = same as safe Ctrl+C when running
                            if running and not requested_incident_end:
                                print(f"[MASTER] Button held for {press_duration:.2f}s -> "
                                      "Stopping Python, requesting incident end")
                                stop_python_processes()
                                append_end_incident_to_input()
                                requested_incident_end = True
                                print("[MASTER] Waiting for 'Incident closed by EMS' "
                                      "in output JSONL for full shutdown...")
                            else:
                                print(f"[MASTER] Long press ({press_duration:.2f}s) but "
                                      "either not running or already requested end.")
                        else:
                            # Short press toggles language_mode flag
                            if started:
                                global language_mode
                                language_mode = 1 - language_mode
                                os.environ["LANGUAGE_MODE"] = str(language_mode)
                                print(f"[MASTER] Button short press ({press_duration:.2f}s) -> "
                                      f"language_mode = {language_mode}")

            # --- 2) JSONL FILE SETUP ----------------------------------
            if running or requested_incident_end:
                # Only care about output file once system has started
                if not output_ready and os.path.exists(MASTER_OUTPUT_JSONL):
                    output_fh = open(MASTER_OUTPUT_JSONL, "r", encoding="utf-8")
                    output_fh.seek(0, os.SEEK_END)  # tail from end
                    output_ready = True
                    print(f"[MASTER] Now monitoring {MASTER_OUTPUT_JSONL} for incident end...")

            # --- 3) JSONL TAIL STEP -----------------------------------
            if output_ready and output_fh is not None:
                incident_end = tail_output_step(output_fh)
                if incident_end:
                    # Auto-shutdown path (timeout or 'Incident closed by EMS')
                    print("[MASTER] Waiting 30 seconds before full shutdown...")
                    time.sleep(30.0)
                    stop_pipeline()
                    print("[MASTER] Exiting master launcher after incident end.")
                    break

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[MASTER] KeyboardInterrupt received.")
        # If running and we haven't requested incident end yet, perform same as long press:
        if running and not requested_incident_end:
            print("[MASTER] Treating Ctrl+C as long press: stopping Python, requesting incident end.")
            stop_python_processes()
            append_end_incident_to_input()
            requested_incident_end = True
            print("[MASTER] Waiting for 'Incident closed by EMS' in output JSONL "
                  "for full shutdown...")
            # We do NOT stop Docker here; we keep looping until JSONL trigger.
            try:
                # keep looping until JSONL incident end is seen
                while True:
                    if not output_ready and os.path.exists(MASTER_OUTPUT_JSONL):
                        output_fh = open(MASTER_OUTPUT_JSONL, "r", encoding="utf-8")
                        output_fh.seek(0, os.SEEK_END)
                        output_ready = True
                        print(f"[MASTER] Now monitoring {MASTER_OUTPUT_JSONL} for incident end...")

                    if output_ready and output_fh is not None:
                        incident_end = tail_output_step(output_fh)
                        if incident_end:
                            print("[MASTER] Waiting 30 seconds before full shutdown...")
                            time.sleep(30.0)
                            stop_pipeline()
                            print("[MASTER] Exiting master launcher after incident end.")
                            break

                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("[MASTER] Second Ctrl+C received; exiting immediately without stopping Docker.")
        else:
            print("[MASTER] Not running or incident end already requested; exiting immediately.")

    finally:
        if HAVE_GPIO:
            GPIO.cleanup()
            print("[MASTER] GPIO cleaned up.")
        if 'output_fh' in locals() and output_fh is not None:
            output_fh.close()

if __name__ == "__main__":
    os.environ["LANGUAGE_MODE"] = str(language_mode)
    main()
