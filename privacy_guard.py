"""
PrivAI - AI Screen Privacy Guard (MVP)

Privacy layer on top of the existing LBPH face-recognition setup.
Flow:
    Webcam -> Haar face detection -> LBPH predict -> USER / UNKNOWN
    UNKNOWN stays for ANOMALY_TIME seconds -> fullscreen privacy overlay
    UNKNOWN gone for RESTORE_DELAY seconds -> overlay removed

Run:   python privacy_guard.py
Quit:  press Q in the camera window
       (if the overlay is covering everything: Ctrl+Shift+Q)
"""

import os
import sys
import time
import ctypes
import tkinter as tk

import cv2

# =====================================================================
# CONFIGURATION  (edit these)
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINER_PATH = os.path.join(BASE_DIR, "trainer", "trainer.yml")

# !!! IMPORTANT !!!
# MY_ID must be the numeric ID your face was saved/trained with in
# capture.py / train.py. If your face was captured as ID 2, set MY_ID = 2.
MY_ID = 1

# LBPH "confidence" is really a DISTANCE: LOWER = better match.
# A face counts as USER only if id == MY_ID and confidence < this value.
# Raise it if you are wrongly flagged as UNKNOWN, lower it if strangers
# are wrongly accepted as you.
CONFIDENCE_THRESHOLD = 70

ANOMALY_TIME = 2.0         # seconds an unknown face must stay before privacy mode
RESTORE_DELAY = 1.0        # seconds with no unknown face before overlay is removed
UNKNOWN_LOSS_GRACE = 0.4   # tolerate brief detection dropouts while the timer runs

CAMERA_INDEX = 0
MIN_FACE_SIZE = 60         # ignore faces smaller than this many pixels (filters noise)
EMERGENCY_EXIT_KEY = "Control-Shift-Q"   # works while the overlay is showing

# =====================================================================
# Constants
# =====================================================================
GREEN = (0, 200, 0)
RED = (0, 0, 255)
ORANGE = (0, 165, 255)
WHITE = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# =====================================================================
# Anomaly tracker (timer + restore logic, no camera / GUI code)
# =====================================================================
class AnomalyTracker:
    """Decides whether privacy mode should be ON."""

    def __init__(self):
        self.unknown_since = None      # when the current unknown streak began
        self.last_unknown_seen = None  # last frame an unknown face was seen
        self.clear_since = None        # when unknown faces disappeared (while ON)
        self.privacy_on = False

    def update(self, unknown_present, now):
        if unknown_present:
            self.last_unknown_seen = now
            self.clear_since = None
            if self.unknown_since is None:
                self.unknown_since = now                    # start timer
            if not self.privacy_on and now - self.unknown_since >= ANOMALY_TIME:
                self.privacy_on = True                      # anomaly confirmed
        else:
            # Timer running but unknown gone -> cancel (after a tiny grace period)
            if (not self.privacy_on and self.unknown_since is not None
                    and now - self.last_unknown_seen > UNKNOWN_LOSS_GRACE):
                self.unknown_since = None
            # Privacy ON and unknown gone -> restore after RESTORE_DELAY
            if self.privacy_on:
                if self.clear_since is None:
                    self.clear_since = now
                if now - self.clear_since >= RESTORE_DELAY:
                    self.privacy_on = False
                    self.unknown_since = None
                    self.clear_since = None
        return self.privacy_on

    @property
    def verifying(self):
        return self.unknown_since is not None and not self.privacy_on

    def elapsed(self, now):
        return 0.0 if self.unknown_since is None else now - self.unknown_since


# =====================================================================
# Fullscreen privacy overlay (tkinter)
# =====================================================================
class PrivacyOverlay:
    BG = "#0b0f1a"

    def __init__(self, root, on_emergency_exit):
        self.visible = False
        self._last_lift = 0.0

        self.win = tk.Toplevel(root)
        self.win.withdraw()
        self.win.overrideredirect(True)          # no title bar / borders
        self.win.configure(bg=self.BG)

        sw = self.win.winfo_screenwidth()
        sh = self.win.winfo_screenheight()
        self.win.geometry(f"{sw}x{sh}+0+0")
        self.win.attributes("-topmost", True)

        box = tk.Frame(self.win, bg=self.BG)
        box.place(relx=0.5, rely=0.5, anchor="center")

        # Negative font sizes = pixels, so it scales with screen height.
        tk.Label(box, text="\U0001F512", font=("Segoe UI Emoji", -(sh // 8)),
                 bg=self.BG, fg="#ffffff").pack()
        tk.Label(box, text="PRIVACY MODE", font=("Segoe UI", -(sh // 14), "bold"),
                 bg=self.BG, fg="#ffffff").pack(pady=(10, 20))
        tk.Label(box, text="  PRIVACY RISK DETECTED  ",
                 font=("Segoe UI", -(sh // 32), "bold"),
                 bg="#c62828", fg="#ffffff").pack(pady=(0, 30))
        tk.Label(box, text="Potential unauthorized viewer detected.",
                 font=("Segoe UI", -(sh // 28)), bg=self.BG, fg="#cfd8dc").pack()
        tk.Label(box, text="Your screen has been protected.",
                 font=("Segoe UI", -(sh // 28)), bg=self.BG, fg="#cfd8dc").pack(pady=(6, 0))

        # Emergency exit so you can never get locked out.
        for seq in (f"<{EMERGENCY_EXIT_KEY}>",
                    f"<{EMERGENCY_EXIT_KEY[:-1]}{EMERGENCY_EXIT_KEY[-1].lower()}>"):
            self.win.bind(seq, lambda e: on_emergency_exit())

    def show(self):
        if not self.visible:
            self.win.deiconify()
            self.win.attributes("-topmost", True)
            self.win.lift()
            self.win.focus_force()
            self.visible = True
            self._last_lift = time.time()
        else:
            # Re-assert topmost twice a second in case another window grabbed it.
            now = time.time()
            if now - self._last_lift > 0.5:
                self.win.attributes("-topmost", True)
                self.win.lift()
                self._last_lift = now

    def hide(self):
        if self.visible:
            self.win.withdraw()
            self.visible = False


# =====================================================================
# Helpers
# =====================================================================
def enable_dpi_awareness():
    """Stops Windows display scaling from leaving part of the screen uncovered."""
    if os.name != "nt":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def classify_face(recognizer, gray, box):
    """Same logic as your existing recognition code."""
    x, y, w, h = box
    face_roi = gray[y:y + h, x:x + w]
    face_id, confidence = recognizer.predict(face_roi)
    is_user = (face_id == MY_ID) and (confidence < CONFIDENCE_THRESHOLD)
    return is_user, confidence


def draw_ui(frame, people, tracker, now):
    h, w = frame.shape[:2]

    # Semi-transparent top banner
    banner = frame.copy()
    cv2.rectangle(banner, (0, 0), (w, 95), (20, 20, 20), -1)
    cv2.addWeighted(banner, 0.7, frame, 0.3, 0, dst=frame)

    if tracker.privacy_on:
        status, color = "PRIVACY RISK DETECTED", RED
    elif tracker.verifying:
        status, color = "UNKNOWN PERSON - VERIFYING", ORANGE
    else:
        status, color = "SAFE", GREEN

    cv2.putText(frame, "PrivAI - Screen Privacy Guard", (10, 25), FONT, 0.7, WHITE, 2)
    cv2.putText(frame, f"People Detected: {people}", (10, 52), FONT, 0.6, WHITE, 1)
    cv2.putText(frame, f"Status: {status}", (10, 80), FONT, 0.65, color, 2)

    pm_text = "ON" if tracker.privacy_on else "OFF"
    cv2.putText(frame, f"Privacy Mode: {pm_text}", (w - 230, 52), FONT, 0.6,
                RED if tracker.privacy_on else GREEN, 2)
    if tracker.verifying:
        cv2.putText(frame, f"Timer: {tracker.elapsed(now):.1f} / {ANOMALY_TIME:.1f}s",
                    (w - 230, 80), FONT, 0.6, ORANGE, 2)

    cv2.putText(frame, "Press Q to exit", (10, h - 12), FONT, 0.5, WHITE, 1)


# =====================================================================
# Main
# =====================================================================
def main():
    enable_dpi_awareness()

    if not os.path.isfile(TRAINER_PATH):
        print(f"[ERROR] Trainer file not found: {TRAINER_PATH}")
        sys.exit(1)

    # LBPH recognizer (needs opencv-contrib-python, which you already use)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_PATH)

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if detector.empty():
        print("[ERROR] Could not load Haar cascade.")
        sys.exit(1)

    cam = cv2.VideoCapture(CAMERA_INDEX)
    if not cam.isOpened():
        print(f"[ERROR] Cannot open camera index {CAMERA_INDEX}.")
        sys.exit(1)

    root = tk.Tk()
    root.withdraw()                      # we only need the overlay window
    state = {"quit": False}
    overlay = PrivacyOverlay(root, lambda: state.update(quit=True))
    tracker = AnomalyTracker()

    print("PrivAI running. Press Q in the camera window to exit.")
    try:
        while not state["quit"]:
            ok, frame = cam.read()
            if not ok:
                print("[ERROR] Failed to read from camera.")
                break

            now = time.time()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5,
                minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))

            unknown_present = False
            for (x, y, w, h) in faces:
                is_user, conf = classify_face(recognizer, gray, (x, y, w, h))
                if is_user:
                    label, color = "USER", GREEN
                else:
                    label, color = "UNKNOWN", RED
                    unknown_present = True
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} ({conf:.0f})", (x, max(y - 8, 12)),
                            FONT, 0.6, color, 2)

            # Timer / restore logic -> overlay on or off
            if tracker.update(unknown_present, now):
                overlay.show()
            else:
                overlay.hide()

            draw_ui(frame, len(faces), tracker, now)
            cv2.imshow("PrivAI - Screen Privacy Guard", frame)

            root.update()                # keep tkinter alive (no mainloop needed)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()
        try:
            root.destroy()
        except tk.TclError:
            pass


if __name__ == "__main__":
    main()