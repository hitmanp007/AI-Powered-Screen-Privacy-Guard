"""
PrivAI - AI Screen Privacy Guard

Privacy layer on top of the existing LBPH face-recognition setup.
Flow:
    Webcam -> Haar face detection -> LBPH predict -> USER / UNKNOWN

    UNKNOWN stays for ANOMALY_TIME seconds        -> fullscreen privacy overlay
    No USER face for USER_ABSENCE_TIME seconds    -> fullscreen privacy overlay  (NEW)
    Overlay is removed ONLY when the authorized user is recognised
    (alone in frame) for RESTORE_DELAY seconds.

Run:   py -3.11 privacy_guard.py
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
RESTORE_DELAY = 1.0        # seconds the USER must be recognised (alone) before unlock
UNKNOWN_LOSS_GRACE = 0.4   # tolerate brief detection dropouts while the timer runs

# --- NEW: user-absence protection -----------------------------------
# Seconds without a recognised USER face before the screen is locked.
# Try 3.0 / 5.0 / 10.0 while testing.
USER_ABSENCE_TIME = 5.0
# While unlocking, a short gap (no face at all) up to this long does not
# reset the RESTORE_DELAY countdown. An UNKNOWN face always resets it.
USER_RETURN_GRACE = 0.5

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

# Tracker states
ACTIVE = "ACTIVE"                    # user present (or nothing wrong yet)
ABSENCE_PENDING = "ABSENCE_PENDING"  # no USER face, absence timer running
PRIVACY_LOCKED = "PRIVACY_LOCKED"    # overlay ON

# Why the screen is locked (only affects overlay text / status line)
REASON_NONE = None
REASON_UNKNOWN = "UNKNOWN_PERSON"
REASON_ABSENT = "USER_ABSENT"

OVERLAY_TEXT = {
    REASON_UNKNOWN: ("PRIVACY RISK DETECTED",
                    "Potential unauthorized viewer detected.",
                    "Your screen has been protected."),
    REASON_ABSENT: ("USER ABSENT - SCREEN LOCKED",
                    "The authorized user has left the camera view.",
                    "The screen unlocks when you return."),
}


# =====================================================================
# Anomaly tracker (timer + restore logic, no camera / GUI code)
# =====================================================================
class AnomalyTracker:
    """
    State machine that decides whether privacy mode should be ON.

      ACTIVE ------ no USER face --------------------> ABSENCE_PENDING
      ABSENCE_PENDING -- USER seen again ------------> ACTIVE
      ACTIVE / ABSENCE_PENDING -- UNKNOWN for ANOMALY_TIME ----> PRIVACY_LOCKED
      ACTIVE / ABSENCE_PENDING -- no USER for USER_ABSENCE_TIME -> PRIVACY_LOCKED
      PRIVACY_LOCKED -- USER recognised alone for RESTORE_DELAY -> ACTIVE

    Inputs each frame:
      user_present    - at least one face matched MY_ID under the threshold
      unknown_present - at least one face did NOT match
    An empty frame is (False, False): "no person", not "unknown person".
    """

    def __init__(self):
        self.state = ACTIVE
        self.lock_reason = REASON_NONE
        self.unknown_since = None       # start of the current unknown streak
        self.last_unknown_seen = None   # last frame an unknown face was seen
        self.absent_since = None        # NEW: start of the current "no USER" streak
        self.clear_since = None         # start of the unlock countdown (while locked)
        self.last_restore_ok = None     # last frame the unlock condition held

    # ------------------------------------------------------------------
    def update(self, user_present, unknown_present, now):
        # ---- 1. Unknown-person streak (existing logic) ----
        if unknown_present:
            self.last_unknown_seen = now
            if self.unknown_since is None:
                self.unknown_since = now                    # start 2 s timer
        elif (self.unknown_since is not None
              and now - self.last_unknown_seen > UNKNOWN_LOSS_GRACE):
            self.unknown_since = None                       # they left: cancel

        # ---- 2. NEW: absence bookkeeping ----
        # The absence timer runs whenever no recognised USER face is in frame
        # (empty frame OR only strangers). ANY frame with the user resets it,
        # so one missed frame can never trigger the lock, and a stranger
        # walking past can never reset it.
        if user_present:
            self.absent_since = None
        elif self.absent_since is None:
            self.absent_since = now

        if self.state != PRIVACY_LOCKED:
            self._update_unlocked(unknown_present, now)
        else:
            self._update_locked(user_present, unknown_present, now)
        return self.privacy_on

    # ------------------------------------------------------------------
    def _update_unlocked(self, unknown_present, now):
        if unknown_present and now - self.unknown_since >= ANOMALY_TIME:
            self._lock(REASON_UNKNOWN)                      # existing rule
        elif (self.absent_since is not None
              and now - self.absent_since >= USER_ABSENCE_TIME):
            self._lock(REASON_ABSENT)                       # NEW rule
        elif self.absent_since is not None:
            self.state = ABSENCE_PENDING
        else:
            self.state = ACTIVE

    def _update_locked(self, user_present, unknown_present, now):
        # A stranger in view: stay locked, cancel any unlock countdown.
        # (Seeing a face is never enough to unlock - it must be the USER.)
        if unknown_present:
            self.lock_reason = REASON_UNKNOWN
            self.clear_since = None
            self.last_restore_ok = None
            return

        if user_present:
            # Authorized user recognised and nobody else in frame:
            # unlock only after RESTORE_DELAY of continuous recognition.
            self.last_restore_ok = now
            if self.clear_since is None:
                self.clear_since = now
            if now - self.clear_since >= RESTORE_DELAY:
                self._unlock()
        elif (self.clear_since is not None
              and now - self.last_restore_ok > USER_RETURN_GRACE):
            self.clear_since = None                         # gap too long: restart

    def _lock(self, reason):
        self.state = PRIVACY_LOCKED
        self.lock_reason = reason
        self.clear_since = None
        self.last_restore_ok = None

    def _unlock(self):
        self.state = ACTIVE
        self.lock_reason = REASON_NONE
        self.unknown_since = None
        self.absent_since = None
        self.clear_since = None
        self.last_restore_ok = None

    # ------------------------------------------------------------------
    @property
    def privacy_on(self):
        return self.state == PRIVACY_LOCKED

    @property
    def verifying(self):
        return self.state != PRIVACY_LOCKED and self.unknown_since is not None

    def elapsed(self, now):
        """Seconds of the current unknown-person streak."""
        return 0.0 if self.unknown_since is None else now - self.unknown_since

    def absence_remaining(self, now):
        """Seconds left before the absence lock fires."""
        if self.absent_since is None:
            return USER_ABSENCE_TIME
        return max(0.0, USER_ABSENCE_TIME - (now - self.absent_since))

    def restore_progress(self, now):
        """Seconds of the unlock countdown, or None if it is not running."""
        return None if self.clear_since is None else now - self.clear_since


# =====================================================================
# Fullscreen privacy overlay (tkinter)
# =====================================================================
class PrivacyOverlay:
    BG = "#0b0f1a"

    def __init__(self, root, on_emergency_exit):
        self.visible = False
        self.reason = None
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
        # These three labels change text depending on WHY the screen is locked.
        self.badge = tk.Label(box, text="", font=("Segoe UI", -(sh // 32), "bold"),
                              bg="#c62828", fg="#ffffff")
        self.badge.pack(pady=(0, 30))
        self.msg1 = tk.Label(box, text="", font=("Segoe UI", -(sh // 28)),
                             bg=self.BG, fg="#cfd8dc")
        self.msg1.pack()
        self.msg2 = tk.Label(box, text="", font=("Segoe UI", -(sh // 28)),
                             bg=self.BG, fg="#cfd8dc")
        self.msg2.pack(pady=(6, 0))
        self._set_reason(REASON_UNKNOWN)

        # Emergency exit so you can never get locked out.
        for seq in (f"<{EMERGENCY_EXIT_KEY}>",
                    f"<{EMERGENCY_EXIT_KEY[:-1]}{EMERGENCY_EXIT_KEY[-1].lower()}>"):
            self.win.bind(seq, lambda e: on_emergency_exit())

    def _set_reason(self, reason):
        if reason == self.reason:
            return
        badge, line1, line2 = OVERLAY_TEXT.get(reason, OVERLAY_TEXT[REASON_UNKNOWN])
        self.badge.config(text=f"  {badge}  ")
        self.msg1.config(text=line1)
        self.msg2.config(text=line2)
        self.reason = reason

    def show(self, reason=REASON_UNKNOWN):
        self._set_reason(reason)
        if not self.visible:
            self.win.deiconify()
            self.win.attributes("-topmost", True)
            self.win.lift()
            self.win.focus_force()
            self.visible = True
            self._last_lift = time.time()
        else:
            # Twice a second: stay on top AND keep keyboard focus, so typing
            # can't reach the application hidden underneath.
            now = time.time()
            if now - self._last_lift > 0.5:
                self.win.attributes("-topmost", True)
                self.win.lift()
                self.win.focus_force()
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

    right_line = None   # (text, color) for the countdown shown on the right
    if tracker.privacy_on:
        if tracker.lock_reason == REASON_ABSENT:
            status, color = "LOCKED - USER ABSENT", RED
        else:
            status, color = "PRIVACY RISK DETECTED", RED
        progress = tracker.restore_progress(now)
        if progress is not None:
            right_line = (f"Unlocking: {progress:.1f} / {RESTORE_DELAY:.1f}s", GREEN)
    elif tracker.verifying:
        status, color = "UNKNOWN PERSON - VERIFYING", ORANGE
        right_line = (f"Timer: {tracker.elapsed(now):.1f} / {ANOMALY_TIME:.1f}s", ORANGE)
    elif tracker.state == ABSENCE_PENDING:
        status, color = "USER ABSENT - LOCKING SOON", ORANGE
        right_line = (f"Lock in: {tracker.absence_remaining(now):.1f}s", ORANGE)
    else:
        status, color = "SAFE", GREEN

    cv2.putText(frame, "PrivAI - Screen Privacy Guard", (10, 25), FONT, 0.7, WHITE, 2)
    cv2.putText(frame, f"People Detected: {people}", (10, 52), FONT, 0.6, WHITE, 1)
    cv2.putText(frame, f"Status: {status}", (10, 80), FONT, 0.6, color, 2)

    pm_text = "ON" if tracker.privacy_on else "OFF"
    cv2.putText(frame, f"Privacy Mode: {pm_text}", (w - 230, 52), FONT, 0.6,
                RED if tracker.privacy_on else GREEN, 2)
    if right_line:
        cv2.putText(frame, right_line[0], (w - 230, 80), FONT, 0.55, right_line[1], 2)

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

            user_present = False         # NEW: was the authorized user recognised?
            unknown_present = False
            for (x, y, w, h) in faces:
                is_user, conf = classify_face(recognizer, gray, (x, y, w, h))
                if is_user:
                    label, color = "USER", GREEN
                    user_present = True
                else:
                    label, color = "UNKNOWN", RED
                    unknown_present = True
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} ({conf:.0f})", (x, max(y - 8, 12)),
                            FONT, 0.6, color, 2)

            # Empty frame => (False, False): "no person", handled by the absence
            # timer. Stranger => unknown_present: handled by the anomaly timer.
            if tracker.update(user_present, unknown_present, now):
                overlay.show(tracker.lock_reason)
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