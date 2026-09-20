# 🔐 Screenshield — AI Screen Privacy Guard

> **Detect. Protect. Stay Private.**

Screenshield is a computer-vision-based desktop privacy system designed to protect a laptop screen from unauthorized viewing in public or shared environments such as cafes, libraries, offices, classrooms, and canteens.

The system uses a webcam to detect and recognize the authorized user. If the authorized user leaves the camera view or an unknown person remains in front of the laptop, Screenshield activates a fullscreen privacy overlay and can send a security notification through Telegram.

---

## 🎯 Problem

When working on a laptop in public places, sensitive information can be exposed through **shoulder surfing**.

For example:

- Someone approaches from behind.
- The laptop owner leaves their seat temporarily.
- An unknown person remains near the laptop.
- Private documents, code, emails, dashboards, or other information may become visible.

Traditional screen-lock mechanisms generally depend on manual action or fixed inactivity timers.

**Screenshield adds an additional computer-vision-based privacy layer.**

---

## 💡 Solution

Screenshield continuously analyzes the webcam feed locally.

The system:

1. Detects faces using **Haar Cascade**.
2. Recognizes the authorized user using **LBPH Face Recognition**.
3. Monitors whether the authorized user is present.
4. Detects unknown people.
5. Activates a fullscreen privacy overlay when a privacy risk is confirmed.
6. Protects the underlying screen from normal interaction while the overlay is active.
7. Restores the screen only after the authorized user is recognized.
8. Sends a **Telegram security alert** when an unknown person is detected while privacy protection is active.

---

# 🏗️ System Architecture

```text
                    ┌─────────────────┐
                    │     Webcam      │
                    └────────┬────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │  Haar Face Detector │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   LBPH Recognition  │
                  └──────────┬──────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
                 USER             UNKNOWN
                    │                 │
                    ▼                 ▼
               Safe Mode        Verification
                                      │
                                      ▼
                              Privacy Risk
                                      │
                           ┌──────────┴──────────┐
                           │                     │
                           ▼                     ▼
                  Fullscreen Overlay       Telegram Alert
                           │                     │
                           ▼                     ▼
                    Screen Protected         📱 Phone