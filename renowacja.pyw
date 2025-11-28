# -*- coding: utf-8 -*-
from __future__ import annotations
"""
EPS AI SOLUTIONS – Renowacja starych zdjęć
Wersja v11: Gemini + ChatGPT (OpenAI) z presetem EPS AI Restoration Pipeline (v11),
obsługą ORIENTACJI (UP/DOWN/LEFT/RIGHT) oraz POST-PROCESSINGIEM cienkich ramek.
"""

import ctypes
import sys

# --- Ukrycie konsoli (gdy uruchamiane jako python.exe/pythonw.exe) ---
try:
    hwnd = ctypes.windll.kernel32.GetConsoleWindow()
    if hwnd:
        ctypes.windll.user32.ShowWindow(hwnd, 0)
except Exception:
    pass

# --- Ikona + DPI (Windows) ---
try:
    myappid = "eps.ai.solutions.renowacja.v11"
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

# --- Importy ogólne ---
import os
import io
import time
import json
import base64
import random
import threading
import concurrent.futures
import re
from pathlib import Path

import urllib3
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops

import customtkinter as ctk
from tkinterdnd2 import DND_FILES, TkinterDnD
from tkinter import filedialog, messagebox

# .env (opcjonalnie)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Gemini SDK
try:
    import google.generativeai as genai
except ImportError:
    genai = None

# OpenAI SDK (ChatGPT)
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- CustomTkinter ---
ctk.set_appearance_mode("system")
ctk.set_default_color_theme("blue")

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config.json"

# =======================
# WYGLĄD – jasnoniebieski
# =======================

COLOR_ACCENT = "#0EA5E9"          # jasny niebieski (sky-500)
COLOR_ACCENT_HOVER = "#0284C7"    # ciemniejszy niebieski (sky-600)
COLOR_BG_MAIN_LIGHT = "#F0F9FF"   # bardzo jasny niebieski (sky-50)
COLOR_BG_MAIN_DARK = "#020617"    # ciemne tło w trybie dark
COLOR_PANEL_LIGHT = "#E0F2FE"     # panel (sky-100)
COLOR_PANEL_DARK = "#020617"
SIDEBAR_BG_LIGHT = "#DBEAFE"      # lekko niebieski (indigo-100)
SIDEBAR_BG_DARK = "#020617"
COLOR_TEXT_MAIN = "#0F172A"       # slate-900
COLOR_TEXT_MUTED = "#64748B"      # slate-500


# =======================
# KONFIGURACJA + PRESETY
# =======================

DEFAULT_CONFIG = {
    # Dostawcy
    "layout_provider": "gemini",
    "restore_provider": "gemini",

    # MODELE – rekomendowane pod EPS AI Restoration Pipeline (v11)
    "layout_model_gemini": "gemini-3-pro-preview",
    "restore_model_gemini": "gemini-3-pro-image-preview",

    # Domyślne dla OpenAI (fallback / analiza)
    "layout_model_openai": "gpt-4.1-mini",
    "restore_model_openai": "gpt-4.1-mini",

    # Parametry generowania – core renowacji
    # Analiza układu i orientacja zawsze wymuszają temperature=0.0
    "temperature": 0.15,          # renowacja – niska, żeby nie puchły twarze
    "top_p": 0.65,                # stabilność wg EPS AI v11
    "max_output_tokens": 2048,
    "max_workers": 5,
    "verify_ssl": False,

    # Rozmiary obrazów
    "max_dim_layout": 1500,
    "max_dim_restore": 3000,

    # Zachowanie plików
    "save_maps": True,
    "overwrite_existing": False,
    "auto_open_output": False,
}

PRESETS = {
    "Domyślne (oryginał)": {
        "layout_provider": "gemini",
        "restore_provider": "gemini",

        "layout_model_gemini": "gemini-3-pro-preview",
        "restore_model_gemini": "gemini-3-pro-image-preview",

        "layout_model_openai": "gpt-4.1-mini",
        "restore_model_openai": "gpt-4.1-mini",

        "temperature": 0.15,
        "top_p": 0.65,
        "max_output_tokens": 2048,
        "max_workers": 5,
        "verify_ssl": False,

        "max_dim_layout": 1500,
        "max_dim_restore": 3000,

        "save_maps": True,
        "overwrite_existing": False,
        "auto_open_output": False,
    },
    "Szybka analiza": {
        "layout_provider": "gemini",
        "restore_provider": "gemini",
        "layout_model_gemini": "gemini-1.5-flash-latest",
        "restore_model_gemini": "gemini-1.5-flash-latest",
        "temperature": 0.25,
        "top_p": 0.9,
        "max_output_tokens": 1024,
        "max_workers": 4,
        "verify_ssl": False,
        "max_dim_layout": 1200,
        "max_dim_restore": 2200,
        "save_maps": False,
        "overwrite_existing": False,
        "auto_open_output": False,
    },
    "Maks. jakość": {
        "layout_provider": "gemini",
        "restore_provider": "gemini",
        "layout_model_gemini": "gemini-3-pro-preview",
        "restore_model_gemini": "gemini-3-pro-image-preview",
        "temperature": 0.15,
        "top_p": 0.65,
        "max_output_tokens": 3072,
        "max_workers": 5,
        "verify_ssl": False,
        "max_dim_layout": 1800,
        "max_dim_restore": 4000,
        "save_maps": True,
        "overwrite_existing": False,
        "auto_open_output": False,
    },
}
PRESET_NAMES = list(PRESETS.keys())


class AppConfig:
    def __init__(self, path: Path = CONFIG_PATH, defaults: dict | None = None):
        self.path = path
        self.defaults = defaults or DEFAULT_CONFIG
        self.data = self.defaults.copy()
        self.load()

    def load(self):
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            if isinstance(user_cfg, dict):
                self.data.update(user_cfg)
        except Exception as exc:
            print(f"[CONFIG] Nie udało się odczytać config.json: {exc}", flush=True)

    def save(self):
        try:
            with self.path.open("w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            print("[CONFIG] Zapisano config.json", flush=True)
        except Exception as exc:
            print(f"[CONFIG] Nie udało się zapisać config.json: {exc}", flush=True)


# =======================
# SILNIK – GEMINI + GPT
# =======================

class RestorationEngine:
    def __init__(self, log_callback=None, stop_event: threading.Event | None = None,
                 config: dict | None = None):
        self.log_callback = log_callback
        self.stop_event = stop_event or threading.Event()
        self.config = config or DEFAULT_CONFIG.copy()

        self.layout_provider = self.config.get("layout_provider", "gemini")
        self.restore_provider = self.config.get("restore_provider", "gemini")

        self.layout_model_gemini = self.config.get("layout_model_gemini", DEFAULT_CONFIG["layout_model_gemini"])
        self.restore_model_gemini = self.config.get("restore_model_gemini", DEFAULT_CONFIG["restore_model_gemini"])
        self.layout_model_openai = self.config.get("layout_model_openai", DEFAULT_CONFIG["layout_model_openai"])
        self.restore_model_openai = self.config.get("restore_model_openai", DEFAULT_CONFIG["restore_model_openai"])

        self.temperature = float(self.config.get("temperature", DEFAULT_CONFIG["temperature"]))
        self.top_p = float(self.config.get("top_p", DEFAULT_CONFIG.get("top_p", 0.65)))
        self.max_output_tokens = int(self.config.get("max_output_tokens", DEFAULT_CONFIG["max_output_tokens"]))
        self.verify_ssl = bool(self.config.get("verify_ssl", DEFAULT_CONFIG["verify_ssl"]))
        self.max_dim_layout = int(self.config.get("max_dim_layout", DEFAULT_CONFIG["max_dim_layout"]))
        self.max_dim_restore = int(self.config.get("max_dim_restore", DEFAULT_CONFIG["max_dim_restore"]))

        cfg_workers = int(self.config.get("max_workers", DEFAULT_CONFIG["max_workers"]))
        self.max_workers = max(1, min(cfg_workers, (os.cpu_count() or 4)))

        self.gemini_api_key = os.environ.get("Google_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")

        if genai and self.gemini_api_key:
            try:
                genai.configure(api_key=self.gemini_api_key)
            except Exception as exc:
                self._log(f"⚠️ Nie udało się skonfigurować Gemini SDK: {exc}")

        if OpenAI and self.openai_api_key:
            try:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
            except Exception as exc:
                self.openai_client = None
                self._log(f"⚠️ Nie udało się utworzyć klienta OpenAI: {exc}")
        else:
            self.openai_client = None

    def apply_config(self, config: dict):
        self.config = config
        self.layout_provider = self.config.get("layout_provider", "gemini")
        self.restore_provider = self.config.get("restore_provider", "gemini")

        self.layout_model_gemini = self.config.get("layout_model_gemini", DEFAULT_CONFIG["layout_model_gemini"])
        self.restore_model_gemini = self.config.get("restore_model_gemini", DEFAULT_CONFIG["restore_model_gemini"])
        self.layout_model_openai = self.config.get("layout_model_openai", DEFAULT_CONFIG["layout_model_openai"])
        self.restore_model_openai = self.config.get("restore_model_openai", DEFAULT_CONFIG["restore_model_openai"])

        self.temperature = float(self.config.get("temperature", DEFAULT_CONFIG["temperature"]))
        self.top_p = float(self.config.get("top_p", DEFAULT_CONFIG.get("top_p", 0.65)))
        self.max_output_tokens = int(self.config.get("max_output_tokens", DEFAULT_CONFIG["max_output_tokens"]))
        self.verify_ssl = bool(self.config.get("verify_ssl", DEFAULT_CONFIG["verify_ssl"]))
        self.max_dim_layout = int(self.config.get("max_dim_layout", DEFAULT_CONFIG["max_dim_layout"]))
        self.max_dim_restore = int(self.config.get("max_dim_restore", DEFAULT_CONFIG["max_dim_restore"]))

        cfg_workers = int(self.config.get("max_workers", DEFAULT_CONFIG["max_workers"]))
        self.max_workers = max(1, min(cfg_workers, (os.cpu_count() or 4)))

    # --- UTIL ---

    def _log(self, msg: str):
        if self.log_callback:
            self.log_callback(msg)

    def encode_image(self, path: str, max_dim: int) -> str | None:
        """Skaluje obraz do max_dim i koduje jako JPEG base64 (jakość 92)."""
        try:
            img = Image.open(path).convert("RGB")
            w, h = img.size
            scale = min(1.0, max_dim / max(w, h))
            if scale < 1.0:
                img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=92)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as exc:
            self._log(f"❌ Nie udało się zakodować obrazu: {exc}")
            return None

    def make_request(self, url: str, payload: dict, headers: dict):
        max_attempts = 8
        for attempt in range(1, max_attempts + 1):
            if self.stop_event.is_set():
                return None
            try:
                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=100,
                    verify=self.verify_ssl,
                )
                status = response.status_code
                if status == 200:
                    return response
                if status in (429, 500, 502, 503):
                    self._log(f"⚠️ Serwer zajęty ({status}) – próba {attempt}/{max_attempts}...")
                    time.sleep(random.uniform(2, 5))
                    continue
                self._log(f"❌ Błąd API ({status}): {response.text[:200]}")
                return None
            except requests.RequestException as exc:
                self._log(f"❌ Błąd sieci: {exc} – próba {attempt}/{max_attempts}")
                time.sleep(1)
            except Exception as exc:
                self._log(f"❌ Nieoczekiwany błąd zapytania: {exc}")
                time.sleep(1)
        self._log("❌ Limit prób API wyczerpany.")
        return None

    # --- DETEKCJA UKŁADU ---

    def detect_layout(self, path: str):
        if self.layout_provider == "openai":
            return self._detect_layout_openai(path)
        else:
            return self._detect_layout_gemini(path)

    def _detect_layout_gemini(self, path: str):
        if not self.gemini_api_key:
            self._log("❌ Brak klucza Gemini – pomijam analizę układu.")
            return None

        model_id = self.layout_model_gemini
        self._log(f"🔎 [Gemini] Analiza układu ({model_id}): {os.path.basename(path)}")
        b64 = self.encode_image(path, self.max_dim_layout)
        if not b64:
            return None

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={self.gemini_api_key}"
        prompt = (
            "Analyze the scanned image. Your task is to detect the EXACT 4 CORNERS of every single photograph "
            "placed on the scanner glass. Ignore paper borders or background scanner noise.\n"
            "Return ONLY valid JSON with structure:\n"
            "{ \"photos\": [ { \"corners\": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] }, ... ] }\n"
            "Coordinates MUST be normalized from 0 to 1000 in both axes (x,y), "
            "corners ordered clockwise starting from top-left."
        )
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"inlineData": {"mimeType": "image/jpeg", "data": b64}},
                    ]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.0,
            },
        }

        r = self.make_request(url, payload, {"Content-Type": "application/json"})
        if not (r and r.status_code == 200):
            return None
        try:
            cand = r.json()["candidates"][0]
            txt = cand["content"]["parts"][0]["text"]
            cleaned = re.sub(r"```json|```", "", txt).strip()
            data = json.loads(cleaned)
            return data
        except Exception as exc:
            self._log(f"❌ Błąd parsowania JSON z detekcji: {exc}")
            return None

    def _detect_layout_openai(self, path: str):
        if not self.openai_client or not self.openai_api_key:
            self._log("❌ Brak klucza OpenAI – nie mogę użyć ChatGPT do analizy układu.")
            return None
        model_id = self.layout_model_openai
        self._log(f"🔎 [OpenAI] Analiza układu ({model_id}): {os.path.basename(path)}")

        b64 = self.encode_image(path, self.max_dim_layout)
        if not b64:
            return None
        image_url = f"data:image/jpeg;base64,{b64}"

        prompt = (
            "You are a vision model extracting layout of old photographs on a scanned page.\n"
            "Task: detect rectangular photos, and return ONLY valid JSON with structure:\n"
            "{ \"photos\": [ { \"corners\": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] }, ... ] }\n"
            "Coordinates MUST be normalized from 0 to 1000 in both axes (x,y), "
            "corners ordered clockwise starting from top-left.\n"
            "No extra keys, no comments, no text outside JSON."
        )
        try:
            resp = self.openai_client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                max_tokens=self.max_output_tokens,
                temperature=0.0,
            )
        except Exception as exc:
            self._log(f"❌ Błąd zapytania do OpenAI: {exc}")
            return None

        try:
            txt = resp.choices[0].message.content
            cleaned = re.sub(r"```json|```", "", txt).strip()
            data = json.loads(cleaned)
            return data
        except Exception as exc:
            self._log(f"❌ Błąd parsowania JSON z OpenAI: {exc}")
            return None

    # --- DETEKCJA ORIENTACJI ---

    def detect_orientation(self, path: str) -> str | None:
        """Zwraca 'UP', 'DOWN', 'LEFT' lub 'RIGHT' albo None przy błędzie."""
        if self.layout_provider == "openai":
            return self._detect_orientation_openai(path)
        else:
            return self._detect_orientation_gemini(path)

    def _detect_orientation_gemini(self, path: str) -> str | None:
        if not self.gemini_api_key:
            self._log("❌ Brak klucza Gemini – pomijam analizę orientacji.")
            return None

        model_id = self.layout_model_gemini
        self._log(f"🧭 [Gemini] Analiza orientacji ({model_id}): {os.path.basename(path)}")
        b64 = self.encode_image(path, self.max_dim_layout)
        if not b64:
            return None

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={self.gemini_api_key}"
        prompt = (
            "Look at this photograph. Based on people, text, faces and gravity, determine where the TOP of the image "
            "is currently pointing.\n"
            "Options: 'UP', 'DOWN', 'LEFT', 'RIGHT'.\n"
            "Return ONLY valid JSON: {\"direction\": \"UP\"} with one of these four exact values.\n"
            "If unsure, choose the most likely orientation."
        )
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"inlineData": {"mimeType": "image/jpeg", "data": b64}},
                    ]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.0,
            },
        }

        r = self.make_request(url, payload, {"Content-Type": "application/json"})
        if not (r and r.status_code == 200):
            return None
        try:
            cand = r.json()["candidates"][0]
            txt = cand["content"]["parts"][0]["text"]
            cleaned = re.sub(r"```json|```", "", txt).strip()
            data = json.loads(cleaned)
            direction = (data.get("direction") or "").strip().upper()
            if direction in ("UP", "DOWN", "LEFT", "RIGHT"):
                return direction
            self._log(f"⚠️ Niepoprawny kierunek z orientacji: {direction!r}")
            return None
        except Exception as exc:
            self._log(f"❌ Błąd parsowania JSON z orientacji: {exc}")
            return None

    def _detect_orientation_openai(self, path: str) -> str | None:
        if not self.openai_client or not self.openai_api_key:
            self._log("❌ Brak klucza OpenAI – pomijam analizę orientacji.")
            return None
        model_id = self.layout_model_openai
        self._log(f"🧭 [OpenAI] Analiza orientacji ({model_id}): {os.path.basename(path)}")

        b64 = self.encode_image(path, self.max_dim_layout)
        if not b64:
            return None
        image_url = f"data:image/jpeg;base64,{b64}"

        prompt = (
            "You are an image orientation detector.\n"
            "Look at the people, text and gravity in this image and decide in which direction "
            "the TOP of the image is currently pointing.\n"
            "Options: 'UP', 'DOWN', 'LEFT', 'RIGHT'.\n"
            "Return ONLY JSON: {\"direction\": \"UP\"} with one of these four values."
        )
        try:
            resp = self.openai_client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                max_tokens=32,
                temperature=0.0,
            )
        except Exception as exc:
            self._log(f"❌ Błąd zapytania do OpenAI (orientacja): {exc}")
            return None

        try:
            txt = resp.choices[0].message.content
            cleaned = re.sub(r"```json|```", "", txt).strip()
            data = json.loads(cleaned)
            direction = (data.get("direction") or "").strip().upper()
            if direction in ("UP", "DOWN", "LEFT", "RIGHT"):
                return direction
            self._log(f"⚠️ Niepoprawny kierunek z orientacji (OpenAI): {direction!r}")
            return None
        except Exception as exc:
            self._log(f"❌ Błąd parsowania JSON z OpenAI (orientacja): {exc}")
            return None

    # --- RENOWACJA ---

    def restore_photo(self, path_src: str, save_path: str) -> bool:
        if self.restore_provider == "openai":
            self._log(
                "⚠️ Renowacja obrazów przez OpenAI (ChatGPT) nie jest w tej wersji "
                "zaimplementowana – używam nadal Gemini do generowania obrazu."
            )
        return self._restore_photo_gemini(path_src, save_path)

    def _restore_photo_gemini(self, path_src: str, save_path: str) -> bool:
        """Renowacja zdjęcia przy użyciu modelu obrazowego Gemini – EPS AI v11."""
        if not self.gemini_api_key:
            self._log("❌ Brak klucza Gemini – pomijam renowację.")
            return False

        model_id = self.restore_model_gemini
        self._log(f"✨ [Gemini] Renowacja ({model_id}): {os.path.basename(save_path)}")

        b64 = self.encode_image(path_src, self.max_dim_restore)
        if not b64:
            return False

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_id}:generateContent?key={self.gemini_api_key}"
        )

        # Master Prompt – EPS AI Restoration Pipeline (v11)
        prompt = (
            "You are an Expert Forensic Photo Restorer.\n"
            "\n"
            "Goal: Transform this scan into a high-quality modern studio photograph "
            "while keeping 100% identity fidelity.\n"
            "\n"
            "RULE 1 – 10% Inpainting Perimeter:\n"
            "- Treat the outer 10% perimeter of the image as potentially damaged or missing.\n"
            "- Generatively reconstruct these edges based on the inner scene context so that the photo "
            "fills the frame perfectly with no white borders, no scanner glass, and no hard edges.\n"
            "\n"
            "RULE 2 – Identity Lock (faces):\n"
            "- STRICT: Do not change facial features (eyes, nose, mouth shape, bone structure).\n"
            "- Do not add makeup, do not beautify, do not change age or expression.\n"
            "- Use existing pixels as absolute ground truth for identity.\n"
            "- Skin texture must remain realistic (visible pores, natural texture), not plastic or over-smoothed.\n"
            "\n"
            "RULE 3 – Studio HDR Look:\n"
            "- Apply subtle HDR-style contrast and color.\n"
            "- Recover highlights and shadows while keeping the image natural.\n"
            "- Neutralize harsh flash glare on faces and skin (matte finish, no oily hotspots).\n"
            "- Reduce ISO noise and grain without destroying fine details like hair and fabric texture.\n"
            "\n"
            "RULE 4 – Physical Damage Removal:\n"
            "- Identify and remove scratches, dust, folds, stains, and scanner dirt.\n"
            "- Repair torn or missing areas in a way that is consistent with the scene.\n"
            "\n"
            "RULE 5 – Composition and Content:\n"
            "- Do not change the number of people, their pose, clothing, background elements, or camera angle.\n"
            "- Do not invent new objects or patterns that are not implied by the existing pixels.\n"
            "\n"
            "Output: Return a single high-quality restored version of the photo that respects all rules above."
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"inlineData": {"mimeType": "image/jpeg", "data": b64}},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "topP": self.top_p,
            },
        }

        r = self.make_request(url, payload, {"Content-Type": "application/json"})
        if not (r and r.status_code == 200):
            return False

        # --- Solidny parser odpowiedzi ---
        try:
            data = r.json()
        except Exception as exc:
            self._log(f"❌ Błąd parsowania JSON z Gemini: {exc}")
            return False

        def find_inline(obj):
            """Rekurencyjnie szuka pierwszego inlineData z polem 'data' w kandydatach."""
            if isinstance(obj, dict):
                if "inlineData" in obj and isinstance(obj["inlineData"], dict) and "data" in obj["inlineData"]:
                    return obj["inlineData"]
                for v in obj.values():
                    res = find_inline(v)
                    if res is not None:
                        return res
            elif isinstance(obj, list):
                for item in obj:
                    res = find_inline(item)
                    if res is not None:
                        return res
            return None

        candidates = data.get("candidates") or []
        inline = find_inline(candidates)

        if not inline:
            # spróbujmy odczytać chociaż tekst (często powód: obraz czarny itp.)
            texts = []

            def collect_text(obj):
                if isinstance(obj, dict):
                    if isinstance(obj.get("text"), str):
                        texts.append(obj["text"])
                    for v in obj.values():
                        collect_text(v)
                elif isinstance(obj, list):
                    for it in obj:
                        collect_text(it)

            collect_text(candidates)
            reason = None
            pf = data.get("promptFeedback")
            if isinstance(pf, dict):
                reason = pf.get("blockReason")

            if reason:
                self._log(f"⚠️ Gemini nie zwrócił obrazu (blockReason={reason}).")
            if texts:
                preview = texts[0].replace("\n", " ")
                if len(preview) > 200:
                    preview = preview[:200] + "…"
                self._log(f"⚠️ Odpowiedź tekstowa modelu (skrót): {preview}")
            else:
                self._log("❌ Brak inlineData w odpowiedzi Gemini – nie zwrócono obrazu.")
            return False

        try:
            img_bytes = base64.b64decode(inline["data"])
            with Image.open(io.BytesIO(img_bytes)) as img:
                img.save(save_path)
            return True
        except Exception as exc:
            snippet = ""
            try:
                snippet = str(data)[:400]
            except Exception:
                pass
            self._log(f"❌ Błąd dekodowania obrazu wyjściowego: {exc}")
            if snippet:
                self._log(f"ℹ️ Fragment surowej odpowiedzi Gemini: {snippet}")
            return False

    # --- CROP ---

    def crop_image(self, img, corners):
        """Perspektywiczne wycięcie fragmentu z już otwartego obrazu."""
        try:
            self._log("✂️ Wycinanie...")
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")

            pts = np.array(corners, dtype="float32")
            s = pts.sum(axis=1)
            diff = np.diff(pts, axis=1)

            tl = pts[np.argmin(s)]
            br = pts[np.argmax(s)]
            tr = pts[np.argmin(diff)]
            bl = pts[np.argmax(diff)]

            w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
            h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

            target = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
            source = np.array([tl, tr, br, bl], dtype="float32")

            matrix = []
            for i in range(4):
                x, y = source[i]
                u, v = target[i]
                matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
                matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])

            A = np.matrix(matrix, dtype=float)
            B = np.array(target).reshape(8)
            res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
            res = np.array(res).reshape(8)

            return img.transform((w, h), Image.PERSPECTIVE, res, Image.BICUBIC)
        except Exception as exc:
            self._log(f"❌ Błąd wycinania: {exc}")
            return None

    # --- POST-PROCESSING (ramki + wyostrzanie) ---

    def postprocess_restored_image(self, path: str, apply_unsharp: bool = True) -> bool:
        """Usuwa cienkie ramki (białe/czarne) i lekko wyostrza obraz."""
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            self._log(f"❌ Nie udało się otworzyć obrazu do post-processingu: {exc}")
            return False

        w, h = img.size
        if w < 4 or h < 4:
            return False

        # kolor tła z rogu – zakładamy, że ramka ma kolor podobny
        bg_color = img.getpixel((0, 0))
        bg = Image.new("RGB", (w, h), bg_color)
        diff = ImageChops.difference(img, bg)
        bbox = diff.getbbox()

        if bbox:
            if bbox != (0, 0, w, h):
                try:
                    img = img.crop(bbox)
                    self._log(f"📐 Post-processing: przycięto ramkę do bbox={bbox}.")
                except Exception as exc:
                    self._log(f"⚠️ Błąd przycinania ramki: {exc}")
        else:
            self._log("ℹ️ Post-processing: nie wykryto ramki do przycięcia.")

        if apply_unsharp:
            try:
                img = img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=110, threshold=3))
                self._log("🔍 Post-processing: zastosowano lekkie wyostrzanie (UnsharpMask).")
            except Exception as exc:
                self._log(f"⚠️ Nie udało się zastosować wyostrzania: {exc}")

        try:
            img.save(path)
            return True
        except Exception as exc:
            self._log(f"❌ Nie udało się zapisać obrazu po post-processingu: {exc}")
            return False


# ===============
# POMOCNICZE
# ===============

def is_image_file(path: str) -> bool:
    return path.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"))


# ===============
# GUI
# ===============

class MainApp(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self):
        ctk.CTk.__init__(self)
        TkinterDnD.DnDWrapper.__init__(self)
        self.TkdndVersion = TkinterDnD._require(self)

        self.title("EPS AI SOLUTIONS – Renowacja starych zdjęć (v11)")
        self.geometry("1150x720")
        self.minsize(980, 640)

        self.config_manager = AppConfig()
        self.stop_event = threading.Event()
        self.engine = RestorationEngine(
            log_callback=self.log_safe,
            stop_event=self.stop_event,
            config=self.config_manager.data,
        )

        cfg = self.config_manager.data
        self.save_maps = bool(cfg.get("save_maps", DEFAULT_CONFIG["save_maps"]))
        self.overwrite_existing = bool(cfg.get("overwrite_existing", DEFAULT_CONFIG["overwrite_existing"]))
        self.auto_open_output = bool(cfg.get("auto_open_output", DEFAULT_CONFIG["auto_open_output"]))

        self.input_paths: list[str] = []
        self.output_dir = str(APP_DIR / "odnowione_eps")
        self.is_running = False
        self.total_files = 0
        self.processed_files = 0

        self.raw_bg_image = None
        self.bg_label = None

        # LISTY MODELI – aktualizacja przy starcie
        self.gemini_models: list[str] = self._load_gemini_models()
        self.openai_models: list[str] = self._load_openai_models()

        self._setup_background()
        self._build_ui()
        self._update_api_indicator()
        self._update_models_label()
        self._center_window()

    # --- TŁO ---

    def _setup_background(self):
        bg_candidates = [
            APP_DIR / "tlo.png",
            APP_DIR / "tlo.jpg",
            APP_DIR / "background.png",
            APP_DIR / "background.jpg",
        ]
        bg_path = None
        for p in bg_candidates:
            if p.exists():
                bg_path = p
                break

        if bg_path:
            try:
                self.raw_bg_image = Image.open(bg_path)
                self.bg_label = ctk.CTkLabel(self, text="")
                self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
                self.bg_label.lower()
                self.bind("<Configure>", self._resize_bg)
            except Exception:
                self.raw_bg_image = None
        else:
            self.configure(fg_color=COLOR_BG_MAIN_LIGHT)

    def _resize_bg(self, event):
        if event.widget == self and self.raw_bg_image:
            w, h = event.width, event.height
            if w > 100 and h > 100:
                i = self.raw_bg_image.resize((w, h), Image.Resampling.LANCZOS)
                ci = ctk.CTkImage(i, size=(w, h))
                self.bg_label.configure(image=ci)
                self.bg_label.image = ci

    # --- BUDOWA UI ---

    def _build_ui(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        main_frame = ctk.CTkFrame(
            self,
            corner_radius=0,
            fg_color=(COLOR_BG_MAIN_LIGHT, COLOR_BG_MAIN_DARK),
        )
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # HEADER
        header = ctk.CTkFrame(main_frame, height=64, corner_radius=0, fg_color=(COLOR_PANEL_LIGHT, "#020617"))
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        header.grid_columnconfigure(1, weight=0)

        logo_frame = ctk.CTkFrame(header, fg_color="transparent")
        logo_frame.grid(row=0, column=0, sticky="w", padx=20, pady=10)

        # logo.png w lewym rogu
        logo_path = APP_DIR / "logo.png"
        if not logo_path.exists():
            if (APP_DIR / "logo.jpg").exists():
                logo_path = APP_DIR / "logo.jpg"
        if logo_path.exists():
            try:
                img = Image.open(logo_path)
                ratio = img.height / img.width
                size = (64, int(64 * ratio))  # trochę większe logo
                logo_img = ctk.CTkImage(light_image=img, dark_image=img, size=size)
                logo_label = ctk.CTkLabel(logo_frame, image=logo_img, text="")
                logo_label.image = logo_img
                logo_label.grid(row=0, column=0, rowspan=2, padx=(0, 12))
            except Exception:
                pass

        title_label = ctk.CTkLabel(
            logo_frame,
            text="EPS AI SOLUTIONS",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=COLOR_TEXT_MAIN,
        )
        title_label.grid(row=0, column=1, sticky="w")

        subtitle_label = ctk.CTkLabel(
            logo_frame,
            text="Automatyczne wycinanie i renowacja starych zdjęć (Pipeline v11)",
            font=ctk.CTkFont(size=13),
            text_color=COLOR_TEXT_MUTED,
        )
        subtitle_label.grid(row=1, column=1, sticky="w")

        header_right = ctk.CTkFrame(header, fg_color="transparent")
        header_right.grid(row=0, column=1, sticky="e", padx=20, pady=10)
        header_right.grid_columnconfigure(0, weight=0)
        header_right.grid_columnconfigure(1, weight=0)
        header_right.grid_columnconfigure(2, weight=0)
        header_right.grid_columnconfigure(3, weight=0)

        self.api_status_dot = ctk.CTkLabel(header_right, text="●", font=ctk.CTkFont(size=16, weight="bold"))
        self.api_status_label = ctk.CTkLabel(header_right, text="API: OFFLINE")
        self.progress_label = ctk.CTkLabel(header_right, text="0 / 0 plików")

        self.api_status_dot.grid(row=0, column=0, padx=(0, 4))
        self.api_status_label.grid(row=0, column=1, padx=(0, 16))
        self.progress_label.grid(row=0, column=2, padx=(0, 16))

        self.appearance_switch = ctk.CTkSwitch(
            header_right,
            text="Tryb ciemny",
            command=self._toggle_appearance_mode,
        )
        self.appearance_switch.select()
        self.appearance_switch.grid(row=0, column=3)

        # Informacja o aktualnych modelach
        self.models_label = ctk.CTkLabel(
            header_right,
            text="",
            font=ctk.CTkFont(size=10),
            text_color=COLOR_TEXT_MUTED,
        )
        self.models_label.grid(row=1, column=0, columnspan=4, sticky="e", pady=(2, 0))

        # BODY
        body = ctk.CTkFrame(main_frame, fg_color=(COLOR_BG_MAIN_LIGHT, COLOR_BG_MAIN_DARK))
        body.grid(row=1, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=0)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        # LEWY PANEL
        left = ctk.CTkFrame(
            body,
            width=320,
            corner_radius=18,
            fg_color=(SIDEBAR_BG_LIGHT, SIDEBAR_BG_DARK),
        )
        left.grid(row=0, column=0, sticky="nsw", padx=16, pady=12)
        left.grid_rowconfigure(4, weight=1)
        left.grid_columnconfigure(0, weight=1)

        header_label = ctk.CTkLabel(
            left,
            text="Wejście i przetwarzanie",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=COLOR_TEXT_MAIN,
        )
        header_label.grid(row=0, column=0, sticky="w", padx=16, pady=(12, 6))

        # Drop zone
        self.drop_frame = ctk.CTkFrame(
            left,
            height=200,
            corner_radius=16,
            border_width=1,
            border_color=(COLOR_ACCENT, "#1d4ed8"),
            fg_color=(COLOR_PANEL_LIGHT, "#1E293B"),
        )
        self.drop_frame.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 8))
        self.drop_frame.grid_rowconfigure(0, weight=1)
        self.drop_frame.grid_columnconfigure(0, weight=1)

        drop_label = ctk.CTkLabel(
            self.drop_frame,
            text="Przeciągnij i upuść skany lub folder tutaj",
            font=ctk.CTkFont(size=13),
            justify="center",
        )
        drop_hint = ctk.CTkLabel(
            self.drop_frame,
            text="Obsługiwane: JPG, PNG, TIFF, ZIP\nMożesz też użyć przycisków poniżej.",
            font=ctk.CTkFont(size=11),
            text_color=COLOR_TEXT_MUTED,
            justify="center",
        )
        drop_label.grid(row=0, column=0, pady=(45, 4))
        drop_hint.grid(row=0, column=0, pady=(90, 0))

        try:
            self.drop_frame.drop_target_register(DND_FILES)
            self.drop_frame.dnd_bind("<<Drop>>", self._on_drop)
        except Exception as exc:
            self.log_safe(f"⚠️ Drag&Drop niedostępny: {exc}")

        self.files_label = ctk.CTkLabel(
            left,
            text="Brak wybranych plików.",
            font=ctk.CTkFont(size=11),
            text_color=COLOR_TEXT_MUTED,
        )
        self.files_label.grid(row=2, column=0, sticky="w", padx=18, pady=(0, 4))

        buttons_frame = ctk.CTkFrame(left, fg_color="transparent")
        buttons_frame.grid(row=3, column=0, sticky="ew", padx=16, pady=(4, 4))
        buttons_frame.grid_columnconfigure((0, 1), weight=1)

        self.btn_add_files = ctk.CTkButton(
            buttons_frame,
            text="➕ Dodaj pliki / folder",
            command=self._on_choose_input,
            fg_color=COLOR_ACCENT,
            hover_color=COLOR_ACCENT_HOVER,
        )
        self.btn_output_dir = ctk.CTkButton(
            buttons_frame,
            text="📂 Folder wyjściowy",
            command=self._on_choose_output,
            fg_color="white",
            hover_color="#e2e8f0",
            text_color=COLOR_TEXT_MAIN,
        )
        self.btn_add_files.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.btn_output_dir.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        processing_frame = ctk.CTkFrame(left, fg_color="transparent")
        processing_frame.grid(row=4, column=0, sticky="ew", padx=16, pady=(4, 12))
        processing_frame.grid_columnconfigure(0, weight=1)

        self.only_layout_switch = ctk.CTkSwitch(
            processing_frame,
            text="Tylko analiza układu (bez renowacji)",
        )
        self.only_layout_switch.grid(row=0, column=0, sticky="w", pady=(0, 6))

        self.progress_bar = ctk.CTkProgressBar(processing_frame)
        self.progress_bar.set(0)
        self.progress_bar.grid(row=1, column=0, sticky="ew", pady=(4, 8))

        action_buttons = ctk.CTkFrame(processing_frame, fg_color="transparent")
        action_buttons.grid(row=2, column=0, sticky="ew")
        action_buttons.grid_columnconfigure((0, 1), weight=1)

        self.btn_start = ctk.CTkButton(
            action_buttons,
            text="▶ Start",
            command=self._on_start_processing,
            fg_color=COLOR_ACCENT,
            hover_color=COLOR_ACCENT_HOVER,
        )
        self.btn_stop = ctk.CTkButton(
            action_buttons,
            text="⏹ Stop",
            fg_color="#b3261e",
            hover_color="#d03b31",
            command=self._on_stop_processing,
            state="disabled",
        )
        self.btn_start.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.btn_stop.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        # PRAWY PANEL – karty: Log, Ustawienia
        right = ctk.CTkTabview(body)
        right.grid(row=0, column=1, sticky="nsew", padx=(4, 16), pady=12)
        log_tab = right.add("Log")
        settings_tab = right.add("Ustawienia")

        log_tab.grid_rowconfigure(0, weight=1)
        log_tab.grid_columnconfigure(0, weight=1)

        # Dwie kolumny: log + zakładka "Porównanie" (przed/po) w przyszłości – placeholder
        self.log_box = ctk.CTkTextbox(log_tab, wrap="word")
        self.log_box.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.log_box.configure(state="disabled")

        self._build_settings_tab(settings_tab)

        # Pasek stanu
        status_bar = ctk.CTkFrame(main_frame, height=26, corner_radius=0,
                                  fg_color=(COLOR_PANEL_LIGHT, COLOR_PANEL_DARK))
        status_bar.grid(row=2, column=0, sticky="ew")
        status_bar.grid_columnconfigure(0, weight=1)
        status_bar.grid_columnconfigure(1, weight=0)

        self.status_label = ctk.CTkLabel(
            status_bar,
            text="Gotowy.",
            anchor="w",
            font=ctk.CTkFont(size=11),
        )
        self.status_label.grid(row=0, column=0, sticky="w", padx=16, pady=4)

        self.open_output_button = ctk.CTkButton(
            status_bar,
            text="Otwórz folder wyjściowy",
            width=170,
            height=24,
            font=ctk.CTkFont(size=11),
            command=self._on_open_output,
            fg_color=COLOR_ACCENT,
            hover_color=COLOR_ACCENT_HOVER,
        )
        self.open_output_button.grid(row=0, column=1, sticky="e", padx=16, pady=4)

    def _build_settings_tab(self, parent):
        parent.grid_rowconfigure(15, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        cfg = self.config_manager.data

        row = 0
        ctk.CTkLabel(
            parent,
            text="Dostawcy i modele",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=row, column=0, sticky="w", padx=12, pady=(10, 4))

        # layout provider & model
        row += 1
        layout_provider_frame = ctk.CTkFrame(parent, fg_color="transparent")
        layout_provider_frame.grid(row=row, column=0, sticky="ew", padx=12, pady=(4, 2))
        layout_provider_frame.grid_columnconfigure(1, weight=1)

        provider_options = ["Gemini (Google)", "ChatGPT (OpenAI)"]
        self.layout_provider_var = ctk.StringVar(
            value="Gemini (Google)" if cfg.get("layout_provider", "gemini") == "gemini" else "ChatGPT (OpenAI)"
        )
        ctk.CTkLabel(layout_provider_frame, text="Analiza (provider):").grid(row=0, column=0, sticky="w")
        self.layout_provider_menu = ctk.CTkOptionMenu(
            layout_provider_frame,
            variable=self.layout_provider_var,
            values=provider_options,
            command=lambda _: self._refresh_layout_model_menu(),
        )
        self.layout_provider_menu.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        row += 1
        layout_model_frame = ctk.CTkFrame(parent, fg_color="transparent")
        layout_model_frame.grid(row=row, column=0, sticky="ew", padx=12, pady=(2, 2))
        layout_model_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(layout_model_frame, text="Model analizy:").grid(row=0, column=0, sticky="w")
        self.layout_model_var = ctk.StringVar()
        self.layout_model_menu = ctk.CTkOptionMenu(
            layout_model_frame,
            variable=self.layout_model_var,
            values=["..."],
        )
        self.layout_model_menu.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        # restore provider & model
        row += 1
        restore_provider_frame = ctk.CTkFrame(parent, fg_color="transparent")
        restore_provider_frame.grid(row=row, column=0, sticky="ew", padx=12, pady=(6, 2))
        restore_provider_frame.grid_columnconfigure(1, weight=1)

        self.restore_provider_var = ctk.StringVar(
            value="Gemini (Google)" if cfg.get("restore_provider", "gemini") == "gemini" else "ChatGPT (OpenAI)"
        )
        ctk.CTkLabel(restore_provider_frame, text="Renowacja (provider):").grid(row=0, column=0, sticky="w")
        self.restore_provider_menu = ctk.CTkOptionMenu(
            restore_provider_frame,
            variable=self.restore_provider_var,
            values=provider_options,
            command=lambda _: self._refresh_restore_model_menu(),
        )
        self.restore_provider_menu.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        row += 1
        restore_model_frame = ctk.CTkFrame(parent, fg_color="transparent")
        restore_model_frame.grid(row=row, column=0, sticky="ew", padx=12, pady=(2, 2))
        restore_model_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(restore_model_frame, text="Model generowania:").grid(row=0, column=0, sticky="w")
        self.restore_model_var = ctk.StringVar()
        self.restore_model_menu = ctk.CTkOptionMenu(
            restore_model_frame,
            variable=self.restore_model_var,
            values=["..."],
        )
        self.restore_model_menu.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        # Info o providerach
        row += 1
        info_text = (
            "Uwaga: renowacja obrazów przez OpenAI (ChatGPT) nie jest w tej wersji w pełni obsługiwana –\n"
            "niezależnie od wyboru providera renowacji, obraz generuje aktualnie Gemini.\n"
            "Do analizy z OpenAI wybierz model z obsługą obrazu (np. gpt-4.1-mini, gpt-4.1)."
        )
        self.providers_info_label = ctk.CTkLabel(
            parent,
            text=info_text,
            font=ctk.CTkFont(size=10),
            text_color=COLOR_TEXT_MUTED,
            wraplength=520,
            justify="left",
        )
        self.providers_info_label.grid(row=row, column=0, sticky="w", padx=12, pady=(0, 8))

        # Parametry generowania
        row += 1
        ctk.CTkLabel(
            parent,
            text="Parametry generowania",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=row, column=0, sticky="w", padx=12, pady=(10, 4))

        row += 1
        temp_frame = ctk.CTkFrame(parent, fg_color="transparent")
        temp_frame.grid(row=row, column=0, sticky="ew", padx=12, pady=(4, 2))
        ctk.CTkLabel(temp_frame, text="Temperatura:").grid(row=0, column=0, sticky="w")
        self.temp_value_label = ctk.CTkLabel(temp_frame, text="")
        self.temp_value_label.grid(row=0, column=2, sticky="e", padx=(6, 0))
        self.temp_slider = ctk.CTkSlider(
            temp_frame,
            from_=0.0,
            to=1.0,
            number_of_steps=20,
            command=self._on_temp_change,
        )
        self.temp_slider.grid(row=0, column=1, sticky="ew", padx=(8, 8))
        temp_frame.grid_columnconfigure(1, weight=1)
        self.temp_slider.set(float(cfg.get("temperature", DEFAULT_CONFIG["temperature"]))
                             if "temperature" in cfg else DEFAULT_CONFIG["temperature"])
        self._on_temp_change(self.temp_slider.get())

        # Max tokens
        row += 1
        tokens_frame = ctk.CTkFrame(parent, fg_color="transparent")
        tokens_frame.grid(row=row, column=0, sticky="ew", padx=12, pady=(4, 2))
        ctk.CTkLabel(tokens_frame, text="Max tokens output:").grid(row=0, column=0, sticky="w")
        self.entry_max_tokens = ctk.CTkEntry(tokens_frame, width=80)
        self.entry_max_tokens.insert(0, str(cfg.get("max_output_tokens", DEFAULT_CONFIG["max_output_tokens"])))
        self.entry_max_tokens.grid(row=0, column=1, sticky="w", padx=(8, 0))

        # Rozmiary
        row += 1
        dims_frame = ctk.CTkFrame(parent, fg_color="transparent")
        dims_frame.grid(row=row, column=0, sticky="ew", padx=12, pady=(4, 2))
        dims_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(dims_frame, text="Max wymiar (layout):").grid(row=0, column=0, sticky="w")
        self.entry_max_dim_layout = ctk.CTkEntry(dims_frame, width=80)
        self.entry_max_dim_layout.insert(
            0, str(cfg.get("max_dim_layout", DEFAULT_CONFIG["max_dim_layout"]))
        )
        self.entry_max_dim_layout.grid(row=0, column=1, sticky="w", padx=(8, 0))
        ctk.CTkLabel(dims_frame, text="Max wymiar (renowacja):").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.entry_max_dim_restore = ctk.CTkEntry(dims_frame, width=80)
        self.entry_max_dim_restore.insert(
            0, str(cfg.get("max_dim_restore", DEFAULT_CONFIG["max_dim_restore"]))
        )
        self.entry_max_dim_restore.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(4, 0))

        # max_workers
        row += 1
        workers_frame = ctk.CTkFrame(parent, fg_color="transparent")
        workers_frame.grid(row=row, column=0, sticky="ew", padx=12, pady=(4, 2))
        ctk.CTkLabel(workers_frame, text="Maks. równoległych zadań:").grid(row=0, column=0, sticky="w")
        self.workers_slider = ctk.CTkSlider(
            workers_frame,
            from_=1,
            to=max(1, os.cpu_count() or 4),
            number_of_steps=max(1, (os.cpu_count() or 4) - 1),
        )
        self.workers_slider.grid(row=0, column=1, sticky="ew", padx=(8, 8))
        self.workers_value_label = ctk.CTkLabel(workers_frame, text="")
        self.workers_value_label.grid(row=0, column=2, sticky="w")
        workers_frame.grid_columnconfigure(1, weight=1)
        self.workers_slider.set(int(cfg.get("max_workers", DEFAULT_CONFIG["max_workers"])))
        self._on_workers_change(self.workers_slider.get())
        self.workers_slider.configure(command=self._on_workers_change)

        # Zachowanie
        row += 1
        toggles_frame = ctk.CTkFrame(parent, fg_color="transparent")
        toggles_frame.grid(row=row, column=0, sticky="w", padx=12, pady=(6, 4))

        self.save_maps_switch = ctk.CTkSwitch(
            toggles_frame,
            text="Zapisuj mapy układu (*.MAP.jpg)",
        )
        self.save_maps_switch.grid(row=0, column=0, sticky="w")

        self.overwrite_switch = ctk.CTkSwitch(
            toggles_frame,
            text="Nadpisuj istniejące pliki wynikowe",
        )
        self.overwrite_switch.grid(row=1, column=0, sticky="w", pady=(4, 0))

        self.autoopen_switch = ctk.CTkSwitch(
            toggles_frame,
            text="Otwórz folder wyjściowy po zakończeniu",
        )
        self.autoopen_switch.grid(row=2, column=0, sticky="w", pady=(4, 0))

        if cfg.get("save_maps", DEFAULT_CONFIG["save_maps"]):
            self.save_maps_switch.select()
        if cfg.get("overwrite_existing", DEFAULT_CONFIG["overwrite_existing"]):
            self.overwrite_switch.select()
        if cfg.get("auto_open_output", DEFAULT_CONFIG["auto_open_output"]):
            self.autoopen_switch.select()

        # SSL
        row += 1
        ssl_frame = ctk.CTkFrame(parent, fg_color="transparent")
        ssl_frame.grid(row=row, column=0, sticky="w", padx=12, pady=(6, 4))
        self.ssl_switch = ctk.CTkSwitch(
            ssl_frame,
            text="Weryfikacja certyfikatów SSL w zapytaniach HTTP",
        )
        if cfg.get("verify_ssl", DEFAULT_CONFIG["verify_ssl"]):
            self.ssl_switch.select()
        else:
            self.ssl_switch.deselect()
        self.ssl_switch.grid(row=0, column=0, sticky="w")

        # PRESETY
        row += 1
        preset_frame = ctk.CTkFrame(parent, fg_color="transparent")
        preset_frame.grid(row=row, column=0, sticky="ew", padx=12, pady=(6, 4))
        preset_frame.grid_columnconfigure(1, weight=1)
        preset_frame.grid_columnconfigure(2, weight=0)

        ctk.CTkLabel(
            preset_frame,
            text="Preset:",
        ).grid(row=0, column=0, sticky="w")

        self.preset_var = ctk.StringVar(value=PRESET_NAMES[0])
        self.preset_menu = ctk.CTkOptionMenu(
            preset_frame,
            variable=self.preset_var,
            values=PRESET_NAMES,
        )
        self.preset_menu.grid(row=0, column=1, sticky="ew", padx=(8, 8))

        self.preset_apply_btn = ctk.CTkButton(
            preset_frame,
            text="Zastosuj",
            width=90,
            fg_color=COLOR_ACCENT,
            hover_color=COLOR_ACCENT_HOVER,
            command=self._apply_preset_from_ui,
        )
        self.preset_apply_btn.grid(row=0, column=2, sticky="e")

        # przycisk zapisu
        row += 1
        save_btn = ctk.CTkButton(
            parent,
            text="Zapisz ustawienia",
            command=self._on_save_settings,
            fg_color=COLOR_ACCENT,
            hover_color=COLOR_ACCENT_HOVER,
        )
        save_btn.grid(row=row, column=0, sticky="e", padx=12, pady=(10, 10))

        # Ustaw menu modeli na podstawie aktualnych wartości
        self._refresh_layout_model_menu()
        self._refresh_restore_model_menu()

    # --- SETTING HELPERS ---

    def _load_gemini_models(self) -> list[str]:
        """Pobiera listę modeli Gemini przy starcie (jeśli dostępny klucz)."""
        if not (genai and self.engine.gemini_api_key):
            return []
        try:
            genai.configure(api_key=self.engine.gemini_api_key)
            all_models = list(genai.list_models())
            models: list[str] = []
            for m in all_models:
                methods = getattr(m, "supported_generation_methods", []) or []
                if "generateContent" in methods:
                    raw_name = getattr(m, "name", None)
                    if raw_name:
                        name = raw_name.split("/")[-1]
                        models.append(name)
            models = sorted(set(models))
            self.log_safe(f"ℹ️ Modele Gemini: {len(models)} dostępnych.")
            return models
        except Exception as exc:
            self.log_safe(f"⚠️ Nie udało się pobrać listy modeli Gemini: {exc}")
            return []

    def _load_openai_models(self) -> list[str]:
        if not self.engine.openai_client:
            return []
        try:
            resp = self.engine.openai_client.models.list()
            data = getattr(resp, "data", []) or []
            models: list[str] = []
            for m in data:
                mid = getattr(m, "id", None)
                if mid and ("gpt" in mid.lower() or "o1" in mid.lower()):
                    models.append(mid)
            models = sorted(set(models))
            self.log_safe(f"ℹ️ Modele OpenAI: {len(models)} dostępnych.")
            return models
        except Exception as exc:
            self.log_safe(f"⚠️ Nie udało się pobrać listy modeli OpenAI: {exc}")
            return []

    def _provider_label_to_key(self, label: str) -> str:
        return "gemini" if "Gemini" in label else "openai"

    def _refresh_layout_model_menu(self):
        cfg = self.config_manager.data
        provider_key = self._provider_label_to_key(self.layout_provider_var.get())
        if provider_key == "gemini":
            options = self.gemini_models or [cfg.get("layout_model_gemini", DEFAULT_CONFIG["layout_model_gemini"])]
            current = cfg.get("layout_model_gemini", DEFAULT_CONFIG["layout_model_gemini"])
        else:
            options = self.openai_models or [cfg.get("layout_model_openai", DEFAULT_CONFIG["layout_model_openai"])]
            current = cfg.get("layout_model_openai", DEFAULT_CONFIG["layout_model_openai"])
        if current not in options:
            options = [current] + options
        self.layout_model_menu.configure(values=options)
        self.layout_model_var.set(current)

    def _refresh_restore_model_menu(self):
        cfg = self.config_manager.data
        provider_key = self._provider_label_to_key(self.restore_provider_var.get())
        if provider_key == "gemini":
            options = self.gemini_models or [cfg.get("restore_model_gemini", DEFAULT_CONFIG["restore_model_gemini"])]
            current = cfg.get("restore_model_gemini", DEFAULT_CONFIG["restore_model_gemini"])
        else:
            options = self.openai_models or [cfg.get("restore_model_openai", DEFAULT_CONFIG["restore_model_openai"])]
            current = cfg.get("restore_model_openai", DEFAULT_CONFIG["restore_model_openai"])
        if current not in options:
            options = [current] + options
        self.restore_model_menu.configure(values=options)
        self.restore_model_var.set(current)

    # --- OGÓLNE HELPERY ---

    def _center_window(self):
        self.update_idletasks()
        w = self.winfo_width() or 1150
        h = self.winfo_height() or 720
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x = int((sw - w) / 2)
        y = int((sh - h) / 2)
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _toggle_appearance_mode(self):
        if self.appearance_switch.get():
            ctk.set_appearance_mode("dark")
            self.appearance_switch.configure(text="Tryb ciemny")
        else:
            ctk.set_appearance_mode("light")
            self.appearance_switch.configure(text="Tryb jasny")

    def _update_api_indicator(self):
        online_labels = []
        if self.engine.gemini_api_key:
            online_labels.append("Gemini")
        if self.engine.openai_api_key:
            online_labels.append("OpenAI")
        if online_labels:
            self.api_status_dot.configure(text="●", text_color="#22c55e")
            self.api_status_label.configure(text="API: " + ", ".join(online_labels))
        else:
            self.api_status_dot.configure(text="●", text_color="#F44336")
            self.api_status_label.configure(text="API: OFFLINE")

    def _update_models_label(self):
        cfg = self.config_manager.data
        layout_provider = cfg.get("layout_provider", "gemini")
        restore_provider = cfg.get("restore_provider", "gemini")

        def prov_name(key: str) -> str:
            return "Gemini" if key == "gemini" else "OpenAI"

        if layout_provider == "gemini":
            layout_model = cfg.get("layout_model_gemini", DEFAULT_CONFIG["layout_model_gemini"])
        else:
            layout_model = cfg.get("layout_model_openai", DEFAULT_CONFIG["layout_model_openai"])

        if restore_provider == "gemini":
            restore_model = cfg.get("restore_model_gemini", DEFAULT_CONFIG["restore_model_gemini"])
        else:
            restore_model = cfg.get("restore_model_openai", DEFAULT_CONFIG["restore_model_openai"])

        text = (
            f"Analiza: {prov_name(layout_provider)} / {layout_model}   |   "
            f"Renowacja: {prov_name(restore_provider)} / {restore_model}"
        )
        self.models_label.configure(text=text)

    def _set_status(self, text: str):
        self.status_label.configure(text=text)

    def _update_progress(self):
        if self.total_files > 0:
            ratio = self.processed_files / self.total_files
        else:
            ratio = 0
        self.progress_bar.set(ratio)
        self.progress_label.configure(text=f"{self.processed_files} / {self.total_files} plików")

    def log_safe(self, message: str):
        """Log do konsoli + GUI (bez wywalania wątków)."""
        try:
            print(message, flush=True)
        except Exception:
            pass

        if not hasattr(self, "log_box") or self.log_box is None:
            return

        def _append():
            try:
                self.log_box.configure(state="normal")
                self.log_box.insert("end", message + "\n")
                self.log_box.see("end")
                self.log_box.configure(state="disabled")
            except Exception:
                pass

        try:
            self.after(0, _append)
        except Exception:
            pass

    # --- ZDARZENIA USTAWIEŃ ---

    def _on_temp_change(self, value: float):
        self.temp_value_label.configure(text=f"{float(value):.2f}")

    def _on_workers_change(self, value: float):
        self.workers_value_label.configure(text=str(int(round(float(value)))))

    def _apply_preset_from_ui(self):
        name = self.preset_var.get()
        preset = PRESETS.get(name)
        if not preset:
            return

        cfg = self.config_manager.data

        layout_provider = preset.get("layout_provider", cfg.get("layout_provider", "gemini"))
        restore_provider = preset.get("restore_provider", cfg.get("restore_provider", "gemini"))

        self.layout_provider_var.set("Gemini (Google)" if layout_provider == "gemini" else "ChatGPT (OpenAI)")
        self.restore_provider_var.set("Gemini (Google)" if restore_provider == "gemini" else "ChatGPT (OpenAI)")

        cfg["layout_provider"] = layout_provider
        cfg["restore_provider"] = restore_provider

        if layout_provider == "gemini" and "layout_model_gemini" in preset:
            cfg["layout_model_gemini"] = preset["layout_model_gemini"]
        if layout_provider == "openai" and "layout_model_openai" in preset:
            cfg["layout_model_openai"] = preset["layout_model_openai"]

        if restore_provider == "gemini" and "restore_model_gemini" in preset:
            cfg["restore_model_gemini"] = preset["restore_model_gemini"]
        if restore_provider == "openai" and "restore_model_openai" in preset:
            cfg["restore_model_openai"] = preset["restore_model_openai"]

        if "temperature" in preset:
            self.temp_slider.set(preset["temperature"])
            self._on_temp_change(self.temp_slider.get())

        if "max_output_tokens" in preset:
            self.entry_max_tokens.delete(0, "end")
            self.entry_max_tokens.insert(0, str(preset["max_output_tokens"]))

        if "max_dim_layout" in preset:
            self.entry_max_dim_layout.delete(0, "end")
            self.entry_max_dim_layout.insert(0, str(preset["max_dim_layout"]))

        if "max_dim_restore" in preset:
            self.entry_max_dim_restore.delete(0, "end")
            self.entry_max_dim_restore.insert(0, str(preset["max_dim_restore"]))

        if "max_workers" in preset:
            self.workers_slider.set(preset["max_workers"])
            self._on_workers_change(self.workers_slider.get())

        if "save_maps" in preset:
            (self.save_maps_switch.select() if preset["save_maps"] else self.save_maps_switch.deselect())

        if "overwrite_existing" in preset:
            (self.overwrite_switch.select() if preset["overwrite_existing"] else self.overwrite_switch.deselect())

        if "auto_open_output" in preset:
            (self.autoopen_switch.select() if preset["auto_open_output"] else self.autoopen_switch.deselect())

        self.log_safe(f"🎛️ Zastosowano preset: {name} (pamiętaj, aby zapisać ustawienia).")
        self._set_status(f"Preset '{name}' załadowany (niezapisany).")

    def _on_save_settings(self):
        cfg = self.config_manager.data

        layout_provider_key = self._provider_label_to_key(self.layout_provider_var.get())
        restore_provider_key = self._provider_label_to_key(self.restore_provider_var.get())
        cfg["layout_provider"] = layout_provider_key
        cfg["restore_provider"] = restore_provider_key

        if layout_provider_key == "gemini":
            cfg["layout_model_gemini"] = self.layout_model_var.get()
        else:
            cfg["layout_model_openai"] = self.layout_model_var.get()

        if restore_provider_key == "gemini":
            cfg["restore_model_gemini"] = self.restore_model_var.get()
        else:
            cfg["restore_model_openai"] = self.restore_model_var.get()

        cfg["temperature"] = float(self.temp_slider.get())
        try:
            cfg["max_output_tokens"] = int(self.entry_max_tokens.get())
        except ValueError:
            cfg["max_output_tokens"] = DEFAULT_CONFIG["max_output_tokens"]

        try:
            cfg["max_dim_layout"] = int(self.entry_max_dim_layout.get())
        except ValueError:
            cfg["max_dim_layout"] = DEFAULT_CONFIG["max_dim_layout"]

        try:
            cfg["max_dim_restore"] = int(self.entry_max_dim_restore.get())
        except ValueError:
            cfg["max_dim_restore"] = DEFAULT_CONFIG["max_dim_restore"]

        cfg["max_workers"] = int(round(self.workers_slider.get()))
        cfg["verify_ssl"] = bool(self.ssl_switch.get())

        cfg["save_maps"] = bool(self.save_maps_switch.get())
        cfg["overwrite_existing"] = bool(self.overwrite_switch.get())
        cfg["auto_open_output"] = bool(self.autoopen_switch.get())

        self.save_maps = cfg["save_maps"]
        self.overwrite_existing = cfg["overwrite_existing"]
        self.auto_open_output = cfg["auto_open_output"]

        self.config_manager.save()
        self.engine.apply_config(cfg)
        self._update_api_indicator()
        self._update_models_label()
        self.log_safe("✅ Zapisano ustawienia i zastosowano w silniku.")
        self._set_status("Zapisano ustawienia.")

    # --- DnD / PLIKI ---

    def _on_drop(self, event):
        paths = self.tk.splitlist(event.data)
        added = self._add_input_paths(paths)
        self.log_safe(f"📁 Dodano {added} ścieżek z Drag&Drop.")
        self._set_status(f"Dodano {added} ścieżek.")

    def _on_choose_input(self):
        files = filedialog.askopenfilenames(
            title="Wybierz skany lub archiwa ZIP",
            filetypes=[
                ("Zdjęcia", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp"),
                ("Archiwa ZIP", "*.zip"),
                ("Wszystkie pliki", "*.*"),
            ],
        )
        if not files:
            return
        added = self._add_input_paths(files)
        self.log_safe(f"🗂 Dodano {added} ścieżek z dialogu.")
        self._set_status(f"Dodano {added} ścieżek.")

    def _add_input_paths(self, paths) -> int:
        before = len(self.input_paths)
        for p in paths:
            p = os.path.abspath(p)
            if p not in self.input_paths:
                self.input_paths.append(p)
        after = len(self.input_paths)
        delta = after - before
        if self.input_paths:
            self.files_label.configure(text=f"Wybrano {len(self.input_paths)} plików / folderów.")
        else:
            self.files_label.configure(text="Brak wybranych plików.")
        return delta

    def _on_choose_output(self):
        d = filedialog.askdirectory(title="Wybierz folder wyjściowy")
        if not d:
            return
        self.output_dir = d
        self.log_safe(f"📂 Zmieniono folder wyjściowy na: {self.output_dir}")
        self._set_status("Folder wyjściowy ustawiony.")

    def _on_open_output(self):
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            if sys.platform.startswith("win"):
                os.startfile(self.output_dir)
            elif sys.platform == "darwin":
                os.system(f"open '{self.output_dir}'")
            else:
                os.system(f"xdg-open '{self.output_dir}'")
        except Exception as exc:
            self.log_safe(f"❌ Nie udało się otworzyć folderu wyjściowego: {exc}")

    # --- START / STOP ---

    def _on_start_processing(self):
        if self.is_running:
            return
        if not self.input_paths:
            messagebox.showinfo("Brak plików", "Najpierw dodaj skany lub folder.")
            return
        file_queue = self._build_file_queue()
        if not file_queue:
            messagebox.showinfo("Brak zdjęć", "Nie znaleziono żadnych plików graficznych.")
            return

        self.total_files = len(file_queue)
        self.processed_files = 0
        self._update_progress()

        self.is_running = True
        self.stop_event.clear()
        self.engine.stop_event = self.stop_event

        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self._set_status("Przetwarzanie rozpoczęte.")

        cfg = self.config_manager.data
        lp = cfg.get("layout_provider", "gemini")
        rp = cfg.get("restore_provider", "gemini")
        lm = cfg.get("layout_model_gemini") if lp == "gemini" else cfg.get("layout_model_openai")
        rm = cfg.get("restore_model_gemini") if rp == "gemini" else cfg.get("restore_model_openai")
        self.log_safe(
            f"▶ START: {self.total_files} plików | "
            f"layout={lp}/{lm} | restore={rp}/{rm} | workers={self.engine.max_workers}"
        )

        t = threading.Thread(target=self._process_queue_thread, args=(file_queue,), daemon=True)
        t.start()

    def _on_stop_processing(self):
        if not self.is_running:
            return
        self.stop_event.set()
        self.is_running = False
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self._set_status("Przerwano przez użytkownika.")
        self.log_safe("⏹ Przerwano przetwarzanie.")

    def _build_file_queue(self) -> list[str]:
        import zipfile
        all_sources: list[str] = []
        for p in self.input_paths:
            if os.path.isdir(p):
                for root, _, files in os.walk(p):
                    for name in files:
                        full = os.path.join(root, name)
                        all_sources.append(full)
            else:
                all_sources.append(p)

        result: list[str] = []
        temp_root = os.path.join(self.output_dir, "_temp_zip")
        os.makedirs(temp_root, exist_ok=True)

        for src in all_sources:
            if src.lower().endswith(".zip"):
                try:
                    with zipfile.ZipFile(src, "r") as zf:
                        subdir = os.path.join(temp_root, Path(src).stem)
                        os.makedirs(subdir, exist_ok=True)
                        zf.extractall(subdir)
                        for root, _, files in os.walk(subdir):
                            for name in files:
                                full = os.path.join(root, name)
                                if is_image_file(full):
                                    result.append(full)
                except Exception as exc:
                    self.log_safe(f"❌ Błąd ZIP ({src}): {exc}")
            else:
                if is_image_file(src):
                    result.append(src)
        return result

    def _process_queue_thread(self, file_queue: list[str]):
        self.log_safe(f"ℹ️ Rozpoczęcie kolejki ({len(file_queue)} plików)...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.engine.max_workers) as ex:
            futures = {ex.submit(self._process_single_file, path, idx): path
                       for idx, path in enumerate(file_queue)}
            for fut in concurrent.futures.as_completed(futures):
                if self.stop_event.is_set():
                    break
                try:
                    fut.result()
                except Exception as exc:
                    path = futures[fut]
                    self.log_safe(f"❌ Błąd podczas przetwarzania {os.path.basename(path)}: {exc}")
                finally:
                    self.processed_files += 1
                    self.after(0, self._update_progress)

        self.is_running = False
        self.after(0, lambda: self.btn_start.configure(state="normal"))
        self.after(0, lambda: self.btn_stop.configure(state="disabled"))
        self.log_safe("✅ Zakończono przetwarzanie.")
        self._set_status("Zakończono.")
        if self.auto_open_output and not self.stop_event.is_set():
            self.after(0, self._on_open_output)

    def _process_single_file(self, path: str, idx: int):
        fname = os.path.basename(path)
        self.log_safe(f"[{idx+1}/{self.total_files}] {fname} → start")

        # 1. analiza układu
        self.log_safe(f"[{fname}] Analiza układu...")
        layout = self.engine.detect_layout(path)
        self.log_safe(f"[{fname}] Analiza zakończona (layout={'OK' if layout else 'BRAK'})")

        map_path = os.path.join(self.output_dir, f"{idx:04d}_{fname}_MAP.jpg")

        try:
            img_src = Image.open(path).convert("RGB")
            draw = ImageDraw.Draw(img_src)
            w, h = img_src.size
            crops: list[tuple[int, list[tuple[float, float]]]] = []

            if layout and isinstance(layout, dict) and "photos" in layout:
                photos = layout.get("photos") or []
                self.log_safe(f"[{fname}] Wykryto: {len(photos)} zdjęć")
                for i, item in enumerate(photos):
                    corners_norm = item.get("corners") or []
                    if len(corners_norm) != 4:
                        self.log_safe(f"[{fname}] ⚠️ pomijam fragment {i+1} – nieprawidłowe rogi")
                        continue
                    corners_px = [
                        ((pt[0] / 1000.0) * w, (pt[1] / 1000.0) * h)
                        for pt in corners_norm
                    ]
                    crops.append((i, corners_px))
                    xs = [c[0] for c in corners_px]
                    ys = [c[1] for c in corners_px]
                    draw.rectangle([min(xs), min(ys), max(xs), max(ys)], outline=COLOR_ACCENT, width=5)
            else:
                self.log_safe(f"[{fname}] Brak layoutu – traktuję jako jedno zdjęcie.")
                crops.append((0, [(0, 0), (w, 0), (w, h), (0, h)]))

            os.makedirs(self.output_dir, exist_ok=True)
            if self.save_maps:
                img_src.save(map_path, quality=90)
                self.log_safe(f"[{fname}] Zapisano mapę układu: {os.path.basename(map_path)}")
        except Exception as exc:
            self.log_safe(f"[{fname}] ❌ Błąd krytyczny podczas analizy: {exc}")
            return

        if self.only_layout_switch.get():
            self.log_safe(f"[{fname}] Tylko analiza układu – pomijam renowację.")
            return

        temp_crops_dir = os.path.join(self.output_dir, "_temp_crops")
        os.makedirs(temp_crops_dir, exist_ok=True)

        for i, corners_px in crops:
            if self.stop_event.is_set():
                return
            try:
                crop_img = self.engine.crop_image(img_src, corners_px)
                if crop_img is not None:
                    tmp_path = os.path.join(temp_crops_dir, f"tmp_{idx}_{i}.png")
                    crop_img.save(tmp_path)

                    # 4. ORIENTACJA – wykryj UP/DOWN/LEFT/RIGHT i obróć
                    direction = self.engine.detect_orientation(tmp_path)
                    if direction:
                        try:
                            oriented_img = Image.open(tmp_path).convert("RGB")
                            if direction == "LEFT":
                                # TOP wskazuje lewo -> obraz obrócony 90° CCW -> trzeba 90° CW
                                oriented_img = oriented_img.rotate(270, expand=True)
                            elif direction == "RIGHT":
                                # TOP wskazuje prawo -> obraz obrócony 90° CW -> trzeba 90° CCW
                                oriented_img = oriented_img.rotate(90, expand=True)
                            elif direction == "DOWN":
                                oriented_img = oriented_img.rotate(180, expand=True)
                            oriented_img.save(tmp_path)
                            self.log_safe(f"[{fname}] 🧭 Orientacja fragmentu {i+1}: {direction}, zastosowano obrót.")
                        except Exception as exc:
                            self.log_safe(f"[{fname}] ⚠️ Błąd podczas obracania fragmentu {i+1}: {exc}")

                    src_for_rest = tmp_path
                else:
                    src_for_rest = path

                out_name = f"{idx:04d}_{Path(fname).stem}_{i+1}.png"
                out_path = os.path.join(self.output_dir, out_name)

                if not self.overwrite_existing and os.path.exists(out_path):
                    self.log_safe(f"[{fname}] Pomijam fragment {i+1} – plik już istnieje.")
                    continue

                self.log_safe(f"[{fname}] Renowacja fragmentu {i+1}...")
                if self.engine.restore_photo(src_for_rest, out_path):
                    self.log_safe(f"[{fname}] ✅ SUKCES: fragment {i+1} → {os.path.basename(out_path)}")
                    # 6. POST-PROCESSING: ramki + wyostrzanie
                    try:
                        self.engine.postprocess_restored_image(out_path, apply_unsharp=True)
                    except Exception as exc:
                        self.log_safe(f"[{fname}] ⚠️ Błąd post-processingu fragmentu {i+1}: {exc}")
                else:
                    self.log_safe(f"[{fname}] ❌ BŁĄD API: fragment {i+1}")

                if crop_img is not None:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
            except Exception as exc:
                self.log_safe(f"[{fname}] Błąd renowacji fragmentu {i+1}: {exc}")


# ===============
# START
# ===============

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
