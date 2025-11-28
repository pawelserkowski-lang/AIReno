import os

# ==========================================
# DEFINICJA STRUKTURY PLIKÓW PROJEKTU EPS AI
# ==========================================

files_structure = {
    # ---------------------------------------------------------
    # 1. PLIKI KONFIGURACYJNE
    # ---------------------------------------------------------
    "requirements.txt": """customtkinter
opencv-python
numpy
Pillow
PyYAML
requests
tkinterdnd2
google-generativeai
python-dotenv""",

    ".gitignore": """.env
__pycache__/
AIReno_Output/
.vscode/
*.pyc
.DS_Store""",

    # ---------------------------------------------------------
    # 2. INTELIGENTNY MAIN.PY (AUTO-INSTALACJA)
    # ---------------------------------------------------------
    "main.py": """import os
import sys
import subprocess
import importlib

def check_and_install_packages():
    # Lista kluczowych bibliotek do sprawdzenia
    required = {
        'yaml': 'PyYAML',
        'cv2': 'opencv-python',
        'customtkinter': 'customtkinter',
        'PIL': 'Pillow',
        'dotenv': 'python-dotenv',
        'google.generativeai': 'google-generativeai',
        'tkinterdnd2': 'tkinterdnd2'
    }
    
    missing = []
    print("--- Sprawdzanie bibliotek ---")
    
    for import_name, install_name in required.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            print(f"[!] Brakuje: {install_name}")
            missing.append(install_name)
    
    if missing:
        print(f"Instalowanie {len(missing)} brakujących pakietów...")
        try:
            # Instalacja przez pip
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
            print("✔ Instalacja zakończona sukcesem!")
            print("-----------------------------------")
        except Exception as e:
            print(f"✘ Błąd instalacji: {e}")
            input("Naciśnij Enter aby zamknąć...")
            sys.exit(1)
    else:
        print("✔ Wszystkie biblioteki są zainstalowane.")

# 1. Najpierw sprawdź/zainstaluj biblioteki
check_and_install_packages()

# 2. Dopiero teraz importuj resztę aplikacji
try:
    from dotenv import load_dotenv
    from gui.app import EPSApp

    # Ładowanie zmiennych środowiskowych
    load_dotenv()
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    if __name__ == "__main__":
        app = EPSApp()
        app.mainloop()

except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    import traceback
    traceback.print_exc()
    input("Naciśnij Enter, aby zamknąć...")
""",

    "config/settings.yaml": """processing:
  default_output_folder: "AIReno_Output"
  model_name: "gemini-1.5-flash" 
""",

    # ---------------------------------------------------------
    # 3. RDZEŃ SYSTEMU (CORE)
    # ---------------------------------------------------------
    "core/__init__.py": "",

    "core/gemini_client.py": """import google.generativeai as genai
import yaml
import os
import time
from PIL import Image
import io

class GeminiClient:
    def __init__(self, config_path="config/settings.yaml"):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        
        self.available_models = [
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-2.0-flash-exp",
            "gemini-3-pro-image-preview" 
        ]
        
        self.current_model_name = "gemini-1.5-flash"
        self._load_config(config_path)
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.set_model(self.current_model_name)
        else:
            self.model = None

    def _load_config(self, path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                cfg = yaml.safe_load(f)
                self.current_model_name = cfg.get('processing', {}).get('model_name', "gemini-1.5-flash")

    def set_model(self, model_name):
        self.current_model_name = model_name
        try:
            self.model = genai.GenerativeModel(model_name)
            return True, f"Silnik zmieniony na: {model_name}"
        except Exception as e:
            self.model = genai.GenerativeModel("gemini-1.5-flash")
            return False, f"Błąd modelu {model_name}, powrót do Flash. ({e})"

    def get_prompt_by_creativity(self, level):
        level = int(level)
        base_tech = " Treat the outer 5% of the image as a buffer zone: use generative fill to reconstruct background seamlessly. "

        if level <= 3:
            return "Strict forensic restoration." + base_tech + "Remove scratches and dust ONLY. Do not change facial features. Keep original film grain. Output high resolution."
        elif level <= 7:
            return ("Balanced restoration. Fix tears, scratches and restore missing textures. "
                    "If B&W, perform HDR colorization. "
                    "Sharpen details but maintain FORENSIC FIDELITY: faces must NOT look swollen (no Frankenstein effect), eyes natural. "
                    + base_tech +
                    "Finally, apply a comprehensive High-End Studio Photography aesthetic. "
                    "Professional color grading, optimized contrast curves, ultra-fine detail rendering, "
                    "while strictly preserving the subject's identity.")
        else:
            return "Artistic restoration." + base_tech + "Hallucinate missing details. Strong studio lighting. Vibrant modern colors. Make it look like a modern 4K digital photo."

    def generate_commit_message(self):
        if not self.model: return "Auto-update by EPS AI"
        try:
            prompt = "Generate a short, professional git commit message (max 7 words) for an update in an AI Image Restoration App."
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except:
            return "Optimization and feature update"

    def restore_image(self, image_path, save_path, creativity=5):
        if not self.model: return False, "Brak AI.", 0
        
        start_time = time.time()
        retries = 0
        max_retries = 5 
        
        while retries < max_retries:
            try:
                img = Image.open(image_path)
                prompt = self.get_prompt_by_creativity(creativity)
                response = self.model.generate_content([prompt, img])
                
                duration = time.time() - start_time
                
                if hasattr(response, 'parts'):
                    for part in response.parts:
                        if hasattr(part, "inline_data") or hasattr(part, "image"):
                            data = part.inline_data.data if hasattr(part, "inline_data") else part.image
                            Image.open(io.BytesIO(data)).save(save_path)
                            return True, "OK", duration
                
                return False, "Model zwrócił tekst (Brak obrazu).", duration

            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "ResourceExhausted" in err_msg:
                    wait_time = (2 ** retries) * 5
                    print(f"⚠️ Limit API! Czekam {wait_time}s...")
                    time.sleep(wait_time)
                    retries += 1
                else:
                    return False, f"Błąd krytyczny API: {err_msg}", time.time() - start_time
        
        return False, "Przekroczono limit prób (Rate Limit).", time.time() - start_time

    def analyze_image_defects(self, image_path):
        if not self.model: return "Brak AI"
        try:
            img = Image.open(image_path)
            return self.model.generate_content(["List defects brief JSON", img]).text
        except: return "Błąd analizy"
""",

    "core/image_processing.py": """import cv2
import numpy as np
import os

class ImageProcessor:
    @staticmethod
    def extract_with_strategy(image_path, output_folder, strategy=0, min_area_percent=0.06):
        strategies = [
            ImageProcessor._strategy_morphology,
            ImageProcessor._strategy_otsu,
            ImageProcessor._strategy_adaptive
        ]
        method = strategies[strategy % len(strategies)]
        return method(image_path, output_folder, min_area_percent)

    @staticmethod
    def _base_processing(image_path, output_folder, edge_image, min_area_percent):
        img = cv2.imread(image_path)
        original = img.copy()
        h, w = img.shape[:2]
        min_area = (h * w) * min_area_percent
        
        contours, _ = cv2.findContours(edge_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        extracted = []
        count = 0
        
        for c in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(c)
            if area < min_area: continue
            if area > (h*w)*0.98: continue 

            x, y, cw, ch = cv2.boundingRect(c)
            ar = float(cw) / ch
            if ar < 0.25 or ar > 4.0: continue 

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            if len(approx) == 4:
                try:
                    warped = ImageProcessor.four_point_transform(original, approx.reshape(4, 2))
                    wh, ww = warped.shape[:2]
                    mh, mw = int(wh*0.01), int(ww*0.01)
                    if mh>0 and mw>0: cropped = warped[mh:wh-mh, mw:ww-mw]
                    else: cropped = warped
                    
                    filename = f"crop_{count}_s{np.random.randint(99)}_{os.path.basename(image_path)}"
                    path = os.path.join(output_folder, filename)
                    cv2.imwrite(path, cropped)
                    extracted.append(path)
                    count += 1
                except: pass
        return extracted

    @staticmethod
    def _strategy_morphology(image_path, output_folder, min_area_percent):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 9)
        edged = cv2.Canny(blurred, 30, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=4)
        return ImageProcessor._base_processing(image_path, output_folder, closed, min_area_percent)

    @staticmethod
    def _strategy_otsu(image_path, output_folder, min_area_percent):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return ImageProcessor._base_processing(image_path, output_folder, thresh, min_area_percent)

    @staticmethod
    def _strategy_adaptive(image_path, output_folder, min_area_percent):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        return ImageProcessor._base_processing(image_path, output_folder, dilated, min_area_percent)

    @staticmethod
    def four_point_transform(image, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
        (tl, tr, br, bl) = rect
        width = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
        height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
        dst = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (width, height))
""",

    "core/pipeline.py": """import time
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from core.image_processing import ImageProcessor
from core.gemini_client import GeminiClient

class RestorationPipeline:
    def __init__(self, log_cb, prog_cb, gallery_cb=None):
        self.log = log_cb
        self.prog = prog_cb
        self.update_gallery = gallery_cb
        self.input_files = []
        self.output_dir = "AIReno_Output"
        self.creativity = 5
        self.min_area_percent = 0.06
        
        try:
            self.ai = GeminiClient()
            self.log(f"System AI Start. Domyślny silnik: {self.ai.current_model_name}")
        except: self.ai = None

    def set_input_files(self, files): self.input_files = files
    
    def update_settings(self, creativity, min_area):
        self.creativity = creativity
        self.min_area_percent = min_area
        
    def change_model(self, model_name):
        if self.ai:
            ok, msg = self.ai.set_model(model_name)
            self.log(msg)
            return ok
        return False

    def git_push_auto(self):
        self.log("--- AI Git Push ---")
        commit_msg = "Auto-update"
        if self.ai:
            self.log("Generowanie opisu zmian (AI)...")
            commit_msg = self.ai.generate_commit_message()
        
        self.log(f"Commit: '{commit_msg}'")
        try:
            if not os.path.exists(".git"):
                self.log("⚠ Brak repozytorium Git.")
                return
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", commit_msg], check=False)
            res = subprocess.run(["git", "push"], capture_output=True, text=True)
            if res.returncode == 0: self.log("✔ Push OK!")
            else: self.log(f"✘ Git Błąd: {res.stderr}")
        except Exception as e:
            self.log(f"✘ Błąd: {e}")

    def retry_crop(self, source_file, strategy_idx):
        self.log(f"Ponawiam crop (Metoda {strategy_idx})...")
        try:
            new_crops = ImageProcessor.extract_with_strategy(source_file, self.output_dir, strategy_idx, self.min_area_percent)
            self.log(f"Znaleziono {len(new_crops)} wycinków.")
            if self.update_gallery:
                self.update_gallery(new_crops, source_file)
        except Exception as e:
            self.log(f"Błąd: {e}")

    def _process_single_image(self, img_path):
        filename = os.path.basename(img_path)
        if self.ai and self.ai.model:
            out = os.path.join(self.output_dir, f"RES_{filename}")
            ok, msg, duration = self.ai.restore_image(img_path, out, self.creativity)
            status_icon = "✔" if ok else "✘"
            log_msg = f"{status_icon} [{duration:.1f}s] {filename}: {msg}"
            if duration > 10 and not ok: log_msg += " (Możliwy Limit API)"
            self.log(log_msg)
            return 1 
        else:
            self.log(f"⚠ Pominięto {filename} (Brak AI)")
            return 0

    def run(self):
        if not self.input_files: return
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)
        
        all_imgs = []
        self.prog(0.1, "Faza 1: Inteligentna Separacja...")
        
        for f in self.input_files:
            try:
                ex = ImageProcessor.extract_with_strategy(f, self.output_dir, 0, self.min_area_percent)
                all_imgs.extend(ex)
                self.log(f"Skan {os.path.basename(f)}: Znaleziono {len(ex)} zdjęć")
                if self.update_gallery: self.update_gallery(ex, f)
            except Exception as e: self.log(str(e))
            
        total = len(all_imgs)
        if not total: 
            self.prog(0, "Koniec (Brak zdjęć)"); return

        self.log(f"--- Start Renowacji AI ({total} plików) ---")
        self.log(f"Silnik: {self.ai.current_model_name}")
        self.log(f"Wątki: 3 (Równoległe przetwarzanie)")
        
        completed = 0
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(self._process_single_image, img) for img in all_imgs]
            for future in futures:
                res = future.result() 
                completed += res
                progress_val = 0.2 + (0.8 * (completed / total))
                self.prog(progress_val, f"Przetwarzanie: {completed}/{total}")

        self.prog(1.0, "Gotowe")
        self.log("--- Proces Zakończony ---")
""",

    # ---------------------------------------------------------
    # 3. INTERFEJS GRAFICZNY (GUI)
    # ---------------------------------------------------------
    "gui/__init__.py": "",
    
    "gui/components.py": """import customtkinter as ctk
from PIL import Image, ImageTk
import os

class CropGallery(ctk.CTkScrollableFrame):
    def __init__(self, master, retry_callback):
        super().__init__(master, label_text="Znalezione zdjęcia")
        self.retry_cb = retry_callback
        self.images_shown = 0
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

    def add_images(self, image_paths, source_file):
        for path in image_paths:
            self.images_shown += 1
            idx = self.images_shown
            card = ctk.CTkFrame(self)
            card.grid(row=(idx-1)//2, column=(idx-1)%2, padx=5, pady=5, sticky="ew")
            try:
                pil_img = Image.open(path)
                pil_img.thumbnail((150, 150))
                ctk_img = ImageTk.PhotoImage(pil_img)
                lbl_img = ctk.CTkLabel(card, image=ctk_img, text="")
                lbl_img.image = ctk_img
                lbl_img.pack(pady=5)
            except: pass
            ctk.CTkLabel(card, text=f"Foto #{idx}", font=("Arial", 10)).pack()
            ctk.CTkButton(card, text="Inna Metoda ↻", width=100, height=20,
                          font=("Arial", 10), fg_color="#555555",
                          command=lambda s=source_file: self.cycle_method(s)).pack(pady=5)

    def cycle_method(self, source_file):
        import random
        next_strat = random.choice([1, 2]) 
        self.retry_cb(source_file, next_strat)

class ComparisonSlider(ctk.CTkFrame):
    def __init__(self, master, width=800, height=600):
        super().__init__(master, width=width, height=height)
        self.w, self.h = width, height
        self.canvas = ctk.CTkCanvas(self, width=self.w, height=self.h, bg="#2b2b2b", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.img_b = self.img_a = None
        self.pos = width // 2
        self.canvas.bind("<B1-Motion>", self.move); self.canvas.bind("<Button-1>", self.move)
    def move(self, e):
        self.pos = max(0, min(e.x, self.w)); self.draw()
    def draw(self):
        if not self.img_b or not self.img_a: return
        self.canvas.delete("all")
        self.p_b = ImageTk.PhotoImage(self.img_b.crop((0, 0, self.pos, self.h)))
        self.canvas.create_image(0, 0, image=self.p_b, anchor="nw")
        self.p_a = ImageTk.PhotoImage(self.img_a.crop((self.pos, 0, self.w, self.h)))
        self.canvas.create_image(self.pos, 0, image=self.p_a, anchor="nw")
        self.canvas.create_line(self.pos, 0, self.pos, self.h, fill="white", width=2)
""",

    "gui/app.py": """import customtkinter as ctk
import threading
from tkinter import filedialog
from tkinterdnd2 import DND_FILES, TkinterDnD
from core.pipeline import RestorationPipeline
from gui.components import ComparisonSlider, CropGallery

ctk.set_appearance_mode("Dark")

class EPSApp(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self):
        super().__init__()
        self.TkdndVersion = TkinterDnD._require(self)
        self.geometry("1200x850")
        self.title("EPS AI Restoration v3.1 (Self-Install)")
        self.pipeline = RestorationPipeline(self.log, self.prog, self.update_gallery)
        
        self.grid_columnconfigure(1, weight=1); self.grid_rowconfigure(0, weight=1)
        self.create_sidebar()
        self.create_main()
        self.drop_target_register(DND_FILES)
        self.dnd_bind('<<Drop>>', self.drop)

    def create_sidebar(self):
        sf = ctk.CTkFrame(self, width=200, corner_radius=0)
        sf.grid(row=0, column=0, sticky="nsew")
        ctk.CTkLabel(sf, text="EPS AI\\nSOLUTIONS", font=("Arial", 20, "bold")).pack(pady=20)
        
        # Silnik AI
        ctk.CTkLabel(sf, text="Silnik AI:").pack(pady=(10,0))
        self.om_model = ctk.CTkOptionMenu(sf, values=self.pipeline.ai.available_models,
                                          command=self.change_engine)
        self.om_model.pack(pady=5)
        self.om_model.set(self.pipeline.ai.current_model_name)

        # Suwaki
        ctk.CTkLabel(sf, text="Kreatywność (1-10):").pack(pady=(20,0))
        self.sl_creat = ctk.CTkSlider(sf, from_=1, to=10, number_of_steps=9); self.sl_creat.pack(pady=5); self.sl_creat.set(5)

        ctk.CTkLabel(sf, text="Min. Obszar Skanu (%):").pack(pady=(20,0))
        self.sl_area = ctk.CTkSlider(sf, from_=0.01, to=0.20, number_of_steps=19); self.sl_area.pack(pady=5); self.sl_area.set(0.06)
        
        ctk.CTkButton(sf, text="⬆ AI Git Push", fg_color="#443355", border_width=1, 
                      command=lambda: threading.Thread(target=self.pipeline.git_push_auto).start()
                      ).pack(side="bottom", pady=20, padx=10)

    def create_main(self):
        tv = ctk.CTkTabview(self); tv.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        t_dash = tv.add("Start"); t_map = tv.add("Mapa Wycięć"); t_logs = tv.add("Logi")
        
        ctk.CTkButton(t_dash, text="Wybierz Skan / Pliki", height=50, command=self.sel).pack(pady=40, padx=40, fill="x")
        self.pb = ctk.CTkProgressBar(t_dash); self.pb.pack(pady=10, fill="x"); self.pb.set(0)
        self.lbl = ctk.CTkLabel(t_dash, text="Gotowy"); self.lbl.pack()
        ctk.CTkButton(t_dash, text="URUCHOM (3 Wątki)", fg_color="green", command=self.run).pack(pady=20)
        
        self.gallery = CropGallery(t_map, self.pipeline.retry_crop); self.gallery.pack(fill="both", expand=True)
        self.txt = ctk.CTkTextbox(t_logs); self.txt.pack(fill="both", expand=True)

    def change_engine(self, choice):
        self.pipeline.change_model(choice)

    def update_gallery(self, images, source_file): self.gallery.add_images(images, source_file)
    def drop(self, e): self.pipeline.set_input_files(self.tk.splitlist(e.data)); self.log(f"Pliki: {len(self.tk.splitlist(e.data))}")
    def sel(self): 
        f = filedialog.askopenfilenames()
        if f: self.pipeline.set_input_files(f); self.log(f"Pliki: {len(f)}")
    def run(self): 
        self.pipeline.update_settings(self.sl_creat.get(), self.sl_area.get())
        threading.Thread(target=self.pipeline.run).start()
    def log(self, m): self.after(0, lambda: self.txt.insert("end", m+"\\n"))
    def prog(self, v, t): self.after(0, lambda: (self.pb.set(v), self.lbl.configure(text=t)))
"""
}

def create_project():
    print("--- Generowanie EPS AI v3.1 (Auto-Fix) ---")
    
    # Tworzenie folderów
    for folder in ["config", "core", "gui"]:
        if not os.path.exists(folder): os.makedirs(folder)

    # Tworzenie plików
    for path, content in files_structure.items():
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Utworzono: {path}")

    # Konfiguracja API
    if not os.path.exists(".env"):
        print("\nPodaj klucz API Google:")
        k = input("Klucz: ").strip()
        with open(".env", "w", encoding="utf-8") as f:
            f.write(f"GOOGLE_API_KEY={k}")
        print("Utworzono .env")
    else:
        print("Plik .env już istnieje.")
    
    print("\nGOTOWE! Teraz uruchom tylko:")
    print("python main.py")
    print("(Program sam zainstaluje biblioteki przy pierwszym starcie)")

if __name__ == "__main__":
    create_project()