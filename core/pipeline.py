import time
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
