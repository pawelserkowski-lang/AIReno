import google.generativeai as genai
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
