import customtkinter as ctk
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
        ctk.CTkLabel(sf, text="EPS AI\nSOLUTIONS", font=("Arial", 20, "bold")).pack(pady=20)
        
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
    def log(self, m): self.after(0, lambda: self.txt.insert("end", m+"\n"))
    def prog(self, v, t): self.after(0, lambda: (self.pb.set(v), self.lbl.configure(text=t)))
