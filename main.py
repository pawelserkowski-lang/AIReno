import os
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
