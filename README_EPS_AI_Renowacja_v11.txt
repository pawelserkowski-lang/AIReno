EPS AI – Renowacja starych zdjęć (v11)

Zawartość paczki:
- renowacja.pyw                 – główny skrypt aplikacji (GUI, obsługa Gemini/OpenAI/Grok)
- logo.png, tlo.png             – grafika interfejsu
- start_renowacja_silent.bat    – start aplikacji bez okna konsoli
- start_renowacja_with_log.bat  – start z logowaniem do katalogu logs
- requirements.txt              – lista wymaganych bibliotek Pythona

Instrukcja:
1. Zainstaluj Pythona (3.11+ zalecane).
2. W tym katalogu uruchom w terminalu:
     pip install -r requirements.txt
3. W pliku .env lub w zmiennych środowiskowych ustaw klucze:
     GOOGLE_API_KEY=...
     OPENAI_API_KEY=...
     GROQ_API_KEY=...    (opcjonalnie, jeśli używasz Groka)
4. Uruchamiaj aplikację plikiem:
     - start_renowacja_silent.bat    lub
     - start_renowacja_with_log.bat
