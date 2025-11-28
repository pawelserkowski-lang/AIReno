@echo off
setlocal
cd /d "%~dp0"
REM Uruchomienie aplikacji bez okna konsoli
start "" pythonw.exe "renowacja.pyw"
endlocal
