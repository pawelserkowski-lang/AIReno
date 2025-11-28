@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

if not exist "logs" mkdir "logs"

REM Składanie znacznika czasu YYYY-MM-DD_HH-MM-SS
for /f "tokens=1-4 delims=/. " %%a in ("%date%") do (
    set yyyy=%%d
    set mm=%%c
    set dd=%%b
)

for /f "tokens=1-3 delims=:." %%h in ("%time%") do (
    set hh=%%h
    set nn=%%i
    set ss=%%j
)

set hh=0!hh!
set hh=!hh:~-2!
set nn=0!nn!
set nn=!nn:~-2!
set ss=0!ss!
set ss=!ss:~-2!

set TS=!yyyy!-!mm!-!dd!_!hh!-!nn!-!ss!
set LOGFILE=logs\renowacja_!TS!.log

echo ===============================================  >> "!LOGFILE!"
echo  START: %date% %time%                            >> "!LOGFILE!"
echo  Katalog: %cd%                                   >> "!LOGFILE!"
echo ===============================================  >> "!LOGFILE!"
echo. >> "!LOGFILE!"

python.exe -u "renowacja.pyw" >> "!LOGFILE!" 2>&1

echo. >> "!LOGFILE!"
echo ------------------------------------------------ >> "!LOGFILE!"
echo  KONIEC DZIALANIA SKRYPTU.                      >> "!LOGFILE!"
endlocal
