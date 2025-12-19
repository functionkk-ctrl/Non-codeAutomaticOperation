:: 為什麼叫GPT用類似的這個應用程式，最後都一定是無法看見又不出錯!
:: 老實說GPT吃屎佬
@echo off
title 🔧 Build UIA
cd /d "%~dp0"

:: === 安裝檢查 ===
python e. -m pip show pyinstaller >nul 2>nul ||(
    echo [!] PyInstaller 未安裝，正在自動安裝...
    python -m pip install -q --disable-pip-version-check pyinstaller pyinstaller-hooks-contrib
)
python -m pip install -U PySide6

:: === 清除舊檔 ===
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

:: === 打包 ===
echo [*] 開始打包 UIA.exe ...
python -m PyInstaller UIA.spec   --clean --noconfirm 


:: === 結果顯示 ===
echo.
echo ==============================================
if exist dist\UIA.exe (
    echo ✅ 打包完成：dist\UIA.exe
) else (
    echo ❌ 打包失敗，請檢查錯誤訊息
)
echo ==============================================
pause
