:: 為什麼叫GPT用類似的這個應用程式，最後都一定是無法看見又不出錯!
:: 老實說GPT吃屎佬
@echo off
title 🔧 Build PySide6 UIA App
cd /d "%~dp0"

:: === 【設定】請在下方修改你要輸出的目標資料夾路徑 ===
set "OUT_DIR=C:\Users\User\Documents\MyCode"

:: === 自動鎖定 .venv 內部的核心執行檔路徑 ===
set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"
set "VENV_PYINSTALLER=%~dp0.venv\Scripts\pyinstaller.exe"

:: 檢查 .venv 是否存在
if not exist "%VENV_PYTHON%" (
    echo [❌ 錯誤] 找不到 .venv 環境，請確保此 .bat 放在專案根目錄下！
    goto ERROR_END
)

echo [*] 正在檢查 .venv 內的套件狀態...
"%VENV_PYTHON%" -m pip install -q pyinstaller pyinstaller-hooks-contrib PySide6

:: === 清除舊的暫存編譯檔 ===
echo [*] 清理舊的緩存...
if exist build rmdir /s /q build

:: === 強制指定導出路徑並執行打包 ===
echo [*] 開始打包 UIA.exe ...
echo [*] 輸出目錄: %OUT_DIR%
echo --------------------------------------------------

:: 修正：改用 .venv 的 pyinstaller.exe 直接啟動
:: 修正：使用 --distpath 強制修改打包輸出位置
"%VENV_PYINSTALLER%" UIA.spec --distpath "%OUT_DIR%" --clean --noconfirm

echo --------------------------------------------------
if exist "%OUT_DIR%\UIA.exe" (
    echo ✅ 打包完成！檔案已成功輸出至: %OUT_DIR%\UIA.exe
) else (
    echo ❌ 打包失敗，請往上捲動查看 PyInstaller 的詳細錯誤阻斷訊息！
)

:ERROR_END
pause
