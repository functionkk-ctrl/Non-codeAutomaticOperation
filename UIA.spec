# -*- mode: python -*-
import os
import shutil
import PySide6
from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_data_files,
    collect_dynamic_libs
)

datas = collect_data_files('PySide6', subdir='qml')
qml_dst = os.path.join("dist", "PySide6", "qml")

block_cipher = None


a = Analysis(
    ['UIA.py'],
    pathex=[],
    binaries=[], # pywin32 通常在隱藏匯入中會自動處理，若噴 DLL 錯誤再加
    datas= [
        ('serviceAccountKey.json', '.'),
        ('templates', 'templates'),
        ('ui.qml', '.'),
        ('ilulu.glb', '.'),
        ('uploads_files_3351752_Rocking_Chair2.obj', '.'),
    ]+ collect_data_files('PySide6', subdir='qml'),
    hiddenimports=[
        'mediapipe',
        'geographiclib', 
        'firebase_admin', 
        'OpenGL.GL', 
        'OpenGL.GLU',
        'pynput.keyboard._win32', # pynput 的隱藏導入
        'pynput.mouse._win32',
        'matplotlib',
        'PySide6.QtCore', 
        'PySide6.QtGui', 
        'PySide6.QtWidgets', 
        'PySide6.QtQuick',
        'shiboken6'
    ],
    excludes=[ 'pandas', 'IPython', 'notebook', 'scipy','tkinter', 'unittest', 'email', 'http'], # 排除掉你沒用到的巨型庫
    cipher=block_cipher,
    noarchive=False,
)

# 為了加快「啟動」與「重複打包」速度，建議使用 PYZ 壓縮
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='UIA',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True, # 使用 UPX 壓縮可以減少體積，但會稍微增加解壓時間
    console=True, # 你的「火星」日誌輸出需要 console
    icon='icon.ico'
)
