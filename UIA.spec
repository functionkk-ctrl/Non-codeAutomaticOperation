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
    pathex=['.'],
    binaries=collect_dynamic_libs("pywin32"),# 加入可能缺失的 DLL
    datas= [
        ('templates', 'templates'),
        ('ui.qml', '.'),
        ('ilulu.glb', '.'),
        ('uploads_files_3351752_Rocking_Chair2.obj', '.'),
    ]+ collect_data_files('PySide6', subdir='qml'),
    hiddenimports=collect_submodules('PySide6') + collect_submodules('shiboken6'),
)

exe = EXE(
    PYZ(a.pure, a.zipped_data, cipher=block_cipher),
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='UIA',
    console=True,
    icon='icon.ico'
)
