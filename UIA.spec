# -*- mode: python -*-
import os
import shutil
import PySide6
from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_data_files,
    collect_dynamic_libs
)

qml_src = os.path.join(os.path.dirname(PySide6.__file__), "qml")
qml_dst = os.path.join("dist", "PySide6", "qml")
shutil.copytree(qml_src, qml_dst, dirs_exist_ok=True)

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
    ],
    hiddenimports=collect_submodules('PySide6'),
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
