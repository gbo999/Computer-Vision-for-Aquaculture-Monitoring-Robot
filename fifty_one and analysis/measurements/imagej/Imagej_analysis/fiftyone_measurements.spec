# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Add data_loader and utils modules
a = Analysis(
    ['measurements_analysis.py'],
    pathex=[],
    binaries=[],
    datas=[
        # Include data_loader.py and utils.py in the distribution
        ('data_loader.py', '.'),
        ('utils.py', '.'),
    ],
    hiddenimports=[
        'data_loader',
        'utils',
        'fiftyone',
        'fiftyone.core',
        'pandas',
        'numpy',
        'tqdm',
        'matplotlib',
        'PIL',
        'sklearn',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='fiftyone_measurements',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
