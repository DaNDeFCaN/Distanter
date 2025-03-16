# -*- mode: python -*-
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# Основные настройки
a = Analysis(
    ['m1.py'],    # Главный скрипт
    pathex=[],             # Дополнительные пути поиска
    binaries=[],           # Внешние бинарные файлы
    datas=collect_data_files('matplotlib') + [('*.csv', '.')],  # Данные
    hiddenimports=[        # Скрытые зависимости
        'scipy.stats._stats',
        'scipy.special.cython_special'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher
)

# Сборка в EXE
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='Distanter',          # Имя выходного файла
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,              # Использовать UPX-сжатие
    console=False,         # Запуск без консоли (True для отладки)
    icon='icon.ico'        # Иконка приложения
)