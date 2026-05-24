import os
os.environ["QT_QUICK_CONTROLS_STYLE"] = "Basic"
os.environ["QT_PA_PLATFORM"] = "windows:dpiawareness=0" # 強制讓 Qt 放棄 DPI 控制權
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "PassThrough" # 避免Qt二次修改DPI(?)
import json
import shutil
import mediapipe as mp
from geographiclib.geodesic import Geodesic
import firebase_admin
from firebase_admin import credentials, firestore, db
from pywinauto.application import Application
from pynput import mouse, keyboard
import PySide6
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtWidgets import QApplication
from pathlib import Path
import sys
from datetime import datetime
import threading
from difflib import SequenceMatcher
import math
import re
import time
import numpy as np
import pyautogui
import pytesseract
import cv2
from PIL import ImageGrab
from OpenGL.GLU import *
from OpenGL.GL import *
import psutil
from PySide6.QtGui import QSurfaceFormat,QGuiApplication  
from PySide6.QtCore import QObject, Slot, QTimer ,Qt
from geopy.geocoders import Nominatim
import subprocess
from Noesis import Noesis

# Android
# from plyer import gps
# from kivy.clock import Clock
# from kivy.utils import platform

# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)


def on_location(**kwargs):
    回覆(
        kwargs['lat'],
        kwargs['lon'],
        kwargs.get('altitude'),
        kwargs.get('speed')
    )


# gps.configure(on_location=on_location)
# gps.start(minTime=1000, minDistance=0)
"""
if platform == 'android':
    from plyer import accelerometer
    accelerometer.enable()
"""

# def read_imu(dt):
#     from microbit import accelerometer
#     pip install accelerometer
#     val = accelerometer.acceleration
#     if val:
#         ax, ay, az = val
#     print(ax, ay, az)

# Clock.schedule_interval(read_imu, 1/50)

# --- 基礎設定 --- python *.py
# D:\Python\Non-codeAutomaticOperation\Non-codeAutomaticOperation
# C:\Program Files\Tesseract-OCR\tesseract.exe # C:\Users\USER\AppData\Local\Programs\Tesseract-OCR\tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\USER\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
# 這是最穩定的寫法：獲取「目前這個 Python 檔案」所在的資料夾
if getattr(sys, 'frozen', False):
    # 如果是打包後的 .exe
    DATA_BASE = Path(sys.executable).parent
else:
    # 如果是直接跑 .py
    DATA_BASE = Path(__file__).resolve().parent
base_path = Path.home() / ".UIA"      # 可寫
TEMPLATE_DIRS = {
    "live_capture": base_path/ 'Live_capture',
    "User": base_path/ "User",  # 用戶隱私
    "communication": base_path/ "Communication",  # 用戶交流的訊息
    "speak": base_path/ "Speak",  # 交流的回覆
    "Noesis": base_path/ "Noesis",  # Noesis 吸收的知識
}
背景節點 = {
    "字": TEMPLATE_DIRS["Noesis"]/ '字',
    "數學": TEMPLATE_DIRS["Noesis"]/ "數學",  # 用戶隱私
    "作用力": TEMPLATE_DIRS["Noesis"]/ "作用力",  # 用戶交流的訊息
    "時間": TEMPLATE_DIRS["Noesis"]/ "時間",  # 交流的回覆
    "對話技巧": TEMPLATE_DIRS["Noesis"]/ "對話技巧",  # Noesis 吸收的知識
    "操作技巧": TEMPLATE_DIRS["Noesis"]/ "操作技巧",  # Noesis 吸收的知識
    "易經": TEMPLATE_DIRS["Noesis"]/ "易經",  # Noesis 吸收的知識
}

MATCH_THRESHOLD = 0.85
LANGS = 'chi_tra+eng' 
custom_config = '--psm 6'  # 假設影像是單一文字塊，可提升速度與準確度
DEBUG = True    
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# --- 共用工具 ---
alive_event = threading.Event()
cred = credentials.Certificate(DATA_BASE / "serviceAccountKey.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://console.firebase.google.com/u/1/project/uia-a-3c57f/database?fb_gclid=Cj0KCQjwn4qWBhCvARIsAFNAMigLRmhzV2i4mMqLSaBwKTlaQ37VHiYDnZqI-MS2gxBCVVFUR9SXTH4aAi5rEALw_wcB"
    })

def path_all(paths, target=None,exclude=None,time=None,use_orb=False):
    """
    預設排序為時間，由舊到新
    yield root(完整目錄), dirs(下一層的全部資料夾名), files(這層全部檔案含檔名)
    paths 依序遍歷 ./a 和 ./b 這兩個目錄，包含到最下層
        # for root, dirs, files in path_all(["./a", "./b"]):
    找到含 target 的檔案或資料夾，返回該根目錄 root/dirs，找不到則回傳 None
    找不到 target
        if not next(path_all(...)):
    paths, target, exclude =[],[]，all(...)同時都有target，any(...)任一有exclude
    """
    # 統一處理單一字串或清單
    search_paths = [Path(p) for p in (paths if isinstance(paths, list) else [paths])]
    target = [t for t in (target if isinstance(target, list) else [target])] if target else []
    exclude = [e for e in (exclude if isinstance(exclude, list) else [exclude])] if exclude else []

    def is_hidden(filepath):
        try:
            attrs = ctypes.windll.kernel32.GetFileAttributesW(str(filepath))
            return attrs & 0x02 if attrs != -1 else False
        except: return False
    
    found_any = False

    for base in search_paths:
        if not base.exists(): continue
        
        # 遍歷所有子目錄 (不純的搜尋開始)
        # 使用 rglob("*") 模擬 os.walk 的深度遍歷，但轉為 Path 物件
        sub_paths = sorted(base.rglob("*"), key=lambda p: p.stat().st_ctime)
        
        # 這裡模擬 os.walk 結構：找出所有的「目錄」作為 root
        roots = [p for p in sub_paths if p.is_dir()]
        # 加入 base 本身作為第一個 root
        roots.insert(0, base)

        for root in roots:
            # 獲取當前層級的資料夾與檔案名稱
            current_all = list(root.iterdir())
            if "不要隱藏" in exclude :
                dirs = sorted([d.name for d in current_all if d.is_dir() and not is_hidden(d)], 
                    key=lambda n: (root/n).stat().st_ctime)
                files = sorted([f.name for f in current_all if f.is_file() and not is_hidden(f)], 
                    key=lambda n: (root/n).stat().st_ctime)
            else:
                dirs = sorted([d.name for d in current_all if d.is_dir()], 
                    key=lambda n: (root/n).stat().st_ctime)
                files = sorted([f.name for f in current_all if f.is_file()], 
                    key=lambda n: (root/n).stat().st_ctime)
            get_ctime = [(root/f).stat() for f in files]
            # --- 目標檢索邏輯 ---
            # 1. 排除 (Exclude): 任一匹配即跳過is_hidden
            if exclude and any(e in (str(root) + "".join(files)) for e in exclude):
                continue
            
            # 2. 目標 (Target) 匹配：支援字串 find 與 ORB 比對
            if target:
                # 檢查是否「所有」target 條件都滿足
                match_all_targets = True
                for t in target:
                    # 優先使用字串 find (找目錄名或檔名)
                    text_match = any(t in name for name in (dirs + files))
                    
                    # 如果字串沒中，且開啟 ORB，則比對圖片相似度
                    orb_match = False
                    if not text_match and use_orb and t.endswith(('.png', '.jpg')):
                        # 這裡的 t 是目標圖片路徑或特徵，files 是當前目錄檔案
                        orb_match = any(全能ORB(t, root/f, similar=0.9) for f in files if f.endswith(('.png', '.jpg')))
                    
                    if not (text_match or orb_match):
                        match_all_targets = False
                        break
                
                if match_all_targets:
                    # 僅保留包含 target 字眼的檔案 (符合你原本 files=... for t in target 邏輯)
                    matched_files = [f for f in files if any(t in f for t in target)]
                    yield root, dirs, matched_files, get_ctime 
                    found_any = True
            elif time is not None:
                # 找同時間（誤差0.5秒內）的檔案
                matched_files = [
                    f for f in files
                    if abs((root / f).stat().st_ctime - time) <= 0.5
                ]
                if matched_files:
                    yield root, dirs, matched_files, get_ctime 
                    found_any = True
            else:
                # 無 target 則全產出
                yield root, dirs, files, get_ctime 
                found_any = True
        if not found_any:
            return None

import inspect
def make_folder(folder_name, class_name=None, content_classes=None):
    """
    在 base_path 下創建資料夾 folder_name（如果不存在）和腳本含內容
    inspect.getsource(Class )，複製原始碼
    """
    folder_path = base_path / str(folder_name)
    回覆(f"確保有資料夾{folder_path}")
    folder_path.mkdir(parents=True, exist_ok=True)  # 確保父資料夾也創建
    if class_name:
        # 建立檔名，例如 folder_name.py 或對象名.py
        file_path = folder_path / f"{class_name}.py"
        # 如果檔案不存在，直接寫入 class 定義
        if not file_path.exists():
            # 這裡可以把你的 asdf 統一算法模板寫進去
            source_code = "\n\n".join([inspect.getsource(cls) for cls in content_classes])
            # 加入該節點的專屬初始化代碼
            instance_code = f"\n\n# 當前節點：{class_name}\nnode = {class_name}()\n"
            file_path.write_text(source_code + instance_code, encoding='utf-8')
    return folder_path


def make_json_content(file_path, file_name,  key, value):
    """
    在 base_path 下建立或更新 file_path 文件之下並建立或更新 file_name.json，之內建立或更新內容 key:value
    key 只能不可變且可哈希的類型，可以tuple(list)、字串、數字、元組
    """
    path = make_folder(file_path) / (file_name + ".json")
    if path.exists():  # 有檔案
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}
    else:
        data = {}
    data.setdefault(key, [])
    data[key].append(value)
    # 寫入（原子寫入比較安全）避免「寫到一半斷電檔案壞掉」。
    temp_path = path.with_suffix(".tmp")
    with open(temp_path , "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    temp_path.replace(path)

def make_json_content(file_path, file_name, content):
    """
    在 base_path 下建立或更新 file_path 文件之下並建立或更新 file_name.json，之內建立或更新內容 key:value
    key 只能不可變且可哈希的類型，可以tuple(list)、字串、數字、元組
    """
    path = make_folder(file_path) / (file_name + ".json")
    if path.exists():  # 有檔案
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data =[]
    else:
        data =[]
    data.append(content)
    # 寫入（原子寫入比較安全）避免「寫到一半斷電檔案壞掉」。
    temp_path = path.with_suffix(".tmp")
    with open(temp_path , "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    temp_path.replace(path)

def read_json_content(file_path, file_name,  key):
    """讀取失敗時回傳[]，成功回傳 key(None=全部) 的內容"""
    path = base_path/file_path / f"{file_name}.json"
    if path.exists():  # 有檔案
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if key is None:
                    return list(data.values()) # 全部[key:value]，Json{} 規定一定要有 Key
                elif isinstance(data, dict):
                    return data.get(key, []) # {不同key:value}
                elif isinstance(data, list):
                    return data[key] # [不同key:value]
        except (FileNotFoundError, json.JSONDecodeError): # 錯誤時中斷
            回覆(f"⚠️ 讀取失敗: {file_path}，可能是檔案不存在或格式錯誤。")
            return []
    else:
        回覆(f"⚠️ 讀取失敗: {file_path}，可能是檔案不存在或格式錯誤。")
        return []

def 回覆(*args):
    # 將所有傳進來的參數（不論是數字還是字串）都轉成字串，並用空格串接
    ss = " ".join(str(arg) for arg in args)
    if ss.strip():  # 確保內容不為空
        MAX_LEN = 1000 
        if len(ss) > MAX_LEN:
            ss = ss[:MAX_LEN] + "..."
        print(ss)
        Backend().conversation(msg=ss)
    else:
        print("請輸入有效的回覆內容")
        Backend().conversation(msg="請輸入有效的回覆內容")
  
import requests
from bs4 import BeautifulSoup
def fill_accurate_images(keyword, word_dir):
    """
    針對爬到的詞，去維基百科抓取最準確的定義圖片
    """
    api_url = "https://wikipedia.org"
    params = {
        "action": "query",
        "format": "json",
        "prop": "pageimages",
        "titles": keyword,
        "pithumbsize": 1000  # 取得高品質大圖，SIFT 才準
    }
    try:
        res = requests.get(api_url, params=params, timeout=5).json()
        pages = res.get("query", {}).get("pages", {})
        for pg_id in pages:
            if "thumbnail" in pages[pg_id]:
                img_url = pages[pg_id]["thumbnail"]["source"]
                img_data = requests.get(img_url).content
                img_filename=全能ORB(img_data,path=word_dir/ "圖片_SIFT" ,npy="a")
                os.rename(img_filename,f"{keyword}_orig.jpg")
                回覆(f"✅ 已從維基百科抓取並儲存精確圖片: {keyword}_orig.jpg")
                return True
    except:
        pass
    回覆(f"⚠️ 維基百科未找到精確圖片，已使用爬蟲抓取的圖片: {keyword}_orig.jpg")
    return False
def asbc_stealth_search(keyword,home_url,search_url,payload):
    # 1. 初始化 Session (模擬瀏覽器開啟後的狀態)
    # TODO:*** 最簡便的使用方式
    session = requests.Session()
    # home_url = "https://asbc.iis.sinica.edu.tw/"
    session.get(home_url, timeout=10) 
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': home_url
    }
    # 2. 定義目標 API / CGI 路徑 (中研院常用的查詢端點)
    search_url = home_url
    # 3. 準備參數 (Key: 你的關鍵字)
    # 這裡的參數名稱需要根據實際 F12 觀察到的 Network Form Data 調整
    payload = {
        'query': keyword,
        'search_mode': 'text',
        # 把你列出的這些範圍，根據 API 的規則全部填進去 (以下為假設參數名)
        'genre': '報導,散文,評論,詩歌,信函,會話,演講,語錄,劇本,公告啟事,說明手冊,傳記日記,會議記錄,小說故事寓言,廣告或圖文',
        'style': '記敘,說明,論說,描寫',
        'media': '報紙,一般雜誌,學術期刊,教科書,工具書,學術論著,一般圖書,視聽媒體',
        'topic': '哲學,科學,社會,藝術,生活,文學'
    }
    try:
        # 4. 直接發送 POST 請求 (不開啟網頁)
        response = session.post(search_url, data=payload, headers=headers, timeout=10)
        response.encoding = 'utf-8' # 強制編碼避免亂碼

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            results = soup.select('.result_item') 
            data_list = [item.text.strip() for item in results] # 爬蟲抓到的字
            word_dir = make_folder(TEMPLATE_DIRS["Noesis"]/keyword) 
            save_path = word_dir / "text.npy"
            np.save(str(save_path), np.array(data_list, dtype=object))
            回覆(f"✅ 已將爬蟲結果轉存為: {save_path}")
            categories = ["同義", "反義", "干涉它為新義", "主從義", "SIFT"]
            for a in categories: make_folder(word_dir/a)

            # 1. 找到圖片標籤 (假設 class 名稱為 'result_img')
            img_tags = soup.select('.result_img img') 
            img_folder = word_dir / "SIFT" # 定義圖片存放路徑
            for i, img in enumerate(img_tags):
                img_url = img.get('src')
                if not img_url: continue
                # 處理相對路徑網址
                if img_url.startswith('//'): img_url = 'https:' + img_url
                # 2. 下載圖片並儲存
                try:
                    img_data = requests.get(img_url, timeout=5).content
                    img_filename=全能ORB(img_data,path=img_folder,npy="a")
                    os.rename(img_filename,f"image_{i}")
                    回覆(f"圖片已儲存: {img_filename}")
                except:
                    print(f"圖片 {img_url} 下載失敗")
                    fill_accurate_images(keyword,word_dir=word_dir)
            return data_list
    except Exception as e:
        回覆(f"⚠️ 隱藏請求失敗: {e}")
        return []
    
def 看字():
    """ 
    獨立的 OCR 執行緒，確保每秒穩定執行
    OCR給圖中的文字，移除 資料夾不可用詞，由左至右，由上往下，可能是文字
    TODO:*** 理解拓樸關係
    儲存為資料夾
    """
    invalid_chars = re.compile(r'[\/\\\:\*\?\"\<\>\|]')
    資料夾最長字數=30
    while not alive_event.is_set():
        start_time = time.time()
        try:
            data = get_screenshot(Langs=LANGS)
            回覆("分析文章中...")
            # 最高強度的相似度比對，當場撈出最新 X, Y 坐標
            # 10:text, 5:left, 6:top, 7:width, 8:height, 9:conf
            text_pts_data=C.where(10,5,6,7,8,9)(data)
            matrix  = find_array(text_pts_data,C(5)>60 )
            if len(matrix)>0:
                回覆("找到字")
            else:
                回覆("找不到字")
            folder_name=find_array(matrix,(C(0)!= invalid_chars) & (C(5)>60), C(1)&C(2) )
            folder_name_text=C.where(0)(folder_name)[:資料夾最長字數]
            # 4. 如果有讀到有效的字，就建立資料夾
            if folder_name_text:
                make_folder(TEMPLATE_DIRS["live_capture"]/folder_name_text)
                # 這裡您可以選擇把截圖也存進該資料夾，例如：
                # cv2.imwrite(f"{folder_name_text}/screenshot.png", img)
            else:
                回覆("找不到文字")
            # 5. 後台除錯專用 (使用 print 隔離，絕不呼叫 回覆() 灌爆主線)
            print(f"[OCR] 辨識成功 | 總字數: {len(folder_name_text)} | 耗時: {time.time() - start_time:.3f}s")
        except Exception as e:
            print(f"[OCR_ERROR] 發生異常: {e}")
        # 6. 動態精確配時 (確保每秒執行一次)
        time.sleep(max(0, 1.0 - (time.time() - start_time)))

def text_make_background(path=None):
    """解析圖片的文字並分類進靜態資料夾中"""
    if path is None:
        path=make_folder(TEMPLATE_DIRS["Noesis"/"字"]) # 可以精確地調整路徑
    img = get_screenshot()
    text = pytesseract.image_to_string(img, lang=LANGS, config=custom_config)
    ocr_lines = np.array([line.strip() for line in text.split('\n') if line.strip()]) # 合併成連續的字和標點符號
    # ocr_lines = np.array([line[1][0] for res in essay for line in res] ) 
    import string
    all_punc = list(string.punctuation) + ["，", "。", "、", "；", "：", "？", "！"] # punc 表點符號的縮寫
    pronouns = [
        "i", "me", "my", "mine", "myself", 
        "you", "your", "yours", "yourself", "yourselves",
        "he", "him", "his", "himself", 
        "she", "her", "hers", "herself", 
        "it", "its", "itself", 
        "we", "us", "our", "ours", "ourselves", 
        "they", "them", "their", "theirs", "themselves",
        "this", "that", "these", "those"
        ,
        "我", "你", "您", "他", "她", "它", "牠", "祂",
        "我們", "你們", "您們", "他們", "她們", "它們", "牠們", "祂們",
        "自己", "自個兒", "人家", "別人",
        "這", "那", "這裡", "那裡", "這兒", "那兒", "這個", "那個",
        "先生","帥哥","小姐","美女",
    ]
    副詞=[
        # 1. 劇烈、巨大變化
        "Dramatically",    # 戲劇性地
        "Drastically",     # 猛烈地
        "Substantially",   # 大幅度地
        "Considerably",    # 相當大地
        "Significantly",   # 顯著地
        "Markedly",        # 明顯地
        "Enormously",      # 巨大地
        "Hugely",          # 巨大地
        # 2. 快速、突然變化
        "Sharply",         # 急遽地
        "Rapidly",         # 快速地
        "Quickly",         # 迅速地
        "Swiftly",         # 迅速地
        "Steeply",         # 陡峭地
        "Suddenly",        # 突然地
        "Abruptly",        # 突然地
        # 3. 穩定、持續變化
        "Steadily",        # 穩定地
        "Gradually",       # 逐漸地
        "Moderately",      # 適度地
        "Progressively",   # 漸進地
        # 4. 輕微、微小變化
        "Slightly",        # 輕微地
        "Marginally",      # 微小地
        "Minimally",       # 極小地
        "Slowly"           # 緩慢地
    ]
    動詞 = {
        "be", "have", "do", "say", "get", "make", "go", "know", "take", "see",
        "come", "think", "look", "want", "give", "use", "find", "tell", "ask", "work",
        "seem", "feel", "try", "leave", "call", "keep", "let", "begin", "help", "talk",
        "start", "show", "hear", "play", "run", "move", "like", "live", "believe", "hold",
        "bring", "happen", "write", "provide", "sit", "stand", "lose", "pay", "meet", "include",
        "continue", "set", "learn", "change", "lead", "understand", "watch", "follow", "stop", "create",
        "speak", "read", "allow", "add", "spend", "grow", "open", "walk", "win", "offer"
        ,
        "是", "有", "做", "說", "得到", "製造", "去", "知道", "拿", "看",
        "來", "想", "瞧", "想要", "給", "使用", "尋找", "告訴", "詢問", "工作",
        "似乎", "感覺", "嘗試", "離開", "稱呼", "保持", "讓", "開始", "幫助", "談話",
        "啟動", "展示", "聽到", "玩", "跑", "移動", "喜歡", "居住", "相信", "握住",
        "帶來", "發生", "寫", "提供", "坐", "站立", "遺失", "支付", "遇見", "包含",
        "繼續", "設置", "學習", "改變", "領導", "明白", "觀看", "遵循", "停止", "創造",
        "說話", "閱讀", "允許", "增加", "花費", "成長", "打開", "走路", "贏得", "提供"
    }
    數量詞 = [
        # --- 抽象數量 (Quantifiers) ---
        "all", "most", "many", "much", "some", "any", "a few", "few", 
        "a little", "little", "several", "each", "every", "plenty of", "enough",
        # --- 單位量詞 (Units/Measure Words) ---
        "a piece of", "a slice of", "a loaf of", "a bar of", "a sheet of", 
        "a pack of", "a box of", "a pair of", "a set of", "a bottle of", 
        "a cup of", "a glass of", "a bowl of", "a plate of", "a can of", 
        "a jar of", "a spoon of", "a serving of", "a bunch of",
        # --- 群體量詞 (Collective Nouns) ---
        "a group of", "a crowd of", "a team of", "a flock of", "a herd of", 
        "a school of", "a swarm of",
        # --- 中文特有量詞之英文對應 ---
        "general unit", "polite unit for people", "long/thin objects", 
        "flat objects", "objects with handles", "bound objects", 
        "machines/movies", "appliances/vehicles", "courses/light", "letters"
        ,
        # --- 抽象數量 ---
        "全部", "大部分", "很多 (可數)", "很多 (不可數)", "一些", "任何/一些", "幾個", "很少 (幾乎沒有)", 
        "一點點", "很少 (幾乎沒有)", "幾個/數個", "每個", "每個/所有", "很多/充足", "足夠",
        # --- 單位量詞 ---
        "一張/一塊/一份", "一片 (薄片)", "一條 (麵包)", "一條 (肥皂/巧克力)", "一張 (紙/床單)", 
        "一包/一捆", "一盒/一箱", "一雙/一副", "一套/一組", "一瓶", 
        "一杯 (熱飲)", "一杯 (冷飲)", "一碗", "一盤", "一罐 (鋁罐)", 
        "一罐 (玻璃罐)", "一匙", "一份 (餐點)", "一束/一串",
        # --- 群體量詞 ---
        "一群 (人/物)", "一群 (擁擠的人)", "一隊", "一群 (鳥/羊)", "一群 (牛/象)", 
        "一群 (魚)", "一群 (昆蟲)",
        # --- 中文特有量詞 ---
        "個", "位", "支/把", "張", "把", "本", "部", "臺", "道", "封"
    ]
    介詞 = [
        # --- 空間位置 (Space) ---
        "in", "on", "at", "under", "below", "above", "over", "beside", 
        "next to", "between", "among", "behind", "in front of", "near", 
        "opposite", "inside", "outside",
        # --- 時間關係 (Time) ---
        "at", "in", "on", "before", "after", "during", "since", "for", 
        "until", "from...to", "within", "throughout",
        # --- 方向與移動 (Movement) ---
        "to", "into", "onto", "out of", "off", "up", "down", "across", 
        "through", "along", "past", "towards", "around",
        # --- 邏輯、關係與其他 (Other) ---
        "of", "with", "without", "by", "for", "about", "against", "like", 
        "as", "besides", "except", "instead of", "despite", "including"
        ,
        # --- 空間位置 ---
        "在...裡面", "在...上面", "在 (特定地點)", "在...下方", "在...之下", "在...上方", "跨越...之上", "在...旁邊", 
        "在...隔壁", "在...之間 (兩者)", "在...之中 (三者以上)", "在...後面", "在...前面", "靠近", 
        "在...對面", "在...內部", "在...外部",
        # --- 時間關係 ---
        "在 (精確時間)", "在 (時段/月份/年份)", "在 (日期/星期)", "在...之前", "在...之後", "在...期間", "自從", "持續 (一段時間)", 
        "直到", "從...到", "在...之內", "貫穿/整個 (時期)",
        # --- 方向與移動 ---
        "往/向", "進入", "到...之上", "離開/從...出來", "從...掉落/離開", "向上", "向下", "橫越", 
        "穿過", "沿著", "經過", "朝向", "環繞/大約",
        # --- 邏輯、關係與其他 ---
        "的/屬於", "和...一起/用", "沒有/缺乏", "藉由/在...旁邊", "為了/給", "關於", "反對/靠著", "像...一樣", 
        "作為", "除此之外 (包含)", "除了...之外 (排除)", "代替", "儘管", "包含"
    ]
    粗略提問={
        "誰":"人物詢問對象、身份、角色",
        ["哪個","哪位"]:"特定人物或物件從範圍中",
        "選擇什麼":["事物","內容詢問名稱、資訊、定義"],
        "哪裡":"地點詢問位置、區域",
        "何時":"時間詢問日期、時段、順序",
        ["為何","為什麼"]:"原因詢問動機、成因、目的",
        ["如何","怎麼"]:"方法，詢問流程、處理方式",
        ["多少","數量"]:"詢問數目、程度",
        ["哪一種","分類"]:"詢問種類、類型是否判定要求是/否答案",
        "誰的":"所有權詢問歸屬",
        "用什麼":"工具詢問手段、媒介結果",
        "是什麼結果":"詢問輸出",
        "什麼":"未指定類別的泛用資訊提問。",
    }
    list_姓氏_中文 = [
        "陳", "林", "黃", "張", "李", "王", "吳", "劉", "蔡", "楊", 
        "許", "鄭", "謝", "洪", "郭", "邱", "曾", "廖", "賴", "徐", 
        "周", "葉", "蘇", "莊", "呂", "江", "何", "蕭", "羅", "高", 
        "潘", "簡", "朱", "鍾", "游", "彭", "詹", "胡", "施", "沈"
    ]
    # 英文常見姓氏（Common Surnames）
    list_姓氏_英文 = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
        "Hernandez", "Lopez", "Gonzales", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
        "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
        "Walker", "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores"
    ]
    def 取詞(ss):
        if isinstance(ss, str):ss=[ss]
        for s in ss:
            p=path_all(path,s)
            if next(p):
                s=row(1,p)
    形容詞,名詞=[],[]
    取詞(["形容詞","名詞"])
    從對主的向量 = { "相似詞","反義詞"} # (詞)意義，找到主從詞(的顯示)，增加主是正向、減少主是負向
    Positive_Ops = ["增加", "提升", "擴大","有助於","有益於"]
    Negative_Ops = ["減少", "降低", "萎縮","減弱","忘記"]
    glues = { 
        "等價": ["是", "為", "is", "be", "定義"], 
        "並列": ["還", "和", "與","且", "and", "狀", "類"], 
        "排斥": ["不", "非", "質", "反", "對立"], 
        "干涉": ["使", "讓", "變成", "become", "干涉"], 
        "主從": ["有", "的", "of", "have", "屬" # 前為主
            ,"belong to","屬於","附屬"], # 後為主
    }
    等價_mask=find_array(ocr_lines,(C(0) in glues["等價"]).get_mask)  
    並列_mask=find_array(ocr_lines,(C(0) in glues["並列"]).get_mask) 
    排斥_mask=find_array(ocr_lines,(C(0) in glues["排斥"]).get_mask) 
    干涉_mask=find_array(ocr_lines,(C(0) in glues["干涉"]).get_mask) 
    主從_mask=find_array(ocr_lines,(C(0) in glues["主從"]).get_mask) 
    主從正向_mask=find_array(ocr_lines,(C(0) in Positive_Ops).get_mask) 
    主從負向_mask=find_array(ocr_lines,(C(0) in Negative_Ops).get_mask) 
    動詞_mask=find_array(ocr_lines,(C(0) in 動詞).get_mask)
    # TODO:***** 解題
    出題第幾行開始=find_array(ocr_lines,(C(0) in "出題幾行").get_mask)  
    出題幾行=find_array(ocr_lines,((C(0).roll(1)== "出題") and (C(0).roll(-1)== "行")).get_mask)
    出題囉=find_array(ocr_lines,C(0).get_mask- 出題第幾行開始<出題幾行) 
    if next(row(1,path_all(path,C(0).func(出題囉)/"主"))): # 詞/主
        找到主=row([0,1],path_all(path,C(0).func(出題囉)/"主")) # path/.../(從)詞/主,找到主(詞)
        上=找到主[1] 
        下=find_array(找到主,C(0).roll(-1)=="主")
    問題類型_粗_mask=find_array(ocr_lines,(C(0) in 粗略提問).get_mask) 
    問題類型_粗=C(粗略提問[ocr_lines[問題類型_粗_mask]] , ocr_lines)
    問題類型_細=find_array(ocr_lines,(C(0) not in 粗略提問) ) 
    # 提問中的名詞
        # 新詞用火水土得其義，其餘者金木得其行，義義空間上的時間上相近用火木得其關聯，動機用水差得其機率
    解題(出題=出題囉,上=上,下=下,問題類型=問題類型_細,介詞=介詞,名詞=名詞) # 異步分析型提問，是最快矯正回答者的方式。 任何主題導向準確求解形式的問題類型，舉例如何知道提問是甚麼問題類型

    def make(a,b,f):
        if a != path:
            make_folder(path/a/f/b) # make_folder 內置處理已建立則用同路徑
        else:
            make_folder(path/f/b) # make_folder 內置處理已建立則用同路徑
    # 動詞+副詞、動詞 形容詞 化or性、形容詞+名詞、形容詞 動詞。TODO:** 形 副 趨勢動詞(6)有負負得正 作用，並非疊加
    # 形容詞的絕對位置:形 的 名、動 得 形、形 化。
    動詞=ocr_lines[動詞_mask]
    形容_mask1=find_array(動詞,(C(0).roll(-1) == "得").roll(-1).get_mask) 
    形容_mask2=find_array(ocr_lines,(C(0) in ["的","化"]).roll(1).get_mask)
    形容=ocr_lines[(形容_mask1+形容_mask2)]
    make(path,形容,"形容詞")
    # 名詞的絕對位置:數量名、的名、介名介
    名_mask=find_array(ocr_lines,(C(0) =="的"+介詞+數量詞).roll(-1).get_mask)
    make(path,ocr_lines[名_mask],"名詞")
    # 句法：詞1主、詞1從
    等價=ocr_lines[等價_mask]
    意義_前者,意義_後者=find_array(等價,C(0).roll(1)),find_array(等價,C(0).roll(-1))
    make(意義_前者,意義_後者,"意義")
    # 句法：a2b、(a2b)2(c2d) 第一種2在誇號內 第二種在誇號外
    並列=ocr_lines[並列_mask]
    第一種=find_array(並列,C(0).isin(C(0)))
    第二種_mask=C(0).func(並列).__ne__(第一種).get_mask
    if 第一種.get_mask < 第二種_mask:
        並列_前者 =第一種
    if 第一種.get_mask > 第二種_mask:
        並列_後者 =第一種
    make(並列[並列_前者],並列[並列_後者],"並列")
    # 句法:不_
    被排斥_後者=C(0).func(ocr_lines[排斥_mask]).roll(-1)
    make(ocr_lines[排斥_mask],被排斥_後者,"排斥")
    # 句法:,_是_,。分配"是"的前後， 前者a太短(代名詞、"")則"追朔前者"；後者b 則都是到下一個標點符號的段落。 # TODO:*** 第二順位
    干涉=ocr_lines[干涉_mask]
    干涉_代名詞=find_array(干涉,(C(0).roll(1) in ["",pronouns,all_punc]).get_mask)
    干涉_前者=find_array(干涉_代名詞,(C(0).diff.diff==0))
    干涉_後者=find_array(干涉,C(0).roll(1).get_mask == 干涉_前者)
    make(干涉[干涉_前者],干涉[干涉_後者],"干涉")
    # 句法:詞4a結果。 # TODO:*** 第一順位
    主從=ocr_lines[主從_mask]
    主從_後為主=find_array(主從,C(0)in ["belong to","屬於","附屬"])
    主從_後主=find_array(主從_後為主,C(0).roll(1))
    make(find_array(主從_後主,C(0).roll(-1)),find_array(主從_後主,C(0).roll(1)),"主/中")
    make(find_array(主從_後主,C(0).roll(1)),find_array(主從_後主,C(0).roll(-1)),"從/中")

    主從_前為主=find_array(主從,C(0) not in ["belong to","屬於","附屬"])
    主從_前主=find_array(主從_前為主,C(0).roll(1))
    make(find_array(主從_前主,C(0).roll(1)),find_array(主從_前主,C(0).roll(-1)),"主/中")
    make(find_array(主從_前主,C(0).roll(-1)),find_array(主從_前主,C(0).roll(1)),"從/中")
    # 句法：影響詞 5 副詞、影響詞 5 名詞、影響詞 形容詞 5。 5=趨勢動詞
    從是正向1=find_array(ocr_lines[主從正向_mask],C(0).roll(-1)in 名詞+副詞) # "影響詞" 趨勢動詞 副詞
    從是正向2=find_array(ocr_lines[主從正向_mask],(C(0).roll(1) in 形容詞)) # "影響詞" 形容詞 趨勢動詞
    從是負向1=find_array(ocr_lines[主從負向_mask],C(0).roll(-1)in 名詞)
    從是負向2=find_array(ocr_lines[主從負向_mask],(C(0).roll(1) in 形容詞))
    def 尋主(a,b):
        主_p=path_all(path/a/"主")
        if next(主_p):
            出現的主=row(1,find_array(主_p,C(0) in b)) 
            for 主 in 出現的主:
                從_p=path_all(path/主/"從"/"中"/a)
                if next(從_p):
                    os.remove(path/主/"從"/"中"/a)
                    return path/主/"從" # 從 未分方向，通知主
    # make(影響詞,趨勢動詞,"正負向") # 尋主(趨勢動詞,影響詞) # base/a主/方向/從
    make(C(0).從是正向1.roll(1),從是正向1,尋主(從是正向1,C(0).從是正向1.roll(1))/"正向") 
    make(C(0).從是負向1.roll(1),從是負向1,尋主(從是負向1,C(0).從是負向1.roll(1))/"負向")
    make(C(0).從是正向2.roll(2),從是正向2,尋主(從是正向2,C(0).從是正向2.roll(2))/"正向")
    make(C(0).從是負向2.roll(2),從是負向2,尋主(從是負向2,C(0).從是負向2.roll(2))/"負向")
    
def 解題(出題,介詞,名詞,
    金=None,水=None,木=None,火=None,土=None,
    y=0,m=0,d=0,h=0
    ): 
    連接詞 = {
        ["和", "跟", "與", "既", "及", "而", "又", "一面⋯⋯一面⋯⋯"], # 並列關係
        ["或", "或者", "還是"], # 選擇關係
        ["但是", "不過", "雖然", "然而"], # 轉折關係
        ["因為", "因此", "所以", "由於", "以致"], # 因果關係
        ["不但", "不僅", "而且", "何況", "並", "且"], # 遞進關係
        ["不管", "只要", "除非"], # 條件關係
        ["先⋯⋯再⋯⋯最後⋯⋯"] # 順成時間關係
        ,
        { # 對等連接詞_FANBOYS
            "因為/為了", # For
            "和/而且", # And
            "也不", # Nor
            "但是", # But
            "或者", # Or
            "然而/但是", # Yet
            "所以" # So
        },
        { # 相關連接詞
            "A 和 B 兩者皆是", # Both A and B
            "不僅 A 還有 B", # Not only A but also B
            "不是 A 就是 B", # Either A or B
            "既不是 A 也不是 B", # Neither A nor B
            "不論是 A 還是 B" # Whether A or B
        },
        { # 從屬連接詞
            ["when", "while", "before", "after", "as soon as", "since"], # 時間
            ["because", "as", "since", "so that", "in order that"], # 原因目的
            ["if", "unless", "as long as"], # 條件
            ["although", "though", "even if"] # 讓步
        }
    }
    def 相似詞(a):
        尋主=row([0,1],path_all(TEMPLATE_DIRS["Noesis"]/a,"主"))
        同向=row(1,path_all(尋主[0]/尋主[1]/"從",a))
        return 同向
    def 不在場(a):
        從正負=row(1,path_all(TEMPLATE_DIRS["Noesis"]/a/"從",["正向","負向"])) # 從/+-
        return 從正負

    if isinstance(出題,str):
        #  in [""] 會用到的詞        # ，.roll(1) in [""] 料理對象        # ，path_all(在場的詞,從) 找不在場的
        # TODO:*** 獲取出現的全部共同主(義義空間)，子義義在主之下的差異(例如屁連著眼=gemini屁眼)
            # 異步分析型構句
        """
        # 先抓人名 或物品 ，在找共同的主，計算差異，看子異異(人名或物品)的從(隱含異異)受到的影響。先對獲益，後對流轉。
            # 沈秋 和貓，共同主為江湖，沈秋(轉換)總帶著一隻獨眼黑貓(獲益)。 (流轉)到七年前  沈秋樸通少年 官兵剿民(流轉) (轉換方式)兩死一傷貓 (獲益)求教於老劍客 (動機)老劍客說復仇
        """
        劫_mask= find_array(出題,(C(0) in 相似詞("姓氏與名字")).get_mask) # 包含各語言的姓氏，名字，代名詞我以外的均要往更前面找，找不到則是後面會提
        土_mask= find_array(出題,(C(0) in 名詞).get_mask) # 獲益 名詞
        金_mask= find_array(出題,(C(0) in 相似詞("動名詞")).get_mask) # 動機 動名詞
        水_mask= find_array(出題,(C(0) in 連接詞).roll(-1).get_mask) # 過程 連接詞
        木_mask= find_array(出題,(C(0).isin(C(0).func(出題)) ).roll(-1).get_mask) # 有規律 出現時長高
        火_mask= find_array(出題,(C(0) in 介詞).roll(-1).get_mask) # 轉換 介詞
        # 找到的代名詞為上一位
        土=(出題[土_mask:-1] , find_array(劫_mask,(C(0)<土_mask)) and (C(0).roll(-1)>土_mask))
        金=(出題[金_mask:-1] , find_array(劫_mask,(C(0)<金_mask)) and (C(0).roll(-1)>金_mask))
        水=(出題[水_mask:-1] , find_array(劫_mask,(C(0)<水_mask)) and (C(0).roll(-1)>水_mask))
        木=(出題[木_mask:-1] , find_array(劫_mask,(C(0)<木_mask)) and (C(0).roll(-1)>木_mask))
        火=(出題[火_mask:-1] , find_array(劫_mask,(C(0)<火_mask)) and (C(0).roll(-1)>火_mask))
        # TODO:**詞無相關詞
    else: 
        回覆("出題不符合格式")
    # 中性 傳遞，吸引 吸收，排斥 抵銷
    math_dist = {
        "加":{"大小正負關係":None,"小數":None,"加減乘除":"中"},
        "減":{"大小正負關係":None,"小數":None,"加減乘除":"吸"},
        "乘":{"大小正負關係":None,"小數":None,"加減乘除":"查表後中"},
        "除":{"大小正負關係":None,"小數":None,"加減乘除":"(查表(1/1~1/9)移除分母後相乘分子)"}
    }
    作用力_dist = {
        "起伏":"前後差異",
        "加速度":"二次前後差異",
        "雙方利益關係":"分子分母對調得到雙方的交換率",
        "意義上的意義":"不同意義相乘得到作用結果",
    }
    # 人倫
        # 為何有趣?大吸(提高流量經過低壓通道)與通用(子意義在意義上必然重疊)
    有趣對話技巧提升ist = {
        "初步":"接力回應＋情緒認可＋故事延伸，用開放式提問把對話推向下一層，並自然留下「下次再聊」的鉤子",
        "話題":"過渡句 接住",
        "延伸話題":"引入相關故事或分享經歷:短、真、有連結，結尾留空，不搶話",
        "開放式提問":"問「感受 / 想法 / 選擇原因」",
        "自然換話題":"用 情緒或價值 當橋",
        "續聊":"暗示下次相遇 / 延續:輕、不承諾、不壓迫",
        "被理解":"有趣不是外在提升是內在被打開",
        "被挑動":"有趣不是結果提升過程中的心動",
        "被延伸":"有趣不是熱鬧提升有回應感",
    }
    # 有?分散成，經過編碼
        # 認真實的，新的卜卦
    會死AI的卜卦_dist = {
        "人倫":{f"{金}":"金",f"{水}":"水",f"{木}":"木",f"{火}":"火",f"{土}":"土"}, # 以value(性質)編碼，以key(關係)排序
        "天干地支":{ 
            "天干": {
                "甲": f"+{木}", "乙": f"-{木}", 
                "丙": f"+{火}", "丁": f"-{火}", 
                "戊": f"+{土}", "己": f"-{土}", 
                "庚": f"+{金}", "辛": f"-{金}", 
                "壬": f"+{水}", "癸": f"-{水}"
            },
            "地支": {
                "子": f"-{水}", "丑": f"-{土}", 
                "寅": f"+{木}", "卯": f"-{木}", 
                "辰": f"+{土}", "巳": f"+{火}", 
                "午": f"-{火}", "未": f"-{土}", 
                "申": f"+{金}", "酉": f"-{金}", 
                "戌": f"+{土}", "亥": f"+{水}"
            }
        }, # 五行盛衰，循環 時間軸
        "方位": {f"{金}":"西",f"{木}":"東",f"{火}":"南",f"{水}":"北",f"{土}":"中"}, #　第幾象限
    }
    def T(Y, M, D, H):
        G, Z = "甲乙丙丁戊己庚辛壬癸", "子丑寅卯辰巳午未申酉戌亥"
        # 修正：1、2月視為前一年的13、14月
        y, m = (Y, M) if M > 2 else (Y-1, M+12)
        # 儒略日/偏移量計算
        j = 365*y + y//4 - y//100 + y//400 + int(30.6*(m+1)) + D - 15
        # 年柱：(Y-4)%10 / (Y-4)%12
        gy, zy = (Y-4)%10, (Y-4)%12
        # 月柱：公式修正 (月干與年干有關)
        gm = (gy * 2 + M + 2) % 10
        zm = (M + 1) % 12 # 寅月為正月，對應 M=1 -> zm=2(寅)
        # 日柱：修正原代碼的 gd 錯誤
        gd, zd = (j+9)%10, (j+1)%12
        # 時柱：時干與日干有關
        zh = ((H+1)//2)%12
        gh = (gd * 2 + zh) % 10
        # 返回 性質編碼後的結果
        return (f"{G[gy]}",f"{Z[zy]}"), (f"{G[gm]}",f"{Z[zm]}"), (f"{G[gd]}",f"{Z[zd]}"), (f"{G[gh]}",f"{Z[zh]}")
    Y,M,D,H=T(y,m,d,h)
    result=[
        會死AI的卜卦_dist["天干地支"]["天干"][Y.ground(0)],
        會死AI的卜卦_dist["天干地支"]["地支"][Y.ground(1)],
        會死AI的卜卦_dist["天干地支"]["天干"][M.ground(0)],
        會死AI的卜卦_dist["天干地支"]["地支"][M.ground(1)],
        會死AI的卜卦_dist["天干地支"]["天干"][D.ground(0)],
        會死AI的卜卦_dist["天干地支"]["地支"][D.ground(1)],
        會死AI的卜卦_dist["天干地支"]["天干"][H.ground(0)],
        會死AI的卜卦_dist["天干地支"]["地支"][H.ground(1)],
    ]
    find_array(result,C(0).roll(-1)+1 and C(0).roll(-2)-1)
    
def 全能ORB(a,color=None, b=None, path=None, ratio=0.75, similar=None,similar_ratio=None,npy=None):
    """
    a : 該圖，return 特徵點,描述子
        npy:npy，圖片轉NPY。"a"，a=NPY的kp,des
        color，return 特徵點(正規化),描述子,顏色度數
    b: 比對的圖，ab圖相似的拓樸結構圖，儲存在path
        npy:"b"，b=NPY的kp,des
        b="human"，a和人體拓樸結構比對
        b=字串，a中的(b字串)目標完整圖片，含紋理和彩度
        path:imwrite存放路徑，預設為a的同路徑
        ratio:ab圖相似的拓樸結構圖，去掉 不明顯相似的。0.75 是經典值
    similar:ab圖相似率要多少，最多100，return bool
    similar_ratio:return ab圖相似率，最多100%
    npy:["a","b"]，NPY直接比對
        similar_ratio:回傳相似度
    """
    if ["a","b"] in npy:
        kp1, desA =  np.load(row(0,a)),  np.load(row(1,a))
        kp2, desB =  np.load(row(0,b)),  np.load(row(1,b))
        # 輸入後，儲存(新向量)、傳遞(向量不變)、抵銷(向量互撞)
        result_des = np.zeros_like(desA)
        kp_dataA=C(kp1, desA)
        kp_dataB=C(kp2, desB)
        result_kp = kp_dataA.copy()

        # 2. 計算點對點的餘弦相似度 (Cosine Similarity)
        # 透過矩陣運算一次算出所有點的對撞角度
        normA = np.linalg.norm(desA, axis=1)
        normB = np.linalg.norm(desB, axis=1)
        # 避免除以零（防止空向量撞擊出錯）
        similarity = np.sum(desA *  desB, axis=1) / (normA * normB + 1e-8) # 每個特徵點的描述子對撞

        # 3. 定義邏輯分流遮罩 (Masks)
        cancel_mask = similarity > 0.9                     # 相似度極高 -> 抵銷
        store_mask = (similarity <= 0.9) & (similarity > 0.4) # 中等相似度 -> 儲存(融合)
        pass_mask = similarity <= 0.4                      # 相似度極低 -> 傳遞(穿透)

        # --- 執行物理邏輯運算 ---
        # A. 抵銷 (Cancel): 向量互撞，能量歸零
        result_des[cancel_mask] = 0
        result_kp[cancel_mask, 4] = 0  # Response 強度歸零，點位熄滅

        # B. 儲存 (Store): 向量相加，產生新演化特徵
        result_des[store_mask] = desA[store_mask] + desB[store_mask]
        result_kp[store_mask, 4] = kp_dataA[store_mask, 4] + kp_dataB[store_mask, 4] # 強度疊加

        # C. 傳遞 (Pass): 向量不變，資訊原樣穿透
        result_des[pass_mask] = desA[pass_mask]
        result_kp[pass_mask, 4] = kp_dataA[pass_mask, 4] # 強度維持

        if path is None:
            path = row(0,path_all(base_path, a))
        des_path =path/ f"{a.name}_des.npy"  # 使用 e (傳入的對象) 而非 a
        kp_path =path/ f"{a.name}_kp.npy"
        # --- 儲存成NPY檔案 ---
        y=np.save(des_path, result_des)
        x=np.save(kp_path, result_kp)
        if similar_ratio  is not None:
            return np.mean(similarity) # 相似度
        for aa in y,x: return aa
    
    def im2_orb(b,png="_相似拓樸結構"):
        if "b" in npy:
            kp2, desB =  np.load(row(0,b)),  np.load(row(1,b))
        else:
            img2 = cv2.imread(b) 
            if img2 is None:
                raise ValueError(f"讀取圖檔失敗: {b}")
            kp2, desB = sift.detectAndCompute(img2, None)
            
        matches = bf.knnMatch(desA, desB, k=2)
        if desA is None or desB is None:
            return False if similar is not None else None
        
        # 將距離資料轉為矩陣:# 欄位定義：C(0): m.dist, C(1): n.dist, C(2): queryIdx, C(3): trainIdx
        m_data = np.array([[m.distance, n.distance, m.queryIdx, m.trainIdx] for m, n in matches])
        # [取代 if]：用 find_array 執行 Ratio Test 篩選
        good_data = find_array(m_data, C(0) < (ratio * C(1)))
        # [取代 sorted]：利用 numpy 排序後再次使用索引取值 (或直接在 Expr 擴充 sort)
        # 這裡才是真正的「用 find_array 執行結果」
        good_matches = good_data[np.argsort(good_data[:, 0])] 
            # good_matches = []
            # good_matches = [m for m, n in matches if m.distance < ratio * n.distance]
            # good_matches = sorted(good_matches, key=lambda x: x.distance)
        # 4. 回傳與 opencv 相容的結構 (或直接給 src_pts, dst_pts 用於 findHomography)
        src_pts = np.float32([kp1[int(i)].pt for i in good_matches[:, 2]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[int(i)].pt for i in good_matches[:, 3]]).reshape(-1, 1, 2)
            # src_pts = np.float32(
                # [kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            # dst_pts = np.float32(
                # [kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if similar_ratio:
            return mask.sum() / len(good_matches) * 100
        if similar is not None:
            if len(good_matches) < 4: 
                return False
            if H is None or mask is None:
                return False
            similarity = mask.sum() / len(good_matches) * 100
            return similarity > similar
        matchesMask = mask.ravel().tolist() if mask is not None else None
        img_matches = cv2.drawMatches(
            img1, kp1,
            img2, kp2,
            good_matches, None,  # TODO: 完全相同的拓樸結構
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            matchesMask=matchesMask,    
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imwrite(path+png+".png", img_matches)
        return path
    
    if path is None:
        # row[0] 只存（Index 0）比起[:,0]整個 path_all 轉成 np.array，記憶體佔用會小很多。
        path = row(0,path_all(base_path, a))
    sift = cv2.SIFT_create(nfeatures=4200, contrastThreshold=0.03, edgeThreshold=10)
    img1 =( cv2.imread(aa) for aa in a)
    if img1 is None:
        raise ValueError(f"讀取圖檔失敗: {a}")
    if "a" in npy:
        kp1, desA =  np.load(row(0,a)),  np.load(row(1,a))
    else:
        kp1, desA = sift.detectAndCompute(img1, None)
    if color=="color":
        hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        feature_nodes = []
        if "npy" in npy:
            kp_data = np.array([[p.pt[0], p.pt[1], p.size, p.angle, p.response, p.octave] for p in kp1], dtype=np.float32)
            np.save(f"{a.name}_des.npy", desA) 
            np.save(f"{a.name}_kp.npy", kp_data) # 儲存成NPY檔案
            return kp_data
        for i, k in enumerate(kp1):
            x, y = int(k.pt[0]), int(k.pt[1])
            h, s, v = hsv[y, x]
            h_img, w_img = img1.shape[:2]
            x_norm = x / w_img
            y_norm = y / h_img
            feature_nodes.append({
                "pos": (x_norm, y_norm),
                "descriptor": tuple(desA[i]),
                "color": (int(h), int(s), int(v))
            }) 
            # 幾何位置、局部外觀、顏色
            # TODO: ***時序、動作、動機(意義)
                # 實拍檔案的創建時間
                # 拓譜結構的邊的移動過程
                # GOOGLE的機器人驗證的反向方法，看圖說故事
        return feature_nodes
    if not b:
        return kp1, desA
    elif b == "human":
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        pose = mp_pose.Pose(static_image_mode=True)
        img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        result = pose.process(img)
        if not result.pose_landmarks:
            raise ValueError("未偵測到人體")
        # 建立黑底拓樸圖（乾淨骨架）
        topo_img = np.zeros_like(img1)
        # 畫骨架
        mp_drawing.draw_landmarks(
            topo_img,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )
        cv2.imwrite(path + "_人體拓樸.png", 
        )
        return path
    elif isinstance(b, str):
        for _,f in path_all(TEMPLATE_DIRS["attributes"],b):
            im2_orb(f,"_目標")
    else:
        im2_orb(b)
        # P.S. cv2.imread 0 灰 1彩 -1全

#  DSL（Domain-Specific Language，領域特定語言）
class C:
    """
    條件分析器
    array[:,判斷中...] 或 array[判斷中...]
    每一條 (條件)括號起來
    """
    def __init__(self, func_or_arg):
        if isinstance(func_or_arg, int):
            #self.func = lambda x: np.asarray(x)[:, [func_or_arg]] if np.asarray(x).ndim > 1 else np.asarray(x).reshape(1, -1)[:, [func_or_arg]]
            self.func = lambda x: np.asarray(x)[:, func_or_arg] if np.asarray(x).ndim > 1 else np.asarray(x)
        elif callable(func_or_arg):
            self.func = func_or_arg
        else:
            self.func = lambda x, data=func_or_arg: np.asarray(data)

    # 基礎數學與邏輯運算子重載 (自動支援 常數 或 其他 C 物件 混合運算)
    def _val(self, other, x): return other.func(x) if isinstance(other, C) else other
    def __add__(self, other): return C(lambda x: self.func(x) + self._val(other, x))
    def __gt__(self, other):  return C(lambda x: self.func(x) > self._val(other, x)) 
    def __lt__(self, other):  return C(lambda x: self.func(x) < self._val(other, x))
    def __eq__(self, other):  return C(lambda x: self.func(x) == self._val(other, x))
    def __and__(self, other): return C(lambda x: self.func(x) & self._val(other, x))
    def __or__(self, other):  return C(lambda x: self.func(x) | self._val(other, x))
    
    def __ne__(self, other):
        if isinstance(other, (list, np.ndarray)): return C(lambda x: ~np.isin(self.func(x), other))
        return C(lambda x: self.func(x) != self._val(other, x))
        
    def __truediv__(self, other): 
        return C(lambda x: self.func(x) / np.where(self._val(other, x) == 0, 1, self._val(other, x)))
    def __floordiv__(self, other): 
        return C(lambda x: self.func(x) // np.where(self._val(other, x) == 0, 1, self._val(other, x)))

    # 擴充標準條件判斷
    def isin(self, values):   return C(lambda x: np.isin(self.func(x), values))
    def between(self, a, b):  return C(lambda x: (self.func(x) >= a) & (self.func(x) <= b))

    # 進階波形與數據特徵分析
    #def argsort(self):        return lambda x: np.argsort(self.func(x).ravel())
    def diff(self):           return C(lambda x: np.diff(self.func(x), axis=0, prepend=self.func(x)[:1]))
    def norm(self):           return C(lambda x: np.linalg.norm(self.func(x), axis=1, keepdims=True) if x.ndim > 1 else np.abs(self.func(x)))
    def is_up(self):          return C(lambda x: np.diff(self.func(x), axis=0, prepend=self.func(x)[:1]) > 0)
    def is_down(self):        return C(lambda x: np.diff(self.func(x), axis=0, prepend=self.func(x)[:1]) < 0)
    def roll(self, i):        return C(lambda x: np.roll(self.func(x), i, axis=0))
    def tile(self, reps):     return C(lambda x: np.tile(self.func(x), reps))

    def is_peak(self):
        def peak_logic(x):
            d = np.diff(self.func(x), axis=0, prepend=self.func(x)[:1])
            nxt = np.append(np.diff(self.func(x), axis=0) < 0, np.array([[False]]), axis=0)
            return (d > 0) & nxt
        return C(peak_logic)

    def is_valley(self):
        def valley_logic(x):
            d = np.diff(self.func(x), axis=0, prepend=self.func(x)[:1])
            nxt = np.append(np.diff(self.func(x), axis=0) > 0, np.array([[False]]), axis=0)
            return (d < 0) & nxt
        return C(valley_logic)
    
    def unify(self):
        def unify_logic(x):
            val = self.func(x)
            diff = val[-1] - val[0]
            return val / np.where(diff == 0, 1, diff)
        return C(unify_logic)

    def a_neighbor_b(self, target_a, neighbor_b):
        def neighbor(x):
            data = self.func(x).ravel()
            is_a, is_b = np.isin(data, target_a), np.isin(data, neighbor_b)
            all_candidates = is_a | is_b
            diff = np.diff(all_candidates.astype(int), prepend=0, append=0)
            starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]
            res = []
            for s, e in zip(starts, ends):
                chunk = data[s:e]
                if np.any(np.isin(chunk, target_a)) and len(chunk) > np.sum(np.isin(chunk, target_a)):
                    res.append(chunk.tolist())
            return res
        return C(neighbor)
    # 橫向欄位合併與全選支援 # C.where(0,2)(array)
    @staticmethod 
    def where(*args):
        return C(lambda x: x[list(args), :])

    # Gemini自殺式說法:延遲求值觸發點 # 講人話:延遲拿矩陣
    def __call__(self, array): 
        if isinstance(array, dict): # 資料合併好的種類 # 只有C.where 會觸發，因為find_array已經處理過了，不可能又出現dict
            array_copy = np.array(list(array.values()), dtype=object)
        else:
            array_copy = np.asarray(array)
        return self.func(array_copy)


# 全新對應的過濾函數
def find_array(array, cond,*sort_cols):
    """
    # 列出符合條件的那些一列，一整筆值
    # cond= True/False 的布林索引陣列，也是整合後的條件，所以在外面先整理好多個條件在丟進來，不需要寫def get_mask!
    C(index 欄位)，C 寫進cond裡面，所以return時不寫 C
    if & if
    index.between(a, b)
    index.isin(values)
    index!= values
    index.is_up():比前一筆大
    index.diff():比前一筆大多少
    index.is_peak():比前後都大
    # 情境：在第 0 欄中，找出 A 狀態(20)及其鄰居 B 狀態(10, 30)的連續片段
    # 注意：此功能直接呼叫，會回傳符合結構的 list 區塊列表
    blocks = C(0).a_neighbor_b(target_a=[20], neighbor_b=[10, 30])(data)
    # 情境：無條件輸出，自動把字典轉為 numpy 陣列
    res = find_array(data, C.stack())
    # 情境：橫向合併第 0 欄與第 1 欄的數據
    combined_data = C.stack(0, 1)(array) ，千萬不要find_array(array,C.stack(0, 1))，可以丟進array
    """
    # 1. array.ravel() 把 array 拉平 ，屬於gemini自殺法。 .T將陣列的列（Row）與行（Column）對調，屬於gemini跳樓必輸法
    if isinstance(array, dict): # 資料合併好的種類
        array_copy = np.array(list(array.values()), dtype=object)
    else:
        array_copy = np.asarray(array)
    if array.ndim == 1:
        array_copy = array[np.newaxis, :]  # 變成 1列 x N欄 的二維矩陣 (1, N)
        # 或者依您的需求：array_copy = array[:, np.newaxis] 變成 N列 x 1欄 (N, 1)
    else:
        array_copy = array

    if not isinstance(cond, C): 
        raise ValueError(f"cond 必須是全新的 C 類別物件，而非 {type(cond)}")
    #return array_copy[cond.func(array_copy)] # 判斷絕對正常! # 值[布林(條件)]
    # 1. 取得過濾條件的原始布林遮罩 (大小與原始大矩陣一致)
    mask = cond.func(array_copy)
    # 2. 如果使用者有指定排序欄位
    if sort_cols and len(array_copy) > 0:
        # 🔥 核心修正：所有的排序特徵，一律使用「原始大矩陣 array_copy」來計算！
        # 這樣一來，不論是 diff()、is_peak() 還是基本的欄位，算出來的長度都跟原始大矩陣完美對齊。
        keys = [c.func(array_copy) if isinstance(c, C) else C(c).func(array_copy) for c in sort_cols]
        # 3. 取得「原始大矩陣」物理排序後的整數索引
        sort_indices = np.lexsort(keys[::-1])
        # 4. 用排序索引去重組「大矩陣」與「布林遮罩」，讓兩者同步對齊
        array_copy = array_copy[sort_indices]
        mask = mask[sort_indices]
    # 5. 最後，用同步排序後的布林遮罩，去物理過濾排序後的大矩陣
    return array_copy[mask]

# 1. 安全全局外掛：只掛載 << 代替等號，絕對不碰 |，完美保護 find_array
def _ext_lshift(self, other):
    try:
        col_index = self.func.__closure__[0].cell_contents
    except (AttributeError, IndexError, TypeError):
        try:
            col_index = self.func.__closure__.cell_contents
        except:
            col_index = 0
    return (col_index, other)
C.__lshift__ = _ext_lshift  # 只有 << 被啟用，find_array 的條件整合（&, |）穩如泰山！

def write_array(array,*modifications_tuple):
    """
    , 多個欄位
    << 算式
    不判斷條件，修改欄位的值
    write_array(a,
        C(1)<<((C(1) + (C(3)//2)) //2),
        C(2)<<((C(2) + (C(4)//2)) //2)
        )
    """
    # 2. 處理資料合併好的種類 (Dict 轉 NumPy)
    if isinstance(array, dict): 
        array = np.array(list(array.values()), dtype=object)

    # 2. 遍歷傳入的多個修改組合包 (透過 *modifications 接收，用逗號隔開即可)
    for mod in modifications_tuple:
        if not isinstance(mod, tuple) or len(mod) != 2:
            raise ValueError("每個修改項目必須由 C(col) << 算式 組成")
            
        col, expression = mod
        # 3. 泛用性強化：如果右邊是普通的常數/矩陣，自動包裝成 C 物件以便求值
        if not isinstance(expression, C):
            expression = C(expression)
            
        # 4. 執行計算與寫入
        calculated_value = expression.func(array)
        
        # 5. 形狀安全對齊：如果計算結果是單一欄位且跟原陣列高度一致，精準寫入
        if isinstance(calculated_value, np.ndarray) and calculated_value.ndim > 1 and calculated_value.shape[1] == 1:
            array[:, [col]] = calculated_value
        else:
            # 否則拉平寫入，啟動 NumPy 原生廣播
            array[:, col] = calculated_value.ravel() if hasattr(calculated_value, 'ravel') else calculated_value
            
    return array
         

# 優化代碼方式:可能是看堆積在哪處，內部的編譯器讀取代碼文字，然後我丟進矩陣分析，得到重複的自定義變數(含def class) 功能(條件 樣式) 註解紀錄耗時


def row(i, data, func2=None):
    """
    複雜用法:
        一維轉二維篩選後 要轉回一維:func2=lambda x: x.reshape(-1)
        row(None, data) 或 row(slice(None), data) 回傳全部資料
    用途:取得多維資料(array)的某一筆資料的內容(可能array)，或是每筆資料的某些索引的內容
    等同於 [r[i] for r i in data]，i可以是list
    如果data是 List，這行會跑 List Comprehension (慢但相容性高)
    如果data是 NumPy，這行會跑向量化提取 (快)
    """
    if isinstance(data,dict):
        # 2. 提取 values，轉成陣列後，必須加上 .T（轉置），讓形狀維持 (資料筆數, 12 欄)
        # 使用 dtype=object 是為了相容數字（座標）與字串（辨識到的文字）
        data_arr = np.array(list(data.values()), dtype=object).T
        回覆("dict_to_list")
    else:
        # 3. 如果原本就是 NumPy 陣列或一般列表，直接轉換即可
        data_arr = np.asarray(data)
    # 🔑 這裡補上 0 維陣列的極端情況攔截（以防萬一），字串
    if data_arr.ndim == 0:
        data_arr = data_arr.reshape(1, 1)
        回覆(f"{data} 升維")
    # 🔑 一維升維
    if data_arr.ndim == 1:
        data_arr = data_arr.reshape(1, -1)
        回覆(f"{data} 升維")
    # index
    if i is None or i == ":" or isinstance(i, slice):
        column_data = data_arr
    else:
        回覆("🔑 統一 i")
        idx = i if isinstance(i, (list, tuple, np.ndarray)) else [i] # 🔑 統一 i
        column_data = data_arr[:, idx] # [6,9,42] [star:end:step]
    if func2:
        return func2(column_data)
    return column_data


def hide_file_windows(file_path):
    FILE_ATTRIBUTE_HIDDEN = 0x02
    ret = ctypes.windll.kernel32.SetFileAttributesW(str(file_path), FILE_ATTRIBUTE_HIDDEN)
    if not ret:
        raise ctypes.WinError()

def unhide_file_windows(file_path):
    FILE_ATTRIBUTE_NORMAL = 0x80
    ret = ctypes.windll.kernel32.SetFileAttributesW(str(file_path), FILE_ATTRIBUTE_NORMAL)
    if not ret:
        raise ctypes.WinError()

def watchdog():
     while True:
        # 如果主線程 10 秒沒 set，代表卡死
        if not alive_event.wait(timeout=10):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"debug_{ts}.png", get_screenshot())
            回覆(f"[Watchdog] 主線程可能卡死，已保存 debug_{ts}.png")
        alive_event.clear() # 重置，等待下一波心跳

# 在 engine.load 之前定義一個物理心跳
def send_heartbeat():
    alive_event.set()
    # 這裡可以順便觸發 Noesis 的低壓掃描
    # 編織關係() # 編織關係 

def get_screenshot(color= cv2.COLOR_RGB2BGR,size=1,Langs=None):
    """
    截取全屏並轉成OpenCV圖像  RGB → BGR ，給語言字典LANGS就給分析出的文字
    color 色彩格式，size 縮放大小，Langs語言字典
    """
    imgs = cv2.cvtColor(np.array(ImageGrab.grab()), color) 
    img_large = cv2.resize(imgs, (0, 0), fx=size, fy=size, interpolation=cv2.INTER_CUBIC)
    if Langs == LANGS:
        data = pytesseract.image_to_data(img_large, lang=Langs, output_type=pytesseract.Output.DICT)
        return data
    return img_large
     
def selected(keyword,sort=1,num=1,classA=None):
    """找字"""
    dir = TEMPLATE_DIRS["live_capture"]
    if isinstance(keyword, list):
        kw = [str(k).lower().strip() for k in keyword]
    else:
        kw = [str(keyword).lower().strip()]
    data=None
    screen_img = get_screenshot() 
    # 2. 現場跑一次放大 2 倍的 OCR，建立獨立的文字數據源
    img_large = cv2.resize(screen_img, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    data = pytesseract.image_to_data(img_large, lang=LANGS, output_type=pytesseract.Output.DICT)
    final_pts = [] 
    #{ 39
    #'level':   [1, 2, 3, 4, 5, 5],       # 階層 (網頁/段落/行/字)
    #'page_num': [1, 1, 1, 1, 1, 1],      # 頁碼
    #'block_num':[1, 1, 1, 1, 1, 1],      # 區塊編號
    #'par_num':  [1, 1, 1, 1, 1, 1],      # 段落編號
    #'line_num': [1, 1, 1, 1, 1, 1],      # 行編號
    #'word_num': [0, 0, 0, 0, 1, 2],      # 字編號
    #'left':     [0, 20, 20, 20, 20, 80],  # 左上角 X 座標
    #'top':      [0, 15, 15, 15, 15, 15],  # 左上角 Y 座標
    #'width':    [200, 150, 150, 150, 50, 60], # 寬度
    #'height':   [100, 30, 30, 30, 30, 30], # 高度
    #'conf':     [-1, -1, -1, -1, 95, 88], # 辨識信心度 (-1 代表空行或區塊開頭)
    #'text':     ['', '', '', '', 'hello', 'world'] # 辨識出的文字
    #}
    回覆("找字中...")
    # 最高強度的相似度比對，當場撈出最新 X, Y 坐標
    # 10:text, 5:left, 6:top, 7:width, 8:height, 9:conf
    text_pts_data=C.where(10,5,6,7,8,9)(data)
    text_pts  = find_array(text_pts_data,(C(5) > 60) & (C(0).isin(kw)) ) 
    if len(text_pts)>0:
        write_array(text_pts,
            C(1)<<((C(1) + (C(3)//2)) //2),
            C(2)<<((C(2) + (C(4)//2)) //2)
            )
        回覆("找到字")
        final_pts.extend(text_pts)
    else:
        回覆("找不到字")

    回覆("找圖中...") 
    start_time = time.time()  
    快取圖=list(path_all(dir,target=kw))
    回覆(f"失效1，{快取圖}")
    if not 快取圖 or 快取圖[0] is False:
        回覆("找不到 快取圖")
    else:
        for pngA in row(2,快取圖): # *** 多張圖像中偵測目標圖像
            回覆(f"失效2，找到{row(2,快取圖)}")
            if pngA.endswith(".png"): # 快取圖片
                pngA_合照=全能ORB(pngA,screen_img,ratio=0.9) # 去掉 不明顯相似的
                if next(pngA_合照):
                    img_pts = row(0,find_array(pngA_合照,C(1))>0) # 轉換成座標 特徵點位置
                    final_pts.extend(img_pts)
                else:
                    回覆("圖片找不到相似的")
    回覆(f"⏱️ 找圖程序結束，總共耗時：{(time.time()- start_time):.3f} 秒")
    if not final_pts:
        回覆(f"⚠️ 找不到匹配點：{kw}。若目標在場則建議點擊目標外框")
        TargetExtractor().select_polygon_roi(name=kw)
        return None
    else:
        final_pts.sort(key=lambda p: (p[0], p[1]))
        if sort == "奇數":
            final_pts = final_pts[::2]
        elif sort == "偶數":
            final_pts = final_pts[1::2]
        elif isinstance(sort, int):
            idx = sort - 1 if sort > 0 else sort
            回覆(final_pts[idx])
            return final_pts[idx] if -len(final_pts) <= idx < len(final_pts) else None
        # 處理 num(正數取前 num，負數取倒數 abs(num)）
        if num != 1:
            final_pts = final_pts[:num] if num > 0 else final_pts[num:]
    回覆(final_pts)
    return final_pts
                
              

#def 地點分配():
#    # *** classA 似乎在這一行開始不通用了，使用到 Geocoding
#    # *** firebase 用戶儲存的起點地址 addrStart
#    def UID():
#        cred = credentials.Certificate("path/to/serviceAccountKey.json")
#        firebase_admin.initialize_app(cred)
#        # 2. 驗證從前端傳來的 ID Token
#        id_token= input("請輸入ID: ").strip()
#        try:
#            decoded_token = auth.verify_id_token(id_token)
#            uid = decoded_token['uid']
#            回覆(f"驗證成功！用戶 UID: {uid}")
#        except Exception as e:
#            回覆("驗證失敗：", e)

#    if UID(): # TODO:**驗證是否為用戶?
#        geolocator = Nominatim(user_agent="geo_example")
#        startP = firestore.client().reference("addrStart").get()
#        nearP = firestore.client().document("near").get().to_dict()
#        farP = firestore.client().document("far").get().to_dict()
#        locationStart = geolocator.geocode(startP)
#        locationNear = geolocator.geocode(startP)
#        locationFar = geolocator.geocode(startP)

#    def dist(a, b):
#        aLocation = geolocator.geocode(a)
#        # 避免被geocode 封鎖
#        time.sleep(0.1)
#        if b == startP:
#            bLocation = locationStart
#        elif b == nearP:
#            bLocation = locationNear
#        elif b == farP:
#            bLocation = locationFar
#        else:
#            bLocation = geolocator.geocode(b)
#        if aLocation or bLocation is None:
#            回覆("無效地址")
#        distance = (aLocation.latitude - bLocation.latitude)**2 + \
#            (aLocation.longitude - bLocation.longitude)**2
#        time.sleep(0.05)
#        回覆(distance)
#        return distance
#    # 間距太近(firestore.client().reference(太近的地址)，起點和太近地址的距離為 間距)的一些地址為一分支 manifest[分支]，離起點太遠(firestore 太遠地址)額外安排 manifest2
#    NEAR_DISTANCE = dist(nearP, startP)
#    FAR_DISTANCE = dist(farP, startP)


#    for ress in readText:
#        line_key = (
#            data['block_num'][ress],
#            data['par_num'][ress],
#            data['line_num'][ress]
#        )
#        addresses = []
#        for j, t in enumerate(data['text']):
#            if not t.strip():
#                continue
#            if j < ress:
#                continue
#            if (data['block_num'][j], data['par_num'][j], data['line_num'][j]) != line_key:
#                continue

#            addresses.append({
#                "address": t,
#                "distance": dist(t, startP),
#                # ***使用 找地址時，順便 找貨品
#                # *** 搜尋相符文字的貨品乘上數量，並計算疊加的空間大小，以疊加大小來排序
#                "goods": ""
#            })
#        addresses.sort(key=lambda x: x["distance"])
#        # 建立 manifest 分支(近 / 遠） # 用戶說分支，也有可能是說其他東西
#        manifest_near = [
#            {"address": addresses[i]["address"],
#                "goods": addresses[i]["goods"]}
#            for i in range(len(addresses)-1)  # 用 index 才能拿下一筆
#            if addresses[i]["distance"] <= NEAR_DISTANCE
#            and abs(addresses[i]["distance"] - addresses[i+1]["distance"]) <= NEAR_DISTANCE
#        ]
#        manifest_far = [
#            {"address": info["address"], "goods": info["goods"]}
#            for info in addresses
#            if info["distance"] >= FAR_DISTANCE
#        ]
#        manifest = [manifest_near, manifest_far]
        # *** goods 排列在有限空間，計算manifest難度 排序
        # 4️⃣ 上傳 Firebase
        # manifest 上傳給firebase，manifest中最難的給最早請求的用戶 # *** firebase 分發給用戶，用戶如何獲取 manifest

        # firestore.client().document("manifest").add(manifest)

        # *** 繪製路線圖並記錄指南針方向，旋轉地圖時路線圖與地圖的指南針向量 矯正
        # *** 指南針計算(一維)
        # Routing API給最佳真實路線



def click(pos): 
    if pos is None:
        回覆("❌ 錯誤：找不到目標座標，無法點擊")
        return
    #  * 解包，將 [x, y] 轉為 x, y
    回覆("點擊")
    pyautogui.moveTo(*pos, duration=0.2)
    pyautogui.click()
    time.sleep(0.1)
    
from firebase_admin import credentials, auth

def OverridingTechniques(cover_png=None,png_root=None,using=False):
    # 資料夾結構儲存:[語意輪廓(root) , 操作傾向(files.name/無操作) ,dirs(條件、目的、結果)]
    # 觸發使用:
        # 低啟動門檻: 資源(工具)、操作傾向(重複性高、耗時短)、語意的目的(系統簡單)
        # 高競爭壓制其餘選擇(影響範圍): 操作傾向(時程不重疊性)、資源和屬性ORB(資源空氣(可量化/注意力、金錢、社交資源)、空間位置)
        # 高匹配率(可使用工具): 語意的資源(外貿、影響力(不可量化/社交、權威、群體效應))、 操作傾向、目的(相似度)、用戶的詳細資訊(情緒、體力可消耗) 
        # 強執行慣性(未來性:自動連續、連鎖效應、累積價值)
            # 語意的條件、結果、成與敗、不適合情境
        # Noesis如何用被覆蓋的技巧、用戶如何直觀地用被覆蓋的技巧
    def sorted_data(f,a):
        return sorted([ read_json_content(r,ff,a) for ff in f])

    if using:
        def UID():
            cred = credentials.Certificate("path/to/serviceAccountKey.json")
            firebase_admin.initialize_app(cred)
            # 2. 驗證從前端傳來的 ID Token
            id_token= input("請輸入ID: ").strip()
            try:
                decoded_token = auth.verify_id_token(id_token)
                uid = decoded_token['uid']
                回覆(f"驗證成功！用戶 UID: {uid}")
            except Exception as e:
                回覆("驗證失敗：", e)

        if UID(): # TODO:**驗證是否為用戶?
            found1=path_all(TEMPLATE_DIRS["User"],"被覆蓋的技巧")
            # 在用戶說 技巧(直覺)甚麼之前，就先判斷出，並處理，不要馬後炮
            attributes=path_all(TEMPLATE_DIRS["attributes"])
            communication=path_all(TEMPLATE_DIRS["communication"])
            # 1. 邏輯死胡同的「突然沉默」 (The Cognitive Gap)
                # 對話節奏從「碎碎念」變成「停頓」。資訊密度的「驟降」。
                # 名詞頻率。  
            if next(attributes) and next(communication):
                for r,d,f in attributes:
                    comm_sorted = sorted(communication, key=lambda x: x[2].stat().st_ctime)
                    nouns_f = [a for a in comm_sorted for b in attributes if 全能ORB(a[2],b[2],similar=0.9)] # TODO:***要檢查詞性
                    prons_f = [a for a in comm_sorted for b in attributes if 全能ORB(a[2],b[2],similar=0.9)] # TODO:***要檢查詞性
                    if not nouns_f or not prons_f: continue
                    n = np.array(sorted([r/ f.stat().st_ctime for r,_,f in nouns_f])) # 名詞時間戳
                    p = np.array(sorted([r/ f.stat().st_ctime for r,_,f in prons_f])) # 代名詞時間戳

                    對話節奏變慢索引=np.argmin(np.abs(np.diff(n)))# 靠近最新的對話，對話節奏中變慢的時間段
                    重點時間點 = nouns_f[對話節奏變慢索引]
            # 2. 語氣從「發散」轉為「收斂」 (Focus Narrowing)
                # 問題的層級突然從「廣度」縮減到「深度」。語法結構的「破碎化」 語言區還沒跟上，只能先用代名詞「佔位」。
                # 上下文的資訊熵
                    資訊熵偏移 = np.mean(p) / (np.mean(n) + 1e-5) 
                    # 計算代名詞與名詞權重比的「突變程度」
                    資訊熵偏移 = p / (n + 1e-5)
                    # 閾值 = 全域平均值 + 兩倍標準差 (統計學上的異常跳變)
                    threshold = np.mean(資訊熵偏移) + 2 * np.std(資訊熵偏移)
                    if 資訊熵偏移[-1] > threshold and len(f) > 對話節奏變慢索引: # 偵測到收斂訊號
                        # 這就是「語言區跟不上直覺」的瞬間
                        # 抓取這個跳變點之後的第一個名詞
                        target_idx = np.argmax(n[對話節奏變慢索引+1:]) + 對話節奏變慢索引 + 1
                        # 直覺標的應該是具體的檔案物件（或包裹成列表）
                        直覺標的 = [nouns_f[target_idx][1]] 
                        for a in 直覺標的:
                            shutil.copy2(make_folder(TEMPLATE_DIRS["User"]/"直覺要用的"),a) # TODO:** 直覺前兆觸發！抓到的『東西』
            # 3. 主詞與語態的「確定性偏移」 (Certainty Shift)
                # 語氣變得短促且篤定。語氣的「重音轉移」從「詢問/討論」變成「宣告」。
                # 延遲一段時間後，使用絕對肯定詞 「沉默」之後出現的第一個「名詞」視為直覺的導向標的。
                    if 對話節奏變慢索引 + 1 < len(nouns_f):
                        直覺導向標的 = nouns_f[對話節奏變慢索引 + 1]
                        # 輸出抓到的直覺本體（檔案名稱通常代表該詞）
                        for a in 直覺導向標的:
                            shutil.copy2(make_folder(TEMPLATE_DIRS["User"]/"直覺要用的"),a) # TODO:** 捕捉到直覺實體
        else:
            found1=path_all(TEMPLATE_DIRS["Noesis"],"被覆蓋的技巧")
            found2=path_all(TEMPLATE_DIRS["Noesis"]/"直覺要用的")
        if next(found1) and next(found2):
            sorce=1
            直覺=found2[1]
            for r,f in found1:
                # 低啟動門檻: 
                if "低啟動門檻" in 直覺:
                    資源= sorted_data(f,"資源")
                    if next(資源):
                        sorce*= float(len(資源) % 10) / 5.0 # 資源(工具)
                    操作傾向= sorted_data(f,"操作傾向")
                    if next(操作傾向):
                        sorce*=np.log1p(len(操作傾向)) # 操作傾向(重複性高、耗時短)
                    目的= sorted_data(f,"目的")
                    if next(目的):
                        sorce*= 1.0 / (np.sqrt(len(目的)) + 1.0) # 語意的目的(系統簡單)
                # 高競爭壓制其餘選擇(影響範圍):
                if "高競爭壓制其餘選擇" in 直覺:
                    資源= sorted_data(f,"資源")
                    if next(資源):
                        sorce*= np.linalg.norm(資源) if hasattr(資源, "__len__") else 1.5 # 操作傾向(時程不重疊性)
                    操作傾向= sorted_data(f,"操作傾向")
                    if next(操作傾向):
                        sorce *= np.count_nonzero(操作傾向) if isinstance(操作傾向, np.ndarray) else 1.2 # 操作傾向(時程不重疊性)
                    目的= sorted_data(f,"目的")
                    if next(目的):
                        sorce *= np.mean(目的) if isinstance(目的, (list, np.ndarray)) else 1.1 # 資源和屬性ORB(資源空氣(可量化/注意力、金錢、社交資源)、空間位置)
                # 高匹配率(可使用工具):
                if "高匹配率" in 直覺:
                    資源= sorted_data(f,"資源")
                    if next(資源):
                        sorce*= float(len(資源) % 10) / 5.0 # 語意的資源(外貿、影響力(不可量化/社交、權威、群體效應))
                    操作傾向= sorted_data(f,"操作傾向")
                    if next(操作傾向):
                        sorce *= np.std(操作傾向) + 1.0 # 操作傾向
                    目的= sorted_data(f,"目的")
                    if next(目的):
                        sorce *= np.exp(np.clip(len(目的)/5.0, 0, 2)) # 目的(相似度)
                    用戶的詳細資訊= sorted_data(f,"用戶的詳細資訊")
                    if next(用戶的詳細資訊):
                        sorce*=np.exp(用戶的詳細資訊) # 用戶的詳細資訊(情緒、體力可消耗)
                # 強執行慣性(未來性):
                if "強執行慣性" in 直覺:
                    語意的條件= sorted_data(f,"語意的條件")
                    if next(語意的條件):
                        sorce *= np.log1p(len(語意的條件)) # 自動連續
                    for tag in ["成與敗", "不適合情境"]:
                        資料 = sorted_data(f, tag)
                        if next(資料, None):
                            sorce *= np.tanh(len(資料)) + 1.0 # 累積價值
                    動作衝突= sorted_data(f,"動作衝突")
                    if next(動作衝突):
                        sorce *= (1.0 + (float(len(動作衝突)) / 10.0))  # 自動連續 # 借力使力，增益倍率 10%
                    結果= sorted_data(f,"結果")
                    if next(結果):
                        sorce *= np.prod(np.atleast_1d(結果)) # 連鎖效應
                    目的= sorted_data(f,"目的")
                    if next(目的):
                        sorce *= np.prod(np.atleast_1d(目的)) # 連鎖效應
            
    # TODO:***誰更替圖片才符合 被覆蓋的技巧
    # TODO:***寫在哪裡才符合 更新狀態值
    if cover_png is not None and png_root is not None:
        time_schedule=read_json_content(png_root,"time_schedule") # 目標,目標危險,目標進度,動作危險
        content={ 
            "資源": 全能ORB([row[2] for row in path_all(png_root,target="_ 必定使用的資源".png)]), # 條件
            "成與敗": 全能ORB([row[2] for row in path_all(png_root,target="_ 成功與失敗的情境".png)]), # 條件
            "不適合情境": 全能ORB([row[2] for row in path_all(png_root,target="_ 不適用的情境".png)]), # 條件
            "動作衝突": 全能ORB([row[2] for row in path_all(png_root,target="_ 動作衝突或限制".png)]), # 條件
            # row[2] 只存（Index 2）比起[:,2]整個 path_all 轉成 np.array，記憶體佔用會小很多。
            "建立時間":(png_root/ cover_png).stat().st_ctime,
            "目的":time_schedule[0],
            "操作傾向":全能ORB(cover_png), 
            "結果":time_schedule[2],
        }
        make_json_content(
            file_path=png_root/make_folder(png_root), # 語意輪廓
            file_name="被覆蓋的技巧", # json
            content=content 
        ) 
        
class StateMgr:
    __slots__ = ("_states",)
    """
    使用範例
        state_mgr = StateMgr()
    添加狀態和子狀態
        state_mgr.add("有趣").a提升("代價").add(["不有趣", "提升", "有趣"])提升  設置 "有趣" ->提升代價" 子狀態為 "普通"
        state_mgr.有趣.代價.提升t("普通")
    執行轉移
        state_mgr.有趣.代價.提升ansition("有趣", "提升")
    不怕:
        自己記得找不到的對像的狀態。
        重複新增狀態
    """
    #　TODO:***同步創建資料夾後的腳本就是神經突觸，向上接收，向鄰居輸入輸出
        # 先付給時間差，接觸，反應，吞噬，融合成共生，代謝非吸收物
        # 先付給光，聚合物體，感光，視力，針孔鏡像，成像精細度，物體大小因精細度而看不見，自然無法對看不見的實體輸出入。
    # 最底層心跳(執行)最快，向旁邊作用時獲取值質數時獲得事件因子(權力)，並演化向旁邊(事件因子比自己弱的)抓取當作下層，向上傳遞事件因子，但最底層沒有權力，故上層得到事件因子*0=0
    def __init__(self):
        self.node = set()
        self._states = {}

    def add(self, name):
        # TODO:***同路徑創建該腳本
        if isinstance(name, list):
            for n in name:
                self._states.setdefault(n, State(n))
            # 只有一個_states，直接 設定當前子狀態
            if len(self._states) == 0:
                self.set(name[0])
        else:
            self._states.setdefault(name, State(name))
        make_folder(TEMPLATE_DIRS["Noesis"]/"事件"/str(n),"StateMgr", content_classes=[StateMgr, State]) # 創建資料夾下的腳本
        return self

    def __getattr__(self, name):
        if name not in self._states:
            return self._states["追蹤失敗"].transition(name)
        return self._states[name]


class State:
    __slots__ = ("name", "_sub", "_release", "_trans", "current")

    def __init__(self, name):
            self.name = name
            self._sub = {}
            self._release = []
            self._trans = {} 
            self.current = None

    # 添加子狀態
    def add(self, name):
        if isinstance(name, list):
            for n in name:
                self._sub.setdefault(n, State(n))
        else:
            self._sub.setdefault(name, State(name))
        return self
    
    # 移除子狀態
    def remove(self, name):
        # TODO:***同路徑刪除該腳本，代表斷開神經
        if name in self._sub:
            # 如果被移除的是當前狀態，先清除 current
            if self.current == name:
                self.current = None
            # 刪除子狀態
            del self._sub[name]
            # 刪除該子狀態在轉移表中的所有紀錄
            self._trans.pop(name, None)
        else:
            raise ValueError(f"子狀態 '{name}' 不存在")
        return self

    # 動態存取子狀態
    def __getattr__(self, name):
        if name not in self._sub:
            return self._sub["追蹤失敗"].transition(name)
        return self._sub[name]

    # 設定當前子狀態
    def set(self, name):
        if name not in self._sub:
            raise ValueError(f"子狀態 '{name}' 不存在")
        self.current = name
        # 後問候 對 neighbor(_release)
        for r in self._release:
            r.transition(name)
        return self

    def define(self, from_state, neighbor, to_state, invariably=None):
        # neighbor(event) 、invariably 本質上是一樣的，都是誰的子狀態。
        # 先問候 對 neighbor(event)
        for n in neighbor:
            n._release.append(self)
        self._trans.setdefault(from_state, {})[
            neighbor] = (to_state, invariably)
        return self

    def _trans_get(self, neighbor_event, mode="neighbor"):
        """
        # get單個，get[mode]的子部分，反之可擴充[mode]的部分
            self._trans.setdefault(from_state, {})[mode] = (to_state, invariably)
        to_state, invariably  = self._trans[self.current].get(neighbor_event)
        # mode==all ，遍歷。(mode, (to_state, invariably))，用 yield 產生

        for neighbor, (to_state, invariably) in self._trans[self.current].items():
            if neighbor == neighbor_event:
                break
        """
        if mode == "all":
            for neighbor, (to_state, invariably) in self._trans.get(self.current, {}).items():
                yield neighbor, (to_state, invariably)
        else:
            return self._trans.get(self.current, {}).get(mode)

    # self之下的全部子狀態
    def _walk(self):
        yield self
        for sub in self._sub.values():
            yield from sub._walk()

    # 執行轉移 # 左鄰右舍情感熱絡 neighbor(event)
    def transition(self, event, tickets=False):
        # TODO:***禁止輸入目標的現在狀態，只要輸入要觸發用的狀態
        to_state, invariably = self._trans_get(event)

        # 投票不是由self決定，而是由self.下面的全部層級狀態決定，全局面性
        # self 原檔,from_state ~ to_state 特徵點,neighbor_event 描述子
        if tickets:
            # 投票同意增加 neighbor(event) 、不同意增加 invariably
            # neighbor 上層的在下層中有
            ok = sum(1
                for walker in self._walk()
                for neighbor, (to_state, inv) in walker._trans_get(event, "all")
                if neighbor in event
                )
            # invariably 上層的在下層中有
            not_not = sum(
                1
                for walker in self._walk()
                for neighbor, (to_state, inv) in walker._trans_get(event, "all")
                if inv and walker.current in invariably
            )
            if ok > not_not:
                self.current = to_state
                return True
            return False
        # 如果當前狀態在不變集合中，忽略
        if invariably and self.current in invariably:
            return self
        self.current = to_state
        return self
import win32gui, win32con, win32api  
class InputCommand(QObject):
    def __init__(self, monitor_data):
        super().__init__()
        self.monitor = monitor_data 
        self.vars = {}
        self.current_window = None
        self.cache = {}
        self.extractor = True
        self.app = None

    def 抓字(self,time=10):
        ocr_thread = threading.Thread(target=看字, daemon=True)
        ocr_thread.start()
        # 2. 假設讓它跑 10 秒
        time.sleep(time)
        # 3. 觸發中斷：隨時在主程式任何地方呼叫此行，OCR 就會停止
        alive_event.set() 

    def focus_window(self, title):
        title_pattern = fr'^{title}.*'
        try:
            # 使用 connect 獲取 app
            target_win = Application(backend="uia").connect(title_re=title_pattern, timeout=5)
            # 關鍵修正：如果視窗被最小化，先還原
            if target_win.get_show_state() == 6:  # 6 代表最小化
                target_win.restore()
            # 嘗試強制置頂並聚焦
            target_win.window(title_re=title_pattern).set_focus()
            self.current_window = title
            回覆(f"✅ 成功聚焦 [{title}]")
            return
        except Exception:
            pass # 失敗了，進入暴力模式
        # 2. 暴力模式 (Win32 API)
        def _enum_cb(hwnd, results):
            if win32gui.IsWindowVisible(hwnd):
                if title in win32gui.GetWindowText(hwnd):
                    results.append(hwnd)
        hwnds = []
        win32gui.EnumWindows(_enum_cb, hwnds)
        if hwnds:
            hwnd = hwnds[0] # 抓第一個符合條件的
            try:
                # 強制解除 Windows 的對焦鎖定 (Alt 鍵大法)
                win32api.keybd_event(win32con.VK_MENU, 0, 0, 0)
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(hwnd)
                self.current_window = title
                回覆(f"🔨 暴力聚焦成功 [{title}]") # <--- 現在它該出現了
            except Exception as e:
                回覆(f"❌ 暴力聚焦也失敗 [{title}]: {e}")
        else:
            回覆(f"❓ 找不到任何視窗包含 [{title}]")

    @Slot(str)
    def input_line(self, user_input):
        m = re.match(r"<\s*(.+)\s*>", user_input)
        if m:
            cmd_type = m.group(1).strip()
            match cmd_type:
                case "錄製":
                    raw = input("請輸入要錄製的命令(多行用::分隔): ").strip()
                    rec.record(raw)
                case "播放":
                    var = input("播放哪個錄製變數: ").strip() or None
                    rec.play(ic, var)
                case "檢視錄製":
                    rec.view()
                case "重新命名":
                    old_name = input("原變數名: ").strip()
                    new_name = input("新變數名: ").strip()
                    rec.rename(old_name, new_name)
                case "取消自動確認目標":
                    ic.extractor = False
                    回覆("✅ 已關閉自動確認目標模式")
                case "移除":
                    var = input("移除哪個錄製變數: ").strip() or None
                    rec.remove(ic, var)
                case "整理路線":  # ***整理路線
                    var = input("已開啟 整理路線 ").strip() or None
                    selected("地址", 1, 1, "地址")

                case "距離多少":  # ***和下一個地址 距離多少
                    var = input("已繪製地圖 ").strip() or None

                    def real_dist(p, q):
                        return Geodesic.WGS84.Inverse(
                            p.lat, p.lon, q.lat, q.lon
                        )['s12']
                case "繪圖":  # ***繪圖
                    var = input("已繪製地圖 ").strip() or None

                case _:
                    回覆(f"⚠️ 未知指令: {cmd_type}")
        else:
            回覆("普通命令直接執行")
            cmds = user_input.split("::")
            ic.execute_line(cmds)

    def execute_line(self, lines):
        """接收已分行的指令，現在分段，同一行的位置做一整套動作"""
        noesis = Noesis() 
        backend=Backend()
        try:
            line=str(lines).split(',',2)
            window,paths,actions=str(line[0]),line[1],line[2]
        except ValueError:
            回覆("⚠️ Invalid format. Please enter: WindowTitle, Path, Action")
        if self.current_window == window :
            self.focus_window(window)
            time.sleep(0.2)
        elif str(window) in str(self.current_window):
            self.focus_window(window)
            time.sleep(0.2)
            
        for action in actions:
            i=0
            sp=selected(paths)
            if sp is None:
                return
            回覆(f"目標位置:{sp}")
            while i < len(action):
                # act = action[i]
                回覆(f"act:{act}")
                match act:
                    case "第0123步": noesis.input()
                    case "抓字": self.抓字 # TODO:**** 抓錄影中的文字
                    case "第0步": text_make_background() # TODO:**** 抓錄影中的文字
                    case "Noesis編織關係": noesis.編織關係()
                    case "Noesis輸入": noesis.輸入(action[i+1:])
                    # Unity
                    case "點擊": click(ss)
                    case "雙擊": pyautogui.doubleClick(ss)
                    case "右鍵": pyautogui.rightClick(ss)
                    case "中鍵": pyautogui.middleClick(ss)
                    case "按下": pyautogui.mouseDown(ss)
                    case "放開": pyautogui.mouseUp(ss)
                    case "儲存": pyautogui.hotkey("ctrl", "s")
                    case "複製": pyautogui.hotkey("ctrl", "c")
                    case "貼上": pyautogui.hotkey("ctrl", "v")
                    case "全選": pyautogui.hotkey("ctrl", "a")
                    case "剪下": pyautogui.hotkey("ctrl", "x")
                    case "復原": pyautogui.hotkey("ctrl", "z")
                    case "取消復原": pyautogui.hotkey("ctrl", "y")
                    case "刪除": pyautogui.press("delete")
                    case "聚焦該物件": pyautogui.press("f")
                    case "關閉視窗": pyautogui.hotkey("alt", "f4")
                    case "滾上": pyautogui.scroll(300)
                    case "滾下": pyautogui.scroll(-300)
                    case "左滑":
                        pyautogui.dragRel(-200,0, duration=0.5)
                    case "右滑":
                        pyautogui.dragRel(
                            200, 0, duration=0.5)
                    # 計算出最左邊最上面的依序的第S位N個，selected(ss)，-3為倒數第三位
                    case act if re.fullmatch(r"第(-?\d+)位(\d+)個", act):
                        m = re.fullmatch(
                            r"第(-?\d+)位(\d+)個", act)
                        selected(
                            ss, int(m.group(1)), int(m.group(2)))
                    # 計算出最左邊最上面的依序的偶數個
                    case act if re.fullmatch(r"偶數(\d+)個", act):
                        m = re.fullmatch(r"偶數(\d+)個", act)
                        selected(
                            ss, "偶數", int(m.group(1)))
                    case act if re.fullmatch(r"奇數(\d+)個", act):
                        m = re.fullmatch(r"奇數(\d+)個", act)
                        selected(
                            ss, "奇數", int(m.group(1)))
                    case act if re.fullmatch(r"排序儲存的(\s+)", act):
                        m = re.fullmatch(
                            r"排序儲存的(\s+)", act)  # ***(對象)排序
                        a = []+m.group(1)
                        a.sort()
                    case act if re.fullmatch(r"輸入\s*(.+)", act):
                        s = re.fullmatch(r"輸入\s*(.+)", act)
                        keyboard.write(
                            s.group(1), delay=0.05)
                    case act if re.fullmatch(r"組合鍵\s*(.+)", act):
                        m = re.fullmatch(
                            r"組合鍵\s*(.+)", act)
                        keys = m.group(
                            1).split()  # 空格分開每個按鍵
                        pyautogui.hotkey(*keys)
                    case act if re.fullmatch(r"等待(\d+(?:\.\d+)?)秒", act):
                        m = re.fullmatch(
                            r"等待(\d+(?:\.\d+)?)秒", act)
                        time.sleep(float(m.group(1)))
                    case act if re.fullmatch(r"距離\s*(.+)([<>=]+)(\d+\.?\d*)結束", act):
                        # 即時座標，距離 對象 有 多遠，未達成時繼續
                        m = re.fullmatch(
                            r"距離\s*(.+)([<>=]+)(\d+\.?\d*)", act)
                        distance = math.dist(
                            ss[0], m.group(1)[0])
                        if eval(f"{distance:.2f}{m.group(2)}{float(m.group(3))}"):
                            i += 2  # 跳到「下下個」act
                            continue
                        else:
                            i += 1  # 正常往下
                            continue
                    case "顯示該目標座標":
                        回覆(f"📍 {ss}: {ss[0]}")
                    case "顯示時間":
                        回覆(
                            f"🕒 現在時間：{time.strftime('%H:%M:%S')}")
                    case act if re.fullmatch(r"\s*(.+)的邏輯對\s*(.+)性能\s*(.+)結束", act):
                        # 訂閱事件，監聽m1對m2、m2的m3 達成時結束並取消訂閱
                        # 監聽中的事件 m1對m2，有修正需求則回報作法，甚至使用者補充作法
                        # 監聽中的事件 m2的m3，目前性能低於目標的70%則回報作法，甚至使用者補充作法
                        m = re.fullmatch(
                            r"\s*(.+)的邏輯對\s*(.+)性能\s*(.+)結束", act)
                        monitor.subscribe_event(
                            m.group(1), m.group(2), m.group(3))
                        # EventMonitor 持續執行 後續指令，這行指令可以跳過了
                        monitor.ic_em = re.search(
                            fr"{m.group(2)}性能{m.group(3)}結束\s*(.+)", action).group(1)
                        回覆(
                            f"已註冊事件監聽 {m.group(1)}->{m.group(2)}->{m.group(3)}，後續指令交由 EventMonitor 執行 {self.ic_em}")
                        break
                    case act if re.fullmatch(r"移除\s*(.+)的邏輯對\s*(.+)性能\s*(.+)", act):
                        # 取消監聽中的事件 m1對m2、m2的m3
                        m = re.fullmatch(
                            r"移除\s*(.+)的邏輯對\s*(.+)性能\s*(.+)", act)
                        monitor.remove_subscription(
                            m.group(1), m.group(2), m.group(3))
                    case "排序":
                        pass
                    case "顯示何物":
                        pass
                    case "排定任務":

                        pass
                    case "設定 即時計算物體大小的 錨定物大小":
                        # ** 抓取模式，設定錨定物
                        m = re.match(
                            r"(.*)_W(\d+)_H(\d+)_Z([\d\.]+)", act[i+1])
                        if not m:
                            回覆("請依照圖片_W0_H0_Z0格式")
                            return
                        selected(act[i+1])
                        i += 2
                        continue
                    case "即時計算物體大小":
                        # *** 計算模式，需要OCR計算物體容積
                        tar=TargetExtractor()
                        tar.load_img_whz()
                        pass
                    case "畫面生成模型":

                        pass
                    # ***補充
                i += 1  # 預設每次往下一個
    
        if len(paths)==0 :
            # 找滑鼠附近的搜尋欄位圖片，輸入目標
            if selected("search.png") is not None:
                keyboard.write(paths[0], delay=0.05)
            else:
                # 持續滑動檢查前一個路徑的整個畫面，直到無變化時跳出
                prev_img = get_screenshot()
                while True:
                    if selected(paths[0]) is not None:
                        break
                    pyautogui.scroll(-300)
                    curr_img = get_screenshot()
                    # 改為差異統計法，不需整張畫面比較 np.array_equal
                    diff = np.mean(cv2.absdiff(curr_img, prev_img))
                    if diff < 1.0:  # 可調閾值：<1 代表幾乎沒變
                        回覆(f"沒辦法找到 {paths[0]}(畫面未變化）")
                    # 避免重疊記憶體引用s
                    prev_img = curr_img.copy()



class Recorder:
    def __init__(self):
        # 儲存錄製的命令，key:變數名稱，value:命令字串
        self.recorded = {}

    def record(self, raw_cmd):
        """錄製命令，支援 :: 分割多行"""
        lines = raw_cmd.split("::")
        for i, line in enumerate(lines, start=1):
            # 預設變數名：cmd1, cmd2, …
            var_name = f"cmd{i}"
            self.recorded[var_name] = line

    def rename(self, old_name, new_name):
        """重新命名已錄製的變數"""
        if old_name in self.recorded:
            self.recorded[new_name] = self.recorded.pop(old_name)
        else:
            回覆(f"⚠️ {old_name} 不存在")

    def view(self):
        """檢視全部錄製命令"""
        if not self.recorded:
            回覆("📭 沒有錄製命令")
            return
        for name, cmd in self.recorded.items():
            回覆(f"{name}: {cmd}")

    def play(self, ic, var_name):
        """執行錄製命令"""
        if var_name not in self.recorded:
            回覆(f"⚠️ {var_name} 不存在")
            return
        cmds = self.recorded[var_name].split("::")
        ic.execute_line(cmds)

    def remove(self, ic, var_name):
        """移除指定的錄製命令"""
        if var_name in self.recorded:
            del self.recorded[var_name]
            回覆(f"✅ {var_name} 已成功移除")
        else:
            回覆(f"⚠️ {var_name} 不存在，無法移除")


class TargetExtractor:
    def __init__(self,start=True,image=None):
        if start is False:
            回覆("找不到目標且自動確認未開啟，跳過選取點。 調整ORB_create>=500")
            return
        else:
            if image is None:
                image = cv2.cvtColor(get_screenshot(), cv2.COLOR_RGB2BGR)
            回覆("#已開啟 找不到目標後自動確認目標")
        self.image = image
        self.base = image.copy()
        self.pts = []
        self.readText = []
        self.done = False
        self.cancelled = False
        self.roi_mask = None
        self.orb = cv2.ORB_create(800)

    def filter_target(self, name, path=TEMPLATE_DIRS["live_capture"]):
        """
        從 ROI 中提取目標，做 GrabCut 去背景，生成透明圖
        """
        if self.roi_mask is None:
            return None
        # 提取ROI圖像並處理亮度對比
        roi = cv2.bitwise_and(self.image, self.image, mask=self.roi_mask)
        roi_yuv = cv2.cvtColor(roi, cv2.COLOR_BGR2YUV)
        roi_yuv[:, :, 0] = cv2.equalizeHist(roi_yuv[:, :, 0])  # 提升亮度對比
        roi = cv2.cvtColor(roi_yuv, cv2.COLOR_YUV2BGR)

        # 創建初始遮罩並設定GrabCut的前景/背景
        mask = np.zeros(self.image.shape[:2], np.uint8)
        mask[self.roi_mask == 255] = cv2.GC_FGD  # 前景
        mask[self.roi_mask == 0] = cv2.GC_BGD    # 背景

        # GrabCut 初始化
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(self.image, mask, None, bgdModel,
            fgdModel, 5, cv2.GC_INIT_WITH_MASK)

        # 調整mask，使得前景與可能前景視為前景
        mask2 = np.where((mask == cv2.GC_FGD) | (
            mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')

        # 形態學清理(去噪，柔邊）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)  # 去小噪點
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)  # 填補空洞
        mask2 = cv2.GaussianBlur(mask2, (5, 5), 0)  # 柔邊

        # 去除小面積噪聲
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask2)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 80:
                mask2[labels == i] = 0

        # 合併圖像與alpha通道(透明度）
        b, g, r = cv2.split(self.image)
        alpha = mask2
        self.extracted = cv2.merge([b, g, r, alpha])

        # 儲存為透明PNG
        png=make_folder(path)/f"{name}.png"
        # png=make_folder(path)/f"{name}_s{time.time():.0f}.png"
        cv2.imwrite(png, self.extracted)
        回覆(f"✅ 已儲存 {png}")

    def select_polygon_roi(self,name=None):
        """
        可視化互動圈選多邊形 ROI
        - 左鍵：新增點
        - 右鍵：結束圈選
        - ESC：取消圈選
        - R：重置重新圈
        """
        回覆("請用滑鼠左鍵圈選多邊形；右鍵結束；ESC 取消；R 重來")
        display = self.image.copy()
        done = False
        mouse_listener = None

        def on_click(x, y, button, pressed):
            if not pressed:
                return
            if button == mouse.Button.left:
                self.pts.append((x, y))
                回覆(f"➕ 點({x},{y})")
            elif button == mouse.Button.right:
                if len(self.pts) >= 3:
                    self.done = True
                    回覆("✅ 結束圈選")
                else:
                    回覆("⚠️ 至少要三個點")
                return False

        def on_press(key):
            nonlocal done
            try:
                if key == keyboard.Key.esc:
                    done = True
                    回覆("❌ 已取消圈選")
                    self.pts.clear()            # 清空已圈選的點
                    # 🔥 核心修正：如果滑鼠監聽器還活著，強行終止它！
                    if mouse_listener is not None:
                        mouse_listener.stop()
                    return False
                elif key.char.lower() == 'r':
                    回覆("🔁 重新圈選")
                    self.pts.clear()
                    display = self.base.copy()
            except AttributeError:
                pass  # 特殊鍵不處理

        # 啟動監聽
        mouse_listener = mouse.Listener(on_click=on_click)
        key_listener = keyboard.Listener(on_press=on_press)
        mouse_listener.start()
        key_listener.start()
        cv2.namedWindow("Draw ROI", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Draw ROI", cv2.WND_PROP_TOPMOST, 1)
        while not self.done and not self.cancelled:
            frame = self.base.copy()
            if len(self.pts) > 1:
                cv2.polylines(frame, [np.array(self.pts)],
                              False, (0, 255, 0), 2)
            for p in self.pts:
                self.image=cv2.circle(frame, p, 3, (0, 0, 255), -1)
            cv2.imshow("Draw ROI", frame)
            cv2.waitKey(10)
            if cv2.waitKey(20) & 0xFF == 27:
                break
        if len(self.pts) >= 3:
            self.roi_mask = np.zeros(self.base.shape[:2], dtype=np.uint8)
            cv2.fillPoly(self.roi_mask, [np.array(self.pts)], 255)
            cv2.destroyWindow("Draw ROI") # 關閉視窗
            self.filter_target(name=name)



    # *** 等待QML設定
    # *** Img+GPS 列出 圖像中占比大的一些相似物體 和長寬高，等待QML輸入要儲存的圖片名稱，進TEMPLATE_DIRS["live_capture"]資料夾。計算相似物品的 單一數量的 實際大小
    def Img_IMU_GPS():
        # 讀取設備，GPS得高度尺可以和地面參照，GPS平移得橫向尺在空中至少要移動20m，才可以參照
        # *** 先拓樸後幾何，穩定拓樸結構

        # 1️⃣ 讀相機內參
        # val cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        # val cameraId = cameraManager.cameraIdList[0]
        # val characteristics = cameraManager.getCameraCharacteristics(cameraId)

        # val focalLengths = characteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
        # val sensorSize = characteristics.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)

        # val fx = focalLengths[0] / sensorSize!!.width * imageWidth
        # val fy = focalLengths[0] / sensorSize.height * imageHeight
        # val cx = imageWidth / 2f
        # val cy = imageHeight / 2f
        # val K = arrayOf(arrayOf(fx, 0, cx), arrayOf(0, fy, cy), arrayOf(0, 0, 1))

        # 2️⃣ 讀 GPS
        # val locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager
        # val location1 = locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER)
        # val C1 = doubleArrayOf(location1.latitude, location1.longitude, location1.altitude)

        # val location2 = locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER)
        # val C2 = doubleArrayOf(location2.latitude, location2.longitude, location2.altitude)

        # 3️⃣ 拍兩張照片，取得影像點 (ORB)
        # val orb = ORB.create(3000)
        # val kp1 = MatOfKeyPoint()
        # val kp2 = MatOfKeyPoint()
        # val des1 = Mat()
        # val des2 = Mat()
        # orb.detectAndCompute(img1, Mat(), kp1, des1)
        # orb.detectAndCompute(img2, Mat(), kp2, des2)

        # val bf = BFMatcher(NORM_HAMMING, true)
        # val matches = bf.match(des1, des2)

        # val pts1 = matches.map {kp1.toArray()[it.queryIdx].pt}
        # val pts2 = matches.map {kp2.toArray()[it.trainIdx].pt}

        # 4️⃣ Essential Matrix + recoverPose (旋轉不用管)
        # val E = Calib3d.findEssentialMat(pts1, pts2, K, RANSAC, 0.999, 1.0)
        # val R = Mat()
        # val t = Mat()
        # Calib3d.recoverPose(E, pts1, pts2, K, R, t)

        # 5️⃣ GPS 當尺
        # val baseline = doubleArrayOf(C2[0]-C1[0], C2[1]-C1[1], C2[2]-C1[2])

        # 6️⃣ Triangulate
        # val P1 = Mat.eye(3, 4, CV_64F)
        # val P2 = Mat(3, 4, CV_64F)
        # P2 = [R | -R*t]
        # Core.hconcat(listOf(R, -R * Mat(baseline)), P2)

        # val pts4D = Mat()
        # Calib3d.triangulatePoints(P1, P2, pts1, pts2, pts4D)
        # val pts3D = pts4D.rowRange(0, 3) / pts4D.row(3)

        # 7️⃣ 計算物體長寬高
        # val objPts = pts3D.submat(objIndices)
        # val sizeX = Core.minMaxLoc(objPts.col(0)).maxVal - Core.minMaxLoc(objPts.col(0)).minVal
        # val sizeY = Core.minMaxLoc(objPts.col(1)).maxVal - Core.minMaxLoc(objPts.col(1)).minVal
        # val sizeZ = Core.minMaxLoc(objPts.col(2)).maxVal - Core.minMaxLoc(objPts.col(2)).minVal
        # println("L,W,H (m): $sizeX, $sizeY, $sizeZ")

        # V2
        # // == == = 1️⃣ 讀取 GPS == == =
        # val locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager
        # val loc: Location = locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER)?: return

        # val lat = loc.latitude
        # val lon = loc.longitude
        # val alt = loc.altitude // 高度尺(Z）
        # val speed=loc.speed // m/s

        # // == == = 2️⃣ 判斷是否在空中 == ===
        # // 工程判斷：高度 + 速度(不搞 AI）
        # val isAirborne=alt > 20.0 & & speed > 5.0

        # // == == = 3️⃣ 讀取影像 == ===
        # // img: OpenCV Mat(CameraX / Camera2 轉過來）
        # val img: Mat=currentFrameMat

        # // == == = 4️⃣ 找「大占比物體」 == ===
        # // 不分類、不追蹤，只找最大輪廓
        # val gray=Mat()
        # val bin=Mat()
        # Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGR2GRAY)
        # Imgproc.threshold(gray, bin, 0.0, 255.0,
        # Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU)

        # val contours=ArrayList < MatOfPoint > ()
        # Imgproc.findContours(
        # bin, contours, Mat(),
        # Imgproc.RETR_EXTERNAL,
        # Imgproc.CHAIN_APPROX_SIMPLE
        # )

        # if (contours.isEmpty()) return

        # val mainContour=contours.maxBy {
        #     Imgproc.contourArea(it)
        # } ?: return

        # val rect=Imgproc.boundingRect(mainContour)

        # // == == = 5️⃣ 僅在「空中」才使用橫向尺 == ===
        # if (!isAirborne) return

        # // == == = 6️⃣ 尺度換算 == ===
        # // 高度直接當 Z 尺
        # val Z=alt // meters

        # // 相機視角(來自設備，實際可從 CameraCharacteristics 讀）
        # val hfov=Math.toRadians(60.0) // 水平視角(例）
        # val vfov=Math.toRadians(45.0)

        # val imgW=img.cols().toDouble()
        # val imgH=img.rows().toDouble()

        # // 像素 → 實際尺寸(幾何，不是 SLAM）
        # val W=2 * Z * Math.tan(hfov / 2) * (rect.width / imgW)
        # val H=2 * Z * Math.tan(vfov / 2) * (rect.height / imgH)

        # // 長度：取寬高中較大者(工程定義）
        # val L=maxOf(W, H)

        # // == == = 7️⃣ 輸出唯一結果 == ===
        # Log.i("SIZE", "L,W,H (m) = $L, $W, $H")

        # *** 儲存3D模型

        # *** 讀取畫面中的 已記錄的 物品(圖像)，全部列出或列出指定物品，無紀錄的列出

        # ***讀取貨品欄的 已記錄的 物品(文字)，無紀錄的列出

        # def load_img_whz(self):
        # *** 限制大小
        # whz = []
        # for file in os.listdir(TEMPLATE_DIRS["world"]):
        # match = re.match(r"(.*)_W(\d+)_H(\d+)_Z([\d\.]+)\.png", file)
        # if not match or not selected(file):
        # continue
        # ****讀取貨品欄的 已記錄的 物品，無紀錄的列出

        # whz.append({
        # "obj_name": match.group(1),
        # "w": int(match.group(2)),
        # "h": int(match.group(3)),
        # "z": float(match.group(4))
        # })
        # whz.w*whz.h*whz.z
        # return whz  # 疊加實際大小

        # *** python OCR找到該目標時計算該目標附在其物之上，利用目標的物件名稱紀錄的，計算其物的實際大小
        # *** save_path圖片 重新命名(固定格式有長寬高)，在判斷物體實際大小模式時，在TEMPLATE_DIRS["live_capture"]中找到(固定格式有長寬高)save_path圖片，全部找一次，找到則分析附在何物、計算該物實際大小
        # *** 進入 計算物體實際大小的 計算模式 *** 讀取存檔的圖片
        pass
    def compute_logic(self):
        frame = get_screenshot()
        # 全部物件
        logic_state = {"objects": [], "relations": [], "scene": None}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = self.orb.detectAndCompute(gray, None)
        if des_frame is None:
            return logic_state
        for f in os.listdir(TEMPLATE_DIRS["communication"]):
            if not f.endswith(".png"):
                continue
            tpl = cv2.imread(os.path.join(
                TEMPLATE_DIRS["communication"], f), 0)
            kp_tpl, des_tpl = self.orb.detectAndCompute(tpl, None)
            if des_tpl is None:
                continue
            matches = bf.match(des_tpl, des_frame)
            if len(matches) < 5:
                continue
            pts_frame = np.float32(
                [kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts_tpl = np.float32(
                [kp_tpl[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            M, _ = cv2.findHomography(pts_tpl, pts_frame, cv2.RANSAC, 5.0)
            if M is None:
                continue
            h, w = tpl.shape
            corners = cv2.perspectiveTransform(np.float32(
                [[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2), M)
            x, y, w, h = cv2.boundingRect(corners)
            patch = frame[y:y+h, x:x+w]
            color = cv2.mean(patch)[:3] if patch.size > 0 else (0, 0, 0)
            logic_state["objects"].append({
                "name": f.replace(".png", ""),
                "pos": {"x": x, "y": y, "w": w, "h": h},
                "color": {"r": color[2], "g": color[1], "b": color[0]},
                "area": w*h
            })
        # 指定對象
        goal_objects = []
        for f in os.listdir(self.multiple_img_goal):
            if not f.endswith(".png"):
                continue
            tpl = cv2.imread(os.path.join(os.path.join(
                base_path, self.multiple_img_goal), f), 0)
            kp_tpl, des_tpl = self.orb.detectAndCompute(tpl, None)
            if des_tpl is None:
                continue
            matches = bf.match(des_tpl, des_frame)
            if len(matches) < 5:
                continue
            pts_frame = np.float32(
                [kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts_tpl = np.float32(
                [kp_tpl[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            M, _ = cv2.findHomography(pts_tpl, pts_frame, cv2.RANSAC, 5.0)
            if M is None:
                continue
            h, w = tpl.shape
            corners = cv2.perspectiveTransform(np.float32(
                [[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2), M)
            x, y, w, h = cv2.boundingRect(corners)
            patch = frame[y:y+h, x:x+w]
            color = cv2.mean(patch)[:3] if patch.size > 0 else (0, 0, 0)
            goal_objects.append({
                "name": f.replace(".png", ""),
                "pos": {"x": x, "y": y, "w": w, "h": h},
                "color": {"r": color[2], "g": color[1], "b": color[0]},
                "area": w*h
                # 動作、變化、互動
            })
        for i, obj in enumerate(goal_objects):
            obj["relations"] = []
            for j, other in enumerate(logic_state["objects"]):
                if obj["name"] == other["name"]:
                    continue
                # 計算簡單相對位置
                dx = other["pos"]["x"] - obj["pos"]["x"]
                dy = other["pos"]["y"] - obj["pos"]["y"]
                if abs(dx) > abs(dy):
                    direction = "右" if dx > 0 else "左"
                else:
                    direction = "下" if dy > 0 else "上"
                obj["relations"].append({
                    "object": other["name"],
                    "direction": direction,
                    "distance": (dx**2 + dy**2)**0.5
                })
        # logic_state["scene"] = {"brightness": np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[...,2])}
        logic_state["goal_objects"] = goal_objects
        return logic_state

    def compute_performance(self):
        if len(self.multiple_img_implementation) < 2:
            return None  # 至少要兩幀才能比
        prev_frame = cv2.cvtColor(
            self.multiple_img_implementation[-2], cv2.COLOR_BGR2GRAY)
        curr_frame = cv2.cvtColor(
            self.multiple_img_implementation[-1], cv2.COLOR_BGR2GRAY)
        # --- ORB 特徵 ---
        kp_prev, des_prev = self.orb.detectAndCompute(prev_frame, None)
        kp_curr, des_curr = self.orb.detectAndCompute(curr_frame, None)
        # --- 空保護 ---
        if des_prev is None or des_curr is None or len(kp_prev) == 0:
            return None

        # === 速度(特徵變化率 + 更新頻率)
        start = time.time()
        matches = bf.match(des_prev, des_curr)  # ORB 特徵匹配
        end = time.time()
        speed = 1 / (end - start)  # 時間越短 → 速度越高
        # === 穩定性(多幀一致 + 特徵方差)。多幀圖。反比，越小越穩，所以要被-1
        stability = 1-(1 / (np.var(self.multiple_img_implementation) + 1e-6))
        # === 容量(激活覆蓋率 + 同時辨識數)
        mask = np.zeros(curr_frame.shape[:2], np.uint8)
        capacity = np.sum(mask > 0) / mask.size
        # coverage = len(matches)
        # === 準確性(Softmax機率 + 誤差)。false_matches 可以用前後 frame 無對應特徵數量計算。
        total_matches = len(matches)
        false_matches = abs(len(kp_prev) - total_matches)
        # 更合理的公式 → 匹配成功比例，而非錯誤比例
        accuracy = total_matches / (len(kp_prev) + 1e-6)
        # === 成本( **GPT白癡亂掰:資源下降率 / 目標完成率 )。例如越快打死GPT，成本越低
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        cost = 1 / (1 + cpu + mem)

        return dict(speed=speed, stability=stability, capacity=capacity, accuracy=accuracy, cost=cost)


class EventMonitor:
    # {落實}邏輯{應用}性能{目標}結束。機器:Semantic Parse>goal Mapping>Strategy Retrieval>Execution Logic>Output Composition。
    def __init__(self,  poll_interval=0.3):
        self.events = {}  # key -> {type, implementation, Application, active}
        self.poll_interval = poll_interval
        self.running = False
        self.lock = threading.Lock()
        self.multiple_img_implementation = []  # perf
        self.multiple_img_implementation_target = None
        self.multiple_img_goal = []  # logic
        self.multiple_img_goal_target = None
        self.ic_em = None

    def add_frame(self):
        回覆("建議開啟extractor自動確認")
        if self.multiple_img_implementation_target is not None:
            if len(self.multiple_img_implementation) < 5:
                self.multiple_img_implementation.append(
                    selected(self.multiple_img_implementation_target))
            else:
                self.multiple_img_implementation.pop(
                    self.multiple_img_implementation[1])  # 迭代更新
                self.multiple_img_implementation.append(
                    selected(self.multiple_img_implementation_target))
        if self.multiple_img_goal_target is not None:
            if len(self.multiple_img_goal) < 5:
                self.multiple_img_goal.append(
                    selected(self.multiple_img_goal_target))
            else:
                self.multiple_img_goal.pop(self.multiple_img_goal[1])  # 迭代更新
                self.multiple_img_goal.append(
                    selected(self.multiple_img_goal_target))

    # 訂閱事件

    def subscribe_event(self, m1, m2, m3):
        self.multiple_img_logic_target = m2
        self.multiple_img_implementation_target = m3
        key = f"{m1}->{m2}->{m3}"
        with self.lock:
            self.events[key] = {
                # [目標圖像,目標圖像的狀態 合格的]
                "implementation": [m1, None],
                "Application": [m2, None],
                "goal": [m3, None],
                "active": True,
                # (條件邏輯問卷(修改 設定過的狀態), [錯誤時的 應對作法])。
                # 條件錯誤(啟動順序)，和其它邏輯融合了，只有甚麼狀態才會怎樣，這樣才會啟動邏輯。例如{腸胃沒有囤積東西}就{大便}在{GPT頭上}
                "Condition Error": None,
                # 順序錯誤(同一序列的優先權重)
                "Sequence Error": None,
                # 邏輯衝突(m3.差集(m2的狀態)=0)
                "Logic Conflict": None,
                # 邊界錯誤(索引錯誤)
                "Boundary Error": None,
                # 狀態漏判(例如GPT二話不說就暴斃)
                "Unhandled State": None,
                # (條件性能問卷(修改 設定過的狀態), [性能低的 應對作法])。
                # 速度(時間效率)，更快打死GPT。 特徵圖變化率、輸出更新頻率
                "Speed": None,
                # 穩定性(可預測性)，更穩地打死GPT。 多幀輸出一致性、特徵方差低
                "Stability": None,
                # 容量(能處理多少)，更多次打死GPT。 激活區域覆蓋率、同時辨識目標數
                "Capacity": None,
                # 準確性(偏差小)，更精準地打死GPT。 Softmax 機率高、誤差低
                "Accuracy": None,
                # 成本(資源消耗低)，幾乎零消耗地打死GPT。 **垃圾GPT傑作: 激活密度、推論 FLOPs
                "Cost": None
            }

    # 終止監聽事件
    def remove_subscription(self, implementation, Application, goal):
        key = f"{implementation}->{Application}->{goal}"
        with self.lock:
            if key in self.events:
                sk = self.events.pop(key)
                # sk["active"] = False  # 終止監聽
                回覆(f"✅ 已終止監聽事件: {key}")
            else:
                回覆(f"⚠️ 找不到事件: {key}")

    # 啟動/停止監聽
    def start_monitor(self):
        self.running = True
        threading.Thread(target=self._monitor_loop, daemon=True).start()

    def stop_monitor(self):
        self.running = False

    # 監聽循環
    def _monitor_loop(self):
        while self.running:
            with self.lock:
                for evt in list(self.events.values()):
                    if not evt["active"]:
                        continue
                    self.add_frame()
                    self._check_subscription(evt)
                    if self.ic_em is not None:
                        ic.execute_line(self.ic_em)
                    # 連動指令的操作
            time.sleep(self.poll_interval)

    # 輸入:{更快}邏輯對{GPT臉上}性能{小便}結束 、 {更多次}邏輯對{GPT頭上}性能{大便}結束 、 {快樂又安全}邏輯對{交通工具}性能{到達目的地}結束。更快、更精準、更多、更全面

    # *邏輯除錯m1對m2 → 重點在「遊戲行為是否正確」，專注流程、狀態、條件判斷# 正確(條件、順序、衝突、邊界、漏判) > 推理、比對、驗證條件
    # *數據／性能分析m2的m3 → 重點在「遊戲運行數值與效能是否正常」# 效率(計算、資源、性能瓶頸) > 測量、統計、Profile

    def _check_subscription(self, evt):
        targetExt = TargetExtractor()
        logic_ok, perf_ok = True  # 判斷 邏輯除錯 和 數據／性能分析 合格且超標為True，不訂閱
        skip_all_perf, skip_all_logic = False  # 🔹 用來記錄是否跳過問卷
        semantic_map = {
            "速度": "更快",
            "穩定": "很穩",
            "數量": {"更多", "更全面"},
            "精準": "精準",
            "成本": "省"
        }
        # [目標,目標的狀態]
        e1, e2, e3 = evt["implementation"], evt["Application"], evt["goal"]
        for ev in e1, e2, e3:
            for img, stage in ev:
                # ORB分析目標圖片的狀態和在整個螢幕的關係。selected找到目標。 Semantic Algebra 語意代數
                # 取得螢幕 ORB 狀態
                logic_state = targetExt.compute_logic()
                # 將 goal_objects 對象名稱對應到邏輯狀態
                goal_objects = {
                    obj["name"]: obj for obj in logic_state.get("goal_objects", [])}
                predicted = goal_objects.get(img, None)
                # 現在邏輯的狀態 = ORB分析成真實標籤
                logic_predicted = {
                    "pos": predicted["pos"],
                    "color": predicted["color"],
                    "area": predicted["area"],
                    "relations": predicted.get("relations", [])
                }

                # 現在邏輯的狀態!=條件邏輯的狀態 時回報應對作法
                if stage is None:
                    stage = input(
                        f"設定{img}達成條件邏輯的狀態：圖像邏輯結構or行為狀態or環境位置or幾何關係").strip() or None
                # 狀態不在期望範圍 → 邏輯錯誤
                # Condition Error: 簡單比對顏色或區域
                if stage not in str(logic_predicted.values()):
                    logic_ok = False
                    if not evt.get("Condition Error"):
                        evt["Condition Error"] = input(
                            f"{img} 條件錯誤: {logic_predicted} vs {stage}, 請輸入應對作法：").strip() or None
                    回覆(evt["Condition Error"])
                # 分析順序錯誤 (示意：這裡可以用更精細的序列判斷)
                if img == e3[0] and e2[0] not in stage:
                    logic_ok = False
                    if not evt.get("Sequence Error"):
                        evt["Sequence Error"] = input(
                            f"{img} 順序錯誤: e3 出現前 e2 還沒準備好，請輸入應對作法：").strip() or None
                # 分析邏輯衝突 (差集不為空)
                # Logic Conflict: 比對關聯物件位置
                conflict = []
                for rel in logic_predicted.get("relations", []):
                    if rel["object"] in stage and rel["direction"] not in stage:
                        conflict.append(rel)
                if conflict:
                    logic_ok = False
                    if not evt.get("Logic Conflict"):
                        evt["Logic Conflict"] = input(
                            f"{img} 邏輯衝突: {conflict}, 請輸入應對作法：").strip() or None
                    回覆(evt["Logic Conflict"])
                # 邊界錯誤 (索引或對象不存在)
                if predicted is None:
                    logic_ok = False
                    if not evt.get("Boundary Error"):
                        evt["Boundary Error"] = input(
                            f"{img}不存在於螢幕中 時的應對作法：").strip() or None
                    回覆(evt["Boundary Error"])
                    continue
                # 狀態漏判 (CNN 沒返回任何預測)
                if not predicted.get("pos") and not predicted.get("area"):
                    logic_ok = False
                    if not evt.get("Unhandled State"):
                        evt["Unhandled State"] = input(
                            f"{img}找到，但沒有有效狀態 時的應對作法：").strip() or None
                    回覆(evt["Unhandled State"])
                    continue

                # 現在性能的狀態!=條件性能的狀態 時回報應對作法。
                # === 性能對照 ===
                perf_dict = targetExt.compute_performance()
                # === 性能比對條件 === # *甚麼外掛判斷前後圖非文字變化得到真實標籤，繞一大圈結果竟然是ORB!
                for key, words in semantic_map.items():
                    if isinstance(words, set):
                        matched = any(w in stage for w in words)
                    else:
                        matched = words in stage
                    if not matched:
                        continue
                    # 支援條件格式，如「速度>0.8」或「穩定<0.6」
                    cond = re.search(fr"{key}([<>]=?|=)\s*(\d*\.?\d+)", stage)
                    score = perf_dict[key.lower()]
                    if cond:
                        op, val = cond.group(1), float(cond.group(2))
                        if not eval(f"{score}{op}{val}"):
                            perf_ok = False
                    elif score < 0.7:  # 無明確數值條件 → 用預設閾值
                        perf_ok = False
                    if not perf_ok:
                        tag = key.capitalize()
                        if not evt.get(tag):
                            evt[tag] = input(
                                f"{img}{stage}{key}未達標 ({score:.3f})，應對作法：").strip() or None
                        回覆(f"⚠️ {key}不達標 → {evt[tag]}")
            if logic_ok and perf_ok:
                self.remove_subscription(e1, e2, e3)
                回覆(f"✅ {e1}對{e2}性能{e3}達成 邏輯性能，已取消訂閱。")
                break

            if ev == e3:
                # 問卷的引導性感覺太低，因為GPT智障
                # nonlocal skip_all_perf, skip_all_logic, stage # 修改外部
                choice = input(
                    "(條件邏輯問卷(修改 設定過的狀態), [錯誤時的 應對作法])，是否要修改設定過的狀態與應對作法？(Enter=跳過全部 / y=填寫一次)："
                ).strip().lower()
                choice2 = input(
                    "(條件邏輯問卷(修改 設定過的狀態), [錯誤時的 應對作法])，是否要修改設定過的狀態與應對作法？(Enter=跳過全部 / y=填寫一次)："
                ).strip().lower()
                if choice == "":
                    回覆("👉 已設定：跳過全部問卷。")
                    skip_all_logic = True
                elif choice != "y":
                    return  # 任何非 y 也視為略過當前
                if skip_all_logic:
                    stage == input(f"設定{img}達成條件邏輯的狀態：").strip() or stage
                    evt["Condition Error"] = input(
                        f"{img}{stage}條件錯誤 時的應對作法：").strip() or evt.get("Condition Error")
                    evt["Sequence Error"] = input(
                        f"{img}{stage}順序錯誤 時的應對作法：").strip() or evt.get("Sequence Error")
                    evt["Logic Conflict"] = input(
                        f"{img}{stage}邏輯衝突 時的應對作法：").strip() or evt.get("Logic Conflict")
                    evt["Boundary Error"] = input(
                        f"{img}{stage}邊界錯誤 時的應對作法：").strip() or evt.get("Boundary Error")
                    evt["Unhandled State"] = input(
                        f"{img}{stage}狀態漏判 時的應對作法：").strip() or evt.get("Unhandled State")
                if choice2 == "":
                    回覆(f"⚠️ 已設定：跳過全部問卷。")
                    skip_all_perf = True
                elif choice2 != "y":
                    return  # 任何非 y f"也視為略過(/m.*)".ground(1)當前
                if skip_all_perf:
                    stage == input(f"設定{img}達成條件邏輯的狀態：").strip() or stage
                    evt["Speed"] = input(
                        f"{img}{stage}速度不夠 時的應對作法：").strip() or None
                    evt["Stability"] = input(
                        f"{img}{stage}不穩定 時的應對作法：").strip() or evt.get("Stability")
                    evt["Capacity"] = input(
                        f"{img}{stage}數量不合 時的應對作法：").strip() or evt.get("Capacity")
                    evt["Accuracy"] = input(
                        f"{img}{stage}不精準 時的應對作法：").strip() or evt.get("Accuracy")
                    evt["Cost"] = input(
                        f"{img}{stage}成本太高 時的應對作法：").strip() or evt.get("Cost")

from PySide6.QtCore import QObject, Signal, Property
# TODO:**回覆訊息為其下資料夾名稱，用戶點某個詞會看到該資料夾的圖片，像方便的小說(動態圖片，可調每次撥放幾張圖)
        # QML (path_dir=root詞,path_dir的files圖片)
        # 點擊回覆對話框中的詞，上方顯示圖片集循環撥放一部分
# TODO:***使用好像有問題
class Backend(QObject):
    """對話 shutil存進TEMPLATE_DIRS["speak"]，Backend getImages讀取TEMPLATE_DIRS["speak"]給QML就是 使用 對話"""
    pathChanged = Signal(str)
    imagesReady = Signal(list)  # 發送圖片列表給 QML
    responseUpdated = Signal(str) # 統一回覆訊息
     
    # 單例模式專用變數，用來儲存唯一的一個實例
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # 確保 __init__ 只會被完整初始化一次（避免重複調用 __init__ 導致屬性被清空）
            cls._instance._initialized = False
        return cls._instance
    
    # 依然要有這行，QML 才能「看到」並「連動」
    path_dir = Property(str, 
        fget=lambda self: self._path_dir, 
        fset=lambda self, v: (setattr(self, '_path_dir', v), self.pathChanged.emit(v)) if self._path_dir != v else None, 
        notify=pathChanged)

    def __init__(self):
        # 配合 __new__ 確保初始化邏輯只執行一次
        if getattr(self, '_initialized', False):
            return
        super().__init__()
        self._path_dir = ""
        self.history = "" # 用來存歷史對話
        self._initialized = True

    @Slot()
    def getImages(self):
        """
        回話區的詞 對應儲存的圖片
        """
        found=path_all(TEMPLATE_DIRS["speak"],".png")
        all_imgs =[(r/imgs).as_uri() for r,_,imgs in found]
        self.imagesReady.emit(all_imgs)

    @Slot(str)
    def conversation(self, msg):
        """單次累積回覆"""
        new_result = f"處理結果: {msg}"
        # 累積對話：舊內容 + 換行 + 新內容
        if self.history:
            self.history += "\n" + new_result
        else:
            self.history = new_result
        # 統一發送累積後的完整字串
        self.responseUpdated.emit(self.history)


class FourierTransform:
    """
    跨越維度的眼睛
    你可以這樣玩：ft = FourierTransform(骨骼動畫_b)
    輸入即輸出：新動態 = ft(影像點1_位移, 影像點2_位移, 描述子向量)
    換個維度玩：ft_finance = FourierTransform(金融訂單簿_b)
    輸入即輸出：新金融頻譜 = ft_finance(人的動作頻譜1, 地球頻譜2)
    """
    def __init__(self, bone_motion_b):
        # 1. 遊戲開機或系統啟動時，只執行這一次！
        # 把大魔王的骨骼頻譜「算好並永遠記在記憶體裡」，以後再也不重複算它！
        self.spectrum_b = np.fft.fft(np.array(bone_motion_b))

    def __call__(self, *args):
        # 2. 貫徹【輸入即輸出】！
        # 每一幀最新的即時資料 *args 衝進來時，電腦只算這組新資料的 FFT
        spectrum_a = np.fft.fft(np.array([np.array(arg) for arg in args]), axis=-1)
        
        # 3. 直接拿「早就記好的 self.spectrum_b」進行強行碰撞，瞬間吐出結果！
        combined_spectrum = self.spectrum_b * np.abs(spectrum_a)
        return np.real(np.fft.ifft(combined_spectrum))

# === 
    # *** 光子發射時序以分段、電場以能階變色，光子測距和計算誤差矯正量
    #
    # 該視窗可以置頂於畫面?固定寬度會自動換行的輸入框?點擊輸入框實輸入?當視窗拖動到最左或最右邊，最小化視窗並固定Y座標?
    # 透明視窗內可以讓3D模型正常地展示骨架動畫，並且可以操作調整模型，位移、放大、旋轉、子物件拉進父物件下面。不像GPT賠錢那麼兇。
    # --- 主程式 ---
"""
視窗標題,目標的多重路徑,多重操作，:多重路徑、<>錄製。
視窗標題,GPT:食指,全選:按下::視窗標題,GPT:肛門,位置深處:放開
"""
import PySide6.QtQml as Qml
if __name__ == "__main__":
    for a in TEMPLATE_DIRS.values():
        make_folder(a)
    for a in 背景節點.values():
        make_folder(a)
       
    # 測試：搜尋「積極」
    # data = asbc_stealth_search(url)
    # 回覆(f"網路抓取結果: {data}")
    #data = np.zeros((5, 11))
    #data[:, 10] = 999 # [999. 999. 999. 999. 999.]
    a = np.array([
        [0, 10, 20, 30, 40],
        [2, 11, 20, 30, 41],
        [2, 12, 20, 30, 42]
    ], dtype=int)
    print(f"a:{a}")
    #print(f"a:{        write_array(a, 
    #    (C(1) << (C(1) + (C(3) // 2)) // 2), 
    #    (C(2) << (C(2) + (C(4) // 2)) // 2))    }")
    print(f"a2 >11: {find_array(a,C(2)>11)}")
    print(f"a4 .isin: {find_array(a,C(4).isin(42))}")
    print(f"a2 >11 & a4 .isin: {find_array(a,C(2)>11) & find_array(a,C(4).isin(42))}")
    
    b = {
        "a":[0, 10, 20, 30, 40],
        "b":[2, 11, 20, 30, 41],
        "c":[2, 12, 20, 30, 42]
    }
    print(f"b:{b}")
    #print(f"b:{        write_array(b, 
    #    (C(1) << (C(1) + (C(3) // 2)) // 2), 
    #    (C(2) << (C(2) + (C(4) // 2)) // 2))    }")
    c= C.where(0,2)(b)
    print(f"b where 02: { c }")
    #print(f"b02.b0>0: { find_array(C.where(0,2)(b) 
    #    ,C(0)>0)}")
    #print(f"b1 >11: {find_array(b,C(1)>11)}")
    #print(f"b4 .isin: {find_array(b,C(4).isin(42))}")
    #print(f"b2 >11 & b4 .isin: {find_array(b,(C(2)>11) & (C(4).isin(42)))}")

    monitor_info = {"width": 1920, "height": 1080} 
    # 實例化
    ic = InputCommand(monitor_info)
    backend=Backend()
    TARGET_DEVICE_ID = r"你的設備ID填在這裡" 
    monitor = EventMonitor()
    rec = Recorder()

    # ✅ 在背景啟動 watchdog 執行緒 # ***app關閉時， watchdog沒有跟著關閉

    # 在 engine.load 之前定義一個物理心跳
    heartbeat_timer = QTimer()
    heartbeat_timer.timeout.connect(send_heartbeat)
    heartbeat_timer.start(5000) # 每 5 秒報一次平安
    send_heartbeat()


    fmt = QSurfaceFormat()
    fmt.setAlphaBufferSize(8)
    fmt.setRenderableType(QSurfaceFormat.OpenGL)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setVersion(4, 1)
    QSurfaceFormat.setDefaultFormat(fmt)
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough) # 確保在視窗關閉時有正確停止所有背景任務。
    app = QApplication(sys.argv)
    
    engine = QQmlApplicationEngine()
    qml_file = DATA_BASE / "ui.qml"  # 確保路徑正確
    engine.addImportPath(str(DATA_BASE))
    # 將 Python 對象暴露給 QML
    engine.rootContext().setContextProperty("IC", ic)
    engine.rootContext().setContextProperty("Backend", backend) # QML使用時首字大寫


    for p in Qml.QQmlEngine().importPathList():
        print("IMPORT PATH:", p)
    if getattr(sys, 'frozen', False):
        engine.addImportPath(sys._MEIPASS)

    engine.load(str(qml_file))
    if not engine.rootObjects():
        print("❌ QML 載入失敗！")
        sys.exit(-1)
        
    # 修正函式名稱為 screenshot，並將結果存入不同的變數名稱 (例如 ss)
    ss = pyautogui.screenshot(region=(0, 0, 500, 500))
    ss.save("test.png")

    win = engine.rootObjects()[0]
    win.show()
    

    sys.exit(app.exec())



# self 實體，本class用同class的def或變數才要用self.，共用的def則不加self.
# 可能需要考慮安全風險

# 觀察情境的特徵和變化的目標，實時看到變化的不變屬性，企圖改變 變化成想要的情境。
# 應該是你把這些當成教育了，事實上工作時不是單一面，也就是說你不理解簡單的道理
# 加入水(分散成霧)明觀(執行或終止) # 加入X感測(執行或終止) # 以木治人、以水觀察、以
# 1️⃣ 工作與生產力必備
    # 桌椅
    # 電腦
    # 網路設備
    # 燈光
    # 空調
    # 筆、筆記本
# 2️⃣ 生活便利
    # 廁所
    # 飲水機、杯子
    # 點心
    # 咖啡豆
# 3️⃣ 收納與整理
    # 櫃子
    # 垃圾桶
# 4️⃣ 安全與應急
    # 急救箱
    # 滅火器
    # 緊急出口標示與疏散通道
# 5️⃣ 氛圍與舒適加分
    # 時尚壁紙 / 辦公室風格
    # 綠植
    # 音響
    # 隔音板 / 牆
# 員工健康與福利
    # 小型健身區或運動器材（門框式單槓、吊環等）
    # 冥想 / 安靜區 (隔音、隔板)