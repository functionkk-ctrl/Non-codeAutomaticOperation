import json
import Noesis
import shutil
import mediapipe as mp
from geographiclib.geodesic import Geodesic
import firebase_admin
from firebase_admin import credentials, firestore, db
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtWidgets import QApplication
from pathlib import Path
import sys
from datetime import datetime
import threading
from difflib import SequenceMatcher
from pywinauto import application
import math
from pynput import mouse, keyboard
import re
import os
import time
import numpy as np
import pyautogui
import pytesseract
import cv2
from PIL import ImageGrab
from OpenGL.GLU import *
from OpenGL.GL import *
import psutil
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtCore import QObject, Slot, QTimer 
from geopy.geocoders import Nominatim
import subprocess

# .venv 一次性安裝 uv add mediapipe geographiclib firebase-admin PySide6 numpy opencv-python Pillow PyOpenGL PyOpenGL-accelerate geopy pynput pyautogui pytesseract psutil pywinauto
# uv run pyinstaller --clean UIA.spec
# uv run UIA.py


# Android
# from plyer import gps
# from kivy.clock import Clock
# from kivy.utils import platform


def on_location(**kwargs):
    print(
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

def read_imu(dt):
    val = accelerometer.acceleration
    if val:
        ax, ay, az = val
    print(ax, ay, az)


# Clock.schedule_interval(read_imu, 1/50)

# --- 基礎設定 --- python "D:\Python\Non-codeAutomaticOperation\UIA.py"
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\USER\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
DATA_BASE = Path(os.getcwd()) 
base_path = Path.home() / ".your_app_name"      # 可寫
TEMPLATE_DIRS = {
    "live_capture": DATA_BASE/ 'live_capture',
    "attributes": DATA_BASE/ "attributes",
    "world": DATA_BASE/ "world",
    "user": DATA_BASE/ "user",  # 用戶隱私
    "communication": DATA_BASE/ "communication",  # 用戶交流的訊息
    "dark_matter": DATA_BASE/ "dark_matter",
    "thinking": DATA_BASE/ "thinking",  # 中轉站
    "thinking2": DATA_BASE/ "thinking2",  # 中轉站
    "speak": DATA_BASE/ "speak",  # 交流的回覆
    "absorb": DATA_BASE/ "absorb",  # Nosis吸收的知識
}

MATCH_THRESHOLD = 0.85
LANGS = "eng+chi_sim"
DEBUG = True    
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# --- 共用工具 ---
alive_event = threading.Event()
cred = credentials.Certificate("serviceAccountKey.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        "databaseURL": "firebase-adminsdk-fbsvc@uia-a-3c57f.iam.gserviceaccount.com"
    })


def path_all(paths, target=None,exclude=None,time=None,use_orb=False):
    """
    預設排序為時間，由舊到新
    yield root(完整目錄), dirs(下一層的全部資料夾名), files(這層全部檔案含檔名)
    paths 依序遍歷 ./a 和 ./b 這兩個目錄，包含到最下層
        # for root, dirs, files in path_all(["./a", "./b"]):
    找到含 target 的檔案或資料夾，返回該根目錄 root/dirs，找不到則回傳 False。
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
            get_ctime = [root/f.stat() for f in files]
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
            yield False

import inspect
def make_folder(folder_name, class_name, content_classes):
    """
    在 base_path 下創建資料夾 folder_name（如果不存在）和腳本含內容
    inspect.getsource(Class )，複製原始碼
    """
    folder_path = base_path / str(folder_name)
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
            print(f"讀取失敗: {file_path}") 
            return []
    else:
        print(f"讀取失敗: {file_path}") 
        return []

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
        np.save(des_path, result_des)
        np.save(kp_path, result_kp)
        if similar_ratio  is not None:
            return np.mean(similarity) # 相似度
        return a 
    
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

class Expr:
    def __init__(self, func):
        self.func = func
    
    def __add__(self, other):
        return Expr(lambda x: self.func(x) + other.func(x))
    
    def __gt__(self, val):
        return Cond(lambda x: self.func(x) > val)
    
    def __lt__(self, val):
        return Cond(lambda x: self.func(x) < val)
    
    def __truediv__(self, other):
        # 支援除法，自動處理除以 0 的風險
        return Expr(lambda x: self.func(x) / np.where(other.func(x) == 0, 1, other.func(x)))

    def argsort(self):
        """回傳由小到大的索引排序"""
        return lambda x: np.argsort(self.func(x))

    def diff(self):
        """
        對該欄位進行差分運算 (x[i] - x[i-1])
        axis=0,第一筆差值必為 0（對應你原本 if i==0: continue 的邏輯）。
        """
        return Expr(lambda x: np.diff(self.func(x), axis=0, prepend=[self.func(x)[0]]))

    def norm(self):
        """計算歐氏距離 (常用於速度向量轉純量)"""
        # 這裡假設傳入的是向量，或者你想對結果取絕對值
        return Expr(lambda x: np.linalg.norm(self.func(x), axis=1) if x.ndim > 1 else np.abs(self.func(x)))

    # np.diff [i]-[i-1]
    def is_peak(self):
        """偵測波峰：i-1 < i > i+1"""
        def peak_logic(x):
            # 用 np.pad 補位，確保回傳的布林陣列跟原陣列一樣長
            # 這樣你就永遠不用手動 +1
            d = np.diff(x, prepend=x[0]) 
            return (d > 0) & (np.append(np.diff(x) < 0, False))
        return Cond(peak_logic)

    def is_valley(self):
        """偵測波谷：i-1 > i < i+1"""
        def valley_logic(x):
            d = np.diff(x, prepend=x[0])
            return (d < 0) & (np.append(np.diff(x) > 0, False))
        return Cond(valley_logic)
    
    def is_up(self):
        """上波：只要這格比前一格大 (diff > 0)"""
        return Cond(lambda x: np.diff(self.func(x), prepend=self.func(x)[0]) > 0) # self.func 把那一欄「抽出來」再做 diff。避免全部起diff

    def is_down(self):
        """下波：只要這格比前一格小 (diff < 0)"""
        return Cond(lambda x: np.diff(self.func(x), prepend=self.func(x)[0]) < 0) # self.func 把那一欄「抽出來」再做 diff。避免全部起diff

    # np
    def unify(self):
        """這就是你的『統一時間』邏輯：將數據壓縮到 0~1 的標準空間"""
        def unify_logic(x):
            val = self.func(x)
            # t / (t_max - t_min)
            diff = val[-1] - val[0]
            return val / (diff if diff != 0 else 1)
        return Expr(unify_logic)

    def tile(self, reps):
        """這就是你的『np.tile』邏輯：實現全向量化的鋪地磚比對"""
        return Expr(lambda x: np.tile(self.func(x), reps))

    def roll(self,i):
        """前i位判斷，>0是前i位，<0是後i位"""
        return Expr(lambda x: np.roll(self.func(x), i))

    def a_neighbor_b(self, target_a, neighbor_b):
        def neighbor(x):
            data = np.asarray(x)
            # 1. 建立標籤（布林陣列）
            is_a = np.isin(data, target_a)
            is_b = np.isin(data, neighbor_b)
            # 2. 核心：只要是 a，或「緊鄰 a 的 b」，都標記為 True
            # np.roll(is_a, 1) | np.roll(is_a, -1) 是 a 的左右鄰居
            keep_mask = is_a | (is_b & (np.roll(is_a, 1) | np.roll(is_a, -1)))
            # 3. 處理「連續多個 b」的情況：如果一個 b 旁邊是「已經被標記的 b」，也把它標記起來
            # 如果你的 b 可能很長 (如 b,b,b,a)，這裡可以用 while 或簡單的兩次 dilation
            # 但最快的方式是直接找「連通區塊」中是否有 a
            all_candidates = is_a | is_b
            diff = np.diff(all_candidates.astype(int), prepend=0, append=0)
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]

            res = []
            for s, e in zip(starts, ends):
                chunk = data[s:e]
                # DSL 邏輯：這組裡面必須「包含 a」且「不只有 a (也就是有旁邊的 b)」
                if np.any(np.isin(chunk, target_a)) and len(chunk) > np.sum(np.isin(chunk, target_a)):
                    res.append(chunk.tolist())
            # data = [10, 10, 20, 30, 50, 20, 10], a=[20], b=[10, 30]
            return res # [[10, 10, 20, 30], [20, 10]]
        return Cond(neighbor)

class Cond:
    def __init__(self, func):
        self.func = func
    
    def __and__(self, other):
        return Cond(lambda x: self.func(x) & other.func(x))
    
    def __or__(self, other):
        return Cond(lambda x: self.func(x) | other.func(x))
    
    def apply(self, array):
        return array[self.func(array)]

    def get_mask(self, array):
        """回傳 True/False 的布林索引陣列"""
        return self.func(array)


class Col(Expr):
    """
    用途:取得多維資料(array)的每一筆的某些幾維資料的內容(可能array)，一維[0,1]生成二維[[0],[1]]
    Col 繼承自 Expr，把後面的函數交給父類別 Expr 的 __init__ 來儲存，這個函數存在 self.func 中
    """
    def __init__(self, arg):
        if isinstance(arg, int):
            def fn(x):
                x_arr = np.asarray(x)
                # 🔑 關鍵：統一升維
                if x_arr.ndim == 1:
                    x_arr = x_arr.reshape(-1, 1)
                return x_arr[:, arg]
            super().__init__(fn)
        else:
            # 如果是直接給資料 (speed)，則封裝成一個直接回傳該資料的函數
            # 這樣它就能參與後面的 .stack() 運算
            arr = np.asarray(arg)
            super().__init__(lambda x, arr=arr: arr)

    def __eq__(self, val):
        return Cond(lambda x: self.func(x) == val)
    
    def isin(self, values):
        return Cond(lambda x: np.isin(self.func(x), values))
    
    def between(self, a, b):
        return Cond(lambda x: (self.func(x) >= a) & (self.func(x) <= b))

    def __ne__(self, val):
        if isinstance(val, (list, np.ndarray)):
            return Cond(lambda x: ~np.isin(self.func(x), val)) # 這裡就是你要的 "排除"
        return Cond(lambda x: self.func(x) != val)

def C(*args):
    """
    【欄位選取器】與【邏輯封裝器】
    這是一個工廠函數，回傳一個 Col(i) 物件。
    核心機制：
    1. 延遲執行 (Lazy Evaluation): 
       呼叫 C(i) 時並不執行計算，而是透過 Col 的 super().__init__ 
       定義一個 'lambda x: x[:, i]' 函數(選取第 i 欄)，存入 self.func 中。
       只有在最後執行 find_array 或 apply 時，才會把真正的 array 丟進去運算。
    2. 運算子重載 (Operator Overloading): 
       Expr 與 Cond 類別重新定義了 Python 原生的符號（如 +, >, <, ==, &, |），
       讓你可以像寫 SQL 或 Pandas 一樣組合篩選條件。
    3. 波形偵測 (Waveform Detection):
       透過 is_peak() 與 is_valley()，在類別內部利用 np.diff 處理差分。
       其中 prepend=x[0] 與 np.append 的設計，解決了 np.diff 長度少 1 的問題，
       確保回傳的布林遮罩 (Mask) 與原陣列長度完全對齊，外層不需要手動修正索引 (+1)。
    用法範例：
        偵測特徵：mask = C(0).is_peak().get_mask(arr) -> 取得所有波峰的布林陣列
        # 必須先寫 find_array:
            find_array(array,C(索引))
            find_array(array,row(索引, array, C(索引)))
            可能的例外:C(int).func(array)
    進化後的 C：
    1. C(0, 1) -> 直接執行 np.column_stack (你的直覺用法)
    2. C(0) -> 建立 Col(0)
    3. C(speed_array) -> 建立資料封裝 Expr
    """
    if len(args) > 1:
        # 這裡就是你說的：C(speed, skel_arr[:, 0]) 
        # 直接執行原本要手寫的 np.column_stack
        # 並且為了維持繼承功能，我們把結果封裝回 Expr
        res = np.column_stack(args)
        return Expr(lambda x: res) 
    return Col(args[0])

def find_array(array,cond):
    """
    用法:find_array(資料陣列,C(順位) 運算符號 數值或文字)，輸入對象請務必維持為 NumPy Array。
    -:find_array(目標,C(0)!="曲線")
    AND:find_array(array,(C(0) == 1) & (C(1) > 5)
    OR:find_array(array,(C(0) == 1) | (C(1) > 5)
    between:find_array(array,C(0).between(1, 5)
    between + isin:find_array(array,(C(0).between(1, 5)) & (C(1).isin([2,3,4])) 
        np.isin(A, B)
        A 裡每個元素
        是否存在於 B
    :find_array(array,((C(0) == 1) & (C(1) > 5)) | (C(2) < 3)
    +:find_array(array,(C(0) + C(1)) > 10)
    expression:find_array(array,((C(0) + C(1)) > 10) & (C(2) == 3)
    符合的列且不提取全部的欄:效果詞,效果值 = row([0,1],find_array(效果,C(0).isin(用戶回饋)))
    """
    if not hasattr(cond, 'func'):
        raise ValueError("cond 必須是 Cond 物件或 Expr 比較運算後的結果")
    return array[cond.func(array)]

    # def array_find_array(a,i,array):
    #     # [[1,2,3,4],[1,2,3,4],,,]。i副位 j主位，j主位的i副位若是同一值a，列出array[此主位]
    #     return array[array[:, i] == a] # array[array([True, True, False])] ，只留下 True 對應的列


def row(i, data, func2=None):
    """
    複雜用法:
        一維轉二維篩選後 要轉回一維:func2=lambda x: x.reshape(-1)
        row(None, data) 或 row(slice(None), data) 回傳全部資料
    用途:取得多維資料(array)的某一筆資料的內容(可能array)
    等同於 [r[i] for r i in data]，i可以是list
    如果data是 List，這行會跑 List Comprehension (慢但相容性高)
    如果data是 NumPy，這行會跑向量化提取 (快)
    """
    data_arr = np.asarray(data)
    # 🔑 一維升維
    if data_arr.ndim == 1:
        data_arr = data_arr.reshape(-1, 1)

    if i is None or i == ":" or isinstance(i, slice):
        column_data = data_arr
    else:
        idx = i if isinstance(i, (list, tuple, np.ndarray)) else [i]# 🔑 統一 i
        column_data = data_arr[:, idx]
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
            cv2.imwrite(f"debug_{ts}.png", screenshot())
            print(f"[Watchdog] 主線程可能卡死，已保存 debug_{ts}.png")
        alive_event.clear() # 重置，等待下一波心跳

# 在 engine.load 之前定義一個物理心跳
def send_heartbeat():
    alive_event.set()
    # 這裡可以順便觸發 Noesis 的低壓掃描
    noesis.編織關係() # 編織關係 

heartbeat_timer = QTimer()
heartbeat_timer.timeout.connect(send_heartbeat)
heartbeat_timer.start(5000) # 每 5 秒報一次平安

def screenshot():
    """截取全屏並轉成OpenCV圖像  RGB → BGR """
    img = np.array(ImageGrab.grab())
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def locate_template_orb(name, sort=1, num=1, extractor=False, dir=TEMPLATE_DIRS["live_capture"]):
    """ORB 特徵匹配找圖像 screenshot() → 灰階 """
    name = name.split("<img>")[1]
    path = os.path.join(dir, f"{name}.png")
    if not os.path.exists(path):
        return None
    tpl = cv2.imread(path, 0)
    screen_gray = cv2.cvtColor(screenshot(), cv2.COLOR_BGR2GRAY)
    # 可調整特徵數，越少越快，100小圖標或按鈕、300一般GUI元素、500複雜畫面(例如 Hierarchy)
    orb = cv2.ORB_create(400)
    kp1, des1 = orb.detectAndCompute(tpl, None)
    kp2, des2 = orb.detectAndCompute(screen_gray, None)
    if des1 is None or des2 is None:
        return None
    matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)
    if len(matches) < 5:
        return None  # 太少特徵配對視為不可靠
    # 取前 10 個最佳匹配點的座標
    pts = np.array([kp2[m.trainIdx].pt for m in matches[:10]], dtype=np.int32).tolist()
        
    pts.sort(key=lambda p: (p[0], p[1]))  # 左上排序
    if not pts:
        TargetExtractor().select_polygon_roi()

    # 選取點
    if sort == "奇數":
        pts = pts[::2]
    elif sort == "偶數":
        pts = pts[1::2]
    elif isinstance(sort, int):
        idx = sort - 1 if sort > 0 else sort
        return [pts[idx]] if -len(pts) <= idx < len(pts) else []
    # 處理 num(正數取前 num，負數取倒數 abs(num)）
    if num != 1:
        pts = pts[:num] if num > 0 else pts[num:]
    return pts

# *** 多張圖像中偵測目標圖像


def locate_template_orb_cached(obj, name, sort=1, num=1):
    obj.extractor=TargetExtractor()
    if name in obj.cache:
        pos = obj.cache[name]
        if validate_cache(name, pos):
            return pos
    pos = locate_template_orb(name, sort, num, extractor=obj.extractor)
    if pos:
        obj.cache[name] = pos
    return pos

def validate_cache(name, pos, tolerance=10, dir=TEMPLATE_DIRS["live_capture"]):
    screen_gray = cv2.cvtColor(screenshot(), cv2.COLOR_BGR2GRAY)
    h, w = screen_gray.shape[:2]
    x, y = pos
    # 安全邊界
    x1, y1 = max(x - tolerance, 0), max(y - tolerance, 0)
    x2, y2 = min(x + tolerance, w), min(y + tolerance, h)
    region = screen_gray[y1:y2, x1:x2]
    tpl = cv2.imread(os.path.join(dir, f"{name}.png"), 0)
    if tpl is None or region.size == 0:
        return False
    res = cv2.matchTemplate(region, tpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return max_val > 0.8


def locate_text(keyword, sort=1, num=1, classA=None):
    """找字"""
    # OCR識別文字
    data = pytesseract.image_to_data(cv2.cvtColor(screenshot(
    ), cv2.COLOR_BGR2GRAY), lang=LANGS, output_type=pytesseract.Output.DICT)
    # 收集匹配點
    pts = [
        (data['left'][i] + data['width'][i] // 2,
         data['top'][i] + data['height'][i] // 2)
        for i, t in enumerate(data['text'])
        # SequenceMatcher 相符比例
        if t.strip() and SequenceMatcher(None, t.lower(), keyword.lower()).ratio() >= 0.7
    ]
    if not pts:
        if DEBUG:
            print(f"⚠️ 找不到匹配點：{keyword}。若目標在場則建議")
        return None
    # 排序(左上優先）
    pts.sort(key=lambda p: (p[0], p[1]))
    # sort 整數 → 指定位置；奇偶 → 篩選；否則回傳前 num 個 # 序列[start:end:step]
    if sort == "奇數":
        pts = pts[::2]
    elif sort == "偶數":
        pts = pts[1::2]
    elif isinstance(sort, int):
        idx = sort - 1 if sort > 0 else sort
        return pts[idx] if -len(pts) <= idx < len(pts) else None
    # 處理 num(正數取前 num，負數取倒數 abs(num)）
    if num != 1:
        pts = pts[:num] if num > 0 else pts[num:]
    if classA is None:
        return pts
    else:
        # * 找classA 的內容，預設是 # 找 classA 的這一行 classA後面
        readText = [
            t
            for t in data['text']
            if t.strip() and SequenceMatcher(None, t.lower(), keyword.lower()).ratio() >= 0.7
        ]

        # *** classA 似乎在這一行開始不通用了，使用到 Geocoding
        # *** firebase 用戶儲存的起點地址 addrStart
        def UID():
            cred = credentials.Certificate("path/to/serviceAccountKey.json")
            firebase_admin.initialize_app(cred)
            # 2. 驗證從前端傳來的 ID Token
            id_token= input("請輸入ID: ").strip()
            try:
                decoded_token = auth.verify_id_token(id_token)
                uid = decoded_token['uid']
                print(f"驗證成功！用戶 UID: {uid}")
            except Exception as e:
                print("驗證失敗：", e)

        if UID(): # TODO:**驗證是否為用戶?
            geolocator = Nominatim(user_agent="geo_example")
            startP = firestore.client().reference("addrStart").get()
            nearP = firestore.client().document("near").get().to_dict()
            farP = firestore.client().document("far").get().to_dict()
            locationStart = geolocator.geocode(startP)
            locationNear = geolocator.geocode(startP)
            locationFar = geolocator.geocode(startP)

        def dist(a, b):
            aLocation = geolocator.geocode(a)
            # 避免被geocode 封鎖
            time.sleep(0.1)
            if b == startP:
                bLocation = locationStart
            elif b == nearP:
                bLocation = locationNear
            elif b == farP:
                bLocation = locationFar
            else:
                bLocation = geolocator.geocode(b)
            if aLocation or bLocation is None:
                print("無效地址")
            distance = (aLocation.latitude - bLocation.latitude)**2 + \
                (aLocation.longitude - bLocation.longitude)**2
            time.sleep(0.05)
            return distance
        # 間距太近(firestore.client().reference(太近的地址)，起點和太近地址的距離為 間距)的一些地址為一分支 manifest[分支]，離起點太遠(firestore 太遠地址)額外安排 manifest2
        NEAR_DISTANCE = dist(nearP, startP)
        FAR_DISTANCE = dist(farP, startP)


        for ress in readText:
            line_key = (
                data['block_num'][ress],
                data['par_num'][ress],
                data['line_num'][ress]
            )
            addresses = []
            for j, t in enumerate(data['text']):
                if not t.strip():
                    continue
                if j < ress:
                    continue
                if (data['block_num'][j], data['par_num'][j], data['line_num'][j]) != line_key:
                    continue

                addresses.append({
                    "address": t,
                    "distance": dist(t, startP),
                    # ***使用 找地址時，順便 找貨品
                    # *** 搜尋相符文字的貨品乘上數量，並計算疊加的空間大小，以疊加大小來排序
                    "goods": ""
                })
            addresses.sort(key=lambda x: x["distance"])
            # 建立 manifest 分支(近 / 遠） # 用戶說分支，也有可能是說其他東西
            manifest_near = [
                {"address": addresses[i]["address"],
                    "goods": addresses[i]["goods"]}
                for i in range(len(addresses)-1)  # 用 index 才能拿下一筆
                if addresses[i]["distance"] <= NEAR_DISTANCE
                and abs(addresses[i]["distance"] - addresses[i+1]["distance"]) <= NEAR_DISTANCE
            ]
            manifest_far = [
                {"address": info["address"], "goods": info["goods"]}
                for info in addresses
                if info["distance"] >= FAR_DISTANCE
            ]
            manifest = [manifest_near, manifest_far]
            # *** goods 排列在有限空間，計算manifest難度 排序
            # 4️⃣ 上傳 Firebase
            # manifest 上傳給firebase，manifest中最難的給最早請求的用戶 # *** firebase 分發給用戶，用戶如何獲取 manifest

            # firestore.client().document("manifest").add(manifest)

            # *** 繪製路線圖並記錄指南針方向，旋轉地圖時路線圖與地圖的指南針向量 矯正
            # *** 指南針計算(一維)
            # Routing API給最佳真實路線


def click(pos): 
    pyautogui.moveTo(*pos, duration=0.2); pyautogui.click(); time.sleep(0.05)

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
                print(f"驗證成功！用戶 UID: {uid}")
            except Exception as e:
                print("驗證失敗：", e)

        if UID(): # TODO:**驗證是否為用戶?
            found1=path_all(TEMPLATE_DIRS["user"],"被覆蓋的技巧")
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
                            shutil.copy2(make_folder(TEMPLATE_DIRS["user"]/"直覺要用的"),a) # TODO:** 直覺前兆觸發！抓到的『東西』
            # 3. 主詞與語態的「確定性偏移」 (Certainty Shift)
                # 語氣變得短促且篤定。語氣的「重音轉移」從「詢問/討論」變成「宣告」。
                # 延遲一段時間後，使用絕對肯定詞 「沉默」之後出現的第一個「名詞」視為直覺的導向標的。
                    if 對話節奏變慢索引 + 1 < len(nouns_f):
                        直覺導向標的 = nouns_f[對話節奏變慢索引 + 1]
                        # 輸出抓到的直覺本體（檔案名稱通常代表該詞）
                        for a in 直覺導向標的:
                            shutil.copy2(make_folder(TEMPLATE_DIRS["user"]/"直覺要用的"),a) # TODO:** 捕捉到直覺實體
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
        state_mgr.add("有趣").add("代價").add(["不有趣", "普通", "有趣"])
    設置 "有趣" -> "代價" 子狀態為 "普通"
        state_mgr.有趣.代價.set("普通")
    執行轉移
        state_mgr.有趣.代價.transition("有趣", "爆炸")
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
            return 自己._states["追蹤失敗"].transition(name)
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
            return 自己._sub["追蹤失敗"].transition(name)
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

class InputCommand(QObject):
    def __init__(self, monitor_data):
        super().__init__()
        self.monitor = monitor_data 
        self.vars = {}
        self.current_window = None
        self.cache = {}
        self.extractor = True
        self.app = None

    def focus_window(self, title):
        title_pattern = fr'^{title}.*'
        try:
            app = application(backend="uia").connect(title_re=title_pattern)
            app.window(title_re=title_pattern).set_focus()
            self.current_window = title
            print(f"🧠 聚焦 [{title}]")
        except Exception as e:
            print(f"❌ 無法聚焦 [{title}]: {e}")

    def selected(self, str, sort=1, num=1, classA=None):
        if "<img>" in str:
            return locate_template_orb_cached(str, sort, num)
        else:
            return locate_text(str, sort, num, classA)

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
                    print("✅ 已關閉自動確認目標模式")
                case "移除":
                    var = input("移除哪個錄製變數: ").strip() or None
                    rec.remove(ic, var)
                case "整理路線":  # ***整理路線
                    var = input("已開啟 整理路線 ").strip() or None
                    self.selected("地址", 1, 1, "地址")

                case "距離多少":  # ***和下一個地址 距離多少
                    var = input("已繪製地圖 ").strip() or None

                    def real_dist(p, q):
                        return Geodesic.WGS84.Inverse(
                            p.lat, p.lon, q.lat, q.lon
                        )['s12']
                case "繪圖":  # ***繪圖
                    var = input("已繪製地圖 ").strip() or None

                case _:
                    print(f"⚠️ 未知指令: {cmd_type}")
        else:
            # 普通命令直接執行
            cmds = user_input.split("::")
            ic.execute_line(cmds)

    def execute_line(self, lines):
        for line in lines:
            try:
                window, path, action = [x.strip() for x in line.split(',', 2)]
                if self.current_window != window:
                    self.focus_window(window)
                    time.sleep(0.3)
                for pa in path.split(":"):
                    # [(x,y),(x,y),(x,y),...]，sp[0]=x,y，sp[0][1]=y，打死GPT
                    sp = self.selected(pa)
                    if sp is not None:
                        if pa != path.split(":")[-1]:
                            click(sp[0])
                        elif pa == path.split(":")[-1]:
                            for act in action.split(":"):
                                i = 0
                                while i < len(action):
                                    act = action[i]
                                    match act:
                                        case "第零一二三步": noesis.input
                                        case "Noesis編織關係": noesis.編織關係()
                                        case "Noesis輸入": noesis.輸入(action[i+1:])
                                        # Unity
                                        case "點擊": click(sp[0])
                                        case "雙擊": pyautogui.doubleClick(sp[0])
                                        case "右鍵": pyautogui.rightClick(sp[0])
                                        case "中鍵": pyautogui.middleClick(sp[0])
                                        case "按下": pyautogui.mouseDown(sp[0])
                                        case "放開": pyautogui.mouseUp(sp[0])
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
                                        # 計算出最左邊最上面的依序的第S位N個，selected(pa)，-3為倒數第三位
                                        case act if re.fullmatch(r"第(-?\d+)位(\d+)個", act):
                                            m = re.fullmatch(
                                                r"第(-?\d+)位(\d+)個", act)
                                            self.selected(
                                                pa, int(m.group(1)), int(m.group(2)))
                                        # 計算出最左邊最上面的依序的偶數個
                                        case act if re.fullmatch(r"偶數(\d+)個", act):
                                            m = re.fullmatch(r"偶數(\d+)個", act)
                                            self.selected(
                                                pa, "偶數", int(m.group(1)))
                                        case act if re.fullmatch(r"奇數(\d+)個", act):
                                            m = re.fullmatch(r"奇數(\d+)個", act)
                                            self.selected(
                                                pa, "奇數", int(m.group(1)))
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
                                                sp[0], m.group(1)[0])
                                            if eval(f"{distance:.2f}{m.group(2)}{float(m.group(3))}"):
                                                i += 2  # 跳到「下下個」act
                                                continue
                                            else:
                                                i += 1  # 正常往下
                                                continue
                                        case "顯示該目標座標":
                                            print(f"📍 {pa}: {sp[0]}")
                                        case "顯示時間":
                                            print(
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
                                            print(
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
                                                print("請依照圖片_W0_H0_Z0格式")
                                                return
                                            self.selected(act[i+1])
                                            i += 2
                                            continue
                                        case "即時計算物體大小":
                                            # *** 計算模式，需要OCR計算物體容積
                                            TargetExtractor().load_img_whz()
                                            pass
                                        case "畫面生成模型":

                                            pass
                                        # ***補充
                                    i += 1  # 預設每次往下一個
                        # else:找下一個路徑
                    else:
                        if pa == path.split(":")[-1]:
                            # 找滑鼠附近的搜尋欄位圖片，輸入目標
                            if self.selected("<img>search") is not None:
                                keyboard.write(pa, delay=0.05)
                            else:
                                print("沒辦法找到{pa}")
                        else:
                            # 持續滑動檢查前一個路徑的整個畫面，直到無變化時跳出
                            prev_img = screenshot()
                            while True:
                                if self.selected(pa) is not None:
                                    break
                                pyautogui.scroll(-300)
                                curr_img = screenshot()
                                # 改為差異統計法，不需整張畫面比較 np.array_equal
                                diff = np.mean(cv2.absdiff(curr_img, prev_img))
                                if diff < 1.0:  # 可調閾值：<1 代表幾乎沒變
                                    print(f"沒辦法找到 {pa}(畫面未變化）")
                                # 避免重疊記憶體引用
                                prev_img = curr_img.copy()
            except ValueError:
                print("⚠️ Invalid format. Please enter: WindowTitle, Path, Action")

    @Slot(str)
    def quitApp(self):
        print("退出應用程式")
        self.app.quit()


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
            print(f"⚠️ {old_name} 不存在")

    def view(self):
        """檢視全部錄製命令"""
        if not self.recorded:
            print("📭 沒有錄製命令")
            return
        for name, cmd in self.recorded.items():
            print(f"{name}: {cmd}")

    def play(self, ic, var_name):
        """執行錄製命令"""
        if var_name not in self.recorded:
            print(f"⚠️ {var_name} 不存在")
            return
        cmds = self.recorded[var_name].split("::")
        ic.execute_line(cmds)

    def remove(self, ic, var_name):
        """移除指定的錄製命令"""
        if var_name in self.recorded:
            del self.recorded[var_name]
            print(f"✅ {var_name} 已成功移除")
        else:
            print(f"⚠️ {var_name} 不存在，無法移除")


class TargetExtractor:
    def __init__(self,start=True,image=None):
        if start is False:
            print("找不到目標且自動確認未開啟，跳過選取點。 調整ORB_create>=500")
            return
        else:
            print("#已開啟 找不到目標後自動確認目標")
        self.image = image
        self.base = image.copy()
        self.pts = []
        self.readText = []
        self.done = False
        self.cancelled = False
        self.roi_mask = None
        self.orb = cv2.ORB_create(800)

    def select_polygon_roi(self):
        """
        可視化互動圈選多邊形 ROI
        - 左鍵：新增點
        - 右鍵：結束圈選
        - ESC：取消圈選
        - R：重置重新圈
        """
        print("🖱️ 請用滑鼠左鍵圈選多邊形；右鍵結束；ESC 取消；R 重來")
        display = self.image.copy()
        done = False
        # ***可能未監聽

        def on_click(x, y, button, pressed):
            if not pressed:
                return
            if button == mouse.Button.left:
                self.pts.append((x, y))
                print(f"➕ 點({x},{y})")
            elif button == mouse.Button.right:
                if len(self.pts) >= 3:
                    self.done = True
                    print("✅ 結束圈選")
                else:
                    print("⚠️ 至少要三個點")
                return False

        def on_press(key):
            nonlocal done
            try:
                if key == keyboard.Key.esc:
                    done = True
                    print("❌ 已取消圈選")
                    return False
                elif key.char.lower() == 'r':
                    print("🔁 重新圈選")
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
                cv2.circle(frame, p, 3, (0, 0, 255), -1)
            cv2.imshow("Draw ROI", frame)
            cv2.waitKey(10)
            if cv2.waitKey(20) & 0xFF == 27:
                break

    def filter_target(self, dir=TEMPLATE_DIRS["live_capture"]):
        """
        從 ROI 中提取目標，做 GrabCut 去背景，生成透明圖
        """
        save_path = os.path.join(dir, f"s{time.time():.0f}.png")
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
        os.makedirs(dir, exist_ok=True)  # 沒有就自動建立
        cv2.imwrite(save_path, self.extracted)
        print(f"✅ 已儲存 {save_path}")

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
        # if not match or not self.selected(file):
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
        frame = screenshot()
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
        self.events = {}  # key -> {type, implementation, application, active}
        self.poll_interval = poll_interval
        self.running = False
        self.lock = threading.Lock()
        self.multiple_img_implementation = []  # perf
        self.multiple_img_implementation_target = None
        self.multiple_img_goal = []  # logic
        self.multiple_img_goal_target = None
        self.ic_em = None

    def add_frame(self):
        print("建議開啟extractor自動確認")
        if self.multiple_img_implementation_target is not None:
            if len(self.multiple_img_implementation) < 5:
                self.multiple_img_implementation.append(
                    self.selected(self.multiple_img_implementation_target))
            else:
                self.multiple_img_implementation.pop(
                    self.multiple_img_implementation[1])  # 迭代更新
                self.multiple_img_implementation.append(
                    self.selected(self.multiple_img_implementation_target))
        if self.multiple_img_goal_target is not None:
            if len(self.multiple_img_goal) < 5:
                self.multiple_img_goal.append(
                    self.selected(self.multiple_img_goal_target))
            else:
                self.multiple_img_goal.pop(self.multiple_img_goal[1])  # 迭代更新
                self.multiple_img_goal.append(
                    self.selected(self.multiple_img_goal_target))

    # 訂閱事件

    def subscribe_event(self, m1, m2, m3):
        self.multiple_img_logic_target = m2
        self.multiple_img_implementation_target = m3
        key = f"{m1}->{m2}->{m3}"
        with self.lock:
            self.events[key] = {
                # [目標圖像,目標圖像的狀態 合格的]
                "implementation": [m1, None],
                "application": [m2, None],
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
    def remove_subscription(self, implementation, application, goal):
        key = f"{implementation}->{application}->{goal}"
        with self.lock:
            if key in self.events:
                sk = self.events.pop(key)
                # sk["active"] = False  # 終止監聽
                print(f"[x] 已終止監聽事件: {key}")
            else:
                print(f"[!] 找不到事件: {key}")

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
        e1, e2, e3 = evt["implementation"], evt["application"], evt["goal"]
        for ev in e1, e2, e3:
            for img, stage in ev:
                # ORB分析目標圖片的狀態和在整個螢幕的關係。self.selected找到目標。 Semantic Algebra 語意代數
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
                    print(evt["Condition Error"])
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
                    print(evt["Logic Conflict"])
                # 邊界錯誤 (索引或對象不存在)
                if predicted is None:
                    logic_ok = False
                    if not evt.get("Boundary Error"):
                        evt["Boundary Error"] = input(
                            f"{img}不存在於螢幕中 時的應對作法：").strip() or None
                    print(evt["Boundary Error"])
                    continue
                # 狀態漏判 (CNN 沒返回任何預測)
                if not predicted.get("pos") and not predicted.get("area"):
                    logic_ok = False
                    if not evt.get("Unhandled State"):
                        evt["Unhandled State"] = input(
                            f"{img}找到，但沒有有效狀態 時的應對作法：").strip() or None
                    print(evt["Unhandled State"])
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
                        print(f"⚠️ {key}不達標 → {evt[tag]}")
            if logic_ok and perf_ok:
                self.remove_subscription(e1, e2, e3)
                print("邏輯性能完成，取消訂閱。")
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
                    print("👉 已設定：跳過全部問卷。")
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
                    print("👉 已設定：跳過全部問卷。")
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
    copyChanged = Signal(str)
    imagesReady = Signal(list)  # 發送圖片列表給 QML

    def __init__(self):
        super().__init__()
        self._path_dir = ""
        self._copy_path = ""

    def getPath(self): return self._path_dir
    def setPath(self, v):
        if self._path_dir != v:
            self._path_dir = v
            self.pathChanged.emit(v)

    def getCopy(self): return self._copy_path
    def setCopy(self, v):
        if self._copy_path != v:
            self._copy_path = v
            self.copyChanged.emit(v)

    path_dir = Property(str, getPath, setPath, notify=pathChanged)
    copy_path = Property(str, getCopy, setCopy, notify=copyChanged)

    @Slot()
    def getImages(self):
        found=path_all(TEMPLATE_DIRS["speak"],".png")
        for _,imgs in found:
            self.imagesReady.emit(imgs)
            
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
    monitor_info = {"width": 1920, "height": 1080} 
    # 實例化
    ic = InputCommand(monitor_info)
    TARGET_DEVICE_ID = r"你的設備ID填在這裡" 
    monitor = EventMonitor()
    rec = Recorder()
    noesis_input=noesis.input
    noesis_編織關係=noesis.編織關係
    noesis_輸入=noesis.輸入
    
    # ✅ 在背景啟動 watchdog 執行緒 # ***app關閉時， watchdog沒有跟著關閉
    send_heartbeat()

    app = QApplication(sys.argv)
    ic.app = app
    fmt = QSurfaceFormat()
    fmt.setAlphaBufferSize(8)
    fmt.setRenderableType(QSurfaceFormat.OpenGL)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setVersion(4, 1)
    QSurfaceFormat.setDefaultFormat(fmt)
    
    # TODO:*** ，提出的問題拆解經過2=128倍得到親子關係，鄰居關係，相差得到答案
    sm=StateMgr
    自己=StateMgr.add("自己")

    engine = QQmlApplicationEngine()
    base = Path(os.path.dirname(os.path.abspath(__file__)))
    qml_file = base / "ui.qml"  # 確保路徑正確
    engine.addImportPath(str(base))

    for p in Qml.QQmlEngine().importPathList():
        print("IMPORT PATH:", p)

    if getattr(sys, 'frozen', False):
        engine.addImportPath(sys._MEIPASS)
    engine.load(str(qml_file))
    if not engine.rootObjects():
        print("❌ QML 載入失敗！")
        sys.exit(-1)

    win = engine.rootObjects()[0]
    win.show()

    # 將 Python 對象暴露給 QML
    engine.rootContext().setContextProperty("IC", ic)

    sys.exit(app.exec())



# self 實體
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