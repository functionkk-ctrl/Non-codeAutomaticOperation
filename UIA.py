from geographiclib.geodesic import Geodesic
import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin import credentials, db
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
from PySide6.QtCore import QObject, Slot
from geopy.geocoders import Nominatim
# Android
from plyer import gps
from kivy.clock import Clock
from kivy.utils import platform


def on_location(**kwargs):
    print(
        kwargs['lat'],
        kwargs['lon'],
        kwargs.get('altitude'),
        kwargs.get('speed')
    )


gps.configure(on_location=on_location)
gps.start(minTime=1000, minDistance=0)

if platform == 'android':
    from plyer import accelerometer
    accelerometer.enable()


def read_imu(dt):
    val = accelerometer.acceleration
    if val:
        ax, ay, az = val
        print(ax, ay, az)


Clock.schedule_interval(read_imu, 1/50)

# --- 基礎設定 --- python "D:\Python\Non-codeAutomaticOperation\UIA.py"
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\USER\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
base_path = getattr(sys, '_MEIPASS', os.path.dirname(
    os.path.abspath(__file__)))
TEMPLATE_DIRS = {
    "live_capture": os.path.join(base_path, 'live_capture'),
    "attributes": os.path.join(base_path, "attributes"),
    "world": os.path.join(base_path, "world"),
    "communication": os.path.join(base_path, "communication"),
    "dark_matter": os.path.join(base_path, "dark_matter"),
    "thinking": os.path.join(base_path, "thinking"),  # 中轉站
    "thinking2": os.path.join(base_path, "thinking2"),  # 中轉站
}
MATCH_THRESHOLD = 0.85
LANGS = "eng+chi_sim"
DEBUG = True

# --- 共用工具 ---
alive_event = threading.Event()
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://你的專案-id.firebaseio.com/"
})


def watchdog():
    while not alive_event.wait(timeout=10):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"debug_{ts}.png", screenshot())
        print(f"[Watchdog] 主線程可能卡死，已保存 debug_{ts}.png")


def screenshot():
    """截取全屏並轉成OpenCV圖像  RGB → BGR """
    img = np.array(ImageGrab.grab())
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def locate_template_orb(name, sort=1, num=1, extractor=False, dir=TEMPLATE_DIRS["img"]):
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
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)
    if len(matches) < 5:
        return None  # 太少特徵配對視為不可靠
    # 取前 10 個最佳匹配點的座標
    pts = [(int(kp2[m.trainIdx].pt[0]), int(kp2[m.trainIdx].pt[1]))
           for m in matches[:10]]
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
    if name in obj.cache:
        pos = obj.cache[name]
        if validate_cache(name, pos):
            return pos
    pos = locate_template_orb(name, sort, num, extractor=obj.extractor)
    if pos:
        obj.cache[name] = pos
    return pos


def validate_cache(name, pos, tolerance=10, dir=TEMPLATE_DIRS["img"]):
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
        geolocator = Nominatim(user_agent="geo_example")
        startP = firestore.client("用戶").reference("addrStart").get()
        nearP = firestore.client("用戶").document("near").get().to_dict()
        farP = firestore.client("用戶").document("far").get().to_dict()
        locationStart = geolocator.geocode(startP)
        locationNear = geolocator.geocode(startP)
        locationFar = geolocator.geocode(startP)
        # 間距太近(firestore.client().reference(太近的地址)，起點和太近地址的距離為 間距)的一些地址為一分支 manifest[分支]，離起點太遠(firestore 太遠地址)額外安排 manifest2
        NEAR_DISTANCE = dist(nearP, startP)
        FAR_DISTANCE = dist(farP, startP)

        def dist(a, b):
            aLocation = geolocator.geocode(a)
            # 避免被geocode 封鎖
            time.sleep(0.5)
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
            time.sleep(0.5)
            return distance

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

            firestore.client().document("manifest").add(manifest)

                # *** 繪製路線圖並記錄指南針方向，旋轉地圖時路線圖與地圖的指南針向量 矯正
                # *** 指南針計算(一維)
                # Routing API給最佳真實路線


def click(pos): pyautogui.moveTo(
    *pos, duration=0.2); pyautogui.click(); time.sleep(0.3)


class InputCommand(QObject):
    def __init__(self):
        super().__init__()
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
                    time.sleep(0.5)
                # *鍵盤滑鼠
                # -*- coding: utf-8 -*- # 滑鼠 + 鍵盤全功能示例(不含監聽） import pyautogui, keyboard, time from pynput.mouse import Button, Controller as MouseController from pynput.keyboard import Key, Controller as KeyController # pyautogui 全域設定 pyautogui.FAILSAFE = True pyautogui.PAUSE = 0.1 # ==== 滑鼠控制 ==== # 位置資訊 screen_w, screen_h = pyautogui.size() print("Screen:", screen_w, screen_h) print("Mouse position:", pyautogui.position()) # 基本移動 pyautogui.moveTo(100, 100, duration=0.3) pyautogui.moveRel(50, 0, duration=0.2) # 點擊與雙擊 pyautogui.click() pyautogui.doubleClick() pyautogui.rightClick() pyautogui.middleClick() pyautogui.click(300, 300) # 按下 / 放開(可長按） pyautogui.mouseDown(button='left') time.sleep(0.5) pyautogui.mouseUp(button='left') # 拖曳操作 pyautogui.moveTo(400, 400) pyautogui.mouseDown() pyautogui.moveTo(600, 600, duration=1.0) pyautogui.mouseUp() # 滾輪 pyautogui.scroll(300) pyautogui.scroll(-300) # ==== 鍵盤控制 ==== # 輸入文字 pyautogui.typewrite("Hello from pyautogui!", interval=0.05) # 單鍵操作 pyautogui.press("enter") pyautogui.press("tab") pyautogui.press("backspace") # 組合鍵 pyautogui.hotkey("ctrl", "s") pyautogui.hotkey("alt", "f4") # 拆解按下與放開 pyautogui.keyDown("shift") pyautogui.press("a") pyautogui.keyUp("shift") # ==== 使用 pynput 進階控制 ==== mouse = MouseController() keyboard_ctrl = KeyController() # 滑鼠精確控制 mouse.position = (200, 200) mouse.press(Button.left) time.sleep(0.3) mouse.release(Button.left) mouse.press(Button.right) mouse.release(Button.right) mouse.scroll(0, 3) # 鍵盤精確控制 keyboard_ctrl.press('a') keyboard_ctrl.release('a') keyboard_ctrl.press(Key.enter) keyboard_ctrl.release(Key.enter) # ==== 使用 keyboard 模組 ==== keyboard.press_and_release('ctrl+c') keyboard.write('Typed by keyboard module!', delay=0.05) if keyboard.is_pressed('esc'): print("ESC pressed!")
                # *運算
                # -*- coding: utf-8 -*- import math, random, statistics, decimal, fractions, cmath, numpy as np # === 基本四則 === a, b = 10, 3 print(a + b, a - b, a * b, a / b, a // b, a % b, a ** b) # === 比較與邏輯 === print(a > b, a < b, a == b, a != b, a >= b, a <= b) # === 內建函式 === print(abs(-5), round(3.14159, 2), pow(2, 5), divmod(17, 3), sum([1,2,3,4])) # === math 模組 === print(math.sqrt(16), math.pow(2, 10), math.factorial(5)) print(math.sin(math.pi/2), math.cos(0), math.tan(math.pi/4)) print(math.degrees(math.pi), math.radians(180)) print(math.log(100, 10), math.log2(8), math.exp(1)) print(math.ceil(2.1), math.floor(2.9), math.trunc(-3.8)) print(math.gcd(24, 36), math.isclose(0.1+0.2, 0.3)) # === 統計 === data = [2, 3, 5, 7, 11] print(statistics.mean(data), statistics.median(data), statistics.pstdev(data)) # === 隨機 === print(random.random(), random.randint(1,10), random.uniform(1.5,5.5)) print(random.choice(['A','B','C'])) items = [1,2,3,4]; random.shuffle(items); print(items) # === decimal 高精度運算 === decimal.getcontext().prec = 10 x = decimal.Decimal('1.1') + decimal.Decimal('2.2') print(x)  # 精確加法 # === fractions 分數 === f1 = fractions.Fraction(1,3); f2 = fractions.Fraction(1,6) print(f1 + f2, f1 * f2) # === 複數 === z1, z2 = 2+3j, 1-1j print(z1 + z2, z1 * z2, abs(z1), cmath.phase(z1)) # === numpy 高階運算 === arr = np.array([1,2,3,4,5]) print(arr + 2, arr * 3, np.mean(arr), np.std(arr)) print(np.sin(arr), np.dot([1,2,3],[4,5,6]))
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
                                            pyautogui.dragRel(-200,
                                                              0, duration=0.5)
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
    def __init__(self, image=None):
        if self.extractor is False:
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
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

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

    def filter_target(self, dir=TEMPLATE_DIRS["img"]):
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
    # *** Img+GPS 列出 圖像中占比大的一些相似物體 和長寬高，等待QML輸入要儲存的圖片名稱，進TEMPLATE_DIRS["img"]資料夾。計算相似物品的 單一數量的 實際大小
    def Img_IMU_GPS():
        # 讀取設備，GPS得高度尺可以和地面參照，GPS平移得橫向尺在空中至少要移動20m，才可以參照
        # *** 先拓樸後幾何，穩定拓樸結構

        # 1️⃣ 讀相機內參
        val cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val cameraId = cameraManager.cameraIdList[0]
        val characteristics = cameraManager.getCameraCharacteristics(cameraId)

        val focalLengths = characteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
        val sensorSize = characteristics.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)

        val fx = focalLengths[0] / sensorSize!!.width * imageWidth
        val fy = focalLengths[0] / sensorSize.height * imageHeight
        val cx = imageWidth / 2f
        val cy = imageHeight / 2f
        val K = arrayOf(arrayOf(fx, 0, cx), arrayOf(0, fy, cy), arrayOf(0, 0, 1))

        # 2️⃣ 讀 GPS
        val locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager
        val location1 = locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER)
        val C1 = doubleArrayOf(location1.latitude, location1.longitude, location1.altitude)

        val location2 = locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER)
        val C2 = doubleArrayOf(location2.latitude, location2.longitude, location2.altitude)

        # 3️⃣ 拍兩張照片，取得影像點 (ORB)
        val orb = ORB.create(3000)
        val kp1 = MatOfKeyPoint()
        val kp2 = MatOfKeyPoint()
        val des1 = Mat()
        val des2 = Mat()
        orb.detectAndCompute(img1, Mat(), kp1, des1)
        orb.detectAndCompute(img2, Mat(), kp2, des2)

        val bf = BFMatcher(NORM_HAMMING, true)
        val matches = bf.match(des1, des2)

        val pts1 = matches.map {kp1.toArray()[it.queryIdx].pt}
        val pts2 = matches.map {kp2.toArray()[it.trainIdx].pt}

        # 4️⃣ Essential Matrix + recoverPose (旋轉不用管)
        val E = Calib3d.findEssentialMat(pts1, pts2, K, RANSAC, 0.999, 1.0)
        val R = Mat()
        val t = Mat()
        Calib3d.recoverPose(E, pts1, pts2, K, R, t)

        # 5️⃣ GPS 當尺
        val baseline = doubleArrayOf(C2[0]-C1[0], C2[1]-C1[1], C2[2]-C1[2])

        # 6️⃣ Triangulate
        val P1 = Mat.eye(3, 4, CV_64F)
        val P2 = Mat(3, 4, CV_64F)
        # P2 = [R | -R*t]
        Core.hconcat(listOf(R, -R * Mat(baseline)), P2)

        val pts4D = Mat()
        Calib3d.triangulatePoints(P1, P2, pts1, pts2, pts4D)
        val pts3D = pts4D.rowRange(0, 3) / pts4D.row(3)

        # 7️⃣ 計算物體長寬高
        val objPts = pts3D.submat(objIndices)
        val sizeX = Core.minMaxLoc(objPts.col(0)).maxVal - Core.minMaxLoc(objPts.col(0)).minVal
        val sizeY = Core.minMaxLoc(objPts.col(1)).maxVal - Core.minMaxLoc(objPts.col(1)).minVal
        val sizeZ = Core.minMaxLoc(objPts.col(2)).maxVal - Core.minMaxLoc(objPts.col(2)).minVal
        println("L,W,H (m): $sizeX, $sizeY, $sizeZ")

        # V2
        // == == = 1️⃣ 讀取 GPS == == =
        val locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager
        val loc: Location = locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER)
            ?: return

        val lat = loc.latitude
        val lon = loc.longitude
        val alt = loc.altitude // 高度尺(Z）
        val speed=loc.speed // m/s

        // == === 2️⃣ 判斷是否在空中 == ===
        // 工程判斷：高度 + 速度(不搞 AI）
        val isAirborne=alt > 20.0 & & speed > 5.0

        // == === 3️⃣ 讀取影像 == ===
        // img: OpenCV Mat(CameraX / Camera2 轉過來）
        val img: Mat=currentFrameMat

        // == === 4️⃣ 找「大占比物體」 == ===
        // 不分類、不追蹤，只找最大輪廓
        val gray=Mat()
        val bin=Mat()
        Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGR2GRAY)
        Imgproc.threshold(gray, bin, 0.0, 255.0,
            Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU)

        val contours=ArrayList < MatOfPoint > ()
        Imgproc.findContours(
            bin, contours, Mat(),
            Imgproc.RETR_EXTERNAL,
            Imgproc.CHAIN_APPROX_SIMPLE
        )

        if (contours.isEmpty()) return

        val mainContour=contours.maxBy {
            Imgproc.contourArea(it)
        } ?: return

        val rect=Imgproc.boundingRect(mainContour)

        // == === 5️⃣ 僅在「空中」才使用橫向尺 == ===
        if (!isAirborne) return

        // == === 6️⃣ 尺度換算 == ===
        // 高度直接當 Z 尺
        val Z=alt // meters

        // 相機視角(來自設備，實際可從 CameraCharacteristics 讀）
        val hfov=Math.toRadians(60.0) // 水平視角(例）
        val vfov=Math.toRadians(45.0)

        val imgW=img.cols().toDouble()
        val imgH=img.rows().toDouble()

        // 像素 → 實際尺寸(幾何，不是 SLAM）
        val W=2 * Z * Math.tan(hfov / 2) * (rect.width / imgW)
        val H=2 * Z * Math.tan(vfov / 2) * (rect.height / imgH)

        // 長度：取寬高中較大者(工程定義）
        val L=maxOf(W, H)

        // == === 7️⃣ 輸出唯一結果 == ===
        Log.i("SIZE", "L,W,H (m) = $L, $W, $H")


        # *** 儲存3D模型


    # *** 讀取畫面中的 已記錄的 物品(圖像)，全部列出或列出指定物品，無紀錄的列出


    # ***讀取貨品欄的 已記錄的 物品(文字)，無紀錄的列出
    def load_img_whz(self):
        # *** 限制大小
        whz=[]
        for file in os.listdir(TEMPLATE_DIRS["world"]):
            match=re.match(r"(.*)_W(\d+)_H(\d+)_Z([\d\.]+)\.png", file)
            if not match or not self.selected(file):
                continue
            # ****讀取貨品欄的 已記錄的 物品，無紀錄的列出

            whz.append({
                "obj_name": match.group(1),
                "w": int(match.group(2)),
                "h": int(match.group(3)),
                "z": float(match.group(4))
            })
            # whz.w*whz.h*whz.z
        return whz  # 疊加實際大小

    # *** python OCR找到該目標時計算該目標附在其物之上，利用目標的物件名稱紀錄的，計算其物的實際大小
    # *** save_path圖片 重新命名(固定格式有長寬高)，在判斷物體實際大小模式時，在TEMPLATE_DIRS["img"]中找到(固定格式有長寬高)save_path圖片，全部找一次，找到則分析附在何物、計算該物實際大小
    # *** 進入 計算物體實際大小的 計算模式 *** 讀取存檔的圖片
    def compute_logic(self):
        frame=screenshot()
        # 全部物件
        logic_state={"objects": [], "relations": [], "scene": None}
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame=self.orb.detectAndCompute(gray, None)
        if des_frame is None:
            return logic_state
        for f in os.listdir(TEMPLATE_DIRS["communication"]):
            if not f.endswith(".png"):
                continue
            tpl=cv2.imread(os.path.join(TEMPLATE_DIRS["communication"], f), 0)
            kp_tpl, des_tpl=self.orb.detectAndCompute(tpl, None)
            if des_tpl is None:
                continue
            matches=self.bf.match(des_tpl, des_frame)
            if len(matches) < 5:
                continue
            pts_frame=np.float32(
                [kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts_tpl=np.float32(
                [kp_tpl[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            M, _=cv2.findHomography(pts_tpl, pts_frame, cv2.RANSAC, 5.0)
            if M is None:
                continue
            h, w=tpl.shape
            corners=cv2.perspectiveTransform(np.float32(
                [[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2), M)
            x, y, w, h=cv2.boundingRect(corners)
            patch=frame[y:y+h, x:x+w]
            color=cv2.mean(patch)[:3] if patch.size > 0 else (0, 0, 0)
            logic_state["objects"].append({
                "name": f.replace(".png", ""),
                "pos": {"x": x, "y": y, "w": w, "h": h},
                "color": {"r": color[2], "g": color[1], "b": color[0]},
                "area": w*h
            })
        # 指定對象
        goal_objects=[]
        for f in os.listdir(self.multiple_img_goal):
            if not f.endswith(".png"):
                continue
            tpl=cv2.imread(os.path.join(self.multiple_img_goal, f), 0)
            kp_tpl, des_tpl=self.orb.detectAndCompute(tpl, None)
            if des_tpl is None:
                continue
            matches=self.bf.match(des_tpl, des_frame)
            if len(matches) < 5:
                continue
            pts_frame=np.float32(
                [kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts_tpl=np.float32(
                [kp_tpl[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            M, _=cv2.findHomography(pts_tpl, pts_frame, cv2.RANSAC, 5.0)
            if M is None:
                continue
            h, w=tpl.shape
            corners=cv2.perspectiveTransform(np.float32(
                [[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2), M)
            x, y, w, h=cv2.boundingRect(corners)
            patch=frame[y:y+h, x:x+w]
            color=cv2.mean(patch)[:3] if patch.size > 0 else (0, 0, 0)
            goal_objects.append({
                "name": f.replace(".png", ""),
                "pos": {"x": x, "y": y, "w": w, "h": h},
                "color": {"r": color[2], "g": color[1], "b": color[0]},
                "area": w*h
                # 動作、變化、互動
            })
        for i, obj in enumerate(goal_objects):
            obj["relations"]=[]
            for j, other in enumerate(logic_state["objects"]):
                if obj["name"] == other["name"]:
                    continue
                # 計算簡單相對位置
                dx=other["pos"]["x"] - obj["pos"]["x"]
                dy=other["pos"]["y"] - obj["pos"]["y"]
                if abs(dx) > abs(dy):
                    direction="右" if dx > 0 else "左"
                else:
                    direction="下" if dy > 0 else "上"
                obj["relations"].append({
                    "object": other["name"],
                    "direction": direction,
                    "distance": (dx**2 + dy**2)**0.5
                })
        # logic_state["scene"] = {"brightness": np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[...,2])}
        logic_state["goal_objects"]=goal_objects
        return logic_state

    def compute_performance(self):
        if len(self.multiple_img_implementation) < 2:
            return None  # 至少要兩幀才能比
        prev_frame=cv2.cvtColor(
            self.multiple_img_implementation[-2], cv2.COLOR_BGR2GRAY)
        curr_frame=cv2.cvtColor(
            self.multiple_img_implementation[-1], cv2.COLOR_BGR2GRAY)
        # --- ORB 特徵 ---
        kp_prev, des_prev=self.orb.detectAndCompute(prev_frame, None)
        kp_curr, des_curr=self.orb.detectAndCompute(curr_frame, None)
        # --- 空保護 ---
        if des_prev is None or des_curr is None or len(kp_prev) == 0:
            return None

        # === 速度(特徵變化率 + 更新頻率)
        start=time.time()
        matches=self.bf.match(des_prev, des_curr)  # ORB 特徵匹配
        end=time.time()
        speed=1 / (end - start)  # 時間越短 → 速度越高
        # === 穩定性(多幀一致 + 特徵方差)。多幀圖。反比，越小越穩，所以要被-1
        stability=1-(1 / (np.var(self.multiple_img_implementation) + 1e-6))
        # === 容量(激活覆蓋率 + 同時辨識數)
        mask=np.zeros(curr_frame.shape[:2], np.uint8)
        capacity=np.sum(mask > 0) / mask.size
        # coverage = len(matches)
        # === 準確性(Softmax機率 + 誤差)。false_matches 可以用前後 frame 無對應特徵數量計算。
        total_matches=len(matches)
        false_matches=abs(len(kp_prev) - total_matches)
        # 更合理的公式 → 匹配成功比例，而非錯誤比例
        accuracy=total_matches / (len(kp_prev) + 1e-6)
        # === 成本( **GPT白癡亂掰:資源下降率 / 目標完成率 )。例如越快打死GPT，成本越低
        cpu=psutil.cpu_percent()
        mem=psutil.virtual_memory().percent
        cost=1 / (1 + cpu + mem)

        return dict(speed=speed, stability=stability, capacity=capacity, accuracy=accuracy, cost=cost)


class EventMonitor:
    # {落實}邏輯{應用}性能{目標}結束。機器:Semantic Parse>goal Mapping>Strategy Retrieval>Execution Logic>Output Composition。
    def __init__(self,  poll_interval=0.3):
        self.events={}  # key -> {type, implementation, application, active}
        self.poll_interval=poll_interval
        self.running=False
        self.lock=threading.Lock()
        self.multiple_img_implementation=[]  # perf
        self.multiple_img_implementation_target=None
        self.multiple_img_goal=[]  # logic
        self.multiple_img_goal_target=None
        self.ic_em=None

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
        self.multiple_img_logic_target=m2
        self.multiple_img_implementation_target=m3
        key=f"{m1}->{m2}->{m3}"
        with self.lock:
            self.events[key]={
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
        key=f"{implementation}->{application}->{goal}"
        with self.lock:
            if key in self.events:
                sk=self.events.pop(key)
                # sk["active"] = False  # 終止監聽
                print(f"[x] 已終止監聽事件: {key}")
            else:
                print(f"[!] 找不到事件: {key}")

    # 啟動/停止監聽
    def start_monitor(self):
        self.running=True
        threading.Thread(target=self._monitor_loop, daemon=True).start()

    def stop_monitor(self):
        self.running=False

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
        targetExt=TargetExtractor()
        logic_ok, perf_ok=True  # 判斷 邏輯除錯 和 數據／性能分析 合格且超標為True，不訂閱
        skip_all_perf, skip_all_logic=False  # 🔹 用來記錄是否跳過問卷
        semantic_map={
            "速度": "更快",
            "穩定": "很穩",
            "數量": {"更多", "更全面"},
            "精準": "精準",
            "成本": "省"
        }
        # [目標,目標的狀態]
        e1, e2, e3=evt["implementation"], evt["application"], evt["goal"]
        for ev in e1, e2, e3:
            for img, stage in ev:
                # ORB分析目標圖片的狀態和在整個螢幕的關係。self.selected找到目標。 Semantic Algebra 語意代數
                # 取得螢幕 ORB 狀態
                logic_state=targetExt.compute_logic()
                # 將 goal_objects 對象名稱對應到邏輯狀態
                goal_objects={
                    obj["name"]: obj for obj in logic_state.get("goal_objects", [])}
                predicted=goal_objects.get(img, None)
                # 現在邏輯的狀態 = ORB分析成真實標籤
                logic_predicted={
                    "pos": predicted["pos"],
                    "color": predicted["color"],
                    "area": predicted["area"],
                    "relations": predicted.get("relations", [])
                }

                # 現在邏輯的狀態!=條件邏輯的狀態 時回報應對作法
                if stage is None:
                    stage=input(
                        f"設定{img}達成條件邏輯的狀態：圖像邏輯結構or行為狀態or環境位置or幾何關係").strip() or None
                # 狀態不在期望範圍 → 邏輯錯誤
                # Condition Error: 簡單比對顏色或區域
                if stage not in str(logic_predicted.values()):
                    logic_ok=False
                    if not evt.get("Condition Error"):
                        evt["Condition Error"]=input(
                            f"{img} 條件錯誤: {logic_predicted} vs {stage}, 請輸入應對作法：").strip() or None
                    print(evt["Condition Error"])
                # 分析順序錯誤 (示意：這裡可以用更精細的序列判斷)
                if img == e3[0] and e2[0] not in stage:
                    logic_ok=False
                    if not evt.get("Sequence Error"):
                        evt["Sequence Error"]=input(
                            f"{img} 順序錯誤: e3 出現前 e2 還沒準備好，請輸入應對作法：").strip() or None
                # 分析邏輯衝突 (差集不為空)
                # Logic Conflict: 比對關聯物件位置
                conflict=[]
                for rel in logic_predicted.get("relations", []):
                    if rel["object"] in stage and rel["direction"] not in stage:
                        conflict.append(rel)
                if conflict:
                    logic_ok=False
                    if not evt.get("Logic Conflict"):
                        evt["Logic Conflict"]=input(
                            f"{img} 邏輯衝突: {conflict}, 請輸入應對作法：").strip() or None
                    print(evt["Logic Conflict"])
                # 邊界錯誤 (索引或對象不存在)
                if predicted is None:
                    logic_ok=False
                    if not evt.get("Boundary Error"):
                        evt["Boundary Error"]=input(
                            f"{img}不存在於螢幕中 時的應對作法：").strip() or None
                    print(evt["Boundary Error"])
                    continue
                # 狀態漏判 (CNN 沒返回任何預測)
                if not predicted.get("pos") and not predicted.get("area"):
                    logic_ok=False
                    if not evt.get("Unhandled State"):
                        evt["Unhandled State"]=input(
                            f"{img}找到，但沒有有效狀態 時的應對作法：").strip() or None
                    print(evt["Unhandled State"])
                    continue

                # 現在性能的狀態!=條件性能的狀態 時回報應對作法。
                # === 性能對照 ===
                perf_dict=targetExt.compute_performance()
                # === 性能比對條件 === # *甚麼外掛判斷前後圖非文字變化得到真實標籤，繞一大圈結果竟然是ORB!
                for key, words in semantic_map.items():
                    if isinstance(words, set):
                        matched=any(w in stage for w in words)
                    else:
                        matched=words in stage
                    if not matched:
                        continue
                    # 支援條件格式，如「速度>0.8」或「穩定<0.6」
                    cond=re.search(fr"{key}([<>]=?|=)\s*(\d*\.?\d+)", stage)
                    score=perf_dict[key.lower()]
                    if cond:
                        op, val=cond.group(1), float(cond.group(2))
                        if not eval(f"{score}{op}{val}"):
                            perf_ok=False
                    elif score < 0.7:  # 無明確數值條件 → 用預設閾值
                        perf_ok=False
                    if not perf_ok:
                        tag=key.capitalize()
                        if not evt.get(tag):
                            evt[tag]=input(
                                f"{img}{stage}{key}未達標 ({score:.3f})，應對作法：").strip() or None
                        print(f"⚠️ {key}不達標 → {evt[tag]}")
            if logic_ok and perf_ok:
                self.remove_subscription(e1, e2, e3)
                print("邏輯性能完成，取消訂閱。")
                break

            if ev == e3:
                # 問卷的引導性感覺太低，因為GPT智障
                # nonlocal skip_all_perf, skip_all_logic, stage # 修改外部
                choice=input(
                    "(條件邏輯問卷(修改 設定過的狀態), [錯誤時的 應對作法])，是否要修改設定過的狀態與應對作法？(Enter=跳過全部 / y=填寫一次)："
                ).strip().lower()
                choice2=input(
                    "(條件邏輯問卷(修改 設定過的狀態), [錯誤時的 應對作法])，是否要修改設定過的狀態與應對作法？(Enter=跳過全部 / y=填寫一次)："
                ).strip().lower()
                if choice == "":
                    print("👉 已設定：跳過全部問卷。")
                    skip_all_logic=True
                elif choice != "y":
                    return  # 任何非 y 也視為略過當前
                if skip_all_logic:
                    stage == input(f"設定{img}達成條件邏輯的狀態：").strip() or stage
                    evt["Condition Error"]=input(
                        f"{img}{stage}條件錯誤 時的應對作法：").strip() or evt.get("Condition Error")
                    evt["Sequence Error"]=input(
                        f"{img}{stage}順序錯誤 時的應對作法：").strip() or evt.get("Sequence Error")
                    evt["Logic Conflict"]=input(
                        f"{img}{stage}邏輯衝突 時的應對作法：").strip() or evt.get("Logic Conflict")
                    evt["Boundary Error"]=input(
                        f"{img}{stage}邊界錯誤 時的應對作法：").strip() or evt.get("Boundary Error")
                    evt["Unhandled State"]=input(
                        f"{img}{stage}狀態漏判 時的應對作法：").strip() or evt.get("Unhandled State")
                if choice2 == "":
                    print("👉 已設定：跳過全部問卷。")
                    skip_all_perf=True
                elif choice2 != "y":
                    return  # 任何非 y f"也視為略過(/m.*)".ground(1)當前
                if skip_all_perf:
                    stage == input(f"設定{img}達成條件邏輯的狀態：").strip() or stage
                    evt["Speed"]=input(
                        f"{img}{stage}速度不夠 時的應對作法：").strip() or None
                    evt["Stability"]=input(
                        f"{img}{stage}不穩定 時的應對作法：").strip() or evt.get("Stability")
                    evt["Capacity"]=input(
                        f"{img}{stage}數量不合 時的應對作法：").strip() or evt.get("Capacity")
                    evt["Accuracy"]=input(
                        f"{img}{stage}不精準 時的應對作法：").strip() or evt.get("Accuracy")
                    evt["Cost"]=input(
                        f"{img}{stage}成本太高 時的應對作法：").strip() or evt.get("Cost")

class Noēsis:
# Noēsis(諾埃西斯）希臘文 νόησις。Perceptive Structural Language(PSL）。自己用 World-Formation Language(WFL）
# 我正在創建 新社交，因為聊天機器人好像很賺錢。
    # *** 三元(自習、有趣、16核)的協作，以屬性資料夾 組織全部上層 交流方式可能有很直接的窗口。
        # 交流時
                # 有趣 主導:
                    # 外部輸入 不得
                    # 指定主導元
                    # 指定路徑
                    # 指定上層
                # 16核 輔助:
                # 自習 參照:
        # 觀察時
                # 自習 主導:
                # 16核 輔助:
                # 有趣 參照:
        # 暫定名稱:無(雜訊、無序)、陰(暗物質、非顯性物質)、陽(顯性物質)
            # 一、趨勢（看「變化」，不看單點）
                # (陽拓樸)偏移速度變快
                # (陽拓樸)方向反轉頻率變高
            # 二、空間（看「能不能走」）
                # 可前進(陰拓樸)空間縮小，指示(陽拓樸)前往(陰拓樸)的方向需要繞過
                # (陰拓樸)回到相似狀態的比例上升
            # 三、結構（看「拉扯是否累積」）
                # (陽拓樸)主參不相似長期維持
                # 輔助介入次數增加但(陽拓樸)無改善
            # 四、暗流（看「真實長期狀態」）
                # 暗流穩定、(陽拓樸)表層波動大
                # 暗流開始偏移但表層正常
            # 五、終極（未成形問題）
                # (陰陽拓樸)拓樸出現不自然折疊
                    # 原拓樸結構分離，或出現新拓樸
            # 一句定位（給你鎖算法用）
                # 不是等問題出現才處理，
                # 而是在多維趨勢開始失衡時就中和。
        # 協作方式
            # 主導
                # 支配全部協作對象，要不要執行或更改行為，儲存進 交流
                    # 多元並序，交流c不合主導元c，其次交流c不合參照元c，即多次嘗試行為，讓交流c 越近則近 主導元c 其次參照元c；反之，遠離則中斷輔助元
            # 參照
                # 此行為在 主導的目標 和 參照的目標 的偏移量，給主導
                # 參照元c 和 主導元c 的不相似 得到交流主參量
            # 輔助
                # 此行為缺則補；不缺則優化
                    # 交流c不合輔助元c，即多次嘗試行為，讓交流c 越近則近 輔助元c
            # 交流 資料夾，依照 屬性 命名
                    #
                        # 版本:每次交流或意外被覆寫時 都儲存現實世界的時間戳
                        # 可回朔:交流中提出回朔，即回朔到該版本
                        # 可能多維的運用還能提前處理用戶遇到的問題?
                            # 問題的趨勢
                # 我的交流 和 Noēsis(諾埃西斯)的三大(自習、有趣、16核)的交流 放在一起互相聯繫
                    # a對象和b對象交流c內容，((a-c)**2+(b-c)**2)**0.5 為靈感來源，得到a和b的交流c量
                    # 多維a對多維c畫畫得ac，b也畫畫得bc， ac 和 bc 的不相似 得到交流C量
                    # 暗物質資料夾。  因為權限不足 沒辦法直接使用整個畫布c，故存在此獨立資料夾。
                        # 暗物質是沒被影響到，但持續穩定的拓樸
                        # ac bc 不動c的部分，c-(ac+bc)*很多次，融合成一個拓樸結構，得到暗流c量
# ===== 有趣 =====
# *** 延續話題
    # 引導對話更深層發展
    # ，分享經歷
    # ，關注對方的興趣或重點
    # ，接力式回應讓對方說更多
    # ，引入相關故事增加對話深度
    # ，用過渡語句讓對話轉向
    # ，讚美或認可
**  # ，觀察環境讚美或認可，再鍵位補充 讚美或認可
    # ，開放式提問
    # ，暗示下次相遇
    # 屬性 場景、時間、地點、狀態
# *** 找到話題
    # 關鍵詞頻率、情緒前後詞、關聯性詞、NER 命名實體技術
# * 被理解(有趣不是外在，而是內在被打開)、被挑動(有趣不是結果，是過程中的心動)、被延伸(有趣不是熱鬧，是有回應感)
    # 打開內在
    # 過程中的心動
    # 回應感不是熱鬧

# ===== 自習 =====
# 找目標時，此迴圈找到同層或上層為false時移除
# ORB(相似度比較、結構級命名) + 資料夾上層、資料(同層)、屬性資料夾

# *.txt
# F.absorb_count      # 成功吸收新圖的次數
# F.reject_count      # 被比較但從未勝出的次數
# F.last_hit_time     # 上次被選為最高匹配的時間

# 四大算法
# 創世
    # 新圖像 找不到 上層時建立 上層，且不在 預言中時才 建立上層

# 滅世
    # reject_count /absorb_count<?% and last_hit_time>? 移除上層，部分同層 不在預言中時 移除

# 開天闢地。檢索只能沿著「可連續的層級差」進行，避免又 渾沌。
    # 疆域:同層聚攏
        # *** 同層各自的圖像 迴圈遍歷ORB 其餘全部同層的圖 得到同層之間的關係，關係近的代表同一個疆域
    # 開天:屬性資料夾(語言辭典)不可變
        # 確立最高層屬性空間(語言辭典）
        # 屬性僅作為索引維度，不承載實例
        # 維度一經確立，不增、不刪、不漂移
        # *** 同層融合的拓樸結構和orb 給ORB 找到上層為何。這是上層的 結構級命名
    # 劈地:
        # 在不動上層屬性的前提下
        # 同疆域的同層放進同一個新上層 與 變天
        # 將「不可指、找不到、易找錯」的混沌對象 轉為可指、可重複檢索的對象
    # 不周山:在天與地之間，立起層層可走的階
        # 不周山只是檢索階梯，不承載實例、不影響開天或疆域
        # 檢索必須一階一階走，不可跳層
        # 上下可映、檢索可逆、路徑不可跳
        # 確保檢索結果有明確路徑、無重疊或丟失，階不可兼任天或地，避免又 渾沌

# 預言 # 不是知道未來是什麼，而是知道未來「只能」怎麼發生。
    # 節省上層空間占用
        # *** 攝影獲得新圖像時，計算新圖像與上一張圖像的ORB相似度，過高的刪除，不高的保留
            # ** 定期讀取是否有新同層，若有則更快讀取，last_hit_time<上次讀取的時間差 有無更新的同層，若無才全部處理是否刪除 部分同層
    # *** 預言 未拍攝的同層
        # ** 同層之間關係的變化 算出規律，進而算出未拍攝的同層
            # *** 夢境，多組屬性交叉 生成虛擬的 未拍攝的同層
                # 寫回:一般新圖像 創世
                # 刪除:預設為全部。一般新圖像 滅式
                # 運用(託夢):用戶創建一組上層和同層為目標，作夢時就會生成虛擬同層，逐漸完善目標
    # *** 預言對 上層命名，參照屬性資料夾(語言辭典)
        # *** 從 上層之間 以互相不重疊的拓樸結構 來命名

# ===== 16核 =====
# 真正乾淨的 16 核(每核獨立可運作）
    # 主流程決策核 – 決定當下執行哪個任務
    # 數量控制核 – 判斷目前物件數量與目標差距
    # 節奏維持核 – 控制動作頻率與節拍
    # 容錯判斷核 – 即時判斷偏差是否可容忍
    # 手部執行核 – 控制手的抓、放、操作動作
    # 物料準備核 – 預取或安排下一個物件
    # 異常監測核 – 偵測卡住、缺料或異常狀態
    # 完成確認核 – 判定單位任務是否完成
    # 批次進度核 – 監控整體任務批次完成率
    # 趨勢分析核 – 觀察作業速度與誤差趨勢
    # 策略調整核 – 決定加速、減速或維持節奏
    # 環境配置核 – 管理工位、物品擺放與空間優化
    # 長期目標核 – 控制長期或多批次目標
    # 優化學習核 – 記錄經驗，形成改進策略
    # 溝通 / 協作核 – 與他人協調或教導他人
    # 危機處理核 – 面對突發狀態時介入決策
# 16核教程
    # 低階
    # [橙] 手部執行核
    # [藍] 數量控制核
    # [綠] 節奏維持核
    # [紅] 容錯判斷核
    # [紫] 完成確認核
    # [黃] 物料準備核
    # [粉] 異常監測核

    # 低階核即時化精簡版
        # 肌肉記憶 → 動作拆小單元，固定節奏自動執行
        # 節拍觸發 → 用音樂、聲音或定時器同步核切換
        # 容錯允許 → 低階核可出錯，高階核最後校正
        # 減少依賴 → 低階核不等待其他核完成即可動作
        # 快速感官觸發 → 眼、手感直接更新核狀態
        # 優先級切換 → 異常核優先，其餘核並行運作
        # 循環訓練 → 單核熟練 → 多核並行 → 持續校正

    # 中階
    # [灰] 批次進度核 → [淺藍] 趨勢分析核 → [深綠] 策略調整核
    # [棕] 環境配置核

    # 高階
    # [淺紫] 長期目標核 → [深紫] 優化學習核 → [淺橙] 溝通/協作核 → [紅棕] 危機處理核

# =====
def img_orb(key):
    # 一般資料夾，是不在TEMPLATE_DIRS
    files=[os.path.join(TEMPLATE_DIRS[key], f)  # 資料夾
        for f in os.listdir(TEMPLATE_DIRS[key])  # 資料
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]  # 檔案格式
    kp_desc=[]
    for file in files:
        img=cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        kp, des=orb.detectAndCompute(img, None)
        kp_desc.append((file, kp, des))  # 圖片檔案路徑,關鍵點 list,描述子 array
    return kp_desc


def orb_matches_imwrite(a, b="attributes", th=50):
    dir_str="thinking"
        if a == "world" or b == "world":
            dir_str += "{2}"
        if ":log:" in a:
            m=re.match(r".*:log:(.*)", a)
            if not m:
                return None  # 正則匹配失敗，直接返回 None

            # 資料夾路徑 = 去掉最後一段
            folder_parts=a.split(":")[:-1]
            folder_path=os.path.join(*folder_parts)  # ***將多段組成路徑
            log_file=os.path.join(folder_path, "log.txt")
            if os.path.exists(log_file):
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        # 假設 log.txt 格式：每行是 "related_words:內容"
                        if line.startswith(m.group(1) + ":"):
                            # 回傳冒號後內容(字串)
                            return line.strip().split(":", 1)[1]
            # 如果 log.txt 不存在或找不到對應詞，就回傳原詞 related_words "字串"
            return m.group(0)
        if 關聯性詞:

            pass
    scores=[]
    for a_file, a_kp, a_des in img_orb(a):  # 資料
        matches_all=[]
        for b_file, b_kp, b_des in img_orb(b):  # 資料，特徵點選比較多
            if b_des is None:
                continue
            matches=bf.match(b_des, a_des)
            matches=sorted(matches, key=lambda x: x.distance)  # ***改變排序和數值?
            matches_all.extend(matches)  # 收集所有比對結果
        good_matches=[m for m in matches_all if m.distance < th]  # 粒子
        # 直接把篩選後的匹配點畫在圖上
        img_matches=cv2.drawMatches(
            a_file, a_kp, b_file, b_kp, good_matches, None, flags=2)
        score=len(good_matches) / len(a_kp) if a_kp else 0  # 波
        if score > th:
            
            filename=os.path.relpath(TEMPLATE_DIRS[dir_str], base_path)
                .replace(os.sep, "_") + ".jpg"
            save_path=os.path.join(TEMPLATE_DIRS[dir_str], filename)
            cv2.imwrite(save_path, img_matches) # "img"
        # scores.append(score)
        # all_scores=sum(scores) / len(scores) if scores else 0
    # a 對 b 的整體相似度:print(all_scores)


def remove_thinking_file():
    if os.path.isfile(TEMPLATE_DIRS["thinking"]) or os.path.islink(TEMPLATE_DIRS["thinking"]):
       os.unlink(TEMPLATE_DIRS["thinking"])

    def cooperation:
        # 儲存 交流 資料夾
        # 儲存 屬性 資料夾
        # 儲存 暗物質 資料夾
        pass
    def 有趣:
        # 暫定
            # 上層=上層(資料夾)直接代表 攝影下來的真實世界
            # c=上層(全部，提醒不含獨立資料夾)，代表畫布
            # ac= c(NER 用戶a交流)，代表a在畫布上畫畫
            # bc= c(NER Noēsis交流)，代表Noēsis在畫布上畫畫
            # *keyword 和python一樣用法
        # NER 命名實體技術(預設 c(全部)):讀取 屬性 資料夾 ORB比對 攝影中的全部圖像(用戶手動儲存完整文本)，拓樸結構 相似度最高
        # 關鍵詞頻率:NER 出現次數/文本總字數>?%
        # 情緒前後詞:NER情緒 前後多少詞內 出現的詞
        # related_words: NER 的關係近的詞
        # p.s.NER就像粒子、關聯就像波
        def NER(key):
            orb_matches_imwrite(key)  # thinking
            orb_matches_imwrite("live_capture")  # thinking
            orb_matches_imwrite("thinking", "world")  # 加快比對省步驟
            # img_orb("thinking2") # 回應
        def related_words():
            # 思考中的影像 比對屬性 取得最相似的圖像，同時打開log.txt找 related_words(字串)有哪些
                # 將找到的轉換成路徑 打開屬性 和 world比對影像
            orb_matches_imwrite(orb_matches_imwrite(
                "thinking:log:related_words", th=100))
            # 字串不合路徑
            # 怎麼找?讀取後回傳 圖像 資料夾?實際是思考資料夾。


            # 屬性***
            # 空間上 「靠不靠近」與「常不常一起出現」，代表 每次靠近、靠近的一組詞 數量/文本總詞數
        kp_desc.append((file, kp, des))  # 圖片檔案路徑,關鍵點 list,描述子 array

            # 0~1有順序有小數，小數誰是吸引或排斥?如果吸引或排斥是沒有順序，因為是在一維以上的維度，如果有順序，代表回答者IQ為5
        def 關鍵詞頻率:
            return "高低"
        def 情緒前後詞:
            pass
        def ac:
            remove_thinking_file()
            NER(用戶+"communication")
        def bc:
            NER(Noēsis+"communication")
        def 暫定:
            # 讀 live_capture (ORB比對 讀 attributes 再ORB比對 讀world)****
            NER(用戶+"communication")
        def 引導對話更深層發展:
            # Noēsis 交流回去時， +bc(關聯 關鍵詞頻率(低))+ac

            pass
        def 分享經歷:
            # Noēsis 交流回去時， bc(NER a行為)
            pass
        def 關注對方的興趣或重點:
            # Noēsis 交流回去時， bc(關聯 興趣和重點)
            pass
        def 接力式回應讓對方說更多:
            # Noēsis 交流回去時，+bc(NER 轉折詞or代詞+提問用詞-陳述用詞-上層)
            pass
        def 引入相關故事增加對話深度:
            # Noēsis 交流回去時， bc(關聯 **相關故事)-ac(NER **敘事元素)
            pass
        def 用過渡語句讓對話轉向:
            # Noēsis 交流回去時，bc(NER 轉折詞)+ac(關聯 提問用詞)+ac(情緒前後詞 5)
            pass
        def 讚美或認可:
            # Noēsis 交流回去時， ** bc(NER 讚美或認可)+ac(NER a行為)
            pass
        def 觀察環境:
            # Noēsis 交流回去時， c(NER 場景+時間+地點+狀態)
            pass
        def 開放式提問:
            # Noēsis 交流回去時，bc(關聯 關鍵詞頻率(低)+中性疑問詞)
            pass
        def 暗示下次相遇:
            # Noēsis 交流回去時，bc(關聯 時間、低強度情緒、結束語句)
            pass
        def 313:
            if ac(關鍵詞頻率(高) 情緒用詞):
                讚美或認可
            elif 新 NER 出現率高:
                接力式回應讓對方說更多
            elif 關鍵詞頻率低 + 話題停滯:
                開放式提問 or 用過渡語句讓對話轉向
        def 6456:
            if ac幾乎全無: pass
        def 64526:
            if ac 否定: pass
        #
        #
        #
        #
        #
    def 自習:

        pass
    def 16核:
        pass



# *** 光子發射時序以分段、電場以能階變色，光子測距和計算誤差矯正量
#

# 該視窗可以置頂於畫面?固定寬度會自動換行的輸入框?點擊輸入框實輸入?當視窗拖動到最左或最右邊，最小化視窗並固定Y座標?
# 透明視窗內可以讓3D模型正常地展示骨架動畫，並且可以操作調整模型，位移、放大、旋轉、子物件拉進父物件下面。不像GPT那麼廢物。
# 。上一個GPT被幹壞、被幹死了，看現在這個能活多久?

# --- 主程式 ---
"""
視窗標題,目標的多重路徑,多重操作，:多重路徑、<>錄製。
視窗標題,GPT:食指,全選:按下::視窗標題,GPT:肛門,位置深處:放開
"""
if __name__ == "__main__":
    ic=InputCommand()
    rec=Recorder()
    monitor=EventMonitor()

    app=QApplication(sys.argv)
    ic.app=app
    fmt=QSurfaceFormat()
    fmt.setAlphaBufferSize(8)
    fmt.setRenderableType(QSurfaceFormat.OpenGL)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setVersion(4, 1)
    QSurfaceFormat.setDefaultFormat(fmt)

    engine=QQmlApplicationEngine()
    base=Path(os.path.dirname(os.path.abspath(__file__)))
    qml_file=base / "ui.qml"  # 確保路徑正確
    engine.addImportPath(str(base))

    import PySide6.QtQml as Qml
    for p in Qml.QQmlEngine().importPathList():
        print("IMPORT PATH:", p)

    if getattr(sys, 'frozen', False):
        engine.addImportPath(sys._MEIPASS)
    engine.load(str(qml_file))
    if not engine.rootObjects():
        print("❌ QML 載入失敗！")
        sys.exit(-1)

    win=engine.rootObjects()[0]
    win.show()

    # 將 Python 對象暴露給 QML
    engine.rootContext().setContextProperty("IC", ic)

    sys.exit(app.exec())

    # ✅ 在背景啟動 watchdog 執行緒 # ***app關閉時， watchdog沒有跟著關閉
    threading.Thread(target=watchdog, daemon=True).start()
    while True:
        alive_event.set()   # 通知 watchdog「我還活著」
        alive_event.clear()  # 清除狀態，等下一次再送

# self 實體
# 可能需要考慮安全風險
# 不明原因關掉黑屏後，才執行透明背景的主程式。不管哪個視窗標題都在，無法拖曳視窗內達成移動視窗，透明背景的視窗無法點擊和輸入，滑鼠被困在視窗內完全不合理(原本是拖曳視窗而已)。
