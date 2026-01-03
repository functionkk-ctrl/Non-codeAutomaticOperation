from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.camera import Camera
from kivy.clock import Clock
from plyer import gps

import numpy as np
import cv2
import math
# 使用手則:開啟攝影，GPS得高度尺可以和地面參照不需要移動，GPS平移得橫向尺在空中至少要移動20m，才可以正確參照。
# 讀取設備資訊有圖像、GPS， *** 分析出圖像中的大占比物體， *** 判斷使用者是否在空中， *** 這些物體計算出單一數量的長寬高， 等待使用者輸入名稱 儲存過的的會顯示
        
class Root(BoxLayout):
    pass

class AppMain(App):
    def build(self):
        self.alt = None
        self.speed = None

        self.cam = Camera(play=True, resolution=(640, 480))
        self.label = Label(text="waiting GPS")

        root = BoxLayout(orientation="vertical")
        root.add_widget(self.cam)
        root.add_widget(self.label)

        gps.configure(on_location=self.on_gps)
        gps.start(minTime=1000, minDistance=1)

        Clock.schedule_interval(self.process, 1/5)
        return root

    def on_gps(self, **kw):
        # *** altitude 不是離地面高度
        self.alt = kw.get("altitude")
        # *** speed*時間 在空中時要移動至少20m
        self.speed = kw.get("speed")

    # 沒有 gps.stop()


    def process(self, dt):
        if self.alt is None or self.speed is None:
            return
        # *** 並非離地面高度 
        if not (self.alt > 20 and self.speed > 5):
            self.label.text = "not airborne"
            return

        tex = self.cam.texture
        if tex is None:
            return

        w, h = tex.size
        frame = np.frombuffer(tex.pixels, np.uint8)
        frame = frame.reshape(h, w, 4)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        # *** 缺少對圖像的精準分析，光線
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, bin_img = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        # *** 相機的形態學，整個畫面是否轉動
        cnts, _ = cv2.findContours(
            bin_img,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        if not cnts:
            return
        # *** boundingRect???改用minAreaRect
        c = max(cnts, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(c)
        # *** FOV???
        Z = self.alt
        hfov = math.radians(60)
        vfov = math.radians(45)

        W = 2 * Z * math.tan(hfov/2) * (bw / w)
        H = 2 * Z * math.tan(vfov/2) * (bh / h)
        L = max(W, H)

        self.label.text = f"L={L:.2f} W={W:.2f} H={H:.2f}"


if __name__ == "__main__":
    AppMain().run()
# ** V2。 分析多張圖像中的目標圖像，分析多張圖像中的結構光並移除，獲得多張目標圖像的拓樸結構變化
import cv2
import numpy as np
import glob
import networkx as nx
from skimage.morphology import skeletonize

# -----------------------------
# Skeleton -> Graph
# -----------------------------
def skeleton_to_graph(skel):
    G = nx.Graph()
    h, w = skel.shape

    for y, x in np.argwhere(skel):
        G.add_node((x, y))
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx_ < w and skel[ny, nx_]:
                    G.add_edge((x, y), (nx_, ny))
    return G


# -----------------------------
# Integrated Operator
# T_M = T ∘ R ∘ M
# -----------------------------
def topology_from_structured_images(
    image_folder: str,
    pure_target_path: str,
    threshold: int = 127
):
    # --- M : target domain ---
    gt = cv2.imread(pure_target_path, cv2.IMREAD_GRAYSCALE)
    _, M = cv2.threshold(gt, threshold, 255, cv2.THRESH_BINARY)
    M = M > 0

    # --- R : multi-image reconstruction ---
    imgs = []
    for f in sorted(glob.glob(f"{image_folder}/*")):
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        imgs.append(img * M)

    stack = np.stack(imgs, axis=-1)
    clean = np.median(stack, axis=-1)

    # --- T : topology / morphology ---
    _, binary = cv2.threshold(clean.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY)
    skeleton = skeletonize(binary > 0)
    G = skeleton_to_graph(skeleton)

    return G, skeleton, clean


G, skeleton, clean_image = topology_from_structured_images(
    image_folder="structured_images",
    pure_target_path="pure_target.png"
)
