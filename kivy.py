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


class darar:
    import cv2
    import numpy as np
    from skimage.morphology import skeletonize
    from scipy.ndimage import convolve
    import networkx as nx
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np
    import glob
    import matplotlib.pyplot as plt

    # 讀入所有圖像
    file_list = glob.glob("images/*.png")
    images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in file_list]

    # 將所有圖像堆疊成 3D array (H x W x N)
    stack = np.stack(images, axis=-1)

    # 方法 1：中位數去除結構光
    reconstructed = np.median(stack, axis=-1).astype(np.uint8)

    # 方法 2：最小值去除亮條紋
    # reconstructed = np.min(stack, axis=-1).astype(np.uint8)

    # 顯示結果
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.imshow(images[0], cmap='gray')
    plt.title("原圖示例")
    plt.subplot(1,2,2)
    plt.imshow(reconstructed, cmap='gray')
    plt.title("去除結構光後")
    plt.show()

    # ------------------------
    # 1. 讀取圖像並二值化
    # ------------------------
    img = cv2.imread("target_image.png", cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    binary_bool = binary > 0

    # ------------------------
    # 2. 骨架化（細化線條）
    # ------------------------
    skeleton = skeletonize(binary_bool)

    # ------------------------
    # 3. 找端點和交叉點
    # ------------------------
    kernel = np.array([[1,1,1],
                    [1,10,1],
                    [1,1,1]])
    conv_result = convolve(skeleton.astype(int), kernel, mode='constant')

    # 端點：中心像素 + 1個鄰居 = 11
    endpoints = np.argwhere(conv_result == 11)

    # 交叉點：中心像素 + >=3個鄰居 >=13
    junctions = np.argwhere(conv_result >= 13)

    # ------------------------
    # 4. 將骨架轉成圖
    # ------------------------
    G = nx.Graph()

    # 將骨架像素加入節點
    for y, x in np.argwhere(skeleton):
        G.add_node((x, y))

    # 連接鄰近像素
    for y, x in np.argwhere(skeleton):
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < skeleton.shape[0] and 0 <= nx_ < skeleton.shape[1]:
                    if skeleton[ny, nx_]:
                        G.add_edge((x, y), (nx_, ny))

    # ------------------------
    # 5. 畫出拓樸圖
    # ------------------------
    plt.figure(figsize=(8, 8))
    pos = {node: (node[0], -node[1]) for node in G.nodes()}  # y軸翻轉方便顯示
    nx.draw(G, pos=pos, node_size=10, node_color='blue', edge_color='gray')

    # 標出端點與交叉點
    plt.scatter(endpoints[:,1], -endpoints[:,0], color='red', s=30, label='Endpoints')
    plt.scatter(junctions[:,1], -junctions[:,0], color='green', s=30, label='Junctions')
    plt.legend()
    plt.title("Graph Topology from Image")
    plt.show()

    # ------------------------
    # 6. 拓樸分析示例
    # ------------------------
    print("節點總數:", G.number_of_nodes())
    print("邊總數:", G.number_of_edges())
    print("端點數:", len(endpoints))
    print("交叉點數:", len(junctions))
    print("連通分量數:", nx.number_connected_components(G))



if __name__ == "__main__":
    AppMain().run()
# ** V2
import cv2
import numpy as np
from skimage.morphology import skeletonize
import networkx as nx
import glob

# 讀入所有圖
file_list = glob.glob("images/*.png")
images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in file_list]

# ==========================
# 1. 分析結構光拓樸
# ==========================
structure_topology = []

for idx, img in enumerate(images):
    # 二值化
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    binary_bool = binary > 0
    # 骨架化
    skeleton = skeletonize(binary_bool)
    # 建立圖
    G = nx.Graph()
    for y,x in np.argwhere(skeleton):
        G.add_node((x,y))
    for y,x in np.argwhere(skeleton):
        for dy in [-1,0,1]:
            for dx in [-1,0,1]:
                if dy==0 and dx==0:
                    continue
                ny,nx_ = y+dy, x+dx
                if 0 <= ny < skeleton.shape[0] and 0 <= nx_ < skeleton.shape[1]:
                    if skeleton[ny,nx_]:
                        G.add_edge((x,y),(nx_,ny))
    # 紀錄拓樸特徵
    structure_topology.append({
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "connected_components": nx.number_connected_components(G)
    })

# ==========================
# 2. 去除結構光 (多圖重建)
# ==========================
stack = np.stack(images, axis=-1)
reconstructed = np.median(stack, axis=-1).astype(np.uint8)

# ==========================
# 3. 去光後拓樸
# ==========================
_, binary_clean = cv2.threshold(reconstructed, 127, 255, cv2.THRESH_BINARY)
skeleton_clean = skeletonize(binary_clean > 0)
G_clean = nx.Graph()
for y,x in np.argwhere(skeleton_clean):
    G_clean.add_node((x,y))
for y,x in np.argwhere(skeleton_clean):
    for dy in [-1,0,1]:
        for dx in [-1,0,1]:
            if dy==0 and dx==0:
                continue
            ny,nx_ = y+dy, x+dx
            if 0 <= ny < skeleton_clean.shape[0] and 0 <= nx_ < skeleton_clean.shape[1]:
                if skeleton_clean[ny,nx_]:
                    G_clean.add_edge((x,y),(nx_,ny))

clean_topology = {
    "nodes": G_clean.number_of_nodes(),
    "edges": G_clean.number_of_edges(),
    "connected_components": nx.number_connected_components(G_clean)
}

# ==========================
# 4. 比較拓樸變化
# ==========================
print("各結構光圖拓樸:", structure_topology)
print("去結構光後拓樸:", clean_topology)
