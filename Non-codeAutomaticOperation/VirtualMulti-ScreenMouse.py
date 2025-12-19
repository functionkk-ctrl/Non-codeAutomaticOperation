import numpy as np
import time

# ===== 虛擬螢幕與虛擬滑鼠 =====
class VirtualMouse:
    def __init__(self, screen_width, screen_height, name="VM"):
        self.name = name
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.x = screen_width // 2
        self.y = screen_height // 2
        self.events = []  # 儲存操作記錄（可選）
    
    def move_to(self, x, y):
        self.x = max(0, min(x, self.screen_width - 1))
        self.y = max(0, min(y, self.screen_height - 1))
        self.events.append(("move", self.x, self.y))
        print(f"[{self.name}] Move to ({self.x}, {self.y})")
    
    def click(self, button="left"):
        self.events.append(("click", self.x, self.y, button))
        print(f"[{self.name}] Click {button} at ({self.x}, {self.y})")
    
    def drag_to(self, x, y, duration=0.5):
        start_x, start_y = self.x, self.y
        steps = int(duration / 0.05)
        for i in range(1, steps + 1):
            nx = start_x + (x - start_x) * i / steps
            ny = start_y + (y - start_y) * i / steps
            self.move_to(int(nx), int(ny))
            time.sleep(0.05)
        self.events.append(("drag", start_x, start_y, x, y))
        print(f"[{self.name}] Dragged from ({start_x},{start_y}) to ({x},{y})")
    
    def type_text(self, text):
        self.events.append(("type", self.x, self.y, text))
        print(f"[{self.name}] Type '{text}' at ({self.x}, {self.y})")

# ===== 管理多虛擬螢幕 =====
class VirtualScreenManager:
    def __init__(self):
        self.screens = {}
    
    def add_screen(self, name, width=800, height=600):
        vm = VirtualMouse(width, height, name)
        self.screens[name] = vm
        return vm
    
    def get_mouse(self, name):
        return self.screens.get(name)

# ===== 使用範例 =====
if __name__ == "__main__":
    manager = VirtualScreenManager()
    
    # 建立兩個虛擬螢幕
    mouseA = manager.add_screen("ScreenA", 800, 600)
    mouseB = manager.add_screen("ScreenB", 640, 480)
    
    # 模擬操作 ScreenA
    mouseA.move_to(100, 100)
    mouseA.click()
    mouseA.drag_to(200, 200)
    mouseA.type_text("Hello A")
    
    # 模擬操作 ScreenB
    mouseB.move_to(50, 50)
    mouseB.click()
    mouseB.drag_to(150, 150)
    mouseB.type_text("Hello B")
    
    # 查看事件記錄
    print("\nScreenA events:", mouseA.events)
    print("ScreenB events:", mouseB.events)
