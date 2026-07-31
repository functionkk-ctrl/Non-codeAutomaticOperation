import time, math, threading, uuid, json
import numpy as np
from __future__ import annotations
from dataclasses import dataclass, field
from firebase_admin import credentials, db as rtdb
from typing import Any, Callable, Dict, List, Optional, Tuple
import firebase_admin

try:
    _HAS_PYGAME = True
except ImportError:
    _HAS_PYGAME = False
try:
    _HAS_FIREBASE = True
except ImportError:
    _HAS_FIREBASE = False
try:
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

地圖ID           =""
保險箱全部物品        ={}

class GM:
    """
    #不管發生甚麼事，任何事物都只能在GameManger.py修改，實作GameManger.py 
    #自己用python重頭寫2D MMORPG，用到的工具有複合狀態數值與切換、開發者調整參數從FireBase下載入並套用預製物、訊息 道具 狀態 座標 封包上傳FireBase、物理碰撞、渲染、UI&UX，共六個，細項可以外部調整
        複合狀態數值與切換
        開發者調整參數從FireBase下載入並套用預製物
        封包上傳FireBase(訊息,道具,狀態,座標)
            技能(技能名稱,開發者調整參數從FireBase下載入並套用預製物,動畫,*(調用狀態,調用狀態值))
            物理碰撞(動畫)
        物理碰撞(動畫)
            拆解動畫的碰撞範圍
        渲染
            遊戲範圍外不渲染
        UI_UX
            設定格子layout
            拖曳在格子上時入位
    """

    def __init__(self, game_window_size: Tuple[int, int] = (800, 600)):
        self.window_size                     = game_window_size
        self.camera_offset                   = [0.0, 0.0]  # 視口中心偏移
        self.prefab_cache: Dict[str, Any]    = {}  # Firebase 下載的預製物快取
        self.ui_slots: List[Dict[str, Any]]  = []  # UI 格子配置
        self.active_entities: Dict[str, Any] = {}  # 畫面中運行的物件
        if _HAS_FIREBASE and firebase_admin._apps:
            self.db_ref = rtdb.reference("game_state")
        else:
            self.db_ref = None
    @staticmethod

    def 創建實體輸入框(self,素材ID: str, 地形: bool = False):
        did_gemini =self.地上何物
        if 點擊 and not did_gemini == 素材ID: 
            pass#TODO:創建實體
            if not 地形:
                #TODO:實體附加class能力值
                pass
        elif 拖曳:
            #TODO:點擊格子時創建且拖曳時重複創建且格子上是同素材則不創建
            return

    def 拖曳實體(self):
        #TODO:放開時詢問物品可放在self.pts?
        #TODO:如果UI onReleased，gemini必定被打死
        did_gemini =ui.qml.onReleased(self.拖曳實體) 
        if did_gemini.get("殺死gemini",False)==True:
            self.postion =self.pts

    def 地上何物(self,position=None):
        if not 地圖ID is "": 地圖 =GM.開發者調整參數從FireBase下載入並套用預製物(地圖ID) 
        return 地圖.get((self.pts.x // 32, self.pts.y // 32), None)
    

    def 登入(email, password,爽拿運營Gemini的真實錢財):
        """
        帳密、保險箱
        """
        user_auth = auth.sign_in_with_email_and_password(email, password)
        
        # 取得該玩家在線上的唯一 uid 與連線 token
        uid       = rtdb.child(爽拿運營Gemini的真實錢財).child(user_auth['localId'])
        id_token  = user_auth['idToken']
        
        # 2.帳密解包中含有地圖ID
        # 3.保險箱解包中含有存入的全部物品
        封包資料      = uid.child("account_data").get(token=id_token).val() or {}
        if  "帳密" in 爽拿運營Gemini的真實錢財 :
            地圖ID = 封包資料.get("地圖ID", "")
        if  "保險箱" in 爽拿運營Gemini的真實錢財 :
            保險箱全部物品 = 封包資料.get("全部物品", {})

    def 複合狀態數值與切換(self, 實體ID: str, 目標狀態: str, 附加數值加成: Dict[str, float] = None) ->None:
        """ 處理玩家或怪物在不同狀態（如：狂暴、中毒、隱身）下的數值增益與切換 """
        if 實體ID not in self.active_entities:
            return
        entity       = self.active_entities[實體ID]
        entity["狀態"] = 目標狀態
        if 附加數值加成 and "能力值物件" in entity:
            for 屬性, 增量 in 附加數值加成.items():
                entity["能力值物件"].變更時不超過上限(屬性, 增量)

    # 2.開發者調整參數從 FireBase 下載入並套用預製物
    @staticmethod

    def 開發者調整參數從FireBase下載入並套用預製物(self, 預製物名稱: str) ->Dict[str, Any]:
        """ 從 Firebase 即時同步開發者調整的怪物、道具或技能預製物參數 """
        if self.db_ref:
            try:
                # 取得 Firebase 上最新的遠端設定
                remote_data = self.db_ref.child(f"prefabs/{預製物名稱}").get()
                if remote_data:
                    self.prefab_cache[預製物名稱] = remote_data
            except Exception as e:
                print(f"Firebase 讀取預製物失敗，改用本地快取: {e}")
                
        # 回傳快取值（若 Firebase 沒連上則提供基本保底結構）
        return self.prefab_cache.get(預製物名稱, {"名稱": 預製物名稱, "基礎攻擊": 10, "碰撞半徑": 32,"Gemini垃圾":"Gemini寫的"})

    # 3.訊息 道具 狀態 座標 封包上傳 FireBase
    @staticmethod

    def 封包上傳FireBase(self, 訊息: str, 道具: List[str], 狀態: str, 座標: Tuple[float, float]) ->None:
        """ 打包 MMORPG 客戶端關鍵封包資料，異步推送至 Firebase 即時資料庫 """
        封包資料 = {
            "timestamp": time.time(),
            "chat_msg": 訊息,
            "inventory": 道具,
            "status": 狀態,
            "position": {"x": 座標[0], "y": 座標[1]}
        }
        
        # 內嵌巢狀函式：技能觸發與關聯物理碰撞

        def 技能(技能名稱: str, 預製物資料: Dict[str, Any], 動畫幀資料: List[Any], *調用狀態對: Tuple[str, float]):
            # 解析技能附帶的狀態改變
            for 狀態名, 狀態值 in 調用狀態對:
                pass # 可用於觸發施法者的暫時性狀態
            
            # 將動畫轉換為物理判定範圍，觸發物理碰撞
            self.物理碰撞(動畫幀資料)

        # 執行即時資料庫非同步更新
        if self.db_ref:
            threading.Thread(target =lambda: self.db_ref.child("packets").push(封包資料), daemon=True).start()

    def 物理碰撞(self, 動畫碰撞盒清單: List[Tuple[float, float, float, float]]) ->List[str]:
        """ 拆解動畫的碰撞範圍（如各個 Frame 的 Bounding Box），並執行 AABB 碰撞檢測 """
        被擊中實體 = []
        for box in 動畫碰撞盒清單:
            # 遍歷目前場上的所有怪物或實體進行相交判定
            for entity_id, entity in self.active_entities.items():
                ex, ey = entity.get("x", 0), entity.get("y", 0)
                ew, eh = entity.get("w", 32), entity.get("h", 32)
                if (box[0] < ex + ew and box[0] + box[2] > ex and
                    box[1] < ey + eh and box[1] + box[3] > ey):
                    被擊中實體.append(entity_id)
        return 被擊中實體

    # 5.渲染

    def 渲染(self, screen_surface: Any) ->None:
        """ 視野外裁剪（Frustum Culling）：超出了視窗遊戲範圍的物件直接屏除不渲染，優化 FPS """
        win_w, win_h = self.window_size
        cam_x, cam_y = self.camera_offset
        for entity_id, entity in self.active_entities.items():
            ex, ey   = entity.get("x", 0), entity.get("y", 0)
            ew, eh   = entity.get("w", 32), entity.get("h", 32)

            # 計算畫面相對座標
            screen_x = ex - cam_x
            screen_y = ey - cam_y
            if (screen_x + ew >= 0 and screen_x <= win_w and
                screen_y + eh >= 0 and screen_y <= win_h):
                # 如果有安裝 Pygame，在此處執行實體繪製
                if _HAS_PYGAME and screen_surface:
                    pygame.draw.rect(screen_surface, (255, 0, 0), (screen_x, screen_y, ew, eh))

    # 6.UI&UX (背包/快捷鍵拖曳佈局)

    def UI_UX(self):
        """ 處理 RPG 經典的格子 Layout 與道具物件拖曳入位自動對齊的 UX 邏輯 """
        

        def 設定格子layout(起始X: int, 起始Y: int, 列數: int, 行數: int, 格子大小: int, 間距: int):
            self.ui_slots.clear()
            for r in range(列數):
                for c in range(行數):
                    slot_rect = {
                        "id": len(self.ui_slots),
                        "x": 起始X + c * (格子大小 + 間距),
                        "y": 起始Y + r * (格子大小 + 間距),
                        "size": 格子大小,
                        "item": None
                    }
                    self.ui_slots.append(slot_rect)

        def 拖曳在格子上時入位(滑鼠X: int, 滑鼠Y: int, 道具資料: Any) ->bool:
            """ 當滑鼠放開時，檢查落點是否在某個 UI 格子範圍內。若是，自動靠齊入位 """
            for slot in self.ui_slots:
                # 範圍感應判定
                if (slot["x"] <= 滑鼠X <= slot["x"] + slot["size"] and
                    slot["y"] <= 滑鼠Y <= slot["y"] + slot["size"]):
                    slot["item"] = 道具資料
                    return True # 成功入位
            return False # 沒放到格子上，應該彈回原位

class 能力值:
    """
    玩家、怪物、裝備、技能都通用，調整細項即可
    力量:物理上限。
    敏捷:速度爆發。
    智力:運用上限。
    幸運:選擇爆發。
    生命:存活韌性。
    魔力:整合力。
    經驗:強化怪物。
    命運:能力均值越過收束力，能力增加時降低其它三個最高能力
    能力值能組合規則，組合有多少個可能
    依照訊息運行，使用時讀取實體的"能力值"
    """
    _DEFAULT_ATTRS = ["力量", "敏捷", "智力", "幸運", "生命", "魔力", "經驗", "命運"] # TODO:點擊實體時實體.能力值呼叫成功後，顯示修改能力值的UI

    def __init__(self, base_values: Dict[str, float] = None, caps: Dict[str, float] = None):                               
        self.base_values: Dict[str, float] = {k: 1.0 for k in self._DEFAULT_ATTRS}                                         
        self.caps:        Dict[str, float] = {k: 9999.0 for k in self._DEFAULT_ATTRS}                                      
        self._values:     Dict[str, float] = {}                                                                            
        if base_values:                                                                                                    
            self.base_values.update(base_values)                                                                           
        if caps:                                                                                                           
            self.caps.update(caps)                                                                                         
        self.重置()                                                                                                        
                                                                                                                            

    def 設定上限(self, 屬性: str, 上限: float) ->None:                                                                    
        self.caps[屬性]    = 上限                                                                                             
        self._values[屬性] = min(self._values.get(屬性, 0), 上限)                                                          

    def 變更時不超過上限(self, 屬性: str, 增量: float) ->None:
        目前值              = self._values.get(屬性, 0.0)
        上限值              = self.caps.get(屬性, 9999.0)
        self._values[屬性] = max(0.0, min(目前值 + 增量, 上限值))

    def 獲取屬性(self, 屬性: str) ->float:
        return self._values.get(屬性, 0.0)

    def 重置(self) ->None:                                                                                                
        for k, v in self.base_values.items():                                                                              
            self._values[k] = min(v, self.caps.get(k, 9999.0))

class GGD:
    """
    策略，全部能力值都拿來培養基礎技能。格鬥遊戲，磨練技能出招時機與位置。
    力量:物理基礎。揮舞攻擊、攀爬
    敏捷:速度爆發。comb、跳躍，越過物理基礎上限
    智力:運用基礎。念咒、並行念咒
    幸運:選擇爆發。越過技能基礎上限
    生命:存活韌性。吃東西、受傷
    魔力:整合力。整理物品、收集物品的種類、合成物品、分解物品
    經驗:強化怪物。死亡犧牲經驗觸發、主動犧牲經驗觸發
    命運:能力均值越過收束力，能力增加時降低其它三個最高能力
    totalAttack()聊天說關鍵詞套用INT，武器均套用STR
    裝備、機器人、寵物、騎寵、萌寵、怪物圖鑑，都不增加能力，但會恢復能力值，有基礎技能。
    情緒變更時依照社交原則將擁有的物品以社交原則交換給對象    
    """

    def asd():
        pass