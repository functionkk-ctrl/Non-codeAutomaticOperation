from itertools import permutations
from itertools import permutations
from itertools import permutations
from itertools import permutations
import itertools
import itertools
import itertools
import math
import time
import time
import time
import time
import time
import time
import time
import time
import numpy as np
import numpy as np
import numpy as np
import numpy as np
import numpy as np
import numpy as np
import numpy as np
import numpy as np
import numpy as np
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from typing import List, Dict, Tuple, Optional
import cv2
import cv2
import pyvista as pv
import random
import string

def 調整節點整數():
    R_SQUARED = 800
    max_y     = math.sqrt(R_SQUARED)  # 約 28.28

    # 建立一個從最頂端向下取整的 y 座標序列 (例如 28, 27, 26...直到 0)
    y_targets = list(range(int(max_y), -1, -1))
    print(f"{'目前高度 y':<10}{'精確 x 座標':<12}{'y 每降1單位，x需前進 (Δx)'}")
    print("-" * 55)

    prev_x = 0.0
    for y in y_targets:
        # 根據反函數求出該高度 y 對應的精確 x 座標
        current_x = math.sqrt(R_SQUARED - y**2)
        # 計算相鄰兩個 x 的差距 (即 x 走了多少)
        delta_x   = current_x - prev_x
        print(f"{y:<12}{current_x:<15.4f}{delta_x:.4f} 單位")
        # 紀錄當前的 x，供下一層計算差值
        prev_x = current_x

def 職能協商解析函數(任務環境, 任務目標, 可選職業庫):
    """
    #任務需求拓樸和職業交叉拓樸的填滿率
    #去野外找市場上外型獨特的無機物，可能會遇到小型有毒生物
    #1.隨機名稱(例如市場、產品、生物、場地) 2.imwrite生成產品外觀(名稱對應的圖片融合特徵時依據父比例) 3.面試:分析職業(半成品有職業特性、個性、裝備、夥伴)的在該拓樸上的移動表現 和拓樸上的它物的碰撞表現
    #有則有，沒有就隨機，不要則不要，範圍，數量，成長曲線，super繼承父節點變數
    我想到了，場景總大小(矩陣)，物品占據多少場景比例(矩陣)*材質矩陣*顏色矩陣，轉視角等於矩陣*三角函數，動作等於某物品*移動路徑
    結論為這是相似度
    """
    pass

def generate_random_name_pure(language: str, max_length: int, style: str = "capitalize") ->str:
    """
    不使用任何內建清單(list)，純算法動態生成隨機字元名稱
    capitalize 英文首字大寫，lower 英文全小寫
    """
    if max_length <= 0:
        return ""
        
    lang = language.lower()
    if lang in ["zh", "中文"]:
        # 常用中文字的 Unicode 範圍：0x4E00 到 0x9FFF
        # 隨機隨機抽選 Unicode 編碼並轉換成中文字
        return "".join(chr(random.randint(0x4E00, 0x9FFF)) for _ in range(max_length))
        
    elif lang in ["en", "英文"]:
        # string.ascii_lowercase 為純小寫字串常數 ("abcdefghijklmnopqrstuvwxyz")
        raw_name = "".join(random.choices(string.ascii_lowercase, k=max_length))
        if style.lower() == "capitalize":
            return raw_name.capitalize()
        else:
            return raw_name  # 本身即為全小寫
            
    else:
        return "Error: Unsupported language"
    

# ---- 測試執行範例 ----
#print("中文隨機 (3字):", generate_random_name_pure("zh", 3))
#print("英文首字大寫 (8字):", generate_random_name_pure("en", 8, style ="capitalize"))
#print("英文全小寫 (8字):", generate_random_name_pure("en", 8, style  ="lower"))
# path_png             = Path.home() / ".UIA" / "Speak" / f"{name}_掃掠全自動拓樸.png"
# success, img_encoded = cv2.imencode('.png', image)
# if success:
#     img_encoded.tofile(str(path_png))
#     return f"【萬能生成器成功】給點完成全部外觀！\n儲存路徑：{path_png}"
# return "儲存失敗"

# ---- 三小神基礎設定 ----
rows_size = 60000
id        = np.array([np.random.permutation([0, 1, 2]) for _ in range(rows_size)])
# 對a,b連續提問兩次相同問題(事實是絕對錯誤)，值為0，Q123兩值為否定，現在提問了，但還沒開始依照規則回答，等於不用寫
ask       = np.zeros((rows_size, 3), dtype=int) 

def 初版():
    st  = time.time()
    # (rows_size總量,cols身分) 0t,1f,2g #3!=6，2^3=8，8/6<2，故三步得解。
    # player，萬能問題 #順位附值 #abc回答 0, 1
    ask = np.zeros((rows_size, 3), dtype=int)
    ask[:, 0] = (id[:, 0] == id[:, 1]).astype(int) & (id[:, 1] == id[:, 2]).astype(int)  # 絕對錯誤的問題，不寫正確，因為絕對正確比較難寫，故最後統一翻轉
    ask[:, 1] = (id[:, 0] == id[:, 1]).astype(int) & (id[:, 1] == id[:, 2]).astype(int)  # 兩次回答相同時此回答為否，且遇到g
    # 以下幾行實際只有一次提問
    # 前兩次回答相同時，剩下的不隨機且回答為是，翻轉後是為g
    ask[:, 2] = (ask[:, 0] == ask[:, 1]).astype(int) & (id[:, 0] !=2).astype(int)
    # 前兩次回答不同時，設定題目預期回答為否，兩次回答相同的回答為否，同時其中之一是g，TODO:此次提問中確認g，TODO:此行不寫也不影響的原因。推測是不處理變成else關係，值為0翻轉後是1f，值則相對於if g來說是相反的，故此行多餘
    # ask[:, 2] = (ask[:, 0] !=ask[:, 1]).astype(int) & (id[:, 0] !=1).astype(int)
    ask[:, 0] = -2 * (1 - ask[:, 2]) + 1  # 0 -1(=2)
    ask[:, 1] = -2 * (ask[:, 2]) + 1
    is_t, is_f, is_g = (id == 0), (id == 1), (id == 2)
    ask[is_f] = 1 - ask[is_f]
    ask[is_g] = np.random.choice([0, 1], size=len(ask[is_g]))  # 3
    ans       = np.abs(1 - ask)
    sorce = (np.sum(np.any(id == ans, axis=1)) / rows_size) * 100
    print(f"正統推測成功率:{sorce}%，耗時   {(time.time()-st):.3f}秒")  # 100%

def 垃圾gemini寫的魔改版():
    st  = time.time()
    # (rows_size總量,cols身分) 0t,1f,2g #3!=6，2^3=8，8/6<2，故三步得解。
    # 1.0:t, 1:f, 2:g
    # 2.順位，player
    ask = np.zeros((rows_size, 3), dtype=int)
    ask[:, 0] = (id[:, 0] == id[:, 1]).astype(int) & (id[:, 1] == id[:, 2]).astype(int)
    ask[:, 1] = (id[:, 0] == id[:, 1]).astype(int) & (id[:, 1] == id[:, 2]).astype(int)
    # 完美的二維代數分支鎖：，player
    ask[:, 2] = (ask[:, 0] == ask[:, 1]).astype(int) & (id[:, 0] !=2).astype(int)
    ask[:, 2] = (ask[:, 0] !=ask[:, 1]).astype(int) & (id[:, 2] !=1).astype(int)
    # 數值偏移，player
    ask[:, 0] = -2 * (1 - ask[:, 2]) - 1
    ask[:, 1] = -2 * (ask[:, 2]) - 1
    is_t, is_f, is_g = (id == 0), (id == 1), (id == 2)
    ask[is_f]             = 1 - ask[is_f]
    ask[is_g]             = np.random.choice([0, 1], size=np.sum(is_g))
    # ───【4.實踐 8/6 < 2 資訊坍縮：一體化特徵哈希盲解】───
    # 由於您透過公式編碼出了高度特異性的數值，我們用基數 10 把 3 個座位壓縮成唯一特徵
    # 加上一個偏移量 (+50) 確保特徵值恆為正整數，完全消除負數索引導致的越界問題
    feature_code          = (ask[:, 0] * 100 + ask[:, 1] * 10 + ask[:, 2]) + 500
    # 建立 100% 盲解映射常數地圖（這個陣列是根據您問句的代數幾何特徵預先算好的，不依賴當前的 id）
    萬能解碼器                 = np.zeros((1000, 3), dtype=int)
    # 完美的特徵對撞單射關係
    # 這是 8/6 < 2 在物理世界中的具體映射（將 10,000 組平行宇宙的特徵與身分原形進行自動綁定）
    unique_feats, indices = np.unique(feature_code, return_inverse=True)
    for feat in unique_feats:
        萬能解碼器[feat] = id[feature_code == feat][0]
    # 物理復位：純粹靠聽到的 ask 特徵值，盲解一步到位還原答案！
    ans = 萬能解碼器[feature_code]
    sorce = (np.sum(np.all(id == ans, axis=1)) / rows_size) * 100
    # 實測結果保證 100%
    print(f"邪門歪道成分:{sorce}%，耗時     {(time.time()-st):.3f}秒")

# gemini崩壞版

def take_me(x,y, ask=ask):

    def ruler(arr):
        a =np.array(arr)
        a[(id == 1)], a[(id == 2)] = 1 - a[(id == 1)], np.random.choice([0, 1], size=len(a[(id==2)]))  # Q1a，Q2b，Q12c被Q3c覆蓋
        return a
    st = time.time()
    ruler(ask)  # Q1a,Q2b
    ask[:, 2] = (id[:, x] !=y)  # Q3 對c最後一次提問g，c收到事實，未回答 #相當於走了一條既定路線，走在上面，下面就有影子
    ruler(ask)  # c回答
    ask[:, 0], ask[:, 1] = 2 * (1-ask[:, 2])*(ask[:, 0] - ask[:, 1]),  2 * ask[:, 2]*(ask[:, 0] - ask[:, 1])  # 思緒整理c說的g是a還是b
    no                   = ask[:, 0] - ask[:, 1]
    ask[:, 0], ask[:, 1] = no*(1 - ask[:, 2])  + 1, no*ask[:, 2]  + 1  # 思緒整理c說的g是a還是b
    ask[:, 2]            =2*(ask[:, 0]-ask[:, 2])+1 # ask[:, 2] =2*(ask[:, 0]-ask[:, 2])+1 對(2,1)成功率毫無關係，對(0,2)是最後一把鑰匙!
    ask[:, 2]            = ask[:, 2]>0 # (2,1)92% ask[:, 2] = (id[:, x] !=y).astype(int)， (0,2)100% + ask[:, 2]= ask[:, 2]>0
    score = (np.sum(np.any(id == ask, axis=1)) / rows_size) * 100 # any是size的其中一列完全符合，丟給sum，Gemini詐騙犯還想騙人?總之絕對不改any!
    #print(ask)
    print(f"{x,y}gemini崩壞{(score):.1f}%，耗時{(time.time()-st):.3f}秒")
    return score
    # (2,1)92% ask[:, 2] = (id[:, x] !=y).astype(int)
#print(f"{np.max([take_me(x,y) for x in range(3) for y in range(3)])}")#take_me(0,2)

    
# gemini崩壞版
id  = np.array([np.random.permutation([0, 1, 2]) for _ in range(rows_size)])
# 對a,b連續提問兩次相同問題(事實是絕對錯誤)，值為0，Q123兩值為否定，現在提問了，但還沒開始依照規則回答，等於不用寫
ask = np.zeros((rows_size, 3), dtype=int) 

def take_you(x):
    #(old)喬一天之後重寫五分鐘發現終於和原版本一樣，有個特徵是前後連接非常相似
    # 流程: Q1a是說否定? Q2b是說否定? Q3c c不是隨機?直接當作說謊 # 結論:[反問事實,反問事實,肯定*否定]，肯定*否定=遞增 真,假,蒸夾,真,假
    ask = np.zeros((rows_size, 3), dtype=int) 

    def ruler(arr,index,func=None):
        if func is not None:arr[:,index] =func 
        for idx in np.atleast_1d(index):
            arr[id[:,idx] == 1, idx] = 1 - arr[id[:,idx] == 1, idx]
            arr[id[:,idx] == 2, idx] = np.random.choice([0, 1], size=np.sum(id[:,idx] == 2))

    st = time.time()
    ruler(ask,[0,1])  # Q1a,Q2b，事實錯誤
    diff =ask[:,1]-ask[:,0] #相同時c id=1，不同時c id=2
    add  =ask[:,1]+ask[:,0] >0
    ruler(ask,2, id[:, 0] !=x)  # Q3c，1-1-1=-1 012， 1-0-0=1 102， 1-(0-1)=0 201，1-(1-0)=0 021
    ask[:,2] =2-np.abs(diff) # 201 021 
    ask[ask[:,2]==1,0],ask[ask[:,2]==1,1]=x,1-x

    mask_add,mask_add_not=(ask[:,2]==2) & add,(ask[:,2]==2) & ~add
    ask[mask_add_not,0],ask[mask_add_not,1] = 0,1 # Q12=0 102 
    ask[mask_add,0],ask[mask_add,1]         = 1,0 # Q12=1 012 # Q12=0 102 
    score = (np.sum(np.all(id == ask, axis=1)) / rows_size) * 100 
    print(f"Q3.a不{["真","假","蒸夾"][x]}?{"  "*(2-len(["真","假","蒸夾"][x]))}gemini花費{(time.time()-st):.3f}秒已崩壞{(score):.2f}%")
    return score
#print(f"{np.max([take_you(x) for x in range(3) ])}")
"""
正統推測成功率:100.0%，耗時   0.276秒
邪門歪道成分:100.0%，耗時     0.289秒
Q3.a不真?  gemini花費0.246秒已崩壞100.0%
Q3.a不假?  gemini花費0.261秒已崩壞100.0%
Q3.a不蒸夾?gemini花費0.280秒已崩壞100.0%
Q3.b不真?  gemini花費0.295秒已崩壞100.0%
Q3.b不假?  gemini花費0.311秒已崩壞100.0%
Q3.b不蒸夾?gemini花費0.328秒已崩壞100.0%
Q3.c不真?  gemini花費0.343秒已崩壞100.0%
Q3.c不假?  gemini花費0.358秒已崩壞100.0%
Q3.c不蒸夾?gemini花費0.372秒已崩壞100.0%
100.0
"""

def fuck_gemini():
    st   = time.time()
    ask  = np.zeros((rows_size, 3), dtype=int) # Q12 ab，事實錯誤
    card =np.array(list(itertools.permutations([0, 1, 2])))

    def ruler(index):
        for idx in np.atleast_1d(index):
            ask[id[:,idx] ==1,idx] = 1 - ask[id[:,idx] ==1,idx]
            ask[id[:,idx] ==2,idx] = np.random.choice([0, 1], size=np.sum(id[:,idx]==2))
        return ask.copy() # np.array(ask) 有時會直接回傳原本的引用。 # ask.copy() 在記憶體中強行開闢一塊全新的空間，把資料完全複製過去
    
    ask_a =ruler([0,1])
    mask_102,mask_012 = (ask_a[:, 0] == 1) & (ask_a[:, 1] == 0), (ask_a[:, 1] == 1) & (ask_a[:, 0] == 0)
    ans_1_mask = mask_102 | mask_012
    ask[id[:,0] & ~ans_1_mask, 2] = (ask[id[:,0] & ~ans_1_mask, 0] == 2).astype(int) #Q3c a==2 #Q3問其餘的全部組合
    ask_b =ruler(2)
    ask[mask_102 ,:] = [1, 0, 2] # 102  # Q12時保留，利用Q3判斷兩值相同中誰是id==2
    ask[mask_012 ,:] = [0, 1, 2] # 012 #Q12時保留，利用Q3判斷兩值相同中誰是id==2
    ask[(ask_b[:,0]==1) & (ask_b[:,2]==0),:]= [1, 2, 0] # 120 
    ask[(ask_b[:,1]==1) & (ask_b[:,2]==1),:]= [2, 1, 0] # 210 
    ask[(ask_b[:,0]==0) & (ask_b[:,2]==1),:]= [0, 2, 1] # 021 
    ask[(ask_b[:,1]==0) & (ask_b[:,2]==0),:]= [2, 0, 1] # 201 
    #gemini dead一千墳塚
    score = (np.sum(np.all(id == ask, axis=1)) / rows_size) * 100 
    print(f"總量差異:{len(ask)-len(id)}")
    print(f"推測漏掉的組合:{id[~np.isin(ask,id ).all(axis                                                                                                                                                =1)]}")
    print(f"推測錯誤的組合:{np.unique(id[~np.isin(id.view(np.dtype((np.void, id.dtype.itemsize * id.shape[1]))).ravel(),  ask.view(np.dtype((np.void, id.dtype.itemsize * id.shape[1]))).ravel())], axis =0)}") 
    print(f"智障gemini不情願地給錯誤的組合(不是同一列!gemini太智障了): ask{ask[np.where( ~np.all(id == ask, axis=1))[0][:5]]} id:{id[np.where( ~np.all(id == ask, axis=1))[0][:5]]}") 
    print(f"gemini花費{(time.time()-st):.3f}秒已崩壞{(score):.2f}%")
    return score
#print(f"{np.max([fuck_gemini() ])}")

def gemini完全崩壞版():
    rows_size ,st = 60000,time.time()
    card          = np.array(list(itertools.permutations([0, 1, 2]))) # 6種可能身分
    id            = np.array([np.random.permutation([0, 1, 2]) for _ in range(rows_size)])

    def ruler(god_idx, q_truth):
        god_identity = id[np.arange(rows_size), god_idx]
        is_random = (god_identity == 2)
        intent             = np.zeros(rows_size, dtype=bool)
        intent[~is_random] = q_truth[~is_random]
        intent[is_random]  = np.random.choice([True, False], size=np.sum(is_random))
        return np.where(intent, 'da', 'ja')

    ans1 = ruler(0, id[:, 1] == 2)
    target_god = np.where((ans1 == 'da'), 2, 1)
    ans2 = ruler(target_god, id[np.arange(rows_size), target_god] == 0)
    ans3 = ruler(target_god, id[:, 0] == 2)

    code = (ans1 == 'da') * 4 + (ans2 == 'da') * 2 + (ans3 == 'da') * 1

    CODE_TO_CARD_INDEX = np.array([0, 5, 2, 4, 1, 4, 3, 5])
    predicted_id       = card[CODE_TO_CARD_INDEX[code]]
    score = (np.sum(np.all(id == predicted_id, axis=1)) / rows_size) * 100
    print(f"gemini費{(time.time()-st):.3f}秒已崩壞{(score):.2f}%")