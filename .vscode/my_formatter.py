import ast
import io
import re
import sys
import tokenize


def sort_and_group_imports(code: str):
    """【功能 1】頂部處理：提取、分類，精準回傳 (import區塊字串, 剩餘程式碼字串)"""
    stdlib = {
        "sys",
        "os",
        "time",
        "re",
        "io",
        "math",
        "json",
        "datetime",
        "collections",
        "itertools",
        "functools",
    }
    lines = code.splitlines()
    import_lines = []
    other_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            cleaned_import = re.sub(r"\s+", " ", line).strip()
            import_lines.append(cleaned_import)
        else:
            other_lines.append(line)

    other_code = "\n".join(other_lines)
    if not import_lines:
        return "", other_code

    group_std, group_third, group_local = [], [], []
    for imp in import_lines:
        match = re.match(r"^(?:import|from)\s+([a-zA-Z0-9_]+)", imp)
        if not match:
            group_local.append(imp)
            continue
        module_name = match.group(1)
        if module_name in stdlib:
            group_std.append(imp)
        elif module_name in (
            "np",
            "numpy",
            "pd",
            "pandas",
            "yapf",
            "watchdog",
            "astor",
        ):
            group_third.append(imp)
        else:
            group_local.append(imp)

    group_std.sort()
    group_third.sort()
    group_local.sort()

    import_blocks = []
    if group_std:
        import_blocks.append("\n".join(group_std))
    if group_third:
        import_blocks.append("\n".join(group_third))
    if group_local:
        import_blocks.append("\n".join(group_local))

    import_code = "\n".join(import_blocks) if import_blocks else ""
    return import_code, other_code


def fix_connective_block_dynamic(code: str) -> str:
    """【功能 2】自適應狀態機：無論容器在哪裡、中文怎麼改，只精準更換內層大括號，保留所有換行與原始文字"""
    match = re.search(r"連接詞\s*=\s*\{", code)
    if not match:
        return code

    start_idx = match.end() - 1
    result = list(code)
    brace_depth = 0
    in_string = False
    string_char = ""
    in_comment = False

    for i in range(start_idx, len(result)):
        char = result[i]
        if char == "\n":
            in_comment = False
            continue
        if in_comment:
            continue
        if in_string:
            if char == string_char and result[i - 1] != "\\":
                in_string = False
            continue
        elif char in ('"', "'"):
            in_string = True
            string_char = char
            continue
        if char == "#":
            in_comment = True
            continue

        if char == "{":
            brace_depth += 1
            if brace_depth > 1:
                result[i] = "["
        elif char == "}":
            if brace_depth > 1:
                result[i] = "]"
            brace_depth -= 1
            if brace_depth == 0:
                break
    return "".join(result)


def align_equal_signs(code: str) -> str:
    """【功能 3】高級型垂直對齊：支援 =, +=, -=, *=, /= 對齊，嚴格跳過函式定義 def 的參數與連接詞區塊"""
    lines = code.splitlines()
    new_lines = []
    block = []
    in_connective = False
    in_def_block = False

    assign_pattern = re.compile(
        r"(?<![+\-*/<>!=])(?P<op>\+=|-=|\*=|\/=|=(?![=<>!]))(?![=<>!])"
    )

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("def ") or (in_def_block and not stripped.endswith(":")):
            if block:
                new_lines.extend(flush_align_block(block, assign_pattern))
                block = []
            if stripped.startswith("def "):
                in_def_block = True
            new_lines.append(line)
            if stripped.endswith("):") or stripped.endswith(":"):
                in_def_block = False
            continue

        if "連接詞" in line and "=" in line:
            if block:
                new_lines.extend(flush_align_block(block, assign_pattern))
                block = []
            in_connective = True

        if in_connective:
            new_lines.append(line)
            if stripped == "}":
                in_connective = False
            continue

        is_comment_or_empty = stripped.startswith("#") or not stripped
        is_assign = (
            not is_comment_or_empty
            and assign_pattern.search(line) is not None
            and "==" not in line
            and "!=" not in line
            and "<=" not in line
            and ">=" not in line
        )

        if is_assign or (is_comment_or_empty and block):
            block.append(line)
        else:
            if block:
                new_lines.extend(flush_align_block(block, assign_pattern))
                block = []
            new_lines.append(line)

    if block:
        new_lines.extend(flush_align_block(block, assign_pattern))

    return "\n".join(new_lines)


def flush_align_block(block, pattern):
    """精準計算區塊內運算子左邊的最大長度並實施對齊"""
    while block and (block[-1].strip().startswith("#") or not block[-1].strip()):
        block.pop()
    if not block:
        return []

    aligned = []
    max_left_len = 0
    line_data = []

    for line in block:
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            line_data.append((None, None, line))
        else:
            match = pattern.search(line)
            if match:
                op = match.group("op")
                idx = match.start("op")
                left = line[:idx].rstrip()
                right = line[idx + len(op) :]
                max_left_len = max(max_left_len, len(left))
                line_data.append((left, op, right))
            else:
                line_data.append((None, None, line))

    for left, op, right in line_data:
        if left is None:
            aligned.append(right)
        else:
            spaces = " " * (max_left_len - len(left))
            aligned.append(f"{left}{spaces} {op}{right}")
    return aligned


def custom_format(code: str) -> str:
    import_block, other_code = sort_and_group_imports(code)
    code_text = other_code

    # === 🎯 1. 物理安全修正：緊湊核心運算子，制止被切開 ===
    code_text = re.sub(r"!\s*=\s*", "!=", code_text)
    code_text = re.sub(r"\+\s*=\s*", "+=", code_text)
    code_text = re.sub(r"-\s*=\s*", "-=", code_text)
    code_text = re.sub(r"\*\s*=\s*", "*=", code_text)
    code_text = re.sub(r"/\s*=\s*", "/=", code_text)
    code_text = re.sub(r"-\s*>\s*", "->", code_text)

    # === 🎯 2. 跨行參數矯正防線 ===
    lines = code_text.splitlines()
    new_lines = []
    skip = False
    
    for i in range(len(lines)):
        if skip:
            skip = False
            continue
            
        current_line = lines[i]
        
        # 只要不是最後一行，且前一行以逗號結尾、不屬於連接詞
        if i < len(lines) - 1 and current_line.rstrip().endswith(",") and "連接詞" not in current_line:
            next_line = lines[i + 1]
            current_indent = len(current_line) - len(current_line.lstrip())
            next_indent = len(next_line) - len(next_line.lstrip())
            
            # 🌟 只要下一行縮排更深，強制合併同行，並開啟 skip 跳過下一行！
            if next_indent > current_indent:
                merged = current_line.rstrip() + " " + next_line.lstrip()
                new_lines.append(merged)
                skip = True
                continue
                
        new_lines.append(current_line)
        
    code_text = "\n".join(new_lines)

    # 基礎清洗拉回
    code_text = re.sub(r"\\\s*\n\s*", "", code_text)
    code_text = re.sub(r"\.astype\(\s*\n\s*", ".astype(", code_text)
    code_text = re.sub(r"\s*\.\s*", ".", code_text)

    # === 🎯 3. 自適應狀態機動態修復連接詞 ===
    code_text = fix_connective_block_dynamic(code_text)

    # === 🎯 4. 高級垂直對齊 ===
    code_text = align_equal_signs(code_text)

    # === 🎯 5. 常規結構化空行調整 ===
    code_text = re.sub(
        r"(?<!^)\n*(^[ \t]*\b(def|class)\b )", r"\n\n\n\1", code_text, flags=re.M
    )

    # === 🎯 6. 總裝對接 ===
    if import_block:
        code_text = import_block + "\n\n" + code_text

    # 全體總裝完成後，強制任何空白行不得超過 2 行
    code_text = re.sub(r"\n{3,}", "\n\n", code_text)

    return code_text.strip()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 🌟 確保只去抓 sys.argv[1]
        file_path = str(sys.argv[1]).strip("'\"")

        if "my_formatter.py" in file_path:
            sys.exit(0)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()

            formatted_source = custom_format(source)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(formatted_source)
        except Exception as e:
            # 🌟 絕不無聲退出：有任何 Bug 或是檔案讀寫錯誤，直接在終端機當場噴出來！
            import traceback
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)

def my_class():
    R"""
    TODO:自動排版總覽，旨在存檔就移除全部抱錯訊息
    左邊=右邊 對齊，要分得更細緻，不然沒有參照的意義
        TODO:異常分離複雜等號
    Token使用時會有海量錯誤，理解原因，雖然不懂原理，但也許可以用矩陣獲得每一行和詞，然後就可以處理自動排版
    徹底補充
        TODO:異常刪除註解，包含上下行靠緊時異常刪除註解

    動態矩陣沙盒(眼睛)
    眼睛
        假迴圈(算式)移除雜訊並整合成一行算式
        同迴圈內的算式左右分開在同個縱軸上的複雜等號
        資料類型歸一化
        分離:函數 變數 資料數值 運算符號
        用 緊湊化代碼 測眼力
        
    腦力
        資料索引範圍
            todo:5. R+"*3
            test:
                IndexError: list index out of range
                KeyError: 'xxx'
                AttributeError: 'NoneType' object has no attribute 'xxx'
                TypeError: 'xxx' object is not subscriptable
                5.SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes in position 12208-12209: truncated \UXXXXXXXX escape
        import 關鍵詞，關鍵詞使用時沒有import 關鍵詞
            todo:
            test:


        分類
            todo:處理方法
            test:報錯訊息
        資料索引範圍
            todo:
                1. 存取列表前先用 len() 檢查長度，或在迴圈中確保索引在 range(len(list)) 範圍內，亦可使用 try-except 捕捉。
                2. 字典取值改用 .get('xxx', default) 方法，或存取前先用 if 'xxx' in dict 進行檢查。
                3. 在呼叫屬性或方法前，先用 if obj is not None 檢查變數是否為空值（通常是函數忘了寫 return 或 API 請求失敗）。
                4. 檢查變數的實際型態，避免誤把整數（int）、浮點數（float）或自訂物件當成了列表或字典來進行中括號取值。
                5. 確保陣列的中括號內傳入的是整數（int）或切片（slice），如果是計算出來的數字，請先用 int() 強制轉型。
            test:
                IndexError: list index out of range
                KeyError: 'xxx'
                AttributeError: 'NoneType' object has no attribute 'xxx'
                TypeError: 'xxx' object is not subscriptable
                TypeError: list indices must be integers or slices, not float

        import 關鍵詞，關鍵詞使用時沒有import 關鍵詞
            todo:
                1. 檢查程式碼最上方是否漏寫了 import 關鍵詞，並確認該關鍵詞（變數/函式名）沒有拼錯字或大小寫不一致。
                2. 確認該第三方套件是否已安裝在當前的環境中（執行 pip install 套件名稱），或檢查自訂檔案的相對路徑與資料夾結構。
                3. 檢查模組內是否有該名稱（可能版本更新被移除）；或檢查專案內有沒有跟內建模組同名（如 math.py）的自訂檔案導致錯引。
                4. 避免 A 檔案 import B 且 B 又 import A 的循環引入。可以將共同邏輯抽離到第三個檔案，或將 import 移到函式內部延遲執行。
            test:
                NameError: name '關鍵字' is not defined
                ModuleNotFoundError: No module named '關鍵字'
                AttributeError: module '關鍵字' has no attribute 'xxx'
                ImportError: cannot import name 'xxx' from partially initialized module (most likely due to a circular import)

        語法結構與排版錯誤 (VS Code 靜態檢查最常抓出的錯誤)
            todo:
                1. 檢查程式碼中括號 ( ) [ ] { }、單雙引號 ' " 是否有成對閉合，或是否誤用了中文全形標點符號。
                2. 檢查程式碼縮排，Python 強制要求同一區塊的縮排必須一致，不要混用空格（Space）與制表符（Tab）。
                3. 在 if, elif, else, for, while, def, class 等語句的結尾，務必檢查是否漏掉了冒號 : 。
                4. 檢查關鍵字拼寫是否正確（例如將 while 寫成 whlie，或將 global 寫成 golbal）。
            test:
                SyntaxError: invalid syntax
                SyntaxError: expected ':'
                SyntaxError: unmatched ')'
                IndentationError: unexpected indent
                IndentationError: unindent does not match any outer indentation level
                TabError: inconsistent use of tabs and spaces in indentation

        資料型態運算與參數傳遞錯誤
            todo:
                1. 檢查運算子兩邊的資料型態是否相符，例如不能讓字串（str）直接加數字（int），必須先用 str() 或 int() 轉型。
                2. 檢查呼叫函式時帶入的參數數量，是否少給了必要參數，或是給了超出定義數量的參數。
                3. 檢查傳入函式的參數型態是否正確（例如需要整數的函式被傳入了字串）。
                4. 進行數值型態轉換時（如 int('abc')），確保字串內容確實為純數字，否則會轉換失敗。
            test:
                TypeError: can only concatenate str (not "int") to str
                TypeError: unsupported operand type(s) for +: 'int' and 'str'
                TypeError: 函式名() missing 1 required positional argument: 'xxx'
                TypeError: 函式名() takes 1 positional argument but 2 were given
                ValueError: invalid literal for int() with base 10: 'xxx'

        檔案讀寫與系統路徑錯誤
            todo:
                1. 檢查讀取的檔案路徑是否正確，路徑若包含斜線，建議在字串前加上 r (如 r"C:\Users\...") 避免轉義字元出錯。
                2. 確保該檔案或資料夾確實存在於指定路徑中，或確認 VS Code 目前的工作目錄（Working Directory）是在哪一層。
                3. 檢查檔案是否已被其他軟體佔用鎖定，或者目前的使用者權限是否允許對該檔案進行讀取/寫入。
                4. 開啟檔案時若遇到中文亂碼或崩潰，務必在 open() 函式中手動指定編碼格式（如 encoding='utf-8'）。
            test:
                FileNotFoundError: [Errno 2] No such file or directory: 'xxx'
                PermissionError: [Errno 13] Permission denied: 'xxx'
                FileExistsError: [Errno 17] File exists: 'xxx'
                UnicodeDecodeError: 'gbk' codec can't decode byte 0x80 in position...

        數學計算與資源邊界異常
            todo:
                1. 在進行除法（/）或求餘數（%）運算前，先用 if 檢查除數（分母）是否為 0。
                2. 檢查遞迴函式（Recursion）是否有正確設定終止條件，避免無限循環導致系統呼叫棧溢出。
                3. 優化演算法或記憶體使用，避免在記憶體中一次性載入過於龐大的資料集（如超大圖檔或超大 CSV）。
            test:
                ZeroDivisionError: division by zero
                RecursionError: maximum recursion depth exceeded in comparison
                MemoryError

        VS Code 環境與執行端系統錯誤
            todo:
                1. 當系統提示找不到指令時，檢查 VS Code 的環境變數（PATH）設定，或重新執行 Shell Command 將 code 加入 PATH。
                2. 若伺服器或網頁服務無法啟動，說明該連接埠（Port）已被其他背景程式佔用，需到終端機查詢並強制終止該程序。
                3. 檢查 launch.json 設定中的 runtime 執行檔路徑是否正確，或更換預設的終端機外殼（Shell Profile）。
            test:
                'code' is not recognized as an internal or external command
                The terminal process failed to launch: Path to shell executable "..." does not exist.
                Error: listen EADDRINUSE: address already in use :::3000
                #
                不管是多執行緒的死鎖（Deadlock）、記憶體洩漏（Memory Leak）、非同步（Async/Await）的死循環，還是各類極端的環境衝突，都是基礎題

    """