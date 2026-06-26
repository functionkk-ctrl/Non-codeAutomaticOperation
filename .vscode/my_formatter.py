import ast  # 加上這行
import io
import re
import sys  # 💡 確保補上這一行！
import tokenize


def sort_and_group_imports(code: str):
    """【功能 1】頂部處理：提取、分類，精準回傳 (import區塊字串, 剩餘程式碼字串)"""
    stdlib = {'sys', 'os', 'time', 're', 'io', 'math', 'json',
              'datetime', 'collections', 'itertools', 'functools'}
    lines = code.splitlines()
    import_lines = []
    other_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(('import ', 'from ')):
            cleaned_import = re.sub(r'\s+', ' ', line).strip()
            import_lines.append(cleaned_import)
        else:
            other_lines.append(line)

    other_code = "\n".join(other_lines)
    if not import_lines:
        return "", other_code

    group_std, group_third, group_local = [], [], []
    for imp in import_lines:
        match = re.match(r'^(?:import|from)\s+([a-zA-Z0-9_]+)', imp)
        if not match:
            group_local.append(imp)
            continue
        module_name = match.group(1)
        if module_name in stdlib:
            group_std.append(imp)
        elif module_name in ('np', 'numpy', 'pd', 'pandas', 'yapf', 'watchdog', 'astor'):
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

    import_code = "\n\n".join(import_blocks) if import_blocks else ""
    return import_code, other_code


def custom_format(code: str) -> str:
    # 1. 剝離頂部 import，其餘「後面全部要處理的字」流向總裝
    import_block, other_code = sort_and_group_imports(code)
    code_text = other_code

    # === 🎯 2. 這是你親手實作、無雜質、強制拉回同行靠緊的最終物理總裝線 ===

    # 物理導正運算子：= = 替換成 ==（精準還原 NumPy 的矩陣符號）
    code_text = re.sub(r'(\=) (\=)', r'\=\=', code_text)

    # [[不找，單[才縮 防線！（精準收攏 NumPy 被扯碎的跨行小括號與等號）
    code_text = re.sub(r'([\(=\[]+)(?!\[)\n\s*', r'\1', code_text)
    code_text = re.sub(r'(,)\n\s*(?!\[)', r'\1', code_text)
    code_text = re.sub(r'\n\s*(?<!\])([\)\]]+)', r'\1', code_text)

    # 你的字串清洗防線：壓縮引號與 print 後面超過兩個以上的垃圾空格
    code_text = re.sub(r'(f?["\'])\s{2,}', r'\1', code_text)
    code_text = re.sub(r'\s{2,}(["\']\s*\))', r'\1', code_text)

    # 🌟 3. 【核心物理降維防線】：精準鎖定「連接詞」大容器！
    # 利用 re.S 穿透換行，抓出整個大括號結構，進行「一絲不掛、強力拉回同行完美靠緊」大洗滌！
    container_match = re.search(
        r'(連接詞\s*=\s*)\{([\s\S]*?)\}', code_text, flags=re.S)
    if container_match:
        prefix = container_match.group(1)
        body = container_match.group(2)

        # 物理大抹平：吸乾內層所有混亂的換行、縮排與垃圾連續空格，揉成一整行！
        body_clean = re.sub(r'\s+', ' ', body).strip()

        # 語法智慧救贖：把內部不管是一維、多維、還是被誤寫的集合大括號 {}，通通更換為合法之中括號 []
        body_clean = body_clean.replace('{', '[').replace('}', ']')

        # 標點符號與中括號靠緊標準化（逗號固定留一格空格，其餘全部完美黏緊）
        body_clean = re.sub(r'\[\s*', '[', body_clean)
        body_clean = re.sub(r'\s*\]', ']', body_clean)
        body_clean = re.sub(r'\s*,\s*', ', ', body_clean)

        # 完美回填：強制還原成最清爽的「單行同一行靠緊」結構！
        code_text = re.sub(
            r'連接詞\s*=\s*\{([\s\S]*?)\}', f"{prefix}[{body_clean}]", code_text, flags=re.S)

    # 4. def|class 上面多空兩行
    code_text = re.sub(
        r'(?<!^)\n*(^[ \t]*\b(def|class)\b )', r'\n\n\n\1', code_text, flags=re.M)

    # 5. 標點符號 \n\n 轉 \n 物理補票（減號 - 扔到中括號集合最末端，徹底吸乾多餘空行）
    code_text = re.sub(r'([+=\(\[\],-])\n{2,}\s*', r'\1\n', code_text)
    code_text = re.sub(r'\n{2,}\s*([\)\]])', r'\n\1', code_text)

    # 3個以上空白行縮成2個\n(一行空白)，保持外部常規結構
    code_text = re.sub(r'\n{3,}', '\n\n', code_text)
    code_text = re.sub(r'\n{4,}', '\n\n\n', code_text)

    # 最終大對接
    if import_block:
        code_text = import_block + "\n\n" + code_text

    return code_text.strip()

    # 按行拆分，將 import 與後面全部要處理的字徹底分流
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(('import ', 'from ')):
            cleaned_import = re.sub(r'\s+', ' ', line).strip()
            import_lines.append(cleaned_import)
        else:
            other_lines.append(line)

    other_code = "\n".join(other_lines)
    if not import_lines:
        return "", other_code

    group_std, group_third, group_local = [], [], []
    for imp in import_lines:
        match = re.match(r'^(?:import|from)\s+([a-zA-Z0-9_]+)', imp)
        if not match:
            group_local.append(imp)
            continue
        module_name = match.group(1)
        if module_name in stdlib:
            group_std.append(imp)
        elif module_name in ('np', 'numpy', 'pd', 'pandas', 'yapf', 'watchdog', 'astor'):
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

    import_code = "\n\n".join(import_blocks) if import_blocks else ""
    return import_code, other_code

    import_code = "\n\n".join(import_blocks) if import_blocks else ""
    return import_code + "\n\n" + other_code
    # 前版本
    if other_lines:
        return "\n\n".join(import_blocks) + "\n".join(other_lines)+"\n"
    return "\n\n".join(import_blocks)


# def custom_format(code: str) -> str:
    # 🌟 1. 頂部剝離：拿到 import 區塊，以及後面全部要集中處理的 other_code
    import_block, other_code = sort_and_group_imports(code)

    # 🌟 2. 徹底解體！步驟 2、步驟 3 舊殘肢完全移除、原地蒸發！
    # 100% 杜絕 Token 走訪對註解的大屠殺，直接把最乾淨、帶有原始註解的 other_code 丟進總裝
    code_text = other_code

    # === 🎯 3. 這是你親手實作、無雜質的最終出廠總裝線 ===

    # = = 替換成 ==（精準搶救 NumPy 條件判斷）
    code_text = re.sub(r'(\=) (\=)', r'\=\=', code_text)

    # 🌟 [[不找，單[才縮 防線！
    code_text = re.sub(r'([\(=\[]+)(?!\[)\n\s*', r'\1', code_text)
    code_text = re.sub(r'(,)\n\s*(?!\[)', r'\1', code_text)
    code_text = re.sub(r'\n\s*(?<!\])([\)\]]+)', r'\1', code_text)

    # 你的字串清洗防線：引號或 print 後面超過兩個以上的垃圾空格直接吸乾
    code_text = re.sub(r'(f?["\'])\s{2,}', r'\1', code_text)
    code_text = re.sub(r'\s{2,}(["\']\s*\))', r'\1', code_text)

    # [往後或下一行找{，此{和後面的}替換成[和] (開啟 re.S 跨行特權)
    code_text = re.sub(
        r'(\b\w+\b\s*=\s*)\{([\s\S]*?)\}', r'\1[\2]', code_text, flags=re.S)

    # 🌟 你的靈魂多維陣列立體化：[["zx"],\n ["zx"],\n[無限]] ── 物理換行精準跳出！
    code_text = re.sub(r'(\]\s*,\s*)(?=\s*\[)', r'\1\n ', code_text)
    code_text = re.sub(r'(\[\[.*?\],)\s*(=\s*\[)', r'\1\n\2', code_text)

    # def|class 上面多空裝兩行
    code_text = re.sub(
        r'(?<!^)\n*(^[ \t]*\b(def|class)\b )', r'\n\n\n\1', code_text, flags=re.M)

    # 🌟 你最無敵的標點符號 \n\n 轉 \n 補票（減號 - 扔到中括號集合最末端）
    # 物理事實：精準吸扁標點符號與運算子周圍多噴出來的空行，因為沒有 # 號，註解 100% 安全繞道放行！
    code_text = re.sub(r'([+=\(\[\],-])\n{2,}\s*', r'\1\n', code_text)
    code_text = re.sub(r'\n{2,}\s*([\)\]])', r'\n\1', code_text)

    # 全局多重空白行正常縮減，維持常規乾淨結構
    code_text = re.sub(r'\n{3,}', '\n\n', code_text)
    code_text = re.sub(r'\n{4,}', '\n\n\n', code_text)

    # 最後一刻，與最開頭剝離好的 import 對接
    if import_block:
        code_text = import_block + "\n\n" + code_text

    return code_text.strip()
    code = sort_and_group_imports(code)

    # 【步驟 2】結構化解析：利用 tokenize 將程式碼依據語法邊界群組化
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except Exception:
        return code

    lines_map = {}
    current_stmt_id = 0
    paren_depth = 0

    for tok in tokens:
        toktype = tok.type
        tokval = tok.string
        if toktype in (tokenize.ENDMARKER, tokenize.ERRORTOKEN):
            continue
        if tokval in ("(", "[", "{"):
            paren_depth += 1
        elif tokval in (")", "]", "}"):
            paren_depth = max(0, paren_depth - 1)
        if toktype in (tokenize.NEWLINE, tokenize.NL):
            if paren_depth > 0:
                # current_stmt_id += 1  # 👈 不管括號有沒有閉合，遇到換行就直接切分語句！
                # continue  # 關鍵防線：大括號未閉合前，不切斷語句，保留多行結構
                pass
            else:
                current_stmt_id += 1
                continue
        if toktype == tokenize.INDENT:
            continue

        if current_stmt_id not in lines_map:
            lines_map[current_stmt_id] = {
                'indent': int(tok.start[1]), 'tokens': []}
        lines_map[current_stmt_id]['tokens'].append(
            (toktype, tokval, tok.start, tok.end))

    processed_statements = []

    # 【步驟 4】對齊輸出與重組
    final_lines = []
    max_left_len = max([len(x['left']) for x in processed_statements if x.get(
        'is_assignment')], default=0)
    for item in processed_statements:
        if item.get('is_assignment'):
            # 💡 靈魂物理公式：用最大長度減掉自己目前的長度，差多少，就用 " " 乘（repeat）多少個空格！
            space_count = max_left_len - len(item['left'])
            padding_spaces = " " * space_count

            # 直球拼接：縮排 + 左邊 + 補滿的空格 + 等號 + 右邊
            line_content = f"{item['indent']}{item['left']}{padding_spaces} {item['op']} {item['right']}"
        else:
            line_content = item['content']
        final_lines.append(line_content)

    # === 🎯 在這裡加入多維陣列的換行asdasdasdasd排版 ===
    code_text = "\n".join(final_lines)
    # 註：你可以根據你想達到的具體換行樣式，微調這個 re.sub 的 pattern
    # = =替換成==
    code_text = re.sub(r'(\=) (\=)', r'\=\=', code_text)
    # 這樣就能精準做到：[[ 開頭的立體結構絕對不找、100% 繞過，只有單個 [, (, = 造成的異常換行才強制縮回！
    # code_text = re.sub(r'([\(=\[]+)(?!\[)\n\s*', r'\1', code_text)
    # 假的
    # code_text = re.sub(r'([(=]+|\[(?!\[))\n\s*', r'\1', code_text)

    # 行尾逗號的精準防線：同樣要確保下一行不是以 [ 開頭（因為立體陣列每行是 [ 開頭），
    # 只有當下一行是普通代碼（非 [ 開頭）時，才能把因逗號切斷的異常換行縮回來！
    # code_text = re.sub(r'(,)\n\s*(?!\[)', r'\1', code_text)
    # 右括號與右中括號的防線：同樣要排除立體陣列結尾閉合的 ]，只有單個括號才縮
    # code_text = re.sub(r'\n\s*(?<!\])([\)\]]+)', r'\1', code_text)

    # 3. 您的字串清洗防線：鎖定 f" 或引號後面超過兩個以上的垃圾空格，直接吸乾
    # code_text = re.sub(r'(f?["\'])\s{2,}', r'\1', code_text)
    # code_text = re.sub(r'\s{2,}(["\']\s*\))', r'\1', code_text)

    # 4. 🌟 [往後或下一行找{，此{和後面的}替換成[和] ── 終極不換行修正版 #TODO:未轉成[
    # 物理導正：最外層等號右邊、或者緊跟在中括號後面的大括號，強制替換為合法中括號
    # code_text = re.sub(r'(\b\w+\b\s*=\s*)\{([\s\S]*?)\}', r'\1[\2]', code_text, flags=re.S)
    # code_text = re.sub(r'=\s*\{\s*\n*', ' = [\n', code_text)
    # code_text = re.sub(r'(\[)\s*\{\s*\n*', r'\1[', code_text)
    # 核心修復：這行是您寫的物理替換！但我們補上 \s* 物理吸乾後面所有的垃圾換行與空格，
    # 100% 確保結尾的 ] 絕對靠緊在同一行閉合，徹底終結右括號被單獨扯到下一行的悲劇！
    # code_text = re.sub(r'(\})\s*\]\s*\n*', r'\1]', code_text)
    # code_text = re.sub(r'(\})\s*\]', r'\1\n]', code_text)  # [往後或下一行找{，此{和後面的}替換成[和]
    # 3. gemini堅持自殺
    # code_text = re.sub(r'=\s*\{\s*\n*', ' = [\n', code_text)
    # code_text = re.sub(r'\}\s*$', ']', code_text.strip())
    # [["zx"], ["zx"], ...]
    # code_text = re.sub(r'(\]\s*,\s*)(?=\s*\[)', r'\1 ', code_text)
    # code_text = re.sub(r'(\]\s*,\s*)(?=\s*\[)', r'\1\n ', code_text)
    # def|class上面多空兩行
    # code_text = re.sub(        r'(?<!^)\n*(^[ \t]*\b(def|class)\b )', r'\n\n\n\1', code_text, flags=re.M)  # (後面異常換行，這是結果
    # [["zx"],\n ["zx"],\n[無限]]
    # code_text = re.sub(r'(\[\[.*?\],)\s*(=\s*\[)', r'\1\n\2', code_text)
    # 只要發現它們後面接著兩個換行符號（\n\n），當場降維打擊，全部吸扁成單一 \n 換行！
    # code_text = re.sub(r'([+=\(\[\],-])\n{2,}\s*', r'\1\n', code_text)
    # 同步修正右括號前面被多噴出來的詭異空行
    # code_text = re.sub(r'\n{2,}\s*([\)\]])', r'\n\1', code_text)
    # code_text = re.sub(r'\n{3,}', '\n\n', code_text)  # 3個以上空白行縮成2個\n(一行空白)

    # === 🎯 這是您要求的：徹底終結異常增多換行（純粹字串物理抽真空） ===
    code_text = "\n".join(final_lines)

    # 1. 物理導正運算子：= = 替換成 ==
    code_text = re.sub(r'(\=) (\=)', r'\=\=', code_text)

    # 2. [[不找，單[才縮 防線！
    code_text = re.sub(r'([\(=\[]+)(?!\[)\n\s*', r'\1', code_text)
    code_text = re.sub(r'(,)\n\s*(?!\[)', r'\1', code_text)
    code_text = re.sub(r'\n\s*(?<!\])([\)\]]+)', r'\1', code_text)

    # 3. 您的字串清洗防線
    code_text = re.sub(r'(f?["\'])\s{2,}', r'\1', code_text)
    code_text = re.sub(r'\s{2,}(["\']\s*\))', r'\1', code_text)

    # 4. [往後或下一行找{，此{和後面的}替換成[和] (開啟 re.S 跨行特權)
    code_text = re.sub(
        r'(\b\w+\b\s*=\s*)\{([\s\S]*?)\}', r'\1[\2]', code_text, flags=re.S)
    code_text = re.sub(r'\{([\s\S]*?)\}', lambda m: f"[{m.group(1)}]" if ':' not in m.group(
        1) else m.group(0), code_text, flags=re.S)

    # 5. 多維陣列立體化：[["zx"],\n ["zx"],\n[無限]]
    code_text = re.sub(r'(\]\s*,\s*)(?=\s*\[)', r'\1\n ', code_text)
    code_text = re.sub(r'(\[\[.*?\],)\s*(=\s*\[)', r'\1\n\2', code_text)

    # 6. def|class 上面多空兩行
    code_text = re.sub(
        r'(?<!^)\n*(^[ \t]*\b(def|class)\b )', r'\n\n\n\1', code_text, flags=re.M)

    # 🌟 7. 【終極事後補票】── 物理抽真空，徹底終結空行無限分裂災難！
    # 物理事實：把減號 - 扔到中括號集合最末端，並加上 \s* 排除所有隱形空格下毒！
    # 不論是容器內部還是 NumPy 運算周圍，只要偵測到連續兩個換行符號（\n{2,}）、且中間夾帶了任何垃圾空格（\s*），
    # 一巴掌全面物理吸乾、降維壓縮成單一的 \n 換行！從源頭死死卡住「異常增多換行」！
    code_text = re.sub(r'([+=\(\[\],-])\n{2,}\s*', r'\1\n', code_text)
    code_text = re.sub(r'\n{2,}\s*([\)\]])', r'\n\1', code_text)

    # 🌟 核心保險：全域夾帶隱形空格的變態多換行，全面強制縮減！
    code_text = re.sub(r'\n\s*\n\s*\n', '\n\n', code_text)
    # 把多維陣列內部所有夾帶空格的髒空行，強制壓回完美的直立靠緊！
    code_text = re.sub(r'\n\s*\n', '\n', code_text)

    return code_text.strip()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 直接精準指定第 1 個引數，確保抓出來的是純路徑字串（str），絕不帶有 list 結構
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
            print(f"Error: {e}", file=sys.stderr)
