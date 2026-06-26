import ast


class BracketTransformer(ast.NodeTransformer):
    """
    這個類別會遍歷整個 Python 檔案的語法樹，
    只要發現賦值語句（A = B）右邊是字典（Dict）或集合（Set），
    就會在保持內容不變的情況下，自動將其降維或轉換成列表（List）。
    """

    def visit_Assign(self, node):
        # 繼續向下處理子節點（確保巢狀結構也能被處理）
        self.generic_visit(node)

        # 如果賦值右側是集合 (Set)，例如 data = {1, 2, 3} -> 轉成列表 [1, 2, 3]
        if isinstance(node.value, ast.Set):
            node.value = ast.List(elts=node.value.elts, ctx=ast.Load())

        # 如果賦值右側是字典 (Dict)，且符合你前面提到的巢狀或是特定的對等關係
        elif isinstance(node.value, ast.Dict):
            # 這裡可以根據你的需求，把 Dict 的 keys 或 values 轉成 List
            # 例如你範例中的多層結構，如果是被解析成 Dict，我們可以提取它的元素
            elements = []
            for k, v in zip(node.value.keys, node.value.values):
                if k is None:  # 集合型態在 Dict 中可能表現為無 Key
                    elements.append(v)
                else:
                    elements.extend([k, v])
            node.value = ast.List(elts=elements, ctx=ast.Load())

        return node


def custom_format(code: str) -> str:
    # 【步驟 1】頂部處理：先排序並分類所有 import 語句（傳入整份原始碼）
    code = sort_and_group_imports(code)

    # 【步驟 2】結構化解析：利用 tokenize 將程式碼依據語法邊界群組化（防範多行斷句被切碎）
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
                continue  # 🔥 關鍵防線：大括號未閉合前，不切斷語句，完美保留多行連接詞結構！
            else:
                current_stmt_id += 1
                continue
        if toktype == tokenize.INDENT:
            continue
        if current_stmt_id not in lines_map:
            lines_map[current_stmt_id] = {
                "indent": int(tok.start[1]),
                "tokens": [],
            }
        lines_map[current_stmt_id]["tokens"].append(
            (toktype, tokval, int(tok.start[1]), int(tok.end[1]))
        )

    processed_statements = []

    # 【步驟 3】逐句（Statement）深度清洗與重組排版
    for stmt_id, data in sorted(lines_map.items()):
        indent_str = " " * data["indent"]
        tok_list = data["tokens"]
        if not tok_list:
            continue

        stmt_str = ""
        prev_type = None
        prev_val = ""
        prev_end = 0
        inline_comment = ""

        # 將該語句的所有 Token 重新串接回完整的字串（跨行大括號此時已被融合成單行長字串）
        for t_type, t_val, t_start, t_end in tok_list:
            tok_name = tokenize.tok_name.get(t_type, "")
            if t_type == tokenize.COMMENT:
                inline_comment = t_val
                continue

            if t_type == tokenize.STRING or tok_name in (
                "FSTRING_TEXT",
                "FSTRING_MIDDLE",
            ):
                t_val_clean = re.sub(r"\s*==\s*", " == ", t_val)
                t_val_clean = re.sub(r"\s*!=\s*", " != ", t_val_clean)
                stmt_str += t_val_clean
                prev_type = t_type
                prev_val = t_val_clean[-1] if t_val_clean else t_val
                prev_end = t_end
                continue

            if stmt_str:
                is_after_prev_end = t_start > prev_end
                if (
                    t_val in (")", "]", "}", ",", ":", ";")
                    or prev_val in ("(", "[", "{", ".", "utf", "encoding")
                    or t_type == tokenize.NUMBER
                    or prev_type == tokenize.NUMBER
                ):
                    if t_val == "=" and prev_val in ("=", "!"):
                        stmt_str = stmt_str.rstrip() + t_val
                    else:
                        stmt_str += t_val
                elif t_val in (
                    "==",
                    "!=",
                    ">=",
                    "<=",
                    "&",
                    "|",
                    "=",
                    "+",
                    "-",
                    "*",
                    "/",
                    "+=",
                    "-=",
                    "*=",
                    "/=",
                ):
                    if t_val == "=" and prev_val in ("=", "!"):
                        stmt_str = stmt_str.rstrip() + t_val
                    else:
                        if not stmt_str.endswith(" "):
                            stmt_str += " "
                        stmt_str += t_val
                elif (
                    t_val == "("
                    and prev_type == tokenize.NAME
                    and prev_val
                    not in (
                        "if",
                        "elif",
                        "while",
                        "for",
                        "return",
                        "and",
                        "or",
                        "not",
                    )
                ):
                    stmt_str += t_val
                else:
                    if not stmt_str.endswith(" ") and is_after_prev_end:
                        stmt_str += " "
                    stmt_str += t_val
            else:
                stmt_str += t_val

            prev_type = t_type
            prev_val = t_val
            prev_end = t_end

        stmt_str = stmt_str.strip()

        # 執行功能 2：清洗 print 語句
        if stmt_str.startswith("print("):
            stmt_str = clean_print_spaces(stmt_str)

        # 🔥 【終極修復核心】：只要有等號和左大括號，就進智慧容器修復器！
        if (
            "=" in stmt_str
            and "{" in stmt_str
            and not stmt_str.startswith(("def ", "return "))
        ):
            # 呼叫升級版修復器，它會把壓扁的字串重新展開成極度漂亮的直式換行結構
            stmt_str = fix_dict_syntax_and_layout(stmt_str, indent_str)

            processed_statements.append({
                "is_assignment": False,
                "content": str(stmt_str),
                "comment": inline_comment,
            })
            continue

        # 絕對防禦與後續一般賦值處理（你原本的後半段邏輯）
        assignment_match = None
        if isinstance(stmt_str, str) and not stmt_str.startswith(
            ("def ", "return ", "print(")
        ):
            assignment_match = re.search(
                r"(?<![=!<>+\-*/])(\+=|-=|\*=|\/=|=(?![=]))", stmt_str
            )

        if assignment_match:
            op = assignment_match.group(1)
            parts = stmt_str.split(op, 1)
            if len(parts) == 2:
                left = str(parts[0]).strip()
                right = str(parts[1]).strip()
                left = re.sub(r"\s*,\s*", ", ", left)
                left = re.sub(r"\s*\[\s*", "[", left)
                left = re.sub(r"\s*\]\s*", "]", left)
                right = re.sub(r"\s*==\s*", " == ", right)
                right = re.sub(r"\s*!=\s*", " != ", right)
                right = re.sub(r"\s*>=\s*", " >= ", right)
                right = re.sub(r"\s*<=\s*", " <= ", right)
                stmt_str = f"{left} {op} {right}"

        # 將一般語句放回陣列（記得補上縮排）
        processed_statements.append({
            "is_assignment": True,
            "content": f"{indent_str}{stmt_str}",
            "comment": inline_comment,
        })

    # 【步驟 4】最後將所有處理完的語句行重新用換行符號串接回檔案原始碼
    final_lines = []
    for s in processed_statements:
        line_content = s["content"]
        if s["comment"]:
            line_content += f"  {s['comment']}"
        final_lines.append(line_content)

    return "\n".join(final_lines)


def fix_dict_syntax_and_layout(stmt: str, indent: str) -> str:
    """【功能 3】智慧容器修復器：完美摧毀嵌套不合法的大括號，並優化為合法的 List 換行排版"""
    if not isinstance(stmt, str) or '{' not in stmt or '}' not in stmt:
        return str(stmt)

    # 判斷是否為真正的字典（包含未被字串包覆的冒號）
    # 因為程式碼可能已被 tokenize 壓成單行，我們精準檢查最外層
    has_real_dict = False
    in_str = None
    str_chars = ("'", '"')

    # 快速檢查整個語句中是否有字典的 key: value 特徵
    # 如果最外層的元素都是 [ ... ] 或 { ... } 且中間沒真正的冒號，就屬於需要修正的容器
    for i, ch in enumerate(stmt):
        if ch in str_chars:
            if not in_str:
                in_str = ch
            elif in_str == ch:
                in_str = None
        if not in_str and ch == ':':
            # 簡單排除切片等干擾，確認是否有字典特徵
            has_real_dict = True
            break

    # 核心遞迴括號轉換器：將所有不合法的 {} 轉為 []
    def convert_brackets(text: str) -> str:
        res = []
        idx = 0
        length = len(text)
        while idx < length:
            if text[idx] == '{':
                start = idx
                depth = 1
                idx += 1
                while idx < length and depth > 0:
                    if text[idx] == '{':
                        depth += 1
                    elif text[idx] == '}':
                        depth -= 1
                    idx += 1
                if depth == 0:
                    inner = text[start+1:idx-1]
                    inner = convert_brackets(inner)  # 遞迴處理內層
                    if inner.strip().startswith('[') and inner.strip().endswith(']'):
                        res.append(inner)
                    else:
                        res.append(f"[{inner}]")
                else:
                    res.append(text[start:idx])
            else:
                res.append(text[idx])
                idx += 1
        return "".join(res)

    # 情況 A：如果是需要導正為 List 的錯位容器（如你給的連接詞範例）
    if not has_real_dict:
        # 分離主體與註解
        comment_part = ""
        if '#' in stmt:
            parts = stmt.split('#', 1)
            stmt = parts[0]
            comment_part = "  #" + parts[1]

        # 提取賦值左側與右側
        if '=' in stmt:
            prefix, rhs = stmt.split('=', 1)
            prefix = prefix.strip() + " ="
        else:
            prefix, rhs = "", stmt

        # 將右側不合法的大括號全數轉為中括號
        fixed_rhs = convert_brackets(rhs.strip())

        # 進行漂亮的排版美化：讓一級子元素獨立換行
        # 清洗掉多餘連續空格，但保持元素完整性
        if fixed_rhs.startswith('[') and fixed_rhs.endswith(']'):
            core = fixed_rhs[1:-1].strip()

            # 聰明切分一級子元素 (依據最外層逗號)
            elements = []
            current = []
            p_depth = 0
            in_s = None

            for char in core:
                if char in str_chars:
                    if not in_s:
                        in_s = char
                    elif in_s == char:
                        in_s = None
                if not in_s:
                    if char in ('[', '{'):
                        p_depth += 1
                    elif char in (']', '}'):
                        p_depth = max(0, p_depth - 1)

                if char == ',' and p_depth == 0 and not in_s:
                    elements.append("".join(current).strip())
                    current = []
                else:
                    current.append(char)
            if current:
                elements.append("".join(current).strip())

            # 組合出極度漂亮的直式排版
            dict_indent = indent + "    "
            formatted_lines = []
            for el in elements:
                if el:
                    # 清洗子元素內部的逗號空格
                    el_clean = re.sub(r'\s*,\s*', ', ', el)
                    formatted_lines.append(f"{dict_indent}{el_clean},")

            final_output = f"{indent}{prefix} [\n" + \
                "\n".join(formatted_lines) + f"\n{indent}]"
            if comment_part:
                final_output += comment_part
            return final_output

    # 情況 B：標準 Dict 語法的美化（保持你原本的標準 Dict 靠齊邏輯，但優化防爆）
    if '=' in stmt:
        prefix, rhs = stmt.split('=', 1)
        prefix_str = prefix.strip() + " ="
    else:
        prefix_str, rhs = "", stmt

    match_dict = re.search(r'\{([\s\S]*?)\}', rhs)
    if not match_dict:
        return str(stmt)

    dict_content = match_dict.group(1)
    pairs = re.split(r',\s*(?=["\'\(\[a-zA-Z0-9_])', dict_content.strip())
    formatted_pairs = []
    max_key_len = 0
    parsed_pairs = []

    for pair in pairs:
        if ':' in pair:
            key, val = pair.split(':', 1)
            max_key_len = max(max_key_len, len(key.strip()))
            parsed_pairs.append((key.strip(), val.strip()))
        elif pair.strip():
            parsed_pairs.append((pair.strip(), None))

    dict_indent = indent + "    "
    for key, val in parsed_pairs:
        if val is not None:
            formatted_pairs.append(
                f"{dict_indent}{key:<{max_key_len}} : {val},")
        else:
            formatted_pairs.append(f"{dict_indent}{key},")

    return f"{indent}{prefix_str} {{\n" + "\n".join(formatted_pairs) + f"\n{indent}}}"
