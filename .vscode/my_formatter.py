import sys
import re
import tokenize
import io


def custom_format(code: str) -> str:
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except Exception:
        return code

    lines_map = {}
    current_stmt_id = 0
    paren_depth = 0

    # --- 階段 1：Token 語法樹拉平語句 ---
    for tok in tokens:
        toktype = tok.type
        tokval = tok.string

        if toktype in (tokenize.ENDMARKER, tokenize.ERRORTOKEN):
            continue

        if tokval in ('(', '[', '{'):
            paren_depth += 1
        elif tokval in (')', ']', '}'):
            paren_depth = max(0, paren_depth - 1)

        if toktype in (tokenize.NEWLINE, tokenize.NL):
            if paren_depth > 0:
                continue
            else:
                current_stmt_id += 1
                continue

        if toktype == tokenize.INDENT:
            continue

        if current_stmt_id not in lines_map:
            # 修正重點：tok.start 是一個 tuple (line, column)，我們只需要欄位號 [1] 來做縮排
            lines_map[current_stmt_id] = {'indent': tok.start[1], 'tokens': []}

        lines_map[current_stmt_id]['tokens'].append((toktype, tokval))

    # --- 階段 2：拼裝並徹底「靠緊」 ---
    processed_statements = []

    for stmt_id, data in sorted(lines_map.items()):
        indent_str = " " * data['indent']
        tok_list = data['tokens']

        if not tok_list:
            continue

        stmt_str = ""
        prev_type = None
        prev_val = ""

        for t_type, t_val in tok_list:
            if t_type == tokenize.COMMENT:
                stmt_str += "  " + t_val
                continue

            if stmt_str:
                if (t_val in ('(', ')', '[', ']', '{', '}', '.', ',', ':', ';') or
                    prev_val in ('(', '[', '{', '.', ':', 'utf', 'encoding') or
                        t_type == tokenize.NUMBER or prev_type == tokenize.NUMBER):

                    if t_val in ('==', '!=', '&', '|', '=', '+', '-'):
                        stmt_str += " " + t_val
                    elif prev_val in ('==', '!=', '&', '|', '=', '+', '-'):
                        stmt_str += " " + t_val
                    else:
                        stmt_str += t_val
                else:
                    stmt_str += " " + t_val
            else:
                stmt_str += t_val

            prev_type = t_type
            prev_val = t_val

        # 篩選單純賦值語句（排除控制流與多重賦值）
        if '=' in stmt_str and '==' not in stmt_str and not stmt_str.startswith(('def ', 'return ', 'print(')) and stmt_str.count('=') == 1:
            parts = stmt_str.split('=', 1)
            left = parts[0].strip()
            right = parts[1].strip()

            # 優化右側位元與比較運算子間距
            right = re.sub(r'\s*==\s*', ' == ', right)
            right = re.sub(r'\s*!=\s*', ' != ', right)
            right = re.sub(r'\s*&\s*', ' & ', right)
            right = re.sub(r'\s*\|\s*', ' | ', right)

            processed_statements.append((indent_str, left, right))
        else:
            processed_statements.append(indent_str + stmt_str)

    # --- 階段 3：連續主等號「垂直對齊（左邊 = 右邊）」 ---
    final_lines = []
    block = []

    def flush_block(current_block):
        if not current_block:
            return
        if len(current_block) == 1:
            indent, left, right = current_block[0]
            final_lines.append(f"{indent}{left} = {right}")
            return

        max_left_len = max(len(item[1]) for item in current_block)
        for indent, left, right in current_block:
            final_lines.append(f"{indent}{left:<{max_left_len}} = {right}")

    for item in processed_statements:
        if isinstance(item, tuple):
            block.append(item)
        else:
            flush_block(block)
            block = []
            final_lines.append(item)
    flush_block(block)

    return "\n".join(final_lines) + "\n"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
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
