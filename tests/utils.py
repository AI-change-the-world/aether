import re


def parse_person_info(text: str, as_tuple: bool = False):
    """
    将大模型输出的字符串解析为 dict 或 tuple，容错处理更健壮

    参数:
        text: 大模型返回的字符串
        as_tuple: 是否输出为 tuple，默认为 False（dict）

    返回:
        dict 或 tuple
    """

    # 目标字段
    fields = ["性别", "发型", "发色", "上身", "下身", "鞋子", "配饰", "特殊动作或状态"]

    # 默认值
    defaults = {
        "性别": "未知",
        "发型": "未知",
        "发色": "未知",
        "上身": "未知",
        "下身": "未知",
        "鞋子": "未知",
        "配饰": "无",
        "特殊动作或状态": "无",
    }

    result = defaults.copy()

    # 按行解析，兼容中英文冒号
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        # 用正则匹配“字段名 + 任意冒号 + 值”
        m = re.match(r"^(.*?)[:：]\s*(.*)$", line)
        if not m:
            continue

        key, value = m.group(1).strip(), m.group(2).strip()
        if key in fields and value:
            result[key] = value

    if as_tuple:
        return tuple(result[field] for field in fields)
    return result


def match_object(obj: dict, query: dict) -> bool:
    """
    判断对象是否匹配检索条件（模糊匹配 + 通配符版）

    规则：
    1. query 中的字段必须存在于 obj，否则 False
    2. query 的值必须明确（不能是 None/""/"未知"/"不确定"），否则 False
    3. obj 的值如果是 None/""/"未知"/"不确定" → 当作通配符，视为匹配
    4. 其余情况使用模糊匹配（忽略大小写，query 在 obj 中出现即可）
    """
    if not isinstance(obj, dict) or not isinstance(query, dict):
        raise ValueError("输入必须是 dict")

    invalid_values = {None, "", "未知", "不确定"}

    for key, q_value in query.items():
        # 条件 key 不存在
        if key not in obj:
            return False

        # query 的值必须明确
        if q_value in invalid_values:
            return False

        obj_value = obj[key]

        # obj 值是通配符 → 自动匹配
        if obj_value in invalid_values:
            continue

        # 转换为字符串做模糊匹配（忽略大小写）
        obj_str = str(obj_value).lower()
        query_str = str(q_value).lower()

        if query_str not in obj_str:  # 这里允许“短发”匹配“黑色短发”
            return False

    return True
