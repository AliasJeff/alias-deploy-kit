"""
streamlit run app_ratio.py
"""
import streamlit as st
import json
import base64
from copy import deepcopy

DEFAULT_RATIO_TEMPLATES = ["16:9", "1:1", "1:2", "2:1", "4:3"]


def merge_tokens(original: str, additions):
    tokens = original.split()
    s = []
    seen = set()
    for t in tokens:
        if t not in seen:
            s.append(t)
            seen.add(t)
    for a in additions:
        if a not in seen:
            s.append(a)
            seen.add(a)
    return " ".join(s)


def compute_beautify_additions(tokens_set):
    add = []
    # 容器适配
    if 'min-h-screen' in tokens_set and 'flex' in tokens_set:
        add += ['md:mx-auto', 'md:max-w-4xl']
    # 网格适配
    if 'grid-cols-2' in tokens_set:
        add += [
            'md:grid-cols-3',
            'lg:grid-cols-[repeat(auto-fit,minmax(220px,1fr))]'
        ]
    # 卡片约束
    if 'rounded-lg' in tokens_set and 'p-4' in tokens_set:
        add += ['max-w-sm', 'w-full', 'mx-auto']
    # 文本改善
    if any(t.startswith('text-') for t in tokens_set):
        add += ['leading-relaxed']
    return list(dict.fromkeys(add))


def get_ratio_tag(w, h):
    ratio = w / h
    if 0.9 <= ratio <= 1.4: return "ratio-4-3"
    if ratio > 1.4: return "ratio-16-9"
    return "ratio-mobile"


def transform_node(node, ratio_tag=None, apply_beautify=True):
    if not isinstance(node, dict): return node
    node = deepcopy(node)

    if 'className' in node and isinstance(node['className'], str):
        orig = node['className']
        tokens_set = set(orig.split())

        final_classes = orig
        if apply_beautify:
            additions = compute_beautify_additions(tokens_set)
            final_classes = merge_tokens(final_classes, additions)
        if ratio_tag:
            final_classes = merge_tokens(final_classes, [ratio_tag])

        node['className'] = final_classes

    if 'children' in node and isinstance(node['children'], list):
        node['children'] = [
            transform_node(c, ratio_tag, apply_beautify)
            for c in node['children']
        ]
    return node


# --- 渲染引擎部分 ---


def json_to_html(node):
    if not isinstance(node, dict): return ""
    tag = node.get("name", "div")
    class_name = node.get("className", "")
    params = node.get("params", {})
    children = node.get("children", [])

    attr_parts = [f'class="{class_name}"'] if class_name else []
    inner_text = ""
    for key, value in params.items():
        if key == "textContent": inner_text = value
        else: attr_parts.append(f'{key}="{value}"')

    attrs_str = " ".join(attr_parts)
    child_content = "".join([json_to_html(child) for child in children])
    return f"<{tag} {attrs_str}>{inner_text}{child_content}</{tag}>"


def wrap_in_template(body_content, width, height, show_frame):
    canvas_style = f"""
        body {{ background-color: #f3f4f6; display: flex; justify-content: center; padding: 20px 0; margin: 0; }}
        .phone-canvas {{
            width: {width}px; height: {height}px;
            background-color: white; overflow-y: auto; position: relative;
            { "box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25); border-radius: 20px;" if show_frame else "" }
        }}
    """
    return f"""<!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <script src="https://cdn.tailwindcss.com"></script>
            <style>{canvas_style}</style>
        </head>
        <body>
            <div class="phone-canvas">{body_content}</div>
        </body>
        </html>"""


# --- Streamlit 界面 ---

st.set_page_config(page_title="智能比例 UI 渲染器", layout="wide")
st.title("📱 自适应终端界面适配")

with st.sidebar:
    st.header("尺寸与比例设置")
    width = st.number_input("目标宽度 (px)", value=375)
    height = st.number_input("目标高度 (px)", value=812)

    st.divider()
    st.header("适配算法配置")
    apply_beautify = st.checkbox("启用智能美化规则", value=True)
    show_frame = st.checkbox("显示模拟器边框", value=True)

    uploaded_file = st.file_uploader("上传 UI JSON 文件", type=["json"])

if uploaded_file:
    try:
        raw_json = json.load(uploaded_file)

        # 执行适配逻辑
        ratio_tag = get_ratio_tag(width, height)
        adapted_json = transform_node(raw_json,
                                      ratio_tag=ratio_tag,
                                      apply_beautify=apply_beautify)

        # 转换为 HTML
        body_inner = json_to_html(adapted_json)
        final_html = wrap_in_template(body_inner, width, height, show_frame)

        # --- 操作区 ---
        st.success(f"已自动匹配布局模式: **{ratio_tag}**")

        col1, col2, col3 = st.columns([2, 2, 4])

        # 预览按钮
        b64_html = base64.b64encode(final_html.encode()).decode()
        preview_href = f'<a href="data:text/html;base64,{b64_html}" target="_blank" style="text-decoration:none;"><button style="width:100%; background-color:#3b82f6; color:white; padding:10px; border:none; border-radius:8px; cursor:pointer;">🚀 新标签页预览</button></a>'
        col1.markdown(preview_href, unsafe_allow_html=True)

        # 下载按钮 (保持原逻辑)
        col2.download_button(label="📥 下载适配后的 HTML",
                             data=final_html,
                             file_name=f"ui_{width}x{height}.html",
                             mime="text/html",
                             use_container_width=True)

        # 实时预览
        st.divider()
        st.components.v1.html(final_html, height=height + 100, scrolling=True)

        # 可视化 JSON 结果 (可选)
        with st.expander("查看适配后的 JSON 结构"):
            st.json(adapted_json)

    except Exception as e:
        st.error(f"处理失败: {e}")
else:
    st.info("请在侧边栏上传 JSON 文件。系统将根据你设定的宽高比例自动应用美化规则。")
