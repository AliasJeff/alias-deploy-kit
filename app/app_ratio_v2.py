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
    # 修改点 1: 增加 overflow-x: hidden 防止内容超出模拟器尺寸
    canvas_style = f"""
        body {{ background-color: #f3f4f6; display: flex; justify-content: center; padding: 20px 0; margin: 0; overflow: hidden; }}
        .phone-canvas {{
            width: {width}px; height: {height}px;
            background-color: white; 
            overflow-y: auto; 
            overflow-x: hidden; /* 严格限制横向溢出 */
            position: relative;
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

st.set_page_config(page_title="智能比例 UI 批量渲染器", layout="wide")
st.title("📱 批量自适应终端界面预览")

with st.sidebar:
    st.header("尺寸与比例设置")
    width = st.number_input("目标宽度 (px)", value=375)
    height = st.number_input("目标高度 (px)", value=812)

    st.divider()
    st.header("适配算法配置")
    apply_beautify = st.checkbox("启用智能美化规则", value=True)
    show_frame = st.checkbox("显示模拟器边框", value=True)

    uploaded_file = st.file_uploader("上传批量 UI JSON 文件", type=["json"])

if uploaded_file:
    try:
        # 读取最外层 JSON 数组
        raw_data = json.load(uploaded_file)

        if not isinstance(raw_data, list):
            st.error("❌ 解析失败：上传的 JSON 文件不是数组格式。请确保最外层是 `[...]`。")
            st.stop()

        total_tasks = sum(
            len(model_item.get("results", [])) for model_item in raw_data
            if isinstance(model_item, dict))

        ratio_tag = get_ratio_tag(width, height)
        st.success(
            f"✅ 已自动匹配布局模式: **{ratio_tag}**，共检测到 **{total_tasks}** 个设计任务。")
        st.divider()

        task_counter = 1

        for model_item in raw_data:
            model_name = model_item.get("model", "未知模型")
            results = model_item.get("results", [])

            for result_item in results:
                prompt_text = result_item.get("question", "未命名界面设计")
                raw_response = result_item.get("response", "{}")

                st.markdown(f"### 🔖 任务 {task_counter} | 模型: `{model_name}`")
                st.markdown(f"**提示词:** {prompt_text}")

                # 先绘制左右两列结构
                col_json, col_ui = st.columns([1, 1], gap="large")

                # 左侧：无论解析是否成功，都尽力展示源代码
                with col_json:
                    st.caption("📄 JSON 原代码 (Response 节点)")
                    with st.container(height=height + 60):
                        try:
                            # 尝试以美化的 JSON 格式展示
                            display_json = json.loads(
                                raw_response) if isinstance(
                                    raw_response, str) else raw_response
                            st.json(display_json)
                        except Exception:
                            # 如果无法解析为标准 JSON，则直接输出原始字符串
                            st.text(raw_response)

                # 修改点 2: 右侧使用 try...except 拦截单个任务的渲染错误
                with col_ui:
                    st.caption("📱 UI 展示结果")
                    try:
                        # 尝试解析并渲染
                        raw_ui_node = json.loads(raw_response) if isinstance(
                            raw_response, str) else raw_response

                        adapted_json = transform_node(
                            raw_ui_node,
                            ratio_tag=ratio_tag,
                            apply_beautify=apply_beautify)
                        body_inner = json_to_html(adapted_json)
                        final_html = wrap_in_template(body_inner, width,
                                                      height, show_frame)

                        b64_html = base64.b64encode(
                            final_html.encode()).decode()
                        preview_href = f'<a href="data:text/html;base64,{b64_html}" target="_blank" style="text-decoration:none;"></a>'
                        st.markdown(preview_href, unsafe_allow_html=True)

                        st.components.v1.html(final_html,
                                              height=height + 80,
                                              scrolling=True)

                    except Exception as e:
                        # 如果单个节点解析或渲染发生崩溃，右侧显示报错，左侧原代码不受影响
                        st.error("❌ 渲染此 UI 节点时发生错误")
                        st.exception(e)

                st.divider()
                task_counter += 1

    except Exception as e:
        # 兜底：处理全局读取文件级别的错误（比如文件本身不是合法的 JSON）
        st.error(f"⚠️ 读取或解析全局文件时发生严重错误: {e}")
