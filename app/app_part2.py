st.set_page_config(page_title="AI UI 生成器", page_icon="📱", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans SC', sans-serif; }
.main-title {
    font-size:2rem; font-weight:700;
    background:linear-gradient(135deg,#6366f1,#06b6d4);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.sub-title { color:#6b7280; font-size:0.95rem; margin-bottom:1.5rem; }
.ok-box  { background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px;
           padding:0.75rem 1rem; color:#166534; font-size:0.9rem; }
.err-box { background:#fef2f2; border:1px solid #fecaca; border-radius:10px;
           padding:0.75rem 1rem; color:#991b1b; font-size:0.9rem; }
.warn-box{ background:#fffbeb; border:1px solid #fde68a; border-radius:10px;
           padding:0.75rem 1rem; color:#92400e; font-size:0.9rem; }
</style>
""", unsafe_allow_html=True)

st.markdown(\'<div class="main-title">📱 AI 界面生成器</div>\', unsafe_allow_html=True)
st.markdown(\'<div class="sub-title">输入页面需求，由 Qwen3-0.6B 微调模型（vLLM 加速）自动生成 UI 并实时渲染</div>\',
            unsafe_allow_html=True)

server_ok = check_server()
if not server_ok:
    st.markdown(
        \'<div class="warn-box">⚠️ vLLM 推理服务未启动，请先运行：\'
        \'<code>python3 /root/autodl-tmp/vllm_server.py</code></div>\',
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.header("🖼️ 画布设置")
    width  = st.number_input("目标宽度 (px)", value=375, min_value=320, max_value=1920)
    height = st.number_input("目标高度 (px)", value=812, min_value=480, max_value=1920)
    st.divider()
    st.header("⚙️ 适配配置")
    apply_beautify = st.checkbox("启用智能美化规则", value=True)
    show_frame     = st.checkbox("显示模拟器边框",   value=True)
    if st.session_state.get("raw_output"):
        st.divider()
        st.caption("修改画布尺寸或适配配置后，点击下方按钮重新渲染（无需重新生成）：")
        apply_changes_btn = st.button(
            "🔄 应用更改",
            use_container_width=True,
            help="使用当前画布/适配设置重新渲染已生成的 JSON，不重新调用模型",
        )
    else:
        apply_changes_btn = False
    st.divider()
    st.header("🎛️ 生成参数")
    max_new_tokens     = st.slider("最大 Token 数",  256, 4096, 2048, 64)
    temperature        = st.slider("Temperature",   0.0,  1.0,  0.1, 0.05)
    top_p              = st.slider("Top-p",         0.5,  1.0,  0.9, 0.05)
    repetition_penalty = st.slider("重复惩罚",       1.0,  1.5,  1.1, 0.05)
    st.divider()
    status_str = "🟢 在线" if server_ok else "🔴 离线"
    st.caption(f"vLLM 服务状态：{status_str}")
    st.caption(f"API 地址：{VLLM_API}")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("✏️ 输入需求")
    st.caption("快捷示例：")
    examples = [
        "设计商品首页，意图：浏览商品",
        "设计搜索页面，意图：输入关键词",
        "设计购物车页面，意图：管理已选商品（增/删）",
        "设计商品详情页面，意图：立即购买",
        "设计订单确认页面，意图：选择收货地址",
        "设计登录页面，意图：用户登录",
    ]
    btn_cols = st.columns(2)
    for i, ex in enumerate(examples):
        if btn_cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state["instruction"] = ex

    instruction = st.text_area(
        "页面描述 / 指令",
        value=st.session_state.get("instruction", ""),
        height=120,
        placeholder="例如：设计商品首页，意图：浏览商品",
    )

    generate_btn = st.button(
        "🚀 生成 UI",
        type="primary",
        use_container_width=True,
        disabled=(not instruction.strip() or not server_ok),
    )

    if st.session_state.get("raw_output"):
        with st.expander("📄 模型原始输出（JSON）", expanded=False):
            st.code(st.session_state["raw_output"], language="json")

with col_right:
    st.subheader("👁️ 实时预览")
    status_slot  = st.empty()
    action_slot  = st.empty()
    divider_slot = st.empty()
    preview_slot = st.empty()

    if st.session_state.get("final_html"):
        final_html = st.session_state["final_html"]
        ratio_tag  = st.session_state.get("ratio_tag", "")
        elapsed    = st.session_state.get("elapsed", 0)
        status_slot.markdown(
            f\'<div class="ok-box">✅ 渲染成功 · 布局：<strong>{ratio_tag}</strong>\'
            f\' · 耗时：<strong>{elapsed:.2f}s</strong> · 引擎：vLLM</div>\',
            unsafe_allow_html=True,
        )
        b64 = base64.b64encode(final_html.encode()).decode()
        c1, c2 = action_slot.columns(2)
        c1.markdown(
            f\'<a href="data:text/html;base64,{b64}" target="_blank" style="text-decoration:none;">\'
            f\'<button style="width:100%;background:#6366f1;color:white;padding:10px 0;\'
            f\'border:none;border-radius:8px;cursor:pointer;font-size:14px;">🔗 新标签页预览</button></a>\',
            unsafe_allow_html=True,
        )
        c2.download_button(
            "📥 下载 HTML",
            data=final_html,
            file_name=f"ui_{width}x{height}.html",
            mime="text/html",
            use_container_width=True,
        )
        divider_slot.divider()
        with preview_slot:
            st.components.v1.html(final_html, height=height + 60, scrolling=True)
        with st.expander("🗂️ 适配后 JSON 结构"):
            st.json(st.session_state.get("adapted_json", {}))
    else:
        preview_slot.info("生成结果将在此处实时显示")

# ── 应用更改触发 ──────────────────────────────────────────────────────────────
if apply_changes_btn and st.session_state.get("raw_output"):
    try:
        final_html, adapted, ratio_tag = rebuild_html(
            st.session_state["raw_output"],
            width, height, show_frame, apply_beautify,
        )
        st.session_state["final_html"]   = final_html
        st.session_state["adapted_json"] = adapted
        st.session_state["ratio_tag"]    = ratio_tag
        st.rerun()
    except json.JSONDecodeError as e:
        st.error(f"JSON 解析失败，无法应用更改：{e}")
    except Exception as e:
        st.error(f"应用更改失败：{e}")

# ── 生成触发 ──────────────────────────────────────────────────────────────────
if generate_btn and instruction.strip() and server_ok:
    st.session_state["instruction"] = instruction.strip()
    with col_right:
        with st.spinner("🤖 vLLM 推理中，请稍候..."):
            try:
                raw_output, elapsed = generate_ui_json(
                    instruction=instruction.strip(),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
                final_html, adapted, ratio_tag = rebuild_html(
                    raw_output, width, height, show_frame, apply_beautify
                )
                st.session_state["raw_output"]   = raw_output
                st.session_state["final_html"]   = final_html
                st.session_state["adapted_json"] = adapted
                st.session_state["ratio_tag"]    = ratio_tag
                st.session_state["elapsed"]      = elapsed
                st.rerun()
            except json.JSONDecodeError as e:
                st.session_state["raw_output"] = raw_output
                st.error(
                    f"JSON 解析失败。\n\n**错误：** {e}\n\n"
                    f"**原始输出（前500字）：**\n```\n{raw_output[:500]}\n```"
                )
            except Exception as e:
                st.error(f"生成失败：{e}")
