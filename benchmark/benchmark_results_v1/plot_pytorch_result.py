import json
import os
import matplotlib.pyplot as plt
import numpy as np
import colorsys

TARGET_JSON_FILE = "benchmark_results_v1/new/2__4090-24g.json"  # 请确保路径正确
IMG_LATENCY = "benchmark_latency_qwen.png"
IMG_THROUGHPUT_SPLIT = "benchmark_throughput_qwen.png"

# 设置中文字体，防止乱码
plt.rcParams['font.sans-serif'] = [
    'SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'DejaVu Sans'
]
plt.rcParams['axes.unicode_minus'] = False


def load_and_sort_data(file_path):
    """读取 JSON 并按 model_name 排序"""
    if not os.path.exists(file_path):
        print(f"❌ 错误: 找不到文件 '{file_path}'")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if "results" in data and isinstance(data["results"], list):
            # 按名称排序，保证图表整齐
            data["results"].sort(key=lambda x: x.get("model_name", ""))
            return data
        return None
    except Exception as e:
        print(f"❌ 读取错误: {e}")
        return None


def get_metric_item(metrics_list, match_dict):
    """在列表中查找匹配项"""
    if not metrics_list: return None
    for item in metrics_list:
        if all(item.get(k) == v for k, v in match_dict.items()):
            return item
    return None


def generate_model_colors(model_names):
    """
    核心修改：针对 Qwen 系列的命名逻辑生成颜色
    逻辑：取前两段 (如 'Qwen3-4B') 作为家族前缀。
    """

    # --- 修改点：自定义前缀提取逻辑 ---
    def get_prefix(name):
        parts = name.split('-')
        # 如果名字里有横杠，取前两段组合 (例如: Qwen3-4B)
        # 这样 Qwen3-4B 和 Qwen3-1.7B 就会被视为不同的家族，分配不同的色相
        if len(parts) >= 2:
            return f"{parts[0]}-{parts[1]}"
        return parts[0]

    # -------------------------------

    # 1. 找出所有唯一的家族 (例如 Qwen3-4B, Qwen3-1.7B, Qwen2-1.5B)
    unique_families = sorted(list(set(get_prefix(m) for m in model_names)))
    family_count = len(unique_families)

    color_map = {}

    print(f"🎨 检测到的模型家族 (将分配不同色系): {unique_families}")

    for fam_idx, family in enumerate(unique_families):
        # 找出该家族的所有变体 (GPTQ, AWQ, Base)
        models_in_fam = [m for m in model_names if get_prefix(m) == family]

        # 将 Base 模型 (名字最短的) 放在中间或特定位置，或者直接按字母排
        # 这里默认按字母排序处理深浅
        models_in_fam.sort()
        count = len(models_in_fam)

        # 分配色相 (Hue): 0~1 之间均匀分布
        hue = fam_idx / max(family_count, 1)
        # 可以稍微偏移一点避免首尾颜色太像
        if family_count > 1:
            hue = (fam_idx * (1.0 / family_count)) % 1.0

        for m_idx, model in enumerate(models_in_fam):
            # 亮度 (Lightness) 变化:
            # 同一色系下，通过深浅区分 GPTQ / AWQ / Base
            if count > 1:
                # 范围从 0.35 (深) 到 0.75 (浅)
                lightness = 0.35 + (0.4 * (m_idx / (count - 1)))
            else:
                lightness = 0.5

            # 饱和度 (Saturation)
            saturation = 0.65

            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            color_map[model] = rgb

    return color_map


def plot_latency(data):
    results = data.get("results", [])
    if not results: return

    models = [r.get('model_name', 'Unnamed') for r in results]
    model_colors = generate_model_colors(models)  # 生成颜色

    try:
        target_prompt_lens = sorted(
            data['meta']['config']['latency_test']['prompt_lens'])
    except:
        all_lens = set()
        for r in results:
            for m in r.get('single_latency_metrics', []):
                all_lens.add(m['prompt_len'])
        target_prompt_lens = sorted(list(all_lens))

    x = np.arange(len(target_prompt_lens))
    width = 0.8 / len(models)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    metrics_config = [
        ("prefill_tokens_per_s", "Prefill Speed (Tokens/s)", axes[0]),
        ("decode_tokens_per_s", "Decode Speed (Tokens/s)", axes[1])
    ]

    for key, title, ax in metrics_config:
        for i, model in enumerate(models):
            metrics = results[i].get('single_latency_metrics', [])
            y_vals, labels = [], []

            for p_len in target_prompt_lens:
                item = get_metric_item(metrics, {'prompt_len': p_len})
                val = 0
                label = ""
                if item:
                    if item.get('error'):
                        label = "Error"
                    else:
                        val = item.get(key, 0)
                        label = f"{int(val)}"
                else:
                    label = "OOM"

                y_vals.append(val)
                labels.append(label)

            offset = (i - len(models) / 2) * width + width / 2

            # 使用生成的颜色
            c = model_colors.get(model, (0.5, 0.5, 0.5))
            bars = ax.bar(x + offset, y_vals, width, label=model, color=c)

            for idx, bar in enumerate(bars):
                txt = labels[idx]
                h = bar.get_height()
                # 标注位置
                text_y = h if h > 0 else 0
                color_txt = 'red' if txt in ["OOM", "Error"] else 'black'
                va = 'bottom'

                # 如果是 OOM 且高度为0，稍微标高一点
                if h == 0: text_y = ax.get_ylim()[1] * 0.02

                ax.text(bar.get_x() + bar.get_width() / 2,
                        text_y,
                        txt,
                        ha='center',
                        va=va,
                        fontsize=8,
                        color=color_txt)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(target_prompt_lens)
        ax.grid(axis='y', alpha=0.3)
        if key == "prefill_tokens_per_s":
            ax.legend(fontsize='small')

    plt.tight_layout()
    plt.savefig(IMG_LATENCY, dpi=300)
    print(f"✅ Latency 图已保存: {IMG_LATENCY}")


def plot_throughput_breakdown(data):
    results = data.get("results", [])
    if not results: return
    models = [r.get('model_name') for r in results]
    model_colors = generate_model_colors(models)

    # 获取输入输出长度配置
    in_s, out_s = set(), set()
    for r in results:
        for m in r.get('max_throughput_metrics', []):
            in_s.add(m['input_len'])
            out_s.add(m['output_len'])
    in_lens, out_lens = sorted(list(in_s)), sorted(list(out_s))

    if not out_lens: return

    n_cols = len(out_lens)
    fig, axes = plt.subplots(3, n_cols, figsize=(max(6, 5 * n_cols), 12))
    if n_cols == 1: axes = np.array([[axes[0]], [axes[1]], [axes[2]]])

    x = np.arange(len(in_lens))
    width = 0.8 / len(models)

    rows = [("calc_prefill", "Prefill RPS", 0),
            ("calc_decode", "Decode RPS", 1), ("raw_rps", "Total RPS", 2)]

    for type_name, label_title, r_idx in rows:
        for c_idx, out_len in enumerate(out_lens):
            ax = axes[r_idx][c_idx]

            for i, model in enumerate(models):
                metrics = results[i].get('max_throughput_metrics', [])
                y_vals, labels = [], []

                for in_len in in_lens:
                    item = get_metric_item(metrics, {
                        'input_len': in_len,
                        'output_len': out_len
                    })
                    val = 0
                    txt = "OOM"
                    if item and not item.get('error'):
                        if type_name == "raw_rps":
                            val = item.get('rps', 0)
                        elif type_name == "calc_prefill":
                            lat = item.get('prefill_latency', 0)
                            if lat > 0:
                                val = item.get('max_batch_size', 0) / lat
                        elif type_name == "calc_decode":
                            lat = item.get('decode_latency', 0)
                            if lat > 0:
                                val = item.get('max_batch_size', 0) / lat

                        txt = f"{int(val)}" if val > 10 else f"{val:.1f}"
                        if val == 0 and item.get('max_batch_size') == 0:
                            txt = "OOM"

                    y_vals.append(val)
                    labels.append(txt)

                offset = (i - len(models) / 2) * width + width / 2
                c = model_colors.get(model, (0.5, 0.5, 0.5))
                bars = ax.bar(x + offset, y_vals, width, label=model, color=c)

                for b_idx, bar in enumerate(bars):
                    t = labels[b_idx]
                    h = bar.get_height()
                    clr = 'red' if t == "OOM" else 'black'
                    y_pos = h if h > 0 else 0.1
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            y_pos,
                            t,
                            ha='center',
                            va='bottom',
                            fontsize=7,
                            color=clr)

            ax.set_title(f"{label_title} (Out={out_len})",
                         fontsize=10,
                         fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(in_lens)
            ax.grid(axis='y', alpha=0.3)
            if r_idx == 0 and c_idx == 0:
                ax.legend(fontsize='x-small')

    plt.tight_layout()
    plt.savefig(IMG_THROUGHPUT_SPLIT, dpi=300)
    print(f"✅ Throughput 图已保存: {IMG_THROUGHPUT_SPLIT}")


if __name__ == "__main__":
    data = load_and_sort_data(TARGET_JSON_FILE)
    if data:
        plot_latency(data)
        plot_throughput_breakdown(data)
