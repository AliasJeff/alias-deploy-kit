import json
import os
import matplotlib.pyplot as plt
import numpy as np

TARGET_JSON_FILE = "benchmark_results/new/2__4090-24g.json"

IMG_LATENCY = "benchmark_latency.png"
IMG_THROUGHPUT_SPLIT = "benchmark_throughput_split.png"

plt.rcParams['font.sans-serif'] = [
    'SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'DejaVu Sans', 'SimSun'
]
plt.rcParams['axes.unicode_minus'] = False


def load_and_sort_data(file_path):
    """读取 JSON 并按 model_name 升序排序"""
    if not os.path.exists(file_path):
        print(f"❌ 错误: 找不到文件 '{file_path}'")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if "results" in data and isinstance(data["results"], list):
            data["results"].sort(key=lambda x: x.get("model_name", ""))
            print(f"📂 成功加载 {len(data['results'])} 个模型，并已按名称升序排序。")
            return data
        else:
            print("⚠️ JSON 结构不正确：缺少 'results' 列表")
            return None
    except Exception as e:
        print(f"❌ 读取错误: {e}")
        return None


def get_metric_item(metrics_list, match_dict):
    """辅助函数：在 list-of-dicts 中查找符合条件的项"""
    if not metrics_list: return None
    for item in metrics_list:
        is_match = True
        for k, v in match_dict.items():
            if item.get(k) != v:
                is_match = False
                break
        if is_match: return item
    return None


def plot_latency(data):
    """
    图1: Latency (Prefill & Decode)
    ✅ 纵坐标: Tokens/s
    """
    results = data.get("results", [])
    if not results: return

    models = [r.get('model_name', 'Unnamed') for r in results]

    # 1. 获取 X 轴 (Prompt Lens)
    try:
        # 尝试从 meta 配置读取，以便发现缺失的测试点 (OOM)
        target_prompt_lens = sorted(
            data['meta']['config']['latency_test']['prompt_lens'])
    except (KeyError, TypeError):
        # Fallback: 如果 meta 缺失，从结果中扫描
        all_lens = set()
        for r in results:
            for m in r.get('single_latency_metrics', []):
                all_lens.add(m['prompt_len'])
        target_prompt_lens = sorted(list(all_lens))

    x = np.arange(len(target_prompt_lens))
    width = 0.8 / len(models)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 配置: (JSON字段, 标题, subplot)
    metrics_config = [
        ("prefill_tokens_per_s", "Single Request - Prefill Speed", axes[0]),
        ("decode_tokens_per_s", "Single Request - Decode Speed", axes[1])
    ]

    for key, title, ax in metrics_config:
        for i, model in enumerate(models):
            metrics = results[i].get('single_latency_metrics', [])
            y_vals, labels = [], []

            for p_len in target_prompt_lens:
                item = get_metric_item(metrics, {'prompt_len': p_len})
                if item:
                    if item.get('error'):
                        y_vals.append(0)
                        labels.append("Error")
                    else:
                        val = item.get(key, 0)
                        y_vals.append(val)
                        labels.append(f"{int(val)}")
                else:
                    y_vals.append(0)
                    labels.append("×\nOOM")

            offset = (i - len(models) / 2) * width + width / 2
            bars = ax.bar(x + offset, y_vals, width, label=model)

            for idx, bar in enumerate(bars):
                label = labels[idx]
                h = bar.get_height()
                x_pos = bar.get_x() + bar.get_width() / 2
                if h == 0 or "OOM" in label or "Error" in label:
                    ax.text(x_pos,
                            0.1,
                            label,
                            ha='center',
                            va='bottom',
                            color='red',
                            fontweight='bold',
                            fontsize=9)
                else:
                    ax.text(x_pos,
                            h,
                            label,
                            ha='center',
                            va='bottom',
                            fontsize=8)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Input Length')
        ax.set_ylabel('Tokens/s')
        ax.set_xticks(x)
        ax.set_xticklabels(target_prompt_lens)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        if key == "prefill_tokens_per_s":
            ax.legend()

    plt.tight_layout()
    plt.savefig(IMG_LATENCY, dpi=300)
    print(f"Latency 图表已保存: {IMG_LATENCY}")


def plot_throughput_breakdown(data):
    """
    图2: Throughput 详情
    """
    results = data.get("results", [])
    if not results:
        return

    models = [r.get('model_name') for r in results]

    try:
        tp_config = data['meta']['config']['throughput_test']
        in_lens = sorted(tp_config['prompt_lens'])
        out_lens = sorted(tp_config['new_tokens'])
    except:
        in_s, out_s = set(), set()
        for r in results:
            for m in r.get('max_throughput_metrics', []):
                in_s.add(m['input_len'])
                out_s.add(m['output_len'])
        in_lens, out_lens = sorted(list(in_s)), sorted(list(out_s))

    n_cols = len(out_lens)
    if n_cols == 0:
        return

    fig, axes = plt.subplots(3, n_cols, figsize=(max(6, 5 * n_cols), 12))

    if n_cols == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]]])

    x = np.arange(len(in_lens))
    width = 0.8 / len(models)

    rows_config = [{
        "type": "calc_prefill",
        "label": "Prefill RPS (Batch Size / Prefill Lat)",
        "row": 0
    }, {
        "type": "calc_decode",
        "label": "Decode RPS (Batch Size / Decode Lat)",
        "row": 1
    }, {
        "type": "raw_rps",
        "label": "Total RPS",
        "row": 2
    }]

    for cfg in rows_config:
        row_idx = cfg['row']
        row_type = cfg['type']

        for col_idx, out_len in enumerate(out_lens):
            ax = axes[row_idx][col_idx]

            for i, model in enumerate(models):
                metrics = results[i].get('max_throughput_metrics', [])
                y_vals, labels = [], []

                for in_len in in_lens:
                    item = get_metric_item(metrics, {
                        'input_len': in_len,
                        'output_len': out_len
                    })

                    if item:
                        if item.get('error'):
                            y_vals.append(0)
                            labels.append(("Error", True))
                        else:
                            val = 0
                            if row_type == "calc_prefill":
                                mbs = item.get('max_batch_size', 0)
                                lat = item.get('prefill_latency', 0)
                                if lat > 0:
                                    val = mbs / lat
                            elif row_type == "calc_decode":
                                mbs = item.get('max_batch_size', 0)
                                lat = item.get('decode_latency', 0)
                                if lat > 0:
                                    val = mbs / lat
                            elif row_type == "raw_rps":
                                val = item.get('rps', 0)

                            is_oom = (val == 0
                                      and item.get('max_batch_size', 0) == 0)

                            y_vals.append(val)
                            fmt_val = f"{int(val)}" if val > 10 else f"{val:.1f}"
                            labels.append(
                                ("OOM" if is_oom else fmt_val, is_oom))
                    else:
                        y_vals.append(0)
                        labels.append(("×\nOOM", True))

                offset = (i - len(models) / 2) * width + width / 2
                bars = ax.bar(x + offset, y_vals, width, label=model)

                for b_idx, bar in enumerate(bars):
                    text, is_err = labels[b_idx]
                    h = bar.get_height()
                    x_pos = bar.get_x() + bar.get_width() / 2

                    if is_err or h == 0:
                        ax.text(x_pos,
                                0.1,
                                text,
                                ha='center',
                                va='bottom',
                                color='red',
                                fontweight='bold',
                                fontsize=8)
                    else:
                        ax.text(x_pos,
                                h,
                                text,
                                ha='center',
                                va='bottom',
                                fontsize=7)

            ax.set_title(f"{cfg['label']} (GenLen={out_len})",
                         fontsize=11,
                         fontweight='bold')

            ax.set_xticks(x)
            ax.set_xticklabels(in_lens)
            ax.grid(axis='y', linestyle='--', alpha=0.3)

            # 只有最后一行显示 X 轴标签
            if row_idx == 2:
                ax.set_xlabel('Input Length')

            # 第一列显示 Y 轴单位
            if col_idx == 0:
                ax.set_ylabel('RPS')

            # 图例仅在第一张图显示
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='upper right', fontsize='small', framealpha=0.9)

    plt.suptitle("Max Throughput Breakdown (Prefill vs Decode vs Total)",
                 fontsize=16,
                 y=0.98)
    plt.tight_layout()
    plt.savefig(IMG_THROUGHPUT_SPLIT, dpi=300)
    print(f"✅ Throughput 分解图已保存: {IMG_THROUGHPUT_SPLIT}")


if __name__ == "__main__":
    print(f"🚀 开始处理文件: {TARGET_JSON_FILE}")

    data = load_and_sort_data(TARGET_JSON_FILE)

    if data:
        plot_latency(data)
        plot_throughput_breakdown(data)

        print("\n🎉 所有任务完成！请查看生成的 .png 图片。")
    else:
        print("⚠️ 程序终止。")
