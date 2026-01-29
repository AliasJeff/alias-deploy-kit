import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

# ================= 配置部分 =================
# 设置字体以支持中文显示
# Windows 用户通常使用 'SimHei'
# macOS 用户通常使用 'Arial Unicode MS' 或 'PingFang SC'
# Linux 用户请替换为系统可用的支持中文的字体
plt.rcParams['font.sans-serif'] = [
    'SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'DejaVu Sans'
]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def load_all_json_results(base_dir="."):
    """
    扫描指定目录下的所有 .json 文件，合并 'results' 字段
    """
    json_files = glob.glob(os.path.join(base_dir, "*.json"))

    if not json_files:
        print(f"❌ 在目录 '{base_dir}' 下没有找到 .json 文件！")
        return []

    print(
        f"📂 发现 {len(json_files)} 个 JSON 文件: {[os.path.basename(f) for f in json_files]}"
    )

    merged_results = []

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # 兼容性检查：确保 JSON 结构里有 "results"
                if "results" in data and isinstance(data["results"], list):
                    # 可选：可以在这里打印读取到的模型名称
                    for res in data["results"]:
                        print(
                            f"   -> 读取模型: {res.get('model_name', 'Unknown')}")
                    merged_results.extend(data["results"])
                else:
                    print(
                        f"⚠️  跳过文件 {os.path.basename(file_path)}: 缺少 'results' 字段或格式不正确"
                    )

        except json.JSONDecodeError:
            print(f"❌ 无法解析 JSON: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"❌ 读取错误 {os.path.basename(file_path)}: {e}")

    # 按模型名称排序，保证画图顺序一致
    merged_results.sort(key=lambda x: x.get("model_name", ""))
    return merged_results


def plot_latency(results):
    """画单请求 Latency (Prefill & Decode)"""
    if not results: return

    models = [r['model_name'] for r in results]
    # 假设所有模型测试的 input_len 是一样的，取第一个作为 X 轴刻度
    try:
        prompt_lens = [
            m['prompt_len'] for m in results[0]['single_latency_metrics']
        ]
    except IndexError:
        print("❌ 数据格式错误：single_latency_metrics 为空")
        return

    # 准备数据容器
    prefill_data = {model: [] for model in models}
    decode_data = {model: [] for model in models}

    for r in results:
        name = r['model_name']
        metrics = r.get('single_latency_metrics', [])
        for m in metrics:
            prefill_data[name].append(m['prefill_tokens_per_s'])
            decode_data[name].append(m['decode_tokens_per_s'])

    x = np.arange(len(prompt_lens))
    total_width = 0.8
    n_bars = len(models)
    width = total_width / n_bars

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- 图1: Prefill Speed ---
    ax = axes[0]
    for i, model in enumerate(models):
        offset = (i - n_bars / 2) * width + width / 2
        # 增加容错：防止某个模型数据缺失导致长度不一致
        if len(prefill_data[model]) == len(x):
            bars = ax.bar(x + offset, prefill_data[model], width, label=model)
            # 标数值
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center',
                            va='bottom',
                            fontsize=8,
                            rotation=45)

    ax.set_ylabel('Speed (Tokens/s)')
    ax.set_xlabel('Input Length (Prompt)')
    ax.set_title('Single Request - Prefill Speed (Higher is Better)',
                 fontsize=12,
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(prompt_lens)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # --- 图2: Decode Speed ---
    ax = axes[1]
    for i, model in enumerate(models):
        offset = (i - n_bars / 2) * width + width / 2
        if len(decode_data[model]) == len(x):
            bars = ax.bar(x + offset, decode_data[model], width, label=model)
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center',
                            va='bottom',
                            fontsize=8)

    ax.set_ylabel('Speed (Tokens/s)')
    ax.set_xlabel('Input Length (Prompt)')
    ax.set_title('Single Request - Decode Speed (Higher is Better)',
                 fontsize=12,
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(prompt_lens)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    output_filename = 'benchmark_latency_summary.png'
    plt.savefig(output_filename, dpi=300)
    print(f"✅ Latency 图表已保存: {output_filename}")
    # plt.show() # 如果在服务器运行，请注释掉此行


def plot_throughput(results):
    """画吞吐量 RPS，按 Output Len 分图"""
    if not results: return

    models = [r['model_name'] for r in results]

    # 获取所有的 input_lens 和 output_lens (基于第一个有效数据)
    try:
        sample_metrics = results[0]['max_throughput_metrics']
        input_lens = sorted(list(set(m['input_len'] for m in sample_metrics)))
        output_lens = sorted(list(set(m['output_len']
                                      for m in sample_metrics)))
    except (IndexError, KeyError):
        print("❌ 数据格式错误：max_throughput_metrics 为空")
        return

    # 动态创建子图
    n_cols = len(output_lens)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
    if n_cols == 1: axes = [axes]

    x = np.arange(len(input_lens))
    total_width = 0.8
    n_bars = len(models)
    width = total_width / n_bars

    for idx, out_len in enumerate(output_lens):
        ax = axes[idx]

        for i, model_obj in enumerate(results):
            model_name = model_obj['model_name']
            rps_values = []

            metrics = model_obj.get('max_throughput_metrics', [])

            for in_len in input_lens:
                # 查找匹配的记录
                val = next((item['rps']
                            for item in metrics if item['input_len'] == in_len
                            and item['output_len'] == out_len), 0)
                rps_values.append(val)

            offset = (i - n_bars / 2) * width + width / 2
            bars = ax.bar(x + offset, rps_values, width, label=model_name)

            # 标数值
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{height:.1f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center',
                                va='bottom',
                                fontsize=8)

        ax.set_title(f'Max Throughput (Gen Len = {out_len})',
                     fontsize=12,
                     fontweight='bold')
        ax.set_xlabel('Input Length')
        ax.set_ylabel('RPS (Requests/s)')
        ax.set_xticks(x)
        ax.set_xticklabels(input_lens)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        # 图例只在第一张图显示，防止遮挡
        if idx == 0:
            ax.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    output_filename = 'benchmark_throughput_summary.png'
    plt.savefig(output_filename, dpi=300)
    print(f"✅ Throughput 图表已保存: {output_filename}")
    # plt.show()


if __name__ == "__main__":
    print("🚀 开始读取并绘制 Benchmark 结果...")

    # 1. 读取数据
    all_results = load_all_json_results(".")

    if all_results:
        # 2. 绘图
        plot_latency(all_results)
        plot_throughput(all_results)
        print("\n🎉 所有任务完成！请查看生成的 .png 图片。")
    else:
        print("⚠️ 未能获取有效数据，终止绘图。")
