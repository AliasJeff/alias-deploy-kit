import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import colorsys  # 引入颜色处理库
from matplotlib.lines import Line2D

JSON_FILENAME = 'benchmark_results_v2/2__4090-24g.json'
OUTPUT_FILENAME_BASE = '2__4090-24g_.png'


def load_data(filename):
    """读取并清洗数据"""
    if not os.path.exists(filename):
        print(f"❌ 错误: 找不到文件 {filename}")
        return pd.DataFrame()

    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rows = []
    for entry in data:
        row = {}
        row['model'] = entry.get('model', 'Unknown')
        config = entry.get('config', {})
        row['batch_size'] = config.get('batch_size')

        row['seq_len'] = config.get('max_new_tokens') or config.get(
            'new_tokens', 0)

        if 'error' in entry and entry['error'] == 'OOM':
            row['is_oom'] = True
            row['gen_speed'] = None
            row['gpu_mem'] = None
            row['latency'] = None
        else:
            row['is_oom'] = False
            metrics = entry.get('metrics', {})
            row['gen_speed'] = metrics.get('tokens_per_second_gen')
            row['gpu_mem'] = metrics.get('gpu_mem_peak_gb')
            row['latency'] = metrics.get('total_time_s')

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        return df.sort_values(by=['model', 'batch_size', 'seq_len'])
    return df


def plot_charts_by_batch_size(df):
    """
    按 Batch Size 循环绘制图表。
    颜色：按模型家族区分色相(Hue)，同家族内亮度(Lightness)不同。
    线型：awq -> ':', gptq -> '--', 其他 -> '-'。
    """
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('ggplot')

    # 获取数据中所有的 Model 和 Batch Size
    all_models = sorted(df['model'].unique())
    batch_sizes = sorted(df['batch_size'].unique())

    # === 1. 颜色与属性生成逻辑 ===

    # 辅助函数：提取家族前缀 (取前两段，例如 "Qwen3-4B")
    def get_prefix(name):
        parts = name.split('-')
        if len(parts) >= 2:
            return f"{parts[0]}-{parts[1]}"
        return parts[0]

    # 找出所有唯一的家族
    unique_families = sorted(list(set(get_prefix(m) for m in all_models)))
    family_count = len(unique_families)

    model_props = {}

    # 遍历每个家族，分配属性
    for fam_idx, family in enumerate(unique_families):
        # 找出属于该家族的所有模型，并排序
        models_in_fam = sorted(
            [m for m in all_models if get_prefix(m) == family])
        count = len(models_in_fam)

        # A. 确定基础色相 (Hue)
        # 在 0.0 ~ 1.0 之间均匀分布
        hue = fam_idx / max(family_count, 1)
        # 稍微偏移一点，避免首尾颜色过于接近
        if family_count > 1:
            hue = (fam_idx * (1.0 / family_count)) % 1.0

        for m_idx, model_name in enumerate(models_in_fam):
            # B. 确定亮度 (Lightness) -> 实现同色系深浅变化
            # 如果家族里只有一个模型，亮度居中(0.5)
            # 如果有多个，亮度从深(0.3)到浅(0.7)渐变
            if count > 1:
                lightness = 0.35 + (0.4 * (m_idx / (count - 1)))
            else:
                lightness = 0.5

            # 饱和度固定，保持色彩鲜艳
            saturation = 0.75

            # 生成 RGB 颜色
            rgb_color = colorsys.hls_to_rgb(hue, lightness, saturation)

            # C. 确定线型 (Line Style)
            name_lower = model_name.lower()
            if 'awq' in name_lower:
                linestyle = ':'
            elif 'gptq' in name_lower:
                linestyle = '--'  # 点划线
            else:
                linestyle = '-'  # 实线

            model_props[model_name] = {
                'color': rgb_color,
                'linestyle': linestyle,
                'marker': 'o'
            }

    # === 图表配置 ===
    chart_configs = [
        ('gen_speed', 'Generation Speed (Decode)', 'Tokens / sec', 'zero'),
        ('gpu_mem', 'Peak GPU Memory Usage', 'Memory (GB)', 'top'),
        ('latency', 'Total Latency', 'Time (s)', 'top'),
        ('overview', 'Stability / OOM Status', 'Model Name', 'categorical')
    ]

    for bs in batch_sizes:
        print(f"正在绘制 Batch Size = {bs} 的图表...")

        df_bs = df[df['batch_size'] == bs]
        if df_bs.empty:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()

        fig.suptitle(f'Benchmark Analysis - Batch Size: {bs}',
                     fontsize=20,
                     fontweight='bold',
                     y=0.98)

        for ax, (col, title, ylabel, mode) in zip(axes, chart_configs):

            # --- 特殊处理：稳定性概览图 ---
            if mode == 'categorical':
                yticks = range(len(all_models))
                ax.set_yticks(yticks)
                ax.set_yticklabels(all_models, fontsize=10)

                for i, model in enumerate(all_models):
                    subset = df_bs[df_bs['model'] == model]
                    if subset.empty: continue

                    # 成功点 (绿色)
                    ok = subset[subset['is_oom'] == False]
                    if not ok.empty:
                        ax.scatter(ok['seq_len'], [i] * len(ok),
                                   c='green',
                                   s=80,
                                   alpha=0.6)

                    # OOM 点 (红色叉号)
                    oom = subset[subset['is_oom'] == True]
                    if not oom.empty:
                        ax.scatter(oom['seq_len'], [i] * len(oom),
                                   c='red',
                                   marker='x',
                                   s=120,
                                   linewidth=2)

                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.set_xlabel('Sequence Length')

                # OOM 图例
                status_lines = [
                    Line2D([0], [0],
                           marker='o',
                           color='w',
                           markerfacecolor='green',
                           markersize=10,
                           label='Success'),
                    Line2D([0], [0],
                           marker='x',
                           color='red',
                           markersize=10,
                           linewidth=2,
                           label='OOM')
                ]
                ax.legend(handles=status_lines, loc='upper right')
                ax.grid(True, linestyle='--', alpha=0.5)
                continue

            # --- 常规折线图 ---
            valid_vals = df_bs[df_bs['is_oom'] == False][col]
            valid_max = valid_vals.max() if not valid_vals.empty else 10

            for model in all_models:
                subset = df_bs[df_bs['model'] == model]
                if subset.empty: continue

                props = model_props[model]

                # 1. 绘制正常数据
                valid_data = subset[subset['is_oom'] == False]
                if not valid_data.empty:
                    ax.plot(valid_data['seq_len'],
                            valid_data[col],
                            color=props['color'],
                            linestyle=props['linestyle'],
                            marker=props['marker'],
                            markersize=6,
                            linewidth=2,
                            label=model)

                # 2. 绘制 OOM 标记
                oom_data = subset[subset['is_oom'] == True]
                if not oom_data.empty:
                    if mode == 'zero':
                        y_vals = [0] * len(oom_data)
                    else:
                        y_vals = [valid_max * 1.1] * len(oom_data)

                    ax.scatter(oom_data['seq_len'],
                               y_vals,
                               marker='x',
                               color='red',
                               s=100,
                               zorder=10,
                               linewidth=2.5)

                    for x, y in zip(oom_data['seq_len'], y_vals):
                        ax.text(x,
                                y,
                                ' OOM',
                                color='red',
                                fontsize=8,
                                va='bottom',
                                fontweight='bold')

            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Output Sequence Length', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)

            # --- 图例 ---
            if col == 'gen_speed':
                # 在第一张图绘制详细图例
                # 提示：由于使用了渐变色，图例会自动展示出深浅不同的颜色
                ax.legend(loc='best',
                          fontsize='small',
                          frameon=True,
                          title="Models",
                          ncol=1)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        base, ext = os.path.splitext(OUTPUT_FILENAME_BASE)
        save_path = os.path.join(os.getcwd(), f"{base}_bs-{bs}{ext}")

        plt.savefig(save_path, dpi=300)
        print(f"✅ 图表已保存: {save_path}")
        plt.close(fig)


if __name__ == "__main__":
    if os.path.exists(JSON_FILENAME):
        df_data = load_data(JSON_FILENAME)
        if not df_data.empty:
            plot_charts_by_batch_size(df_data)
        else:
            print("数据为空，无法绘图。")
    else:
        print(f"未找到文件: {JSON_FILENAME}")
