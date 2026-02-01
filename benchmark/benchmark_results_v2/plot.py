import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib.lines import Line2D

JSON_FILENAME = 'benchmark_results_v2/1__4070-8g.json'
OUTPUT_FILENAME_BASE = 'benchmark_results_v2.png'


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
    逻辑修改：按 Batch Size 循环绘制图表。
    每个 Batch Size 生成一张独立图片。
    """
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('ggplot')

    # 获取数据中所有的 Model 和 Batch Size
    all_models = sorted(df['model'].unique())
    batch_sizes = sorted(df['batch_size'].unique())

    # === 视觉映射 ===
    # 策略：一张图是一个固定的 BS，图里的线条对比不同的 Model
    # 所以我们用【颜色】来区分【Model】
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_models)))
    model_color_map = {m: color for m, color in zip(all_models, colors)}

    # 辅助：不同模型也可以用不同线型
    line_styles = ['-']
    model_style_map = {
        m: line_styles[i % len(line_styles)]
        for i, m in enumerate(all_models)
    }

    # 定义4个图表的配置
    # (数据列名, 标题, Y轴标签, OOM显示位置模式)
    chart_configs = [
        ('gen_speed', 'Generation Speed (Decode)', 'Tokens / sec', 'zero'),
        ('gpu_mem', 'Peak GPU Memory Usage', 'Memory (GB)', 'top'),
        ('latency', 'Total Latency', 'Time (s)', 'top'),
        ('overview', 'Stability / OOM Status', 'Model Name', 'categorical')
    ]

    for bs in batch_sizes:
        print(f"正在绘制 Batch Size = {bs} 的图表...")

        # 1. 筛选当前 Batch Size 的数据
        df_bs = df[df['batch_size'] == bs]
        if df_bs.empty:
            continue

        # 2. 创建 2x2 画布
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()

        # 3. 设置整张大图的标题 (包含 Batch Size)
        fig.suptitle(f'Benchmark Analysis - Batch Size: {bs}',
                     fontsize=20,
                     fontweight='bold',
                     y=0.98)

        # 4. 遍历4个指标绘制子图
        for ax, (col, title, ylabel, mode) in zip(axes, chart_configs):

            # --- 特殊处理第4张图：稳定性概览 (Model vs Length) ---
            if mode == 'categorical':
                # Y轴是模型名称，X轴是序列长度
                yticks = range(len(all_models))
                ax.set_yticks(yticks)
                ax.set_yticklabels(all_models)

                for i, model in enumerate(all_models):
                    subset = df_bs[df_bs['model'] == model]
                    if subset.empty: continue

                    # 画成功的点 (绿色圆点)
                    ok = subset[subset['is_oom'] == False]
                    if not ok.empty:
                        ax.scatter(ok['seq_len'], [i] * len(ok),
                                   c='green',
                                   s=80,
                                   alpha=0.6)

                    # 画 OOM 的点 (红色叉号)
                    oom = subset[subset['is_oom'] == True]
                    if not oom.empty:
                        ax.scatter(oom['seq_len'], [i] * len(oom),
                                   c='red',
                                   marker='x',
                                   s=120,
                                   linewidth=2)

                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.set_xlabel('Sequence Length')

                # 手动创建图例
                custom_lines = [
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
                ax.legend(handles=custom_lines, loc='upper right')
                ax.grid(True, linestyle='--', alpha=0.5)
                continue

            # 计算当前子图数据的最大值 (用于确定 OOM 标记画在多高)
            valid_vals = df_bs[df_bs['is_oom'] == False][col]
            valid_max = valid_vals.max() if not valid_vals.empty else 10

            for model in all_models:
                subset = df_bs[df_bs['model'] == model]
                if subset.empty: continue

                # A. 绘制正常数据的连线
                valid_data = subset[subset['is_oom'] == False]
                if not valid_data.empty:
                    ax.plot(valid_data['seq_len'],
                            valid_data[col],
                            color=model_color_map[model],
                            linestyle=model_style_map[model],
                            marker='o',
                            markersize=6,
                            linewidth=2,
                            label=model)  # 图例显示模型名

                # B. 绘制 OOM 数据
                oom_data = subset[subset['is_oom'] == True]
                if not oom_data.empty:
                    # 确定 OOM 标记的 Y 轴位置
                    if mode == 'zero':
                        y_vals = [0] * len(oom_data)
                    else:
                        # 显存/延迟图中，把叉画在最大值的上方 10% 处
                        y_vals = [valid_max * 1.1] * len(oom_data)

                    ax.scatter(oom_data['seq_len'],
                               y_vals,
                               marker='x',
                               color='red',
                               s=100,
                               zorder=10,
                               linewidth=2.5)

                    # 标注 OOM 文字
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

            if col == 'gen_speed':
                ax.legend(loc='best',
                          fontsize='small',
                          frameon=True,
                          title="Models")

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        base, ext = os.path.splitext(OUTPUT_FILENAME_BASE)
        save_path = os.path.join(os.getcwd(), f"{base}_bs_{bs}{ext}")

        plt.savefig(save_path, dpi=300)
        print(f"✅ 图表已保存: {save_path}")

        plt.close(fig)  # 关闭当前画布，释放内存


if __name__ == "__main__":
    if os.path.exists(JSON_FILENAME):
        df_data = load_data(JSON_FILENAME)
        if not df_data.empty:
            plot_charts_by_batch_size(df_data)
        else:
            print("数据为空，无法绘图。")
    else:
        print(f"未找到文件: {JSON_FILENAME}")
