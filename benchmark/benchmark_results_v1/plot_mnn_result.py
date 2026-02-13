import platform
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os

INPUT_FILE = "3__llm_benchmark_results.json"
OUTPUT_DIR = "bench_plots_bar"

QUANT_PALETTE = {
    "FP16 (No Quant)": "#4C72B0",  # 蓝
    "Int8 QK": "#DD8452",  # 橙
    "Int8 QKV": "#55A868",  # 绿
    "Unknown": "#C0C0C0"
}

# 连续变量（线程 / 长度）统一用冷色渐变
SEQUENTIAL_PALETTE = sns.color_palette("Blues", as_cmap=False)

HEATMAP_CMAP = "Blues"


def load_and_clean_data(filepath):
    """加载 JSON 并转换为适合绘图的 DataFrame"""
    if not os.path.exists(filepath):
        print(f"❌ 错误: 找不到文件 {filepath}。请先运行 run_bench.py 生成数据。")
        return None

    print(f"📂 正在读取 {filepath} ...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 读取 JSON 失败: {e}")
        return None

    if not data:
        print("⚠️ 数据为空，无法绘图。")
        return None

    df = pd.json_normalize(data)

    rename_map = {
        'inputs.threads': 'Threads',
        'inputs.n_prompt': 'Prompt_Len',
        'inputs.n_gen': 'Gen_Len',
        'inputs.quant_attention': 'Quant_Code',
        'metrics.speed_tps': 'Speed',
        'metrics.stage': 'Stage',
        'metrics.seq_len': 'Seq_Len',
        'metrics.model_name': 'Model'
    }
    df = df.rename(columns=rename_map)

    def map_quant(x):
        if x == 1: return "Int8 QK"
        if x == 2: return "Int8 QKV"
        return "FP16 (No Quant)"

    if 'Quant_Code' in df.columns:
        df['Quant_Label'] = df['Quant_Code'].apply(map_quant)
    else:
        df['Quant_Label'] = "Unknown"

    df['Stage_Label'] = df['Stage'].map({
        'prefill': 'Prefill (Prompt Processing)',
        'decode': 'Decode (Token Generation)'
    })

    df['Threads_Str'] = df['Threads'].astype(str) + " Threads"
    df['Seq_Len_Int'] = df['Seq_Len'].astype(int)
    df['Seq_Len_Str'] = df['Seq_Len'].astype(str)

    return df


def setup_style():
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)

    system = platform.system()
    if system == "Darwin":
        fonts = ['PingFang SC', 'Heiti SC', 'STHeiti']
    elif system == "Windows":
        fonts = ['Microsoft YaHei', 'SimHei']
    else:
        fonts = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei']

    plt.rcParams['font.sans-serif'] = fonts
    plt.rcParams['axes.unicode_minus'] = False


def save_plot(filename):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  --> 图表已保存: {path}")
    plt.close()


def add_bar_labels(ax):
    for container in ax.containers:
        labels = [
            f'{v.get_height():.1f}' if v.get_height() > 0 else ''
            for v in container
        ]
        ax.bar_label(container, labels=labels, padding=3, fontsize=9)


# ================= 📈 绘图逻辑 =================


def plot_thread_scaling(df):
    print("📊 [1/4] 正在绘制线程扩展性分析图...")

    max_p = df['Prompt_Len'].max() if 'Prompt_Len' in df.columns else 0
    max_g = df['Gen_Len'].max() if 'Gen_Len' in df.columns else 0

    subset = df[((df['Stage'] == 'prefill') & (df['Seq_Len_Int'] == max_p)) |
                ((df['Stage'] == 'decode') &
                 (df['Seq_Len_Int'] == max_g))].copy()

    if subset.empty:
        subset = df

    g = sns.catplot(data=subset,
                    kind="bar",
                    x="Threads_Str",
                    y="Speed",
                    hue="Quant_Label",
                    col="Stage_Label",
                    palette=QUANT_PALETTE,
                    height=5,
                    aspect=1.2,
                    sharey=False,
                    legend_out=True)

    g.fig.suptitle(f"CPU 线程扩展性 (Max Len: P={max_p}, G={max_g})",
                   y=1.05,
                   fontsize=16)
    g.set_axis_labels("", "Speed (Tokens/s)")

    for ax in g.axes.flat:
        add_bar_labels(ax)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    save_plot("1_thread_scaling_bar.png")


def plot_length_impact(df):
    print("📊 [2/4] 正在绘制序列长度影响分析图...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    target_quants = ["Int8 QKV", "Int8 QK", "FP16 (No Quant)"]
    selected_quant = next(
        (q for q in target_quants if q in df['Quant_Label'].values), "Unknown")

    print(f"    (注: 仅展示 '{selected_quant}' 模式)")
    filtered_df = df[df['Quant_Label'] == selected_quant].copy()

    def draw_subplot(stage, ax, title):
        data = filtered_df[filtered_df['Stage'] == stage].copy()
        if data.empty:
            return

        data = data.sort_values(['Seq_Len_Int', 'Threads'])

        sns.barplot(data=data,
                    x="Seq_Len_Str",
                    y="Speed",
                    hue="Threads_Str",
                    palette=SEQUENTIAL_PALETTE,
                    ax=ax,
                    edgecolor="white",
                    linewidth=0.5)

        ax.set_title(title)
        ax.set_xlabel("Length (Tokens)")
        ax.set_ylabel("Speed (Tokens/s)")
        add_bar_labels(ax)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    draw_subplot('prefill', axes[0], f"Prefill 速度 vs 长度 ({selected_quant})")
    draw_subplot('decode', axes[1], f"Decode 速度 vs 长度 ({selected_quant})")

    plt.tight_layout()
    save_plot("2_length_impact_bar.png")


def plot_quantization_gain(df):
    print("📊 [3/4] 正在绘制量化收益对比图...")

    max_gen = df['Gen_Len'].max() if 'Gen_Len' in df.columns else None
    if max_gen is None:
        return

    data = df[(df['Stage'] == 'decode')
              & (df['Seq_Len_Int'] == max_gen)].copy()
    if data.empty:
        return

    plt.figure(figsize=(10, 6))

    ax = sns.barplot(data=data,
                     x="Threads_Str",
                     y="Speed",
                     hue="Quant_Label",
                     palette=QUANT_PALETTE,
                     edgecolor="white")

    add_bar_labels(ax)

    plt.title(f"Decode 阶段量化收益对比 (Gen Length={max_gen})", fontsize=14)
    plt.ylabel("Speed (Tokens/s)")
    plt.xlabel("")
    plt.legend(title="Quantization", bbox_to_anchor=(1.0, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    save_plot("3_quantization_gain_bar.png")


def plot_heatmap(df):
    print("📊 [4/4] 正在绘制性能热力图...")

    data = df[df['Stage'] == 'decode'].copy()
    if data.empty:
        return

    data['Config'] = data['Threads_Str'] + " | " + data['Quant_Label']

    pivot = data.pivot_table(index='Config',
                             columns='Seq_Len_Int',
                             values='Speed',
                             aggfunc='mean')

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot,
                annot=True,
                fmt=".1f",
                cmap=HEATMAP_CMAP,
                linewidths=.5,
                cbar_kws={'label': 'Tokens / sec'})

    plt.title("Decode Speed Heatmap", fontsize=14)
    plt.xlabel("Generation Length")
    plt.ylabel("Configuration")

    save_plot("4_performance_heatmap.png")


def main():
    setup_style()
    df = load_and_clean_data(INPUT_FILE)
    if df is None:
        return

    print(f"✅ 数据加载成功，共 {len(df)} 条记录。\n")

    plot_thread_scaling(df)
    plot_length_impact(df)
    plot_quantization_gain(df)
    plot_heatmap(df)

    print(f"\n✨ 所有图表已保存至: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
