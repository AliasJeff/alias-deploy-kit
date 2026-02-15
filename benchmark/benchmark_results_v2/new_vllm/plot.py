import json
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def get_prefix(model_name):
    """
    提取模型名称的前缀，用于判断同类型模型。
    这里以 '-' 作为分隔符，提取前两部分作为前缀。
    例如 'Qwen3-4B' 和 'Qwen3-4B-GPTQ-4bit' 都会提取为 'Qwen3-4B'。
    """
    parts = model_name.split('-')
    if len(parts) >= 2:
        return '-'.join(parts[:2])
    return model_name


def main():
    # 1. 读取 JSON 数据
    input_file = 'benchmark_vllm_results.json'
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"找不到文件 {input_file}，请确保 JSON 文件路径正确。")
        return

    # 2. 解析和组织数据
    # 数据结构: data_map[max_new_tokens][model_name] = [(batch_size, rps), ...]
    data_map = defaultdict(lambda: defaultdict(list))
    models_set = set()

    for item in data:
        # --- 修改点 1: 如果存在 error 字段且不为空，直接跳过不显示 ---
        if item.get('error'):
            continue

        model = item['model']
        batch_size = item['config']['batch_size']
        max_new_tokens = item['config']['max_new_tokens']
        rps = item['metrics']['request_per_second']

        data_map[max_new_tokens][model].append((batch_size, rps))
        models_set.add(model)

    # 按名称对模型和 max_new_tokens 进行排序（保证模型按名称排序）
    sorted_models = sorted(list(models_set))
    sorted_tokens = sorted(list(data_map.keys()))

    if not sorted_tokens:
        print("未找到有效的数据（可能数据为空或全部包含 error）！")
        return

    # 3. 分配颜色（同类模型分配同色系的不同深浅）
    prefix_groups = defaultdict(list)
    for m in sorted_models:
        prefix = get_prefix(m)
        prefix_groups[prefix].append(m)

    # 预设几个经典的 matplotlib 连续渐变色系
    cmaps = [
        'Blues', 'Reds', 'Greens', 'Oranges', 'Purples', 'Greys', 'YlOrBr',
        'PuBuGn'
    ]
    model_colors = {}

    for i, (prefix, group_models) in enumerate(prefix_groups.items()):
        cmap_name = cmaps[i % len(cmaps)]
        cmap = plt.get_cmap(cmap_name)

        # 为了保证颜色在白底上清晰可见，取色范围在 0.4（较深）到 0.9（极深）之间
        if len(group_models) == 1:
            model_colors[group_models[0]] = cmap(0.7)
        else:
            shades = np.linspace(0.4, 0.9, len(group_models))
            for j, m in enumerate(group_models):
                model_colors[m] = cmap(shades[j])

    # 4. 计算子图布局 (一行最多 3 个)
    n_plots = len(sorted_tokens)
    cols = min(3, n_plots)
    rows = math.ceil(n_plots / cols)

    # 创建画布，squeeze=False 确保 axes 始终是二维矩阵，方便索引和平铺
    fig, axes = plt.subplots(rows,
                             cols,
                             figsize=(6 * cols, 5 * rows),
                             squeeze=False)
    axes_flat = axes.flatten()

    # 5. 绘制各个子图
    for idx, tokens in enumerate(sorted_tokens):
        ax = axes_flat[idx]

        # 在 y=50 处画红色的目标虚线
        ax.axhline(y=50,
                   color='red',
                   linestyle='--',
                   linewidth=2,
                   label='Target (50 req/s)',
                   zorder=1)

        # 遍历所有模型并绘制折线
        for model in sorted_models:
            if model in data_map[tokens]:
                # 按照 batch_size 从小到大排序当前折线的点，防止折线连线乱穿插
                points = sorted(data_map[tokens][model], key=lambda x: x[0])
                x = [p[0] for p in points]
                y = [p[1] for p in points]

                # 画出该模型的折线和数据点圆圈
                ax.plot(x,
                        y,
                        marker='o',
                        color=model_colors[model],
                        linewidth=2,
                        markersize=6,
                        label=model,
                        zorder=2)

        # 设置子图的标题和坐标轴名称
        ax.set_title(f'max_new_tokens = {tokens}',
                     fontsize=13,
                     fontweight='bold')
        ax.set_xlabel('Batch Size', fontsize=11)
        ax.set_ylabel('Req / s', fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.7)

        # --- 修改点 2: 仅在第一张图（idx == 0）显示图例 ---
        if idx == 0:
            # 提取图例并去重（确保 Target 线始终排在第一个）
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))

            # 构建排序后的图例列表
            sorted_labels = ['Target (50 req/s)'
                             ] + [m for m in sorted_models if m in by_label]
            sorted_handles = [
                by_label[lbl] for lbl in sorted_labels if lbl in by_label
            ]

            ax.legend(sorted_handles, sorted_labels, fontsize=9)

    # 隐藏因为网格布局而多出来的空白占位子图
    for idx in range(n_plots, len(axes_flat)):
        fig.delaxes(axes_flat[idx])

    # 6. 调整布局防遮挡并保存图表
    plt.tight_layout()
    output_filename = 'benchmark_results.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ 绘图完成！图表已合并并保存至: {output_filename}")


if __name__ == '__main__':
    main()
