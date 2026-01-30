import subprocess
import json
import itertools
import re
import os
import datetime
import sys

# ==================== ⚙️ 配置区域 ⚙️ ====================

# 1. llm_bench 可执行文件路径
BENCH_BIN = "./libs/llm_bench"

# 2. 模型配置文件路径 (请修改为你实际的路径)
MODEL_CONFIG = "./models/Qwen3-1.7B-MNN/config.json"

# 3. 输出文件名
OUTPUT_JSON = "llm_benchmark_results.json"

# 4. 测试参数网格 (Grid Search)
# 脚本会自动运行以下列表中所有参数的排列组合
TEST_CONFIGS = {
    # 线程数 (-t): 测试多核扩展性
    "threads": [8, 6, 4, 2],

    # 精度 (-c): 0=Normal(FP32/FP16), 1=High, 2=Low(通常为Int8/BF16, 端侧推荐)
    "precision": [2],

    # 提示词长度 (-p): 影响 Prefill (预填充) 速度
    # 对应输出中的 pp (Prompt Processing)
    "n_prompt": [64, 128, 256],

    # 生成长度 (-n): 影响 Decode (解码) 速度
    # 对应输出中的 tg (Token Generation)
    "n_gen": [16, 64, 256, 512],

    # KV Cache 量化 (-qatten): 0=FP16, 1=Int8 QK, 2=Int8 QKV
    "quant_attention": [0, 1],

    # 后端 (-a): cpu, opencl, metal
    "backend": ["cpu"]
}

# ========================================================


def parse_markdown_table(stdout_str):
    """
    解析 llm_bench 的 Markdown 输出。
    采用动态表头匹配，不依赖固定列索引。
    """
    lines = stdout_str.strip().split('\n')
    results = []

    header_map = {}  # {列索引: 列名}
    table_started = False

    for line in lines:
        line = line.strip()

        # 1. 识别表头行 (必须包含 model 和 t/s)
        if line.startswith("|") and "model" in line and "t/s" in line:
            # 移除首尾的 |，然后分割
            cols = [c.strip() for c in line.strip('|').split('|')]
            # 建立索引映射
            header_map = {i: col_name for i, col_name in enumerate(cols)}
            table_started = True
            continue

        # 2. 跳过 Markdown 分隔行 (如 |---|---|)
        if table_started and line.startswith("|") and set(line).issubset(
            {'|', '-', ' ', ':'}):
            continue

        # 3. 解析数据行
        if table_started and line.startswith("|"):
            parts = [p.strip() for p in line.strip('|').split('|')]

            # 如果分割出的列数与表头不一致，跳过
            if len(parts) != len(header_map):
                continue

            row_data = {}
            # 根据表头映射数据
            for idx, val in enumerate(parts):
                if idx in header_map:
                    row_data[header_map[idx]] = val

            # 清洗并转换数据类型
            cleaned_entry = process_row_data(row_data)
            if cleaned_entry:
                results.append(cleaned_entry)

    return results


def process_row_data(row):
    """
    将原始字符串数据清洗为 JSON 友好的格式
    """
    try:
        entry = {}

        # === 基础字段 ===
        entry["model_name"] = row.get("model", "Unknown")
        entry["model_size_str"] = row.get("modelSize", "")

        # === 动态字段 (可能不存在) ===
        if "quantAttention" in row:
            entry["quant_attn_str"] = row["quantAttention"]
        if "backend" in row:
            entry["backend_str"] = row["backend"]

        # === 解析测试类型 (pp vs tg) ===
        # 格式示例: "pp32", "tg32", "pp32+tg32"
        test_str = row.get("test", "")
        entry["test_label"] = test_str

        if test_str.startswith("pp"):
            entry["stage"] = "prefill"
            # 提取数字
            nums = re.findall(r'\d+', test_str)
            entry["seq_len"] = int(nums[0]) if nums else 0
        elif test_str.startswith("tg"):
            entry["stage"] = "decode"
            nums = re.findall(r'\d+', test_str)
            entry["seq_len"] = int(nums[0]) if nums else 0
        else:
            entry["stage"] = "mixed"
            entry["seq_len"] = 0

        # === 解析速度 (t/s) ===
        # 格式示例: "316.00 ± 1.55"
        speed_str = row.get("t/s", "")
        if "±" in speed_str:
            mean, std = speed_str.split("±")
            entry["speed_tps"] = float(mean.strip())
            entry["speed_std"] = float(std.strip())
        else:
            # 尝试直接解析
            try:
                entry["speed_tps"] = float(speed_str.strip())
                entry["speed_std"] = 0.0
            except:
                entry["speed_tps"] = 0.0

        return entry
    except Exception as e:
        print(f"Warning: 解析行数据失败: {row}, Error: {e}")
        return None


def run_benchmarks():
    if not os.path.exists(BENCH_BIN):
        print(f"Error: 找不到可执行文件 {BENCH_BIN}，请检查路径。")
        return

    # 生成参数笛卡尔积
    keys = list(TEST_CONFIGS.keys())
    values = list(TEST_CONFIGS.values())
    combinations = list(itertools.product(*values))

    all_records = []

    print(f"🚀 开始测试，总计 {len(combinations)} 组参数配置...")
    print(f"📂 结果将保存至: {os.path.abspath(OUTPUT_JSON)}\n")

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))

        # 构造命令
        # 注意: 使用 -kv false 以分离 Prefill 和 Decode 测试
        # C++ 代码中，当 -kv false 时，它会分别跑 pp 和 tg 循环
        cmd = [
            BENCH_BIN,
            "-m",
            MODEL_CONFIG,
            "-a",
            params["backend"],
            "-t",
            str(params["threads"]),
            "-c",
            str(params["precision"]),
            "-p",
            str(params["n_prompt"]),
            "-n",
            str(params["n_gen"]),
            "-qatten",
            str(params["quant_attention"]),
            "-kv",
            "false",
            "-rep",
            "3",  # 重复3次取平均
            "-load",
            "false"  # 不重复测试加载时间，节省时间
        ]

        # 进度提示
        print(
            f"[{i+1}/{len(combinations)}] Config: Thr={params['threads']}, Prompt={params['n_prompt']}, Gen={params['n_gen']}, QAttn={params['quant_attention']}"
        )

        try:
            # 执行命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'  # 防止特殊字符报错
            )

            if result.returncode != 0:
                print(f"  ⚠️  命令执行返回非0: {result.stderr}")

            # 解析输出
            parsed_rows = parse_markdown_table(result.stdout)

            if not parsed_rows:
                print("  ⚠️  未解析到数据，可能运行输出异常。")

            # 合并 输入参数(params) 和 输出结果(parsed_rows)
            for row in parsed_rows:
                record = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "inputs": params,  # 输入配置
                    "metrics": row  # 输出指标
                }
                all_records.append(record)

                # 打印简报
                print(
                    f"   -> [{row['stage'].upper()}] Len: {row['seq_len']}, Speed: {row['speed_tps']} t/s"
                )

        except Exception as e:
            print(f"   ❌ 执行出错: {e}")

    # 保存 JSON
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_records, f, indent=4, ensure_ascii=False)

    print(f"\n✅ 测试完成！详细数据已写入 {OUTPUT_JSON}")


if __name__ == "__main__":
    run_benchmarks()
