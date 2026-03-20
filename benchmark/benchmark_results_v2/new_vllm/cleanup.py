import json


def filter_json_data(input_filename, output_filename):
    try:
        # 1. 读取原始JSON文件
        with open(input_filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 确保顶层数据是一个列表
        if not isinstance(data, list):
            print("错误：JSON的顶层数据不是一个列表格式。")
            return

        # 2. 过滤掉 max_new_tokens 为 160 的项
        filtered_data = [
            item for item in data
            if item.get("config", {}).get("max_new_tokens") != 500
        ]

        # 3. 将过滤后的数据写入新的JSON文件
        with open(output_filename, 'w', encoding='utf-8') as f:
            # indent=2 让输出的json格式化美观，ensure_ascii=False 保证中文字符正常显示
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)

        print(f"处理完成！")
        print(f"原始数据共有: {len(data)} 项")
        print(f"过滤后剩余: {len(filtered_data)} 项")
        print(f"删除了: {len(data) - len(filtered_data)} 项")
        print(f"新文件已保存为: {output_filename}")

    except FileNotFoundError:
        print(f"找不到文件: {input_filename}，请检查文件路径。")
    except json.JSONDecodeError:
        print(f"文件 {input_filename} 不是有效的JSON格式，请检查。")


# --- 使用示例 ---
if __name__ == "__main__":
    # 替换为你实际的文件名
    INPUT_FILE = "2__4090-24g-vllm.json"
    # 过滤后保存的文件名
    OUTPUT_FILE = "2__4090-24g-vllm-filtered.json"

    filter_json_data(INPUT_FILE, OUTPUT_FILE)
