```bash
cd libs/
./llm_demo ../models/qwen3-ui-1.7b-mnn-w4-b256/config.json prompt.txt
./llm_bench -m ../models/qwen3-ui-1.7b-mnn-w4-b256/config.json -a cpu -t 4 -p 32 -n 32 -rep 3 -kv false
```

`llm_demo`的用法如下：

```bash
# 使用config.json
## 交互式聊天
./llm_demo ../models/qwen3-ui-1.7b-mnn-w4/config.json
## 针对prompt中的每行进行回复
./llm_demo ../models/qwen3-ui-1.7b-mnn-w4/config.json prompt.txt

# 不使用config.json, 使用默认配置
## 交互式聊天
./llm_demo ../models/qwen3-ui-1.7b-mnn-w4/llm.mnn
## 针对prompt中的每行进行回复
./llm_demo ../models/qwen3-ui-1.7b-mnn-w4/llm.mnn prompt.txt
```

llm_bench args:

```bash
usage: ./llm_bench [options]

options:
  -h, --help
  -m, --model <filename>                    (default: ./Qwen2.5-1.5B-Instruct/config.json)
  -a, --backends <cpu,opencl,metal>         (default: cpu)
  -c, --precision <n>                       (default: 2) | Note: (0:Normal(for cpu backend, 'Normal' is 'High'),1:High,2:Low)
  -t, --threads <n>                         (default: 4)
  -p, --n-prompt <n>                        (default: 512)
  -n, --n-gen <n>                           (default: 128)
  -pg <pp,tg>                               (default: 0,0)
  -mmp, --mmap <0|1>                        (default: 0)
  -rep, --n-repeat <n>                      (default: 5)
  -kv, --kv-cache <true|false>              (default: false) | Note: if true: Every time the LLM model generates a new word, it utilizes the cached KV-cache
  -fp, --file-print <stdout|filename>       (default: stdout)
  -scn, --sme-core-num <n>                  (default: 2) | Note: Specify the number of smeCoreNum to use.
  -load, --loading-time <true|false>        (default: true)
  -dyo, --dynamicOption <n>                 (default: 0) | Note: if set 8, trades higher memory usage for better decoding performance
  -mr, --mixedSme2NeonRatio <n>             (default: 41) | Note: This parameter is intended to optimize multi-threaded inference performance on backends that support Arm SME instructions. The optimal ratio may vary across different models; we recommend trying values such as 41, 49, 33.
  -qatten, --quant-attention <0|1>           (default: 0) | Note: if 1, quantize attention's key value to int8; default 0
```
