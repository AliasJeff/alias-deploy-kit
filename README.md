# alias-deploy-kit

## Environment

## Basic process

### 1.

## Optimized process

### 1. Install basic dependencies

```bash
sudo apt-get update && sudo apt-get install -y cmake build-essential git python3-pip

python -m venv venv

pip install -r requirements.txt
```

### 2. Run prune script

```bash
python prune_qwen3.py
```

### 3. Export target model (NPU) using llmexport (Sparse + Int4)

```bash
python llmexport.py \
    --path ./outputs/models/qwen3-1.7b-sparse-2by4 \
    --mnn_model_name qwen3_4b_sparse_w4.mnn \
    --export_mnn \
    --quant_bit 4 \
    --quant_block 128 \
    --dhm_model
```

### 4. Export draft model (CPU) using llmexport

```bash
python llmexport.py \
    --path Qwen/Qwen2.5-0.5B \
    --mnn_model_name qwen2.5_0.5b_w4.mnn \
    --export_mnn \
    --quant_bit 4
```

### 5. Compile MNN engine (with QNN backend)

Prepare Android NDK & Qualcomm QNN SDK

- Android NDK: NDK r25c or r26b
- Qualcomm AI Engine Direct (QNN) SDK

```bash
export ANDROID_NDK=/path/to/android-ndk
export QNN_SDK_ROOT=/opt/qcom/qnn-sdk

cd MNN
rm -rf build_android
mkdir build_android
cd build_android

cmake.. \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DANDROID_ABI="arm64-v8a" \
    -DANDROID_STL=c++_static \
    -DANDROID_NATIVE_API_LEVEL=android-28 \
    -DMNN_BUILD_QUANTOOLS=ON \
    -DMNN_ARM82=ON \
    -DMNN_OPENCL=ON \
    -DMNN_QNN=ON \
    -DQNN_SDK_ROOT=$QNN_SDK_ROOT \
    -DMNN_BUILD_LLM=ON \
    -DMNN_SUPPORT_TRANSFORMER_FUSE=ON

make -j8
```

Result:
cd build_android

- libMNN.so
- libMNN_QNN.so
- libQnnHtp.so / libQnnHtpV73Skel.so
