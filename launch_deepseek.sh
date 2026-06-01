#!/bin/bash

module load rocm/7.2.0
module load cuda/13.0
module load gcc/13.2

max_num_seqs=16
max_num_batched_tokens=1024
tensor_parallel_size=2
#export VLLM_ROCM_USE_AITER=1
source /home/users/andrewka/sw/vLLM/vllm_envs/deepseek-v4_rocm_vllm_rocm/bin/activate

export FLASH_ATTENTION_TRITON_AMD_AUTOTUNE="TRUE"
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
vllm serve deepseek-ai/DeepSeek-V4-Flash \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype auto \
    --tensor-parallel-size ${tensor_parallel_size} \
    --max-num-seqs ${max_num_seqs} \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --moe-backend "triton_unfused" \
    --tokenizer-mode "deepseek_v4" \
    --enforce-eager \
    --tool-call-parser deepseek_v4  \
    --enable-auto-tool-choice  \
    --reasoning-parser deepseek_v4 \
    --kv-cache-dtype fp8 \
    --kernel-config '{"moe_backend":"triton"}'

