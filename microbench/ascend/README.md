# AscendC microbenchmark suite

AscendC C++ benchmarks (`bisheng -xcce --npu-arch=dav-2201`) that measure Ascend 910B M1 and M2 hardware characteristics: HBM/L2 bandwidth scaling across active AI Cores, L2 working-set capacity cliff, pointer-chasing L2/HBM idle and loaded latency curves (`*_latency_by_bw`, `*_latency_loaded_ns`), MLP bytes-in-flight bandwidth curves (`dram_bw_by_bytes_in_flight`, `l2_bw_by_bytes_in_flight`), 2D strided `GM ND -> L1 Fractal NZ` row-issue rate (`aic_nd2nz_issue_kernel` -> `gmem_load_issue_controlled`), L2 miss-merging (`l2_merge_by_sharing`), on-chip UB bandwidth and latency, Cube FP16/BF16 `Mmad` throughput, Vector FP32 FMA and `Exp` throughput, AIC/AIV per-stage pipeline sync overhead (`fixed_sync_ns`, `cv_sync_hop_ns_by_active_sms`, `stage_sync_overhead_ns`, `min_async_roundtrip_ns`), queued device launch overhead, and first-wave dispatch timing.

## Build and Run (on an Ascend 910B host with CANN 8+/9+)

```bash
source /usr/local/Ascend/cann/set_env.sh
ARCH_DIR="${ASCEND_HOME_PATH}/aarch64-linux"
bisheng -xcce microbench/ascend/vgpu_ascend_bench.cpp \
  --npu-arch=dav-2201 -O3 -std=c++17 \
  -I"${ARCH_DIR}/include" \
  -I"${ARCH_DIR}/tikcpp/tikcfw" \
  -I"${ARCH_DIR}/tikcpp/tikcfw/interface" \
  -I"${ARCH_DIR}/tikcpp/tikcfw/impl" \
  -L"${ARCH_DIR}/lib64" -lascendcl -lruntime -lstdc++ -lm \
  -o vgpu_ascend_bench

./vgpu_ascend_bench --device 0 --samples 15 > raw_bench.json

python3 microbench/ascend/collect_ascend_profile.py \
  --raw-json raw_bench.json \
  --platform-ini "${ARCH_DIR}/data/platform_config/Ascend910B3.ini" \
  --out profiles/ascend910b3.json
```
