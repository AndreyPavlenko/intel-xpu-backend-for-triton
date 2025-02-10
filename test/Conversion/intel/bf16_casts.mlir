// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s --dump-input-context 20

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_bf16_conversion, triton_intel_gpu.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @extf_spirv
  tt.func public @extf_spirv(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<bf16>) attributes {noinline = false} {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %1 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>, #blocked>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<bf16>, #blocked>, tensor<128xi32, #blocked>
    %3 = tt.load %2 : tensor<128x!tt.ptr<bf16>, #blocked>
    // CHECK: __spirv_ConvertBF16ToFINTEL
    %4 = arith.extf %3 fastmath<fast> : tensor<128xbf16, #blocked> to tensor<128xf32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %6 = tt.addptr %5, %0 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %6, %4 : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_bf16_conversion, triton_intel_gpu.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @truncf_spirv
  tt.func public @truncf_spirv(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %1 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %3 = tt.load %2 : tensor<128x!tt.ptr<f32>, #blocked>
    // CHECK: __spirv_ConvertFToBF16INTEL
    %4 = arith.truncf %3 fastmath<fast> : tensor<128xf32, #blocked> to tensor<128xbf16, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>, #blocked>
    %6 = tt.addptr %5, %0 : tensor<128x!tt.ptr<bf16>, #blocked>, tensor<128xi32, #blocked>
    tt.store %6, %4 : tensor<128x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_bf16_conversion, triton_intel_gpu.target_arch = "pisa", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @extf_pisa
  tt.func public @extf_pisa(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<bf16>) attributes {noinline = false} {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %1 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>, #blocked>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<bf16>, #blocked>, tensor<128xi32, #blocked>
    %3 = tt.load %2 : tensor<128x!tt.ptr<bf16>, #blocked>
    // CHECK: llvm.fpext
    %4 = arith.extf %3 fastmath<fast> : tensor<128xbf16, #blocked> to tensor<128xf32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %6 = tt.addptr %5, %0 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %6, %4 : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_bf16_conversion, triton_intel_gpu.target_arch = "pisa", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @truncf_pisa
  tt.func public @truncf_pisa(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %1 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %3 = tt.load %2 : tensor<128x!tt.ptr<f32>, #blocked>
    // CHECK: llvm.fptrunc
    %4 = arith.truncf %3 fastmath<fast> : tensor<128xf32, #blocked> to tensor<128xbf16, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>, #blocked>
    %6 = tt.addptr %5, %0 : tensor<128x!tt.ptr<bf16>, #blocked>, tensor<128xi32, #blocked>
    tt.store %6, %4 : tensor<128x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}

