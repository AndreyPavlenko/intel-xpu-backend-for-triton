// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s

module attributes {triton_intel_gpu.target_arch = "spirv", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @kernel
  tt.func public @kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %0 = tt.load %arg0 : !tt.ptr<f32>
    %1 = tt.load %arg1 : !tt.ptr<f32>
    // CHECK:    llvm.call @add_fn
    tt.call @add_fn(%0, %1, %arg2) : (f32, f32, !tt.ptr<f32>) -> ()
    tt.return
  }
  // CHECK: llvm.func internal @add_fn
  tt.func private @add_fn(%arg0: f32, %arg1: f32, %arg2: !tt.ptr<f32>) {
    %0 = arith.addf %arg0, %arg1 fastmath<fast> : f32
    tt.store %arg2, %0 : !tt.ptr<f32>
    tt.return
  }
}

// -----

module attributes {triton_intel_gpu.target_arch = "pisa", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func pisa_kernelcc @kernel
  tt.func public @kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %0 = tt.load %arg0 : !tt.ptr<f32>
    %1 = tt.load %arg1 : !tt.ptr<f32>
    // CHECK:    llvm.call pisa_funccc @add_fn
    tt.call @add_fn(%0, %1, %arg2) : (f32, f32, !tt.ptr<f32>) -> ()
    tt.return
  }
  // CHECK: llvm.func internal pisa_funccc @add_fn
  tt.func private @add_fn(%arg0: f32, %arg1: f32, %arg2: !tt.ptr<f32>) {
    %0 = arith.addf %arg0, %arg1 fastmath<fast> : f32
    tt.store %arg2, %0 : !tt.ptr<f32>
    tt.return
  }
}
