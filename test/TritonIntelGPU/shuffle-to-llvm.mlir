// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s

module attributes {triton_intel_gpu.target_arch = "pisa", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: llvm.func pisa_funccc @llvm.pisa.shfl.xor
  // CHECK-SAME: convergent
  // CHECK-SAME: no_unwind
  // CHECK-SAME: will_return
  llvm.func pisa_funccc @llvm.pisa.localid.x() -> i32
  // CHECK-LABEL: @copy_kernel
  llvm.func pisa_kernelcc @copy_kernel(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) {
    %id = llvm.call pisa_funccc @llvm.pisa.localid.x() : () -> i32
    %in_ptr = llvm.getelementptr %arg0[%id] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i64
    // CHECK: %[[VAL:.*]] = llvm.load
    %val = llvm.load %in_ptr {alignment = 8 : i64} : !llvm.ptr<1> -> i64
    %offset = llvm.mlir.constant(16 : index) : i32
    %width = llvm.mlir.constant(32 : index) : i32
    // CHECK: %[[VEC_VAL:.*]] = llvm.bitcast %[[VAL]] : i64 to vector<2xi32>
    // CHECK: %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[VAL_0:.*]] = llvm.extractelement %[[VEC_VAL]][%[[CST_0]] : i32] : vector<2xi32>
    // CHECK: %[[CST_1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: %[[VAL_1:.*]] = llvm.extractelement %[[VEC_VAL]][%[[CST_1]] : i32] : vector<2xi32>
    // CHECK: %[[NEW_VAL_0:.*]] = llvm.call pisa_funccc @llvm.pisa.shfl.xor(%[[VAL_0]]
    // CHECK: %[[NEW_VAL_1:.*]] = llvm.call pisa_funccc @llvm.pisa.shfl.xor(%[[VAL_1]]
    // CHECK: %[[VEC_UNDEF:.*]] = llvm.mlir.undef : vector<2xi32>
    // CHECK: %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[NEW_VEC_0:.*]] = llvm.insertelement %[[NEW_VAL_0]], %[[VEC_UNDEF]][%[[CST_0]] : i32]
    // CHECK: %[[CST_1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: %[[NEW_VEC_1:.*]] = llvm.insertelement %[[NEW_VAL_1]], %[[NEW_VEC_0]][%[[CST_1]] : i32]
    // CHECK: %[[RES:.*]] = llvm.bitcast %[[NEW_VEC_1]] : vector<2xi32> to i64
    // CHECK: llvm.store %[[RES]]
    %shfl_val, %valid = gpu.shuffle xor %val, %offset, %width : i64
    %out_ptr = llvm.getelementptr %arg1[%id] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i64
    llvm.store %shfl_val, %out_ptr {alignment = 8 : i64} : i64, !llvm.ptr<1>
    llvm.return
  }
}
