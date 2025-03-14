//===- PISATargetInfo.cpp - PISATargetInfo implementation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PISATargetInfo.h"

#include "intel/lib/Utils/LLVMIntr.h"

using namespace mlir;

namespace mlir::triton::intel {

bool PISATargetInfo::isSupportedWarpReduceOp(Operation *op,
                                             unsigned numLanesToReduce,
                                             unsigned warpSize) const {
  // Only uniform reductions are supported.
  if (numLanesToReduce != warpSize)
    return false;

  // Expect a single output value.
  if (op->getNumResults() != 1)
    return false;

  Type resTy = op->getResult(0).getType();
  // pISA fred operation has following restictions:
  //   Types: fp32, fp16, bf16, fp16x2, bf16x2
  //   Reduction operations: min, max
  if (resTy.isF32() || resTy.isF16() || resTy.isBF16()) {
    return isa<arith::MaxNumFOp, arith::MinNumFOp>(op);
  }

  // pISA ired operation has following restrictions:
  //   Types: i16, i32
  //   Reduction operations: sum, smin, smax, umin, umax, and, or, xor
  if (resTy.isInteger(32) || resTy.isInteger(16)) {
    return isa<arith::AddIOp, arith::MaxSIOp, arith::MaxUIOp, arith::MinSIOp,
               arith::MinUIOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp>(op);
  }

  return false;
}

Value PISATargetInfo::genWarpReduce(RewriterBase &rewriter, Location loc,
                                    Value acc, Operation *reduceOp,
                                    unsigned numLanesToReduce,
                                    unsigned warpSize) const {
  std::string funcName = std::string("llvm.pisa.");

  // Select reduction op.
  if (isa<arith::MaxNumFOp>(reduceOp))
    funcName += "fred.max";
  else if (isa<arith::MinNumFOp>(reduceOp))
    funcName += "fred.min";
  else if (isa<arith::AddIOp>(reduceOp))
    funcName += "ired.sum";
  else if (isa<arith::MaxSIOp>(reduceOp))
    funcName += "ired.smax";
  else if (isa<arith::MaxUIOp>(reduceOp))
    funcName += "ired.umax";
  else if (isa<arith::MinSIOp>(reduceOp))
    funcName += "ired.smin";
  else if (isa<arith::MinUIOp>(reduceOp))
    funcName += "ired.umin";
  else if (isa<arith::AndIOp>(reduceOp))
    funcName += "ired.and";
  else if (isa<arith::OrIOp>(reduceOp))
    funcName += "ired.or";
  else if (isa<arith::XOrIOp>(reduceOp))
    funcName += "ired.xor";
  else
    llvm_unreachable("PISATargetInfo::genWarpReduce: unexpected reduction op");

  // Select data type.
  Type resTy = reduceOp->getResult(0).getType();
  if (resTy.isInteger(16))
    funcName += ".i16";
  else if (resTy.isInteger(32))
    funcName += ".i32";
  else if (resTy.isF32())
    funcName += ".f32";
  else if (resTy.isF16())
    funcName += ".f16";
  else if (resTy.isBF16())
    funcName += ".bf16";
  else
    llvm_unreachable("PISATargetInfo::genWarpReduce: unexpected data types");

  Value mask = rewriter.create<LLVM::ConstantOp>(
      loc, rewriter.getI32Type(), (1ULL << numLanesToReduce) - 1);

  auto funcAttrs = triton::gpu::intel::convergentNoUnwindWillReturnAttrs;
  funcAttrs.memEffectsAttr = rewriter.getAttr<LLVM::MemoryEffectsAttr>(
      /*other=*/LLVM::ModRefInfo::NoModRef,
      /*argMem=*/LLVM::ModRefInfo::NoModRef,
      /*inaccessibleMem=*/LLVM::ModRefInfo::NoModRef);
  LLVM::CallOp call = createDeviceFunctionCall(
      rewriter, funcName, resTy, {resTy, rewriter.getI32Type()}, {acc, mask},
      {}, funcAttrs, {}, LLVM::cconv::CConv::PISA_FUNC);
  return call.getResult();
}

} // namespace mlir::triton::intel
