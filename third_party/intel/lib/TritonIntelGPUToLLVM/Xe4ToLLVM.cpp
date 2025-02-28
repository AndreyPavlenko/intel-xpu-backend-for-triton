#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "Utils/LLVMIntr.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
using namespace mlir::triton::gpu;
using namespace mlir::triton::gpu::intel;

namespace {

template <typename Op>
struct IndexLowering : public ConvertOpToLLVMPattern<Op> {
  explicit IndexLowering(const LLVMTypeConverter &converter, StringRef name)
      : ConvertOpToLLVMPattern<Op>(converter), name(name) {}
  using ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto dim = unsigned(op.getDimension());
    // clang-format off
    std::string funcName = std::string("llvm.pisa.") + name;
    switch (dim) {
    case 0: funcName = funcName + ".x"; break;
    case 1: funcName = funcName + ".y"; break;
    case 2: funcName = funcName + ".z"; break;
    default: llvm_unreachable("invalid dimension");
    }
    // clang-format on
    Type resTy =
        this->getTypeConverter()->convertType(op.getResult().getType());
    LLVM::CallOp call =
        createDeviceFunctionCall(rewriter, funcName, resTy, {}, {}, {}, {}, {},
                                 LLVM::cconv::CConv::PISA_FUNC);
    rewriter.replaceOp(op, call);
    return success();
  }

  std::string name;
};

struct BarrierLowering : public ConvertOpToLLVMPattern<mlir::gpu::BarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mlir::gpu::BarrierOp op,
                  mlir::gpu::BarrierOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    // In pISA we use a fence to sync memory and a barrier to sync control flow.
    // So gpu.barrier is replaced with a sequence:
    //   fence syncscope("workgroup-shared") acq_rel
    //   llvm.pisa.workgroup.barrier
    rewriter.create<LLVM::FenceOp>(loc, LLVM::AtomicOrdering::acq_rel,
                                   "workgroup-shared");
    createDeviceFunctionCall(rewriter, "llvm.pisa.workgroup.barrier",
                             LLVM::LLVMVoidType::get(getContext()), {}, {}, {},
                             {}, {}, LLVM::cconv::CConv::PISA_FUNC);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ShuffleLowering : public ConvertOpToLLVMPattern<mlir::gpu::ShuffleOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  // Check that width operand is an integer constant matching subgroup size.
  bool checkWidth(mlir::gpu::ShuffleOp op) const {
    auto mod = op->getParentOfType<ModuleOp>();
    auto expectedWidth = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    llvm::APInt actualWidth;
    return matchPattern(op.getWidth(), m_ConstantInt(&actualWidth)) &&
           expectedWidth == actualWidth;
  }

  // pISA shuffle works only on i32 values. Smaller values should be bitcasted
  // to integer values and extended to 32 bits. 64-bit values should be
  // bitcasted to i64 and split into a pair of 32-bit values.
  SmallVector<Value> prepareValue(Value val,
                                  ConversionPatternRewriter &rewriter) const {
    TritonLLVMOpBuilder b(val.getLoc(), rewriter);
    Type ty = val.getType();
    // First, bitcast to integer value.
    if (isa<LLVM::LLVMPointerType>(ty))
      val = b.ptrtoint(i64_ty, val);
    else if (!ty.isInteger())
      val = b.bitcast(val, int_ty(ty.getIntOrFloatBitWidth()));
    // Now, extend or split value.
    SmallVector<Value> res;
    if (val.getType().getIntOrFloatBitWidth() < 32)
      res.push_back(b.zext(i32_ty, val));
    else if (val.getType().getIntOrFloatBitWidth() == 32)
      res.push_back(val);
    else {
      assert(val.getType().isInteger(64));
      val = b.bitcast(val, vec_ty(i32_ty, 2));
      res.push_back(b.extract_element(val, b.i32_val(0)));
      res.push_back(b.extract_element(val, b.i32_val(1)));
    }
    return res;
  }

  // Restore value of the original type after shuffle.
  Value restoreValue(Type ty, ArrayRef<Value> vals,
                     ConversionPatternRewriter &rewriter) const {
    Value val = vals.front();
    TritonLLVMOpBuilder b(val.getLoc(), rewriter);
    // Concatenate values if required.
    if (vals.size() > 1) {
      assert(vals.size() == 2);
      val = b.undef(vec_ty(i32_ty, 2));
      b.insert_element(val, vals[0], b.i32_val(0));
      b.insert_element(val, vals[1], b.i32_val(1));
      val = b.bitcast(val, i64_ty);
    }

    if (isa<LLVM::LLVMPointerType>(ty))
      return b.inttoptr(ty, val);

    if (ty.getIntOrFloatBitWidth() < val.getType().getIntOrFloatBitWidth())
      val = b.trunc(int_ty(ty.getIntOrFloatBitWidth()), val);

    if (!ty.isInteger())
      val = b.bitcast(val, ty);

    return val;
  }

  SmallVector<Value> genShuffle(Location loc, ArrayRef<Value> vals,
                                Value offset, mlir::gpu::ShuffleMode mode,
                                ConversionPatternRewriter &rewriter) const {
    TritonLLVMOpBuilder b(loc, rewriter);
    std::string fnName = "llvm.pisa.shfl.";
    switch (mode) {
    case mlir::gpu::ShuffleMode::DOWN:
      fnName += "down";
      break;
    case mlir::gpu::ShuffleMode::UP:
      fnName += "up";
      break;
    case mlir::gpu::ShuffleMode::XOR:
      fnName += "xor";
      break;
    case mlir::gpu::ShuffleMode::IDX:
      fnName += "idx";
      break;
    default:
      llvm_unreachable("unexpected shuffle mode");
    }

    SmallVector<Value> res;
    for (auto val : vals) {
      auto call = createDeviceFunctionCall(
          rewriter, fnName, val.getType(), {i32_ty, i32_ty, i32_ty, i32_ty},
          {val, offset, b.i32_val(0), b.i32_val(-1)}, {}, {}, {},
          LLVM::cconv::CConv::PISA_FUNC);
      res.push_back(call.getResult());
    }
    return res;
  }

  LogicalResult
  matchAndRewrite(mlir::gpu::ShuffleOp op,
                  mlir::gpu::ShuffleOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    TritonLLVMOpBuilder b(op.getLoc(), rewriter);
    if (!checkWidth(op))
      return rewriter.notifyMatchFailure(op, "unexpected shuffle width value");

    auto vals = prepareValue(adaptor.getValue(), rewriter);
    auto shuffled = genShuffle(op.getLoc(), vals, adaptor.getOffset(),
                               adaptor.getMode(), rewriter);
    auto res = restoreValue(adaptor.getValue().getType(), shuffled, rewriter);
    rewriter.replaceOp(op, {res, b.true_val()});
    return success();
  }
};

} // namespace

void mlir::triton::intel::populateXe4ToLLVMPatterns(
    TritonIntelGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<IndexLowering<mlir::gpu::BlockIdOp>>(typeConverter, "groupid");
  patterns.add<IndexLowering<mlir::gpu::BlockDimOp>>(typeConverter,
                                                     "localsize");
  patterns.add<IndexLowering<mlir::gpu::ThreadIdOp>>(typeConverter, "localid");
  patterns.add<IndexLowering<mlir::gpu::GridDimOp>>(typeConverter,
                                                    "groupcount");
  patterns.add<BarrierLowering>(typeConverter);
  patterns.add<ShuffleLowering>(typeConverter);
}
