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

} // namespace

void mlir::triton::intel::populateXe4ToLLVMPatterns(
    TritonIntelGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<IndexLowering<mlir::gpu::BlockIdOp>>(typeConverter, "groupid");
  patterns.add<IndexLowering<mlir::gpu::BlockDimOp>>(typeConverter,
                                                     "localsize");
  patterns.add<IndexLowering<mlir::gpu::ThreadIdOp>>(typeConverter, "localid");
  patterns.add<BarrierLowering>(typeConverter);
}
