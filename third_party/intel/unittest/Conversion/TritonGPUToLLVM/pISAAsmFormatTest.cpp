#include "intel/include/TritonIntelGPUToLLVM/pISAAsmFormat.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/Signals.h"

#include <gtest/gtest.h>

namespace mlir {
namespace triton {
class pISAAsmFormatTest : public ::testing::Test {
protected:
  static constexpr int numValues = 4;

  pISAAsmFormatTest() {
    ctx.loadDialect<arith::ArithDialect>();

    createValues();
  }

  // Creates the test values.
  void createValues() {
    OpBuilder builder(&ctx);
    builder.setInsertionPointToStart(&block);

    // a b1 value for predicate.
    v[0] = builder.create<arith::ConstantIntOp>(builder.getUnknownLoc(), 1, 1);
    for (int i = 0; i < numValues; i++) {
      v[i + 1] =
          builder.create<arith::ConstantIntOp>(builder.getUnknownLoc(), i, 32);
    }
  }

  MLIRContext ctx;
  Block block;
  Value v[numValues + 1];
};

TEST_F(pISAAsmFormatTest, basic) {
  intel::pISABuilder builder;

  // Create the operands needed by the instructions in the pISA code.
  auto *cst = builder.newConstantOperand(1);
  auto *val = builder.newOperand(v[1], "=r");

  // create an instruction
  auto &mov = *builder.create("mov.16b");

  mov(val, cst).predicate(v[0]);
  ASSERT_EQ(builder.dump(), "@$1 mov.16b $0, 0x1;");

  auto values = builder.getAllMLIRArgs();
  ASSERT_EQ(values[0], v[1]); // $0 -> v[1]
  ASSERT_EQ(values[1], v[0]); // $1 -> v[0]

  auto constraints = builder.getConstraints();
  ASSERT_EQ(constraints, "=r,b"); // $0 -> =r, $1 -> b
}

} // namespace triton
} // namespace mlir

int main(int argc, char *argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
