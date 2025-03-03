#include "intel/include/TritonIntelGPUToLLVM/pISAAsmFormat.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/AsmFormat.h"
#include "llvm/Support/raw_ostream.h"

#include <sstream>

namespace mlir::triton::intel {

pISAInstr::Operand *
pISABuilder::newOperand(mlir::Value value, StringRef constraint,
                        std::function<std::string(int)> formatter) {
  argArchive.emplace_back(std::make_unique<Operand>(value, constraint));
  auto *opr = argArchive.back().get();
  opr->repr = formatter;
  opr->idx = oprCounter++;
  return opr;
}

void pISABuilder::initOperand(Operand *opr) {
  auto numBits = 0;
  // Derive numBits from the constraint.
  if (opr->constraint[1] == 'c' || opr->constraint[1] == 'h')
    numBits = 16;
  else if (opr->constraint[1] == 'r')
    numBits = 32;
  else if (opr->constraint[1] == 'l')
    numBits = 64;
  else
    llvm_unreachable(("Unknown constraint: " + opr->constraint).c_str());
  // If numBits is less than 16, we use 16 as default because pISA does not
  // support 8-bit mov.
  numBits = numBits < 16 ? 16 : numBits;
  auto *zero = newConstantOperand(0);
  auto &init = create<>("mov")->o("u" + std::to_string(numBits));
  init(opr, zero);
}

pISABuilder::Operand *pISABuilder::newOperand(StringRef constraint, bool init) {
  // Constraint should be something like "=r"
  assert(constraint.size() == 2 && constraint[0] == '=');
  auto *opr = newOperand();
  opr->idx = oprCounter++;
  opr->constraint = constraint;
  if (init) {
    initOperand(opr);
  }
  return opr;
}

pISABuilder::Operand *pISABuilder::newOperand(unsigned operandIndex) {
  assert(operandIndex < oprCounter && "operand index out of range");
  auto *opr = newOperand();
  opr->idx = operandIndex;
  opr->constraint = std::to_string(operandIndex);
  return opr;
}

pISABuilder::Operand *pISABuilder::newConstantOperand(const std::string &v) {
  argArchive.emplace_back(std::make_unique<Operand>());
  argArchive.back()->repr = [v](int idx) { return v; };
  return argArchive.back().get();
}

pISABuilder::Operand *pISABuilder::newConstantOperand(int64_t v) {
  std::stringstream ss;
  ss << "0x" << std::hex << v;
  return newConstantOperand(ss.str());
}

std::string pISABuilder::getConstraints() const {
  auto args = getAllArgs();
  llvm::SmallVector<std::string, 4> argReprs;
  for (auto arg : args)
    argReprs.push_back(arg->constraint);
  return strJoin(argReprs, ",");
}

llvm::SmallVector<Value, 4> pISABuilder::getAllMLIRArgs() const {
  llvm::SmallVector<Value, 4> res;
  for (auto &arg : argArchive) {
    if (!arg->isList() && arg->value)
      res.push_back(arg->value);
  }
  return res;
}

SmallVector<pISABuilder::Operand *, 4> pISABuilder::getAllArgs() const {
  llvm::SmallVector<Operand *, 4> res;
  for (auto &x : argArchive)
    if (!x->isList())
      res.push_back(x.get());
  return res;
}

mlir::Value pISABuilder::launch(OpBuilder &rewriter, Location loc, Type resTy,
                                bool hasSideEffect, bool isAlignStack,
                                ArrayRef<Attribute> attrs) const {
  auto ctx = rewriter.getContext();
  auto inlineAsm = rewriter.create<LLVM::InlineAsmOp>(
      loc, resTy, getAllMLIRArgs(), // operands
      dump(),                       // asm_string
      getConstraints(),             // constraints
      hasSideEffect,                // has_side_effects
      isAlignStack,                 // is_align_stack
      LLVM::AsmDialectAttr::get(ctx,
                                LLVM::AsmDialect::AD_ATT), // asm_dialect
      ArrayAttr::get(ctx, attrs)                           // operand_attrs
  );

  return inlineAsm.getRes();
}

std::string pISAInstr::Operand::dump() const {
  if (repr)
    return repr(idx);
  if (!isList())
    return "$" + std::to_string(idx);

  llvm::SmallVector<std::string> oprs;
  for (auto *opr : list)
    oprs.push_back(opr->dump());
  return "{" + strJoin(oprs, ", ") + " }";
}

pISAInstr::Operand *pISABuilder::newAddrOperand(mlir::Value addr,
                                                StringRef constraint, int off) {
  auto *opr = newOperand(addr, constraint);
  opr->repr = [off](int idx) -> std::string {
    std::stringstream ss;
    ss << "[ $" << idx << " + " << off << " ]";
    return ss.str();
  };

  return opr;
}

std::string pISABuilder::dump() const {
  llvm::SmallVector<std::string> lines;
  for (auto &exec : executions) {
    lines.push_back(exec->dump());
  }

  return strJoin(lines, "\n\t");
}

pISAInstrExecution &pISAInstrCommon::call(ArrayRef<Operand *> oprs,
                                          bool onlyAttachMLIRArgs) {
  if (onlyAttachMLIRArgs) {
    // Nearly impossible to make the $0,$1 in two pISA code snippets to point to
    // the same MLIR values in onlyAttachMLIRArgs mode.
    assert(builder->executions.empty() &&
           "builder can only hold a single execution when onlyAttachMIIRArgs "
           "is true.");
    builder->reorderArgArchive(oprs);
  }

  builder->executions.emplace_back(
      std::make_unique<pISAInstrExecution>(this, oprs, onlyAttachMLIRArgs));

  return *builder->executions.back();
}

pISAInstrExecution &pISAInstrCommon::operator()(ArrayRef<Operand *> oprs,
                                                bool onlyAttachMLIRArgs) {
  return call(oprs, onlyAttachMLIRArgs);
}

std::string pISAInstrExecution::dump() const {
  std::string osStr;
  llvm::raw_string_ostream os(osStr);

  if (pred) {
    if (!pred->repr)
      os << "@" << pred->dump() << " ";
    else
      os << pred->repr(pred->idx) << " ";
  }

  std::string instrRepr = strJoin(instr->instrParts, ".");
  if (onlyAttachMLIRArgs) {
    os << instrRepr;
    os.flush();
    return osStr;
  }

  llvm::SmallVector<std::string, 4> argReprs;
  for (auto *arg : argsInOrder) {
    argReprs.push_back(arg->dump());
  }

  std::string argsRepr = strJoin(argReprs, ", ");

  os << instrRepr << " " << argsRepr << ";";
  os.flush();
  return osStr;
}

SmallVector<pISAInstrExecution::Operand *>
pISAInstrExecution::getArgList() const {
  SmallVector<Operand *> args;
  for (auto *arg : argsInOrder) {
    if (arg->isList())
      args.insert(args.end(), arg->list.begin(), arg->list.end());
    else
      args.push_back(arg);
  }
  return args;
}

pISAInstr &pISAInstr::global() {
  o("global");
  return *this;
}

pISAInstr &pISAInstr::shared() {
  o("shared");
  return *this;
}

pISAInstr &pISAInstr::v(int vecWidth, bool predicate) {
  if (vecWidth > 1) {
    o("v" + std::to_string(vecWidth), predicate);
  }
  return *this;
}

pISAInstr &pISAInstr::b(int width) {
  o("b" + std::to_string(width));
  return *this;
}

} // namespace mlir::triton::intel
