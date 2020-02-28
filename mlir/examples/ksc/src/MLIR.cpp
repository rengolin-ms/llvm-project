#include <cstdlib>
#include <numeric>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SourceMgr.h"

#include "mlir/Analysis/Verifier.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Parser.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"

#include "MLIR.h"

using namespace Knossos::MLIR;
using namespace std;

//============================================================ Helpers

// Convert from AST type to MLIR
mlir::Type Generator::ConvertType(AST::Expr::Type type) {
  switch (type) {
  case AST::Expr::Type::None:
    return builder.getNoneType();
  case AST::Expr::Type::Bool:
    return builder.getI1Type();
  case AST::Expr::Type::Integer:
    return builder.getIntegerType(64);
  case AST::Expr::Type::Float:
    return builder.getF64Type();
  default:
    assert(0 && "Unsupported type");
  }
}


// Inefficient but will do for now
static void dedup_declarations(vector<mlir::FuncOp> &decl, vector<mlir::FuncOp> def) {
  for (auto &d: def) {
    auto it = std::find(decl.begin(), decl.end(), d);
    if (it != decl.end())
      decl.erase(it);
  }
}

//============================================================ MLIR Generator

// Build module, all its global functions, recursing on their
// implementations. Return the module.
const mlir::ModuleOp Generator::build(const AST::Expr* root) {
  module = mlir::ModuleOp::create(UNK);
  assert(root->kind == AST::Expr::Kind::Block);
  auto rB = llvm::dyn_cast<AST::Block>(root);
  buildGlobal(rB);
  assert(!mlir::failed(mlir::verify(*module)) && "Validation failed!");
  return module.get();
}

// Global context, allowing declarations and definitions.
void Generator::buildGlobal(const AST::Block* block) {
  // First we need to make sure we don't emit declarations when definitions
  // are available.
  // FIXME: Find a way to use the functions map instead
  vector<mlir::FuncOp> declarations;
  vector<mlir::FuncOp> definitions;
  for (auto &op: block->getOperands()) {
    switch (op->kind) {
    case AST::Expr::Kind::Declaration:
      declarations.push_back(buildDecl(llvm::dyn_cast<AST::Declaration>(op.get())));
      continue;
    case AST::Expr::Kind::Definition:
      definitions.push_back(buildDef(llvm::dyn_cast<AST::Definition>(op.get())));
      continue;
    default:
      assert(0 && "unexpected node");
    }
  }

  // Types were already checked by the AST, so if there's an
  // outline declaration already, we only insert the definition.
  dedup_declarations(declarations, definitions);

  // Now we add declarations first, then definitions
  for (auto decl: declarations)
    module->push_back(decl);
  for (auto def: definitions)
    module->push_back(def);
}

// Declaration only, no need for basic blocks
mlir::FuncOp Generator::buildDecl(const AST::Declaration* decl) {
  assert(!functions.count(decl->getName()) && "Duplicated function declaration");
  llvm::SmallVector<mlir::Type, 4> argTypes;
  for (auto &t: decl->getArgTypes())
    argTypes.push_back(ConvertType(t));
  auto retTy = ConvertType(decl->getType());
  auto type = builder.getFunctionType(argTypes, retTy);
  auto func = mlir::FuncOp::create(UNK, decl->getName(), type);
  functions.insert({decl->getName(), func});
  return func;
}

// Definition of the whole function starts here
// TODO: Add scope for variables
mlir::FuncOp Generator::buildDef(const AST::Definition* def) {
  // Make sure we have its declaration cached
  if (!functions.count(def->getName()))
    buildDecl(def->getProto());
  auto func = functions[def->getName()];
  assert(func);

  // First basic block, with args
  auto &entryBlock = *func.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);
  for (const auto &arg :
       llvm::zip(def->getArguments(), entryBlock.getArguments()))
    declareVariable(llvm::dyn_cast<AST::Variable>(std::get<0>(arg).get()),
                    std::get<1>(arg));

  // Lower body
  currentFunc = func;
  auto last = buildBlock(llvm::dyn_cast<AST::Block>(def->getImpl()));

  // Return the last value
  builder.create<mlir::ReturnOp>(UNK, last);
  return func;
}

// Declare a variable
// TODO: Add scope for variables
void Generator::declareVariable(const AST::Variable* var,
                                     mlir::Value val) {
  assert(!variables.count(var->getName()) && "Duplicated variable declaration");
  if (!val && var->getInit())
    val = buildNode(var->getInit());
  assert(val);
  variables.insert({var->getName(), val});
}

// Get the variable assigned value
mlir::Value Generator::buildVariable(const AST::Variable* var) {
  assert(variables.count(var->getName()) && "Variable not declared");
  return variables[var->getName()];
}

// Build node by type
mlir::Value Generator::buildNode(const AST::Expr* node) {
  if (AST::Block::classof(node))
    return buildBlock(llvm::dyn_cast<AST::Block>(node));
  if (AST::Literal::classof(node))
    return buildLiteral(llvm::dyn_cast<AST::Literal>(node));
  if (AST::Operation::classof(node))
    return buildOp(llvm::dyn_cast<AST::Operation>(node));
  if (AST::Let::classof(node))
    return buildLet(llvm::dyn_cast<AST::Let>(node));
  if (AST::Condition::classof(node))
    return buildCond(llvm::dyn_cast<AST::Condition>(node));
  if (AST::Variable::classof(node))
    return buildVariable(llvm::dyn_cast<AST::Variable>(node));
  // TODO: Implement all node types
  assert(0 && "unexpected node");
}

// Build function body
mlir::Value Generator::buildBlock(const AST::Block* block) {
  mlir::Value last;
  for (auto &op: block->getOperands())
    last = buildNode(op.get());
  return last;
}

// Builds literals
mlir::Value Generator::buildLiteral(const AST::Literal* op) {
  mlir::Type type = ConvertType(op->getType());
  return builder.create<mlir::ConstantOp>(UNK, type, getAttr(op));
}

// Builds operations/calls
mlir::Value Generator::buildOp(const AST::Operation* op) {
  auto operation = op->getName();

  // Function call
  if (functions.count(operation)) {
    auto func = functions[operation];
    assert(func.getNumArguments() == op->size() && "Arguments mismatch");

    // Operands
    llvm::SmallVector<mlir::Value, 4> operands;
    for (auto &arg: op->getOperands())
      operands.push_back(buildNode(arg.get()));

    // Function
    auto call = builder.create<mlir::CallOp>(UNK, func, operands);
    return call.getResult(0);
  }

  if (op->size() == 1) {
    // Unary operations
    //auto arg = buildNode(op->getOperand(0));
    assert(0 && "Unary operations not implemented yet");

  } else if (op->size() == 2) {
    // Binary operations
    auto lhs = buildNode(op->getOperand(0));
    auto rhs = buildNode(op->getOperand(1));
    if (operation == "add@ii")
      return builder.create<mlir::AddIOp>(UNK, lhs, rhs);
    else if (operation == "sub@ii")
      return builder.create<mlir::SubIOp>(UNK, lhs, rhs);
    else if (operation == "mul@ii")
      return builder.create<mlir::MulIOp>(UNK, lhs, rhs);
    else if (operation == "div@ii")
      return builder.create<mlir::SignedDivIOp>(UNK, lhs, rhs);
    else if (operation == "add@ff")
      return builder.create<mlir::AddFOp>(UNK, lhs, rhs);
    else if (operation == "sub@ff")
      return builder.create<mlir::SubFOp>(UNK, lhs, rhs);
    else if (operation == "mul@ff")
      return builder.create<mlir::MulFOp>(UNK, lhs, rhs);
    else if (operation == "div@ff")
      return builder.create<mlir::DivFOp>(UNK, lhs, rhs);

    assert(0 && "Unknown binary operation");
  }
  assert(0 && "Unknown operation");
}

// Builds lexical blocks
mlir::Value Generator::buildLet(const AST::Let* let) {
  // Bind the variable to an expression
  // TODO: Use context
  for (auto &v: let->getVariables())
    declareVariable(llvm::dyn_cast<AST::Variable>(v.get()));
  // Lower the body, using the variable
  return buildNode(let->getExpr());
}

// Builds conditions, create new basic blocks
mlir::Value Generator::buildCond(const AST::Condition* cond) {
  // Check for the boolean result of the conditional block
  auto condValue = buildNode(cond->getCond());

  // Create all basic blocks and the condition
  auto ifBlock = currentFunc.addBlock();
  auto elseBlock = currentFunc.addBlock();
  auto tailBlock = currentFunc.addBlock();
  mlir::ValueRange emptyArgs;
  builder.create<mlir::CondBranchOp>(UNK, condValue, ifBlock, emptyArgs, elseBlock, emptyArgs);

  // Lower if/else in their respective blocks
  builder.setInsertionPointToStart(ifBlock);
  auto ifValue = buildNode(cond->getIfBlock());
  builder.create<mlir::BranchOp>(UNK, tailBlock, ifValue);

  builder.setInsertionPointToStart(elseBlock);
  auto elseValue = buildNode(cond->getElseBlock());
  builder.create<mlir::BranchOp>(UNK, tailBlock, elseValue);

  // Merge the two blocks into a value and return
  assert(ifValue.getType() == elseValue.getType() && "Type mismatch");
  tailBlock->addArgument(ifValue.getType());
  builder.setInsertionPointToEnd(tailBlock);
  return tailBlock->getArgument(0);
}

// Lower constant literals
mlir::Attribute Generator::getAttr(const AST::Expr* op) {
  auto lit = llvm::dyn_cast<AST::Literal>(op);
  assert(lit && "Can only get attributes from lits");
  switch (lit->getType()) {
  case AST::Expr::Type::Bool:
    if (lit->getValue() == "true")
      return builder.getBoolAttr(true);
    else
      return builder.getBoolAttr(false);
  case AST::Expr::Type::Float:
    return builder.getFloatAttr(builder.getF64Type(),
                                std::atof(lit->getValue().str().c_str()));
  case AST::Expr::Type::Integer:
    return builder.getI64IntegerAttr(std::atol(lit->getValue().str().c_str()));
  case AST::Expr::Type::String:
    return builder.getStringAttr(lit->getValue());
  default:
    assert(0 && "Unimplemented literal type");
  }
}

//============================================================ MLIR round-trip

const mlir::ModuleOp Generator::build(const std::string& mlir) {
  // Parse the input mlir
  llvm::SourceMgr sourceMgr;
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> srcOrErr =
      llvm::MemoryBuffer::getMemBufferCopy(mlir);
  sourceMgr.AddNewSourceBuffer(std::move(*srcOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile(sourceMgr, builder.getContext());
  assert(!mlir::failed(mlir::verify(*module)) && "Validation failed!");
  return module.get();
}

//============================================================ LLVM IR Lowering

unique_ptr<llvm::Module> Generator::emitLLVM() {
  // The lowering pass manager
  mlir::PassManager pm(&context);
  if (optimise > 0) {
    pm.addPass(mlir::createInlinerPass());
    pm.addPass(mlir::createSymbolDCEPass());
    mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
  }
  pm.addPass(mlir::createLowerToLLVMPass());

  // First lower to LLVM dialect
  if (mlir::failed(pm.run(module.get())))
    return nullptr;

  // Then lower to LLVM IR
  auto llvmModule = mlir::translateModuleToLLVMIR(module.get());
  assert(llvmModule);
  return llvmModule;
}
