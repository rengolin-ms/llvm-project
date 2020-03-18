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
mlir::Type Generator::ConvertType(AST::Type type, size_t dim) {
  switch (type) {
  case AST::Type::None:
    return builder.getNoneType();
  case AST::Type::Bool:
    return builder.getI1Type();
  case AST::Type::Integer:
    return builder.getIntegerType(64);
  case AST::Type::Float:
    return builder.getF64Type();
  case AST::Type::Vector:
    // FIXME: support nested vectors, ex: (Vec (Vec Ty))
    assert(type.getSubType() != AST::Type::Vector);
    if (dim)
      return mlir::MemRefType::get(dim, ConvertType(type.getSubType()));
    else
      return mlir::UnrankedMemRefType::get(ConvertType(type.getSubType()), 0);
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
  if (mlir::failed(mlir::verify(*module)))
    return nullptr;
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
  auto last = buildNode(def->getImpl());

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
  if (AST::Build::classof(node))
    return buildBuild(llvm::dyn_cast<AST::Build>(node));
  if (AST::Index::classof(node))
    return buildIndex(llvm::dyn_cast<AST::Index>(node));
  if (AST::Size::classof(node))
    return buildSize(llvm::dyn_cast<AST::Size>(node));
  // TODO: Implement all node types
  assert(0 && "unexpected node");
}

// Builds blocks
mlir::Value Generator::buildBlock(const AST::Block* block) {
  if (block->size() == 0)
    return mlir::Value();
  if (block->size() == 1)
    return buildNode(block->getOperand(0));
  for (auto &op: block->getOperands())
    buildNode(op.get());
  return mlir::Value();
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

// Builds variable declarations
mlir::Value Generator::buildLet(const AST::Let* let) {
  // Bind the variable to an expression
  // TODO: Use context
  for (auto &v: let->getVariables())
    declareVariable(llvm::dyn_cast<AST::Variable>(v.get()));
  // Lower the body, using the variable
  if (let->getExpr())
    return buildNode(let->getExpr());
  // Otherwise, the let is just a declaration, return void
  return mlir::Value();
}

// Builds conditions using select
mlir::Value Generator::buildCond(const AST::Condition* cond) {

  // Constant booleans aren't allowed on selects / cond_branch in LLVM
  auto lit = llvm::dyn_cast<AST::Literal>(cond->getCond());
  if (lit) {
    if (lit->getValue() == "true")
      return buildNode(cond->getIfBlock());
    else
      return buildNode(cond->getElseBlock());
  }

  // Check for the boolean result of the conditional block
  auto i = buildNode(cond->getIfBlock());
  auto e = buildNode(cond->getElseBlock());
  auto c = buildNode(cond->getCond());
  assert(i.getType() == e.getType() && "Type mismatch");
  return builder.create<mlir::SelectOp>(UNK, c, i, e);
}

// Builds loops creating vectors [WIP]
mlir::Value Generator::buildBuild(const AST::Build* b) {
  // Declare the bounded vector variable and allocate it
  // FIXME: Allow dim to be a dynamic runtime expression
  auto lit = llvm::dyn_cast<AST::Literal>(b->getRange());
  assert(lit && "Dynamic range not supported");
  size_t dim = std::atol(lit->getValue().str().c_str());
  auto ivTy = ConvertType(lit->getType());
  auto vecTy = mlir::MemRefType::get(dim, ivTy);
  auto vec = builder.create<mlir::AllocOp>(UNK, vecTy);

  // Declare the range, initialised with zero
  auto zeroAttr = builder.getIntegerAttr(ivTy, 0);
  auto zero = builder.create<mlir::ConstantOp>(UNK, ivTy, zeroAttr);
  auto range = buildNode(b->getRange());

  // Create all basic blocks and the condition
  auto headBlock = currentFunc.addBlock();
  headBlock->addArgument(ivTy);
  auto bodyBlock = currentFunc.addBlock();
  bodyBlock->addArgument(ivTy);
  auto exitBlock = currentFunc.addBlock();
  mlir::ValueRange indArg {zero};
  builder.create<mlir::BranchOp>(UNK, headBlock, indArg);

  // HEAD BLOCK: Compare induction with range, exit if equal or greater
  builder.setInsertionPointToEnd(headBlock);
  auto headIv = headBlock->getArgument(0);
  auto cond = builder.create<mlir::CmpIOp>(UNK, mlir::CmpIPredicate::slt, headIv, range);
  mlir::ValueRange bodyArg {headIv};
  mlir::ValueRange exitArgs {};
  builder.create<mlir::CondBranchOp>(UNK, cond, bodyBlock, bodyArg, exitBlock, exitArgs);

  // BODY BLOCK: Lowers expression, store and increment
  builder.setInsertionPointToEnd(bodyBlock);
  auto bodyIv = bodyBlock->getArgument(0);
  // Declare the local induction variable before using in body
  auto var = llvm::dyn_cast<AST::Variable>(b->getVariable());
  declareVariable(var);
  variables[var->getName()] = bodyIv;
  // Build body and store result
  auto expr = buildNode(b->getExpr());
  auto indTy = builder.getIndexType();
  auto indIv = builder.create<mlir::IndexCastOp>(UNK, bodyIv, indTy);
  mlir::ValueRange indices{indIv};
  builder.create<mlir::StoreOp>(UNK, expr, vec, indices);
  // Increment induction and loop
  auto oneAttr = builder.getIntegerAttr(ivTy, 1);
  auto one = builder.create<mlir::ConstantOp>(UNK, ivTy, oneAttr);
  auto incr = builder.create<mlir::AddIOp>(UNK, bodyIv, one);
  mlir::ValueRange headArg {incr};
  builder.create<mlir::BranchOp>(UNK, headBlock, headArg);

  // EXIT BLOCK: change insertion point before returning the final vector
  builder.setInsertionPointToEnd(exitBlock);
  return vec;
}

// Builds index access to vectors
mlir::Value Generator::buildIndex(const AST::Index* i) {
  auto idx = buildNode(i->getIndex());
  auto var = llvm::dyn_cast<AST::Variable>(i->getVariable());
  auto vec = variables[var->getName()];
  auto indTy = builder.getIndexType();
  auto indIdx = builder.create<mlir::IndexCastOp>(UNK, idx, indTy);
  mlir::ValueRange rangeIdx {indIdx};
  return builder.create<mlir::LoadOp>(UNK, vec, rangeIdx);
}

// Builds size of vector operator
mlir::Value Generator::buildSize(const AST::Size* s) {
  auto var = llvm::dyn_cast<AST::Variable>(s->getVariable());
  auto vec = variables[var->getName()];
  // FIXME: Support multi-dimensional vectors
  auto dim = builder.create<mlir::DimOp>(UNK, vec, 0);
  auto intTy = builder.getIntegerType(64);
  return builder.create<mlir::IndexCastOp>(UNK, dim, intTy);
}

// Lower constant literals
mlir::Attribute Generator::getAttr(const AST::Expr* op) {
  auto lit = llvm::dyn_cast<AST::Literal>(op);
  assert(lit && "Can only get attributes from lits");
  switch (lit->getType()) {
  case AST::Type::Bool:
    if (lit->getValue() == "true")
      return builder.getBoolAttr(true);
    else
      return builder.getBoolAttr(false);
  case AST::Type::Float:
    return builder.getFloatAttr(builder.getF64Type(),
                                std::atof(lit->getValue().str().c_str()));
  case AST::Type::Integer:
    return builder.getI64IntegerAttr(std::atol(lit->getValue().str().c_str()));
  case AST::Type::String:
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
  if (mlir::failed(mlir::verify(*module)))
    return nullptr;
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
