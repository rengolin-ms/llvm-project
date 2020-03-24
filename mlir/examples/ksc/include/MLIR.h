/* Copyright Microsoft Corp. 2020 */
#ifndef _MLIR_H_
#define _MLIR_H_

#include <map>

#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/InitAllDialects.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"

#include "AST.h"

namespace Knossos {
namespace MLIR {

// MLIR generator
using Types = llvm::SmallVector<mlir::Type, 4>;
using Values = llvm::SmallVector<mlir::Value, 4>;
class Generator {
  // The main module
  mlir::OwningModuleRef module;
  // The current builder
  mlir::OpBuilder builder;
  // The global context
  mlir::MLIRContext context;
  // A default location, ignoring source loc for now
  mlir::Location UNK;
  // Current function (for basic block placement)
  mlir::FuncOp currentFunc;

  // Cache for functions and variables
  std::map<llvm::StringRef, mlir::FuncOp> functions;
  std::map<llvm::StringRef, Values> variables;

  // Helpers
  Types ConvertType(const AST::Type &type, size_t dim=0);
  mlir::Value memrefCastForCall(mlir::Value orig);
  mlir::Attribute getAttr(const AST::Expr* op);

  // Module level builders
  void buildGlobal(const AST::Block* block);
  mlir::FuncOp buildDecl(const AST::Declaration* decl);
  mlir::FuncOp buildDef(const AST::Definition* def);

  // Function level builders
  Values buildNode(const AST::Expr* node);
  Values buildBlock(const AST::Block* block);
  Values buildOp(const AST::Operation* op);
  Values buildCond(const AST::Condition* cond);
  Values buildLet(const AST::Let* let);
  Values buildLiteral(const AST::Literal* lit);
  Values buildVariable(const AST::Variable* var);
  void declareVariable(llvm::StringRef name, Values vals);
  void declareVariable(const AST::Variable* var, Values vals = {});
  Values buildBuild(const AST::Build* b);
  Values buildIndex(const AST::Index* i);
  Values buildSize(const AST::Size* s);
  Values buildTuple(const AST::Tuple* t);
  Values buildGet(const AST::Get* g);
  Values buildFold(const AST::Fold* g);

  void serialiseArgs(const AST::Definition *def, mlir::Block &entry);


public:
  Generator() : builder(&context), UNK(builder.getUnknownLoc()) { }

  // Build from MLIR source
  const mlir::ModuleOp build(const std::string& mlir);
  // Build from KSC AST
  const mlir::ModuleOp build(const AST::Expr* root);
  // Emit LLVM IR
  std::unique_ptr<llvm::Module> emitLLVM(int optLevel=0);
};

} // namespace MLIR
} // namespace Knossos

#endif // _MLIR_H_
