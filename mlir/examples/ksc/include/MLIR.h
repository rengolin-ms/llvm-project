/* Copyright Microsoft Corp. 2020 */
#ifndef _MLIR_H_
#define _MLIR_H_

#include <map>

#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/StringRef.h"

#include "AST.h"

namespace Knossos {
namespace MLIR {

// MLIR generator
class Generator {
  // The main module
  mlir::OwningModuleRef module;
  // The current builder
  mlir::OpBuilder builder;
  // The lowering pass manager
  mlir::PassManager pm;
  // A default location, ignoring source loc for now
  mlir::Location UNK;
  // Current function (for basic block placement)
  mlir::FuncOp currentFunc;

  // Cache for functions and variables
  std::map<llvm::StringRef, mlir::FuncOp> functions;
  std::map<llvm::StringRef, mlir::Value> variables;

  // Helpers
  mlir::Type ConvertType(AST::Expr::Type type);
  mlir::Attribute getAttr(const AST::Expr* op);

  // Module level builders
  void buildGlobal(const AST::Block* block);
  mlir::FuncOp buildDecl(const AST::Declaration* decl);
  mlir::FuncOp buildDef(const AST::Definition* def);

  // Function level builders
  mlir::Value buildNode(const AST::Expr* node);
  mlir::Value buildBlock(const AST::Block* block);
  mlir::Value buildOp(const AST::Operation* op);
  mlir::Value buildCond(const AST::Condition* cond);
  mlir::Value buildLet(const AST::Let* let);
  mlir::Value buildLiteral(const AST::Literal* lit);
  mlir::Value buildVariable(const AST::Variable* var);
  void declareVariable(const AST::Variable* var,
                            mlir::Value val = nullptr);

public:
  Generator(mlir::MLIRContext &context)
      : builder(&context), pm(&context), UNK(builder.getUnknownLoc()) {}

  const mlir::ModuleOp build(const AST::Expr* root);
};

} // namespace MLIR
} // namespace Knossos

#endif // _MLIR_H_
