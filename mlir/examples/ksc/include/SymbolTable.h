/* Copyright Microsoft Corp. 2020 */
#ifndef _SYMBOLTABLE_H_
#define _SYMBOLTABLE_H_

#include <cassert>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "AST.h"

namespace Knossos {
namespace AST {

//================================================ Symbol Table (vars, funcs)

/// Symbol Table: holds symbols for operations, functions and variables
///
/// Overall structure:
///  * List of reserved words: language constructs, operations
///  * List of functions and variables declared (names can't clash)
///
/// TODO:
///  - Add lexical context to variables (now they're all globals)
///  - Perform recursive search on parent contexts
class SymbolTable {
  std::map<llvm::StringRef, Expr*> declared;

  bool isVariable(const Expr* expr) const {
    return expr && llvm::dyn_cast<Variable>(expr);
  }

  bool isFunction(const Expr* expr) const {
    return expr && llvm::dyn_cast<Declaration>(expr);
  }

public:
  SymbolTable() { }
  void clear() { declared.clear(); }

  /// Function symbols
  bool addFunction(llvm::StringRef name, Declaration* expr) {
    const auto result = declared.insert({name, expr});
    return result.second;
  }
  bool addFunction(llvm::StringRef name, Definition* expr) {
    auto fun = declared.find(name);

    // If already exists, must be a declaration, types must match
    if (fun != declared.end()) {
      auto cur = llvm::dyn_cast<Declaration>(fun->second);
      if (!cur)
        return false;
      // Make sure the types are the same
      if (cur->getType() != expr->getType())
        return false;
      if (cur->size() != expr->size())
        return false;
      for (auto arg : llvm::zip(cur->getArgTypes(), expr->getArguments())) {
        if (std::get<0>(arg) != std::get<1>(arg)->getType())
          return false;
      }
      return true;
    }

    // Add the declaration
    const auto result = declared.insert({name, expr->getProto()});
    return result.second;
  }
  bool existsFunction(llvm::StringRef name) const {
    auto fun = declared.find(name);
    return (fun != declared.end() && isFunction(fun->second));
  }
  Declaration* getFunction(llvm::StringRef name) {
    auto fun = declared.find(name);
    if (fun == declared.end())
      return nullptr;
    if (!isFunction(fun->second))
      return nullptr;
    return llvm::dyn_cast<Declaration>(fun->second);
  }

  /// Variable symbols
  bool addVariable(llvm::StringRef name, Variable* expr) {
    auto fun = declared.find(name);

    // If already exists, must not have init, types must match
    if (fun != declared.end()) {
      auto cur = llvm::dyn_cast<Variable>(fun->second);
      if (cur->getType() != expr->getType())
        return false;
      if (cur->getInit())
        return false;
    }

    const auto result = declared.insert({name, expr});
    return result.second;
  }
  bool existsVariable(llvm::StringRef name) const {
    auto var = declared.find(name);
    return (var != declared.end() && isVariable(var->second));
  }
  Variable* getVariable(llvm::StringRef name) {
    auto fun = declared.find(name);
    if (fun == declared.end())
      return nullptr;
    if (!isVariable(fun->second))
      return nullptr;
    return llvm::dyn_cast<Variable>(fun->second);
  }
};

} // namespace AST
} // namespace Knossos
#endif /// _SYMBOLTABLE_H_
