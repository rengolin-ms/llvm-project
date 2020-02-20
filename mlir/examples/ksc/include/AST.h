/* Copyright Microsoft Corp. 2020 */
#ifndef _AST_H_
#define _AST_H_

#include <cassert>
#include <memory>
#include <string>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

namespace Knossos {
namespace AST {

/// Code location, for MLIR generation
struct Location {
  std::string filename = "";
  size_t line = 0;
  size_t column = 0;
};

/// A node in the AST.
struct Expr {
  using Ptr = std::unique_ptr<Expr>;

  /// Expr type, for quick checking in switch/cases
  enum class Kind {
    Invalid,
    Block,
    Type,
    Literal,
    Variable,
    Let,
    Declaration,
    Definition,
    Condition,
    Operation,
    /// Unused below (TODO: Implement those)
    Fold,
    Lambda,
    Tuple,
    Apply,
    Assert
  };

  /// Type checking occurs from bottom up. Each node is responsible for its own
  /// checks and, if valid, their return type can be used for the parents.
  /// Function and variable types are checked by the symbol table
  enum class Type {
    None,
    String,
    Bool,
    Integer,
    Float,
    LAST = Float,
    /// Unused below (TODO: Implement those)
    Tuple,
    Vec,
    Lambda,
    LM
  };

  /// valid types, for safety checks
  bool isValidType() const { return type > Type::None && type <= Type::LAST; }
  Type getType() const { return type; }

  /// Type of the node, for quick access
  const Kind kind;
  /// Future place for source location
  const Location loc;

  virtual ~Expr() = default;
  virtual void dump(size_t tab = 0) const;

protected:
  Expr(Type type, Kind kind) : kind(kind), type(type) {}

  /// Type it returns, for type checking
  Type type;
};

/// Block node has nothing but children
struct Block : public Expr {
  using Ptr = std::unique_ptr<Block>;
  Block() : Expr(Expr::Type::None, Expr::Kind::Block) {}
  Block(Expr::Ptr op) : Expr(Expr::Type::None, Expr::Kind::Block) {
    operands.push_back(std::move(op));
  }

  void addOperand(Expr::Ptr node) {
    type = node->getType();
    operands.push_back(std::move(node));
  }
  llvm::ArrayRef<Expr::Ptr> getOperands() const { return operands; }
  Expr *getOperand(size_t idx) {
    assert(idx < operands.size() && "Offset error");
    return operands[idx].get();
  }

  void dump(size_t tab = 0) const override;

  /// LLVM RTTI
  static bool classof(const Expr *c) { return c->kind == Expr::Kind::Block; }

private:
  std::vector<Expr::Ptr> operands;
};

/// Types, ex: Float, String, Bool
///
/// These are constant immutable objects. If no type match,
/// the type remains None and this is not a type.
struct Type : public Expr {
  using Ptr = std::unique_ptr<Type>;
  Type(llvm::StringRef value, Expr::Type type)
      : Expr(type, Expr::Kind::Type), value(value) {}

  llvm::StringRef getValue() { return value; }

  void dump(size_t tab = 0) const override;

  /// LLVM RTTI
  static bool classof(const Expr *c) { return c->kind == Expr::Kind::Type; }

private:
  std::string value;
};

/// Literals, ex: "Hello", 10.0, 123, false
///
/// These are constant immutable objects.
/// Type is determined by the parser.
struct Literal : public Expr {
  using Ptr = std::unique_ptr<Literal>;
  Literal(llvm::StringRef value, Expr::Type type)
      : Expr(type, Expr::Kind::Literal), value(value) {
    assert(isValidType() && "Invalid type");
  }

  llvm::StringRef getValue() const { return value; }

  void dump(size_t tab = 0) const override;

  /// LLVM RTTI
  static bool classof(const Expr *c) { return c->kind == Expr::Kind::Literal; }

private:
  std::string value;
};

/// Named variables, ex:
///   str in (let (str "Hello") (print str))
///     x in (def a (x : Float) x)
///
/// Variables have a contextual name (scope::name) and an optional
/// initialisation expression.
struct Variable : public Expr {
  using Ptr = std::unique_ptr<Variable>;
  /// Definition: (x 10) in ( let (x 10) (expr) )
  Variable(llvm::StringRef name, Expr::Type type, Expr::Ptr init)
      : Expr(type, Expr::Kind::Variable), name(name), init(std::move(init)) {
    if (this->init)
      assert(this->init->getType() == type && "Variable type mismatch");
  }
  /// Declaration: (x : Integer) in ( def name Type (x : Integer) (expr) )
  Variable(llvm::StringRef name, Expr::Type type)
      : Expr(type, Expr::Kind::Variable), name(name), init(nullptr) {
    assert(isValidType() && "Invalid variable type");
  }

  /// No value == nullptr
  Expr *getInit() const { return init.get(); }
  llvm::StringRef getName() const { return name; }

  void dump(size_t tab = 0) const override;

  /// LLVM RTTI
  static bool classof(const Expr *c) { return c->kind == Expr::Kind::Variable; }

private:
  std::string name;
  Expr::Ptr init;
};

/// Lexical bloc, ex: (let (x 10) (add x 10))
///
/// Defines a variable to be used insode the scope.
/// TODO: Can we declare more than one variable?
struct Let : public Expr {
  using Ptr = std::unique_ptr<Let>;
  Let(Expr::Ptr var, Expr::Ptr expr)
      : Expr(expr->getType(), Expr::Kind::Let), var(std::move(var)),
        expr(std::move(expr)) {}

  Variable *getVariable() const { return llvm::dyn_cast<Variable>(var.get()); }
  Expr *getExpr() const { return expr.get(); }

  void dump(size_t tab = 0) const override;

  /// LLVM RTTI
  static bool classof(const Expr *c) { return c->kind == Expr::Kind::Let; }

private:
  Expr::Ptr var;
  Expr::Ptr expr;
};

/// Operation, ex: (add x 3), (neg (mul@ff (sin x) d_dcos)))
/// Call, ex: (fwd$to_float 10 dx)
///
/// Represent native operations (add, mul) and calls.
///
/// For native, all types must match and that's the return type.
/// For calls, return type and operand types must match declaration.
struct Operation : public Expr {
  using Ptr = std::unique_ptr<Operation>;
  Operation(llvm::StringRef name, Expr::Type type)
      : Expr(type, Expr::Kind::Operation), name(name) {}

  void addOperand(Expr::Ptr op) { operands.push_back(std::move(op)); }
  llvm::ArrayRef<Expr::Ptr> getOperands() const { return operands; }
  llvm::StringRef getName() const { return name; }
  size_t size() const { return operands.size(); }
  Expr *getOperand(size_t idx) const {
    assert(idx < operands.size() && "Offset error");
    return operands[idx].get();
  }

  void dump(size_t tab = 0) const override;

  /// LLVM RTTI
  static bool classof(const Expr *c) {
    return c->kind == Expr::Kind::Operation;
  }

private:
  std::string name;
  std::vector<Expr::Ptr> operands;
};

/// Declaration, ex: (edef max Float (Float Float))
///
/// Declares a function (external or posterior). Will be used for lookup during
/// the declaration (to match signature) and also used to emit declarations in
/// the final IR.
struct Declaration : public Expr {
  using Ptr = std::unique_ptr<Declaration>;
  Declaration(llvm::StringRef name, Expr::Type type)
      : Expr(type, Expr::Kind::Declaration), name(name) {}

  void addArgType(Expr::Type opt) { argTypes.push_back(opt); }
  llvm::ArrayRef<Expr::Type> getArgTypes() const { return argTypes; }
  Expr::Type getArgType(size_t idx) const {
    assert(idx < argTypes.size() && "Offset error");
    return argTypes[idx];
  }
  llvm::StringRef getName() const { return name; }
  size_t size() const { return argTypes.size(); }

  void dump(size_t tab = 0) const override;

  /// LLVM RTTI
  static bool classof(const Expr *c) {
    return c->kind == Expr::Kind::Declaration;
  }

private:
  std::string name;
  std::vector<Expr::Type> argTypes;
};

/// Definition, ex: (def fwd$to_float Float ((x : Integer) (dx : (Tuple))) 0.0)
///
/// Implementation of a function. Declarations populate the prototype,
/// definitions complete the variable list and implementation, while also
/// validating the arguments and return types.
struct Definition : public Expr {
  using Ptr = std::unique_ptr<Definition>;
  Definition(llvm::StringRef name, Expr::Type type, Expr::Ptr impl)
      : Expr(type, Expr::Kind::Definition), impl(std::move(impl)) {
    proto = std::make_unique<Declaration>(name, type);
    for (auto &a : arguments)
      proto->addArgType(a->getType());
  }

  /// Arguments and return type (for name and type validation)
  void addArgument(Expr::Ptr node) { arguments.push_back(std::move(node)); }
  llvm::ArrayRef<Expr::Ptr> getArguments() const { return arguments; }
  Expr *getArgument(size_t idx) {
    assert(idx < arguments.size() && "Offset error");
    return arguments[idx].get();
  }
  Block *getImpl() const { return llvm::dyn_cast<Block>(impl.get()); }
  Declaration *getProto() const { return proto.get(); }
  llvm::StringRef getName() const { return proto->getName(); }
  size_t size() const { return arguments.size(); }

  void dump(size_t tab = 0) const override;

  /// LLVM RTTI
  static bool classof(const Expr *c) {
    return c->kind == Expr::Kind::Definition;
  }

private:
  Expr::Ptr impl;
  Declaration::Ptr proto;
  std::vector<Expr::Ptr> arguments;
};

/// Condition, ex: (if (or x y) (add x y) 0)
struct Condition : public Expr {
  using Ptr = std::unique_ptr<Condition>;
  Condition(Expr::Ptr cond, Expr::Ptr ifBlock, Expr::Ptr elseBlock)
      : Expr(Expr::Type::None, Expr::Kind::Condition), cond(std::move(cond)),
        ifBlock(std::move(ifBlock)), elseBlock(std::move(elseBlock)) {
    assert(this->cond->getType() == Expr::Type::Bool &&
           "Condition should be boolean");
    assert(this->ifBlock->getType() == this->elseBlock->getType() &&
           "Type mismatch");
    type = this->ifBlock->getType();
  }

  Expr *getIfBlock() const { return ifBlock.get(); }
  Expr *getElseBlock() const { return elseBlock.get(); }
  Expr *getCond() const { return cond.get(); }

  void dump(size_t tab = 0) const override;

  /// LLVM RTTI
  static bool classof(const Expr *c) {
    return c->kind == Expr::Kind::Condition;
  }

private:
  Expr::Ptr cond;
  Expr::Ptr ifBlock;
  Expr::Ptr elseBlock;
};

} // namespace AST
} // namespace Knossos
#endif /// _AST_H_
