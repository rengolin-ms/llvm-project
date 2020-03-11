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

/// Type checking occurs from bottom up. Each node is responsible for its own
/// checks and, if valid, their return type can be used for the parents.
/// Function and variable types are checked by the symbol table
struct Type {
  /// Not enum class because we're already inside Type
  ///// so we can already use AST::Type::Integer
  enum ValidType {
    None,
    String,
    Bool,
    Integer,
    Float,
    LAST_SCALAR=Float,
    Tuple,
    Vector,
    Lambda,
    /// This is not a valid IR type, but can be parsed
    LM
  };

  /// Scalar constructor
  Type(ValidType type) : type(type) {
    assert(isScalar() && "Wrong ctor");
  }
  /// Vector constructor
  Type(ValidType type, Type subTy) : type(type) {
    assert(type == Vector && "Wrong ctor");
    subTypes.push_back(subTy);
  }
  /// Tuple constructor
  Type(ValidType type, std::vector<Type> &&subTys) : type(type) {
    assert(type == Tuple && subTys.size() > 1 && "Wrong ctor");
    subTypes = std::move(subTys);
  }
  /// Utilities
  bool isScalar() const {
    return type >= None && type <= LAST_SCALAR;
  }
  operator ValidType() const { return type; }
  bool operator ==(ValidType oTy) const {
    return type == oTy;
  }
  static bool isScalar(ValidType type) {
    return type >= None && type <= LAST_SCALAR;
  }
  void dump() const;
  // Vector accessor
  const Type &getSubType() const {
    assert(type == Vector);
    return subTypes[0];
  }
  // Tuple accessor
  const Type &getSubType(size_t idx) const {
    assert(type == Tuple);
    return subTypes[idx];
  }
  llvm::ArrayRef<Type> getSubTypes() const {
    assert(type == Tuple);
    return subTypes;
  }

protected:
  ValidType type;
  std::vector<Type> subTypes;
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
    Rule,
    Build,
    Index,
    Size,
    Tuple,
    Get,
    /// Unused below (TODO: Implement those)
    Fold,
    Lambda,
    Apply,
    Assert
  };

  /// valid types, for safety checks
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
///
/// This should only be used for the program scope. Redundant parenthesis
/// may create single blocks, which will then just pass-through the evaluation
/// of the contained token (ie. multi-token blocks inside code are illegal).
struct Block : public Expr {
  using Ptr = std::unique_ptr<Block>;
  Block() : Expr(Type::None, Kind::Block) {}
  Block(Expr::Ptr op) : Expr(Type::None, Kind::Block) {
    operands.push_back(std::move(op));
  }

  void addOperand(Expr::Ptr node) {
    type = node->getType();
    operands.push_back(std::move(node));
  }
  llvm::ArrayRef<Expr::Ptr> getOperands() const { return operands; }
  Expr *getOperand(size_t idx) const {
    assert(idx < operands.size() && "Offset error");
    return operands[idx].get();
  }
  size_t size() const { return operands.size(); }

  void dump(size_t tab = 0) const override;

  /// LLVM RTTI
  static bool classof(const Expr *c) { return c->kind == Kind::Block; }

private:
  std::vector<Expr::Ptr> operands;
};

/// Literals, ex: "Hello", 10.0, 123, false
///
/// These are constant immutable objects.
/// Type is determined by the parser.
struct Literal : public Expr {
  using Ptr = std::unique_ptr<Literal>;
  Literal(llvm::StringRef value, Type type)
      : Expr(type, Kind::Literal), value(value) {}

  llvm::StringRef getValue() const { return value; }

  void dump(size_t tab = 0) const override;

  /// LLVM RTTI
  static bool classof(const Expr *c) { return c->kind == Kind::Literal; }

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
  /// Declaration: (x : Integer) in ( def name Type (x : Integer) (expr) )
  /// We need to bind first, then assign to allow nested lets
  Variable(llvm::StringRef name, Type type=Type::None)
      : Expr(type, Kind::Variable), name(name), init(nullptr) {}

  void setInit(Expr::Ptr &&expr) {
    assert(!init);
    init = std::move(expr);
    if (type != Type::None)
      assert(type == init->getType());
    else
      type = init->getType();
  }
  /// No value == nullptr
  Expr *getInit() const { return init.get(); }
  llvm::StringRef getName() const { return name; }

  void dump(size_t tab = 0) const override;

  /// LLVM RTTI
  static bool classof(const Expr *c) { return c->kind == Kind::Variable; }

private:
  std::string name;
  Expr::Ptr init;
};

/// Let, ex: (let (x 10) (add x 10))
///
/// Defines a variable.
struct Let : public Expr {
  using Ptr = std::unique_ptr<Let>;
  Let(std::vector<Expr::Ptr> &&vars)
      : Expr(Type::None, Kind::Let), vars(std::move(vars)),
      expr(nullptr) {}
  Let(std::vector<Expr::Ptr> &&vars, Expr::Ptr expr)
      : Expr(expr->getType(), Kind::Let), vars(std::move(vars)),
        expr(std::move(expr)) {}

  llvm::ArrayRef<Expr::Ptr> getVariables() const { return vars; }
  Expr *getVariable(size_t idx) const {
    assert(idx < vars.size() && "Offset error");
    return vars[idx].get();
  }
  Expr *getExpr() const { return expr.get(); }
  size_t size() const { return vars.size(); }

  void dump(size_t tab = 0) const override;

  /// LLVM RTTI
  static bool classof(const Expr *c) { return c->kind == Kind::Let; }

private:
  std::vector<Expr::Ptr> vars;
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
  Operation(llvm::StringRef name, Type type)
      : Expr(type, Kind::Operation), name(name) {
    parseName(name);
  }

  void addOperand(Expr::Ptr op) { operands.push_back(std::move(op)); }
  llvm::ArrayRef<Expr::Ptr> getOperands() const { return operands; }
  llvm::StringRef getName() const { return name; }
  llvm::StringRef getPrefix() const { return prefix; }
  llvm::StringRef getRoot() const { return root; }
  llvm::StringRef getSuffix() const { return suffix; }
  size_t size() const { return operands.size(); }
  Expr *getOperand(size_t idx) const {
    assert(idx < operands.size() && "Offset error");
    return operands[idx].get();
  }

  void dump(size_t tab = 0) const override;

  /// LLVM RTTI
  static bool classof(const Expr *c) {
    return c->kind == Kind::Operation;
  }

private:
  void parseName(llvm::StringRef name) {
    size_t dollar = name.find('$');
    size_t at = name.find('@');
    size_t rootFrom = 0;
    size_t rootTo = name.size();
    bool changed = false;
    if (dollar != llvm::StringRef::npos) {
      prefix = name.substr(0, dollar-1).str();
      rootFrom = dollar+1;
      changed = true;
    }
    if (at != llvm::StringRef::npos) {
      suffix = name.substr(at+1).str();
      rootTo = at;
      changed = true;
    }
    if (changed)
      root = name.substr(rootFrom, rootTo - rootFrom).str();
    else
      root = name.str();
  }
  std::string name;
  std::vector<Expr::Ptr> operands;
  std::string prefix;
  std::string root;
  std::string suffix;
};

/// Declaration, ex: (edef max Float (Float Float))
///
/// Declares a function (external or posterior). Will be used for lookup during
/// the declaration (to match signature) and also used to emit declarations in
/// the final IR.
struct Declaration : public Expr {
  using Ptr = std::unique_ptr<Declaration>;
  Declaration(llvm::StringRef name, Type type)
      : Expr(type, Kind::Declaration), name(name) {}

  void addArgType(Type opt) { argTypes.push_back(opt); }
  llvm::ArrayRef<Type> getArgTypes() const { return argTypes; }
  Type getArgType(size_t idx) const {
    assert(idx < argTypes.size() && "Offset error");
    return argTypes[idx];
  }
  llvm::StringRef getName() const { return name; }
  size_t size() const { return argTypes.size(); }

  void dump(size_t tab = 0) const override;

  /// LLVM RTTI
  static bool classof(const Expr *c) {
    return c->kind == Kind::Declaration;
  }

private:
  std::string name;
  std::vector<Type> argTypes;
};

/// Definition, ex: (def fwd$to_float Float ((x : Integer) (dx : (Tuple))) 0.0)
///
/// Implementation of a function. Declarations populate the prototype,
/// definitions complete the variable list and implementation, while also
/// validating the arguments and return types.
struct Definition : public Expr {
  using Ptr = std::unique_ptr<Definition>;
  Definition(llvm::StringRef name, Type type, Expr::Ptr impl)
      : Expr(type, Kind::Definition), impl(std::move(impl)) {
    proto = std::make_unique<Declaration>(name, type);
  }

  /// Arguments and return type (for name and type validation)
  void addArgument(Expr::Ptr node) {
    proto->addArgType(node->getType());
    arguments.push_back(std::move(node));
  }
  llvm::ArrayRef<Expr::Ptr> getArguments() const { return arguments; }
  Expr *getArgument(size_t idx) {
    assert(idx < arguments.size() && "Offset error");
    return arguments[idx].get();
  }
  Expr *getImpl() const { return impl.get(); }
  Declaration *getProto() const { return proto.get(); }
  llvm::StringRef getName() const { return proto->getName(); }
  size_t size() const { return arguments.size(); }

  void dump(size_t tab = 0) const override;

  /// LLVM RTTI
  static bool classof(const Expr *c) {
    return c->kind == Kind::Definition;
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
      : Expr(Type::None, Kind::Condition), cond(std::move(cond)),
        ifBlock(std::move(ifBlock)), elseBlock(std::move(elseBlock)) {
    assert(this->cond->getType() == Type::Bool &&
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
    return c->kind == Kind::Condition;
  }

private:
  Expr::Ptr cond;
  Expr::Ptr ifBlock;
  Expr::Ptr elseBlock;
};

/// Build, ex: (build N (lam (var) (expr)))
///
/// Loops over range N using lambda L
struct Build : public Expr {
  using Ptr = std::unique_ptr<Build>;
  Build(Expr::Ptr range, Expr::Ptr var, Expr::Ptr expr)
      : Expr(Type(Type::Vector, expr->getType()), Kind::Build), range(std::move(range)),
        var(std::move(var)), expr(std::move(expr)) {}

  Expr *getRange() const { return range.get(); }
  Expr *getVariable() const { return var.get(); }
  Expr *getExpr() const { return expr.get(); }

  void dump(size_t tab = 0) const override;

  /// LLVM RTTI
  static bool classof(const Expr *c) { return c->kind == Kind::Build; }

private:
  Expr::Ptr range;
  Expr::Ptr var;
  Expr::Ptr expr;
};

/// Tuple, ex: (tuple 10.0 42 (add@ff 1.0 2.0))
///
/// Builds a tuple, inferring the types
struct Tuple : public Expr {
  using Ptr = std::unique_ptr<Tuple>;
  Tuple(std::vector<Expr::Ptr> &&elements)
      : Expr(Type::None, Kind::Tuple), elements(std::move(elements)) {
    std::vector<Type> types;
    for (auto &el: this->elements)
      types.push_back(el->getType());
    type = { Type::Tuple, std::move(types) };
  }

  llvm::ArrayRef<Expr::Ptr> getElements() const { return elements; }
  Expr *getElement(size_t idx) {
    assert(idx < elements.size() && "Offset error");
    return elements[idx].get();
  }
  size_t size() const { return elements.size(); }

  void dump(size_t tab = 0) const override;

  /// LLVM RTTI
  static bool classof(const Expr *c) { return c->kind == Kind::Tuple; }

private:
  std::vector<Expr::Ptr> elements;
};

/// Index, ex: (index N vector)
///
/// Extract the Nth index from a vector
/// Note: index range is [0, N-1]
struct Index : public Expr {
  using Ptr = std::unique_ptr<Index>;
  Index(Expr::Ptr index, Expr::Ptr var)
      : Expr(Type::None, Kind::Index), index(std::move(index)),
        var(std::move(var)) {
    assert(this->index->getType() == Type::Integer && "Invalid index type");
    assert(this->var->getType() == Type::Vector && "Invalid variable type");
    type = this->var->getType().getSubType();
  }

  Expr *getIndex() const { return index.get(); }
  Expr *getVariable() const { return var.get(); }

  void dump(size_t tab = 0) const override;

  /// LLVM RTTI
  static bool classof(const Expr *c) { return c->kind == Kind::Index; }

private:
  Expr::Ptr index;
  Expr::Ptr var;
};

/// Size, ex: (index N vector)
///
/// Extract the Nth index from a vector
struct Size : public Expr {
  using Ptr = std::unique_ptr<Size>;
  Size(Expr::Ptr var)
      : Expr(Type::Integer, Kind::Size), var(std::move(var)) {
    assert(this->var->getType() == Type::Vector && "Invalid variable type");
  }

  Expr *getVariable() const { return var.get(); }

  void dump(size_t tab = 0) const override;

  /// LLVM RTTI
  static bool classof(const Expr *c) { return c->kind == Kind::Size; }

private:
  Expr::Ptr var;
};

/// Get, ex: (get$7$9 tuple)
///
/// Extract the Nth index from a vector.
/// Note: index range is [1, N]
struct Get : public Expr {
  using Ptr = std::unique_ptr<Get>;
  Get(size_t index, size_t max, Expr::Ptr expr)
      : Expr(Type::None, Kind::Get), index(index), expr(std::move(expr)) {
    assert(this->expr->getType() == Type::Tuple && "Invalid expriable type");
    assert(index > 0 && index <= max && "Out of bounds tuple index");
    type = this->expr->getType().getSubType(index-1);
  }

  size_t getIndex() const { return index; }
  Expr *getExpr() const { return expr.get(); }
  Expr *getElement() const {
    auto tuple = llvm::dyn_cast<Tuple>(expr.get());
    return tuple->getElement(index-1);
  }

  void dump(size_t tab = 0) const override;

  /// LLVM RTTI
  static bool classof(const Expr *c) { return c->kind == Kind::Get; }

private:
  size_t index;
  Expr::Ptr expr;
};

/// Rule, ex: (rule "mul2" (v : Float) (mul@ff v 2.0) (add v v))
///
/// Rules express ways to transform the graph. They need a special
/// MLIR dialect to be represented and cannot be lowered to LLVM.
struct Rule : public Expr {
  using Ptr = std::unique_ptr<Condition>;
  Rule(llvm::StringRef name, Expr::Ptr variable, Expr::Ptr pattern,
       Expr::Ptr result)
      : Expr(Type::None, Kind::Rule), name(name),
        variable(std::move(variable)), pattern(std::move(pattern)),
        result(std::move(result)) {}

  llvm::StringRef getName() const { return name; }
  Expr *getExpr() const { return variable.get(); }
  Expr *getPattern() const { return pattern.get(); }
  Expr *getResult() const { return result.get(); }

  void dump(size_t tab = 0) const override;

  /// LLVM RTTI
  static bool classof(const Expr *c) {
    return c->kind == Kind::Rule;
  }

private:
  std::string name;
  Expr::Ptr variable;
  Expr::Ptr pattern;
  Expr::Ptr result;
};

} // namespace AST
} // namespace Knossos
#endif /// _AST_H_
