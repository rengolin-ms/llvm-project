/* Copyright Microsoft Corp. 2020 */
#include <iostream>

#include "Parser.h"

using namespace std;
using namespace Knossos::AST;

//================================================ Helpers

static Expr::Type Str2Type(llvm::StringRef ty) {
  if (ty == "String")
    return Expr::Type::String;
  if (ty == "Bool")
    return Expr::Type::Bool;
  if (ty == "Integer")
    return Expr::Type::Integer;
  if (ty == "Float")
    return Expr::Type::Float;
  if (ty == "Tuple")
    return Expr::Type::Tuple;
  if (ty == "Vec")
    return Expr::Type::Vec;
  if (ty == "Lambda")
    return Expr::Type::Lambda;
  if (ty == "LM")
    return Expr::Type::LM;
  return Expr::Type::None;
}

static const string Type2Str(Expr::Type type) {
  switch (type) {
  case Expr::Type::None:
    return "none";
  case Expr::Type::String:
    return "String";
  case Expr::Type::Bool:
    return "Bool";
  case Expr::Type::Integer:
    return "Integer";
  case Expr::Type::Float:
    return "Float";
  case Expr::Type::Tuple:
    return "Tuple";
  case Expr::Type::Vec:
    return "Vec";
  case Expr::Type::Lambda:
    return "Lambda";
  case Expr::Type::LM:
    return "LM";
  }
}

static Expr::Type LiteralType(llvm::StringRef str) {
  // String
  if (str[0] == '"' && str[str.size() - 1] == '"')
    return Expr::Type::String;
  // Bool
  if (str == "true" || str == "false")
    return Expr::Type::Bool;
  // Number
  bool isNumber = true;
  bool isFloat = false;
  for (auto c : str) {
    if (c == '.')
      isFloat = true;
    else if (!::isdigit(c)) {
      isNumber = false;
      break;
    }
  }
  if (isNumber) {
    if (isFloat)
      return Expr::Type::Float;
    else
      return Expr::Type::Integer;
  }

  // TODO: detect Tuple, Vec, Lambda, LM
  return Expr::Type::None;
}

static bool isLiteralOrType(llvm::StringRef str) {
  return Str2Type(str) != Expr::Type::None ||
         LiteralType(str) != Expr::Type::None;
}

static llvm::StringRef unquote(llvm::StringRef original) {
  size_t start = 0, len = original.size();
  if (original.front() == '"') { start++; len--; }
  if (original.back() == '"') len--;
  return original.substr(start, len);
}

//================================================ Lex source into Tokens

// Lex a token out, recurse if another entry point is found
size_t Lexer::lexToken(Token *tok, size_t pos) {
  size_t tokenStart = pos;
  bool isInString = false;
  while (pos < len) {
    switch (code[pos]) {
      case ';':
        // Comment, to the end of the line
        while (code[pos] != '\n')
          tokenStart = ++pos;
        break;
      case ' ':
        // Spaces are allowed inside strings
        if (isInString) {
          pos++;
          break;
        }
        // Maybe end of a value
        if (tokenStart != pos) {
          tok->addChild(
              make_unique<Token>(code.substr(tokenStart, pos - tokenStart)));
        }
        // Or end of a token, which we ignore
        tokenStart = ++pos;
        break;
      case ')':
        // Maybe end of a value
        if (tokenStart != pos) {
          tok->addChild(
              make_unique<Token>(code.substr(tokenStart, pos - tokenStart)));
        }
        // Finished parsing this token
        tokenStart = ++pos;
        return pos;
      case '(': {
        // Recurse into sub-tokens
        auto t = make_unique<Token>();
        tokenStart = pos = lexToken(t.get(), pos + 1);
        tok->addChild(move(t));
        break;
      }
      case '"':
        if (isInString) {
          // Strings need to capture the quotes, too
          size_t start = tokenStart - 1;
          size_t length = (pos - start + 1);
          tok->addChild(
              make_unique<Token>(code.substr(start, length)));
        }
        tokenStart = ++pos;
        isInString = !isInString;
        break;
      case '\n':
      case '\r':
        // Ignore
        tokenStart = ++pos;
        break;
      default:
        // These are text, so we keep reading
        pos++;
    }
  }
  return pos;
}

//================================================ Parse Tokens into Exprs

// Creates an Expr for each token, validating
Expr::Ptr Parser::parseToken(const Token *tok) {
  // Values are all strings (numbers will be converted later)
  // They could be variable use, type names, literals
  if (tok->isValue)
    return parseValue(tok);

  // Empty block
  if (tok->size() == 0)
    return unique_ptr<Expr>(new Block());

  // If the first expr is not a value, this is a block of blocks
  if (!tok->getHead()->isValue)
    return parseBlock(tok);

  // First child is a value, can be all sorts of block types
  // Check first value for type
  string value = tok->getHead()->getValue().str();

  // Constructs: let, edef, def, if, fold, etc.
  if (isReservedWord(value)) {
    if (value == "let") {
      return parseLet(tok);
    } else if (value == "edef") {
      return parseDecl(tok);
    } else if (value == "def") {
      return parseDef(tok);
    } else if (value == "if") {
      return parseCond(tok);
    } else if (value == "rule") {
      return parseRule(tok);
    }
    // TODO: implement fold, lambda, tuple, apply
    assert(0 && "Not implemented yet");
  }

  // Operations: reserved keywords like add, mul, etc.
  if (isReservedOp(value))
    return parseOperation(tok);

  // Function call: (fun 10.0 42 "Hello")
  if (functions.exists(value))
    return parseCall(tok);

  // Variable declaration/definition
  // TODO: Do variables always start with letters?
  if (!isLiteralOrType(value) && !variables.exists(value))
    return parseVariable(tok);

  // Nothing recognised, return as an opaque block
  // TODO: Make sure this is semantically valid in ksc
  return parseBlock(tok);
}

// Values (variable names, literals, type names)
Expr::Ptr Parser::parseBlock(const Token *tok) {
  Block *b = new Block();
  for (auto &c : tok->getChildren())
    b->addOperand(parseToken(c.get()));
  return unique_ptr<Expr>(b);
}

// Values (variable names, literals, type names)
Expr::Ptr Parser::parseValue(const Token *tok) {
  assert(tok->isValue);
  string value = tok->getValue().str();

  // Type names: Integer, Float, Bool, String...
  Expr::Type ty = Str2Type(value);
  if (ty != Expr::Type::None) {
    return unique_ptr<Expr>(new Type(value, ty));
  }
  // Literals: 10.0 42 "Hello" (not hello, that's a variable use)
  ty = LiteralType(value);
  if (ty != Expr::Type::None) {
    // Trim quotes "" from strings before creating constant
    if (ty == Expr::Type::String)
      value = value.substr(1, value.length()-2);
    return unique_ptr<Expr>(new Literal(value, ty));
  }
  // Variable use: name (without quotes)
  assert(variables.exists(value) && "Variable not declared");
  auto val = variables.get(value);
  // For now, we duplicate the variable
  // TODO: Maybe call a function with the init inside?
  assert(Variable::classof(val));
  return unique_ptr<Expr>(new Variable(value, val->getType()));
}

// Calls (fun arg1 arg2 ...)
// Checks types agains symbol table
Expr::Ptr Parser::parseCall(const Token *tok) {
  string name = tok->getHead()->getValue().str();
  assert(functions.exists(name));
  Declaration* decl = llvm::dyn_cast<Declaration>(functions.get(name));
  assert(decl);

  vector<Expr> operands;
  Expr::Type type = decl->getType();
  Operation *o = new Operation(name, type);
  for (auto &c : tok->getTail())
    o->addOperand(parseToken(c.get()));
  // Validate types
  for (const auto &it : llvm::zip(o->getOperands(), decl->getArgTypes()))
    assert(get<0>(it)->getType() == get<1>(it));
  assert(o->getType() == decl->getType());
  return unique_ptr<Expr>(o);
}

// Operations (op arg1 arg2 ...), retTy = argnTy
Expr::Ptr Parser::parseOperation(const Token *tok) {
  assert(!tok->isValue && tok->size() > 1);
  assert(tok->getHead()->isValue);
  llvm::StringRef op = tok->getHead()->getValue();

  // Get all operands to validate type
  vector<Expr::Ptr> operands;
  for (auto &c : tok->getTail())
    operands.push_back(parseToken(c.get()));
  // Validate types
  Expr::Type type = operands[0]->getType();
  for (auto &o : operands)
    assert(o->getType() == type);
  // Create the op and return
  Operation *o = new Operation(op, type);
  for (auto &op : operands)
    o->addOperand(move(op));
  return unique_ptr<Expr>(o);
}

// Variables can be declarations, definitions or use, depending on the arguments
Expr::Ptr Parser::parseVariable(const Token *tok) {
  assert(tok->size() > 1);
  llvm::ArrayRef<Token::Ptr> children = tok->getChildren();
  string value = children[0]->getValue().str();
  // Variable declaration: (name : Type) : add name to SymbolTable
  if (tok->size() == 3 && children[1]->getValue() == ":") {
    Expr::Type type = Str2Type(children[2]->getValue());
    assert(type != Expr::Type::None);
    auto var = unique_ptr<Variable>(new Variable(value, type));
    variables.add(value, var.get());
    assert(variables.exists(value));
    return var;
  }

  // Variable definition: (name value) : check name on SymbolTable
  // TODO: This is naive, we need to check context, too
  if (tok->size() == 2) {
    Expr::Ptr expr = parseToken(children[1].get());
    auto type = expr->getType();
    auto var = unique_ptr<Variable>(new Variable(value, type, move(expr)));
    variables.add(value, var.get());
    assert(variables.exists(value));
    return var;
  }

  assert(0 && "Invalid variable declaration/definition");
}

// Sub-expr with contextual variables: (let (x 10) (add x 10))
// TODO: can we bind more than one variable?
Expr::Ptr Parser::parseLet(const Token *tok) {
  assert(tok->size() == 3);
  const Token *bond = tok->getChild(1);
  const Token *expr = tok->getChild(2);
  assert(!bond->isValue && bond->size() == 2);
  auto var = parseVariable(bond);
  auto body = parseToken(expr);
  return make_unique<Let>(move(var), move(body));
}

// Declares a function (and add it to symbol table)
Expr::Ptr Parser::parseDecl(const Token *tok) {
  assert(tok->size() == 4);
  const Token *name = tok->getChild(1);
  const Token *type = tok->getChild(2);
  const Token *args = tok->getChild(3);
  assert(name->isValue && type->isValue);
  assert(!args->isValue);

  auto decl =
      make_unique<Declaration>(name->getValue(), Str2Type(type->getValue()));
  assert(decl);
  assert(!args->isValue);
  for (auto &c : args->getChildren())
    decl->addArgType(Str2Type(c->getValue()));
  functions.add(name->getValue().str(), decl.get());
  assert(functions.exists(name->getValue().str()));
  return decl;
}

// Defines a function (checks from|adds to symbol table)
Expr::Ptr Parser::parseDef(const Token *tok) {
  assert(tok->size() == 5);
  const Token *name = tok->getChild(1);
  const Token *type = tok->getChild(2);
  const Token *args = tok->getChild(3);
  const Token *expr = tok->getChild(4);
  assert(name->isValue && type->isValue && !args->isValue);
  vector<Expr::Ptr> arguments;
  for (auto &a : args->getChildren())
    arguments.push_back(parseToken(a.get()));
  for (auto &a : arguments)
    assert(a->kind == Expr::Kind::Variable);

  // Function body is a block, create one if single expr
  auto body = parseToken(expr);
  if (!Block::classof(body.get()))
    body = make_unique<Block>(move(body));

  auto node = make_unique<Definition>(name->getValue(),
                                      Str2Type(type->getValue()), move(body));
  assert(node);
  assert(!args->isValue);
  for (auto &arg : arguments)
    node->addArgument(move(arg));
  functions.add(name->getValue().str(), node->getProto());
  assert(functions.exists(name->getValue().str()));
  return node;
}

// Conditional: (if (cond) (true block) (false block))
Expr::Ptr Parser::parseCond(const Token *tok) {
  assert(tok->size() == 4);
  const Token *cond = tok->getChild(1);
  const Token *ifBlock = tok->getChild(2);
  const Token *elseBlock = tok->getChild(3);
  assert(!cond->isValue && !ifBlock->isValue && !elseBlock->isValue);
  auto c = parseToken(cond);
  auto i = parseToken(ifBlock);
  auto e = parseToken(elseBlock);
  return make_unique<Condition>(move(c), move(i), move(e));
}

// Rule: (rule "mul2" (v : Float) (mul@ff v 2.0) (add v v))
Expr::Ptr Parser::parseRule(const Token *tok) {
  assert(tok->size() == 5);
  const Token *name = tok->getChild(1);
  assert(name->isValue);
  llvm::StringRef unquoted = unquote(name->getValue());
  auto var = parseToken(tok->getChild(2));
  auto pat = parseToken(tok->getChild(3));
  auto res = parseToken(tok->getChild(4));
  auto rule =
      make_unique<Rule>(unquoted, move(var), move(pat), move(res));
  rules.add(unquoted.str(), rule.get());
  return rule;
}

//================================================ Dumps tokens, nodes to stdout

void Token::dump(size_t tab) const {
  if (isValue) {
    cout << string(tab, ' ') << "tok(" << getValue().data() << ")\n";
    return;
  }

  cout << string(tab, ' ') << "enter\n";
  tab += 2;
  for (auto &t : children)
    t->dump(tab);
  tab -= 2;
  cout << string(tab, ' ') << "exit\n";
}

void Expr::dump(size_t tab) const {
  cout << string(tab, ' ') << "type [" << Type2Str(type) << "]" << endl;
}

void Block::dump(size_t tab) const {
  cout << string(tab, ' ') << "Block:" << endl;
  for (auto &op : operands)
    op->dump(tab + 2);
}

void Type::dump(size_t tab) const {
  cout << string(tab, ' ') << "Type:" << endl;
  Expr::dump(tab + 2);
}

void Literal::dump(size_t tab) const {
  cout << string(tab, ' ') << "Literal:" << endl;
  cout << string(tab + 2, ' ') << "value [" << value << "]" << endl;
  Expr::dump(tab + 2);
}

void Variable::dump(size_t tab) const {
  cout << string(tab, ' ') << "Variable:" << endl;
  cout << string(tab + 2, ' ') << "name [" << name << "]" << endl;
  Expr::dump(tab + 2);
  if (init)
    init->dump(tab + 2);
}

void Let::dump(size_t tab) const {
  cout << string(tab, ' ') << "Let:" << endl;
  Expr::dump(tab + 2);
  var->dump(tab + 2);
  expr->dump(tab + 2);
}

void Operation::dump(size_t tab) const {
  cout << string(tab, ' ') << "Operation:" << endl;
  cout << string(tab + 2, ' ') << "name [" << name << "]" << endl;
  Expr::dump(tab + 2);
  for (auto &op : operands)
    op->dump(tab + 2);
}

void Declaration::dump(size_t tab) const {
  cout << string(tab, ' ') << "Declaration:" << endl;
  cout << string(tab + 2, ' ') << "name [" << name << "]" << endl;
  Expr::dump(tab + 2);
  cout << string(tab + 2, ' ') << "Types: [ ";
  for (auto ty : argTypes)
    cout << Type2Str(ty) << " ";
  cout << "]" << endl;
}

void Definition::dump(size_t tab) const {
  cout << string(tab, ' ') << "Definition:" << endl;
  cout << string(tab + 2, ' ') << "name [" << proto->getName().data() << "]"
       << endl;
  Expr::dump(tab + 2);
  cout << string(tab + 2, ' ') << "Arguments:" << endl;
  for (auto &op : arguments)
    op->dump(tab + 4);
  cout << string(tab + 2, ' ') << "Implementation:" << endl;
  impl->dump(tab + 4);
}

void Condition::dump(size_t tab) const {
  cout << string(tab, ' ') << "Condition:" << endl;
  cond->dump(tab + 2);
  cout << string(tab + 2, ' ') << "True branch:" << endl;
  ifBlock->dump(tab + 4);
  cout << string(tab + 2, ' ') << "False branch:" << endl;
  elseBlock->dump(tab + 4);
}

void Rule::dump(size_t tab) const {
  cout << string(tab, ' ') << "Rule:" << endl;
  cout << string(tab + 2, ' ') << "name [" << name << "]"
       << endl;
  Expr::dump(tab + 2);
  cout << string(tab + 2, ' ') << "Variable:" << endl;
  variable->dump(tab + 4);
  cout << string(tab + 2, ' ') << "Pattern:" << endl;
  pattern->dump(tab + 4);
  cout << string(tab + 2, ' ') << "Result:" << endl;
  result->dump(tab + 4);
}
