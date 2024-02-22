from lark import Lark, Transformer, v_args, Token, Tree
from lark.visitors import CollapseAmbiguities

import logging
from lark import logger
logger.setLevel(logging.DEBUG)

# maybe :; should be implemented as a builtin macro (parse as separator symbol)
# each ()[]{}"" define their own parser?
parser = Lark(
r"""
    start: expr+
    ?expr: SYM | NUMBER | STRING | paren | bracket | braces | chunk
    _S: WS
    mydef: expr _S* ":" _S* expr
    _listof{x}: (_S* x (_S+ x)* _S*)?
    _defex: expr|mydef
    paren: "(" _listof{_defex} ")"
    bracket: "[" _listof{_defex} "]"
    braces: "{" _listof{_defex} "}"
    chunk: expr ~ 2
    SYM: CNAME | /[*+-\/=`!#$%&?@^~,.;<>|]/
    %import common.CNAME
    %import common.NUMBER
    %import common.ESCAPED_STRING -> STRING
    %import common.WS
"""
# %ignore WS
# "'_.\
,parser="earley", debug=True)


def t(x):
    if isinstance(x,Token ):return x.type
    if isinstance(x,Tree  ):return x.data.value

def v(x):
    if isinstance(x,Token ):return x.value
    if isinstance(x,Tree  ):return x.children

# Define a transformer to convert the parse tree into Python code
@v_args(inline=True)
class MyTransformer(Transformer):
    NUMBER = lambda self, tok: Number(float(v(tok)))
    SYM    = lambda self, tok: Symbol(v(tok))
    STRING = lambda self, tok: String(v(tok))
    def paren   (self, *args):return Paren   (args)
    def bracket (self, *args):return Bracket (args)
    def braces  (self, *args):return Braces  (args)
    def chunk   (self, *args):return Paren   (args)
    # mydef?

class Base():
    def run(self, ps):
        ps.stack.append(self)
    # def __repr__(self):
    #     return f"({type(self)} {str(self)})"

class Symbol  (str   ,Base):pass
    # # does this go here?
    # def eval(self, ps): return ps.dict[self] if self in ps.dict else self

class Number  (float ,Base):pass
class String  (str   ,Base):pass
class Paren   (list  ,Base):pass
class Bracket (list  ,Base):pass
class Braces  (list  ,Base):pass
# Maybe?
class Block (list ,Base):pass
class Line  (list ,Base):pass
class Stack (list ,Base):
    def run(self, ps):
        ps.stack.extend(self)

class Function(Base):
    def __init__(self, f, arity):
        self.f=f
        self.arity=arity

    def run(self, ps):
        if self.arity==1:
            Sx=ps.getx()
            ps.stack.extend(
                [Number(self.f(x)) for x in Sx] if Sx else\
                [self]
            )
        elif self.arity==2:
            Sx,Sy=ps.getxy()
            # need to capture in lambdas?
            ps.stack.extend(
                [Number(self.f(x,y))     for x in Sx for y in Sy] if Sx and Sy else\
                [Function(lambda   y:self.f(x,y), 1) for x in Sx] if Sx        else\
                [Function(lambda x  :self.f(x,y), 1) for y in Sy] if        Sy else\
                [self]
            )

# builtin ops
Plus = Function(lambda x,y:x+y, 2)

def asstack(x):return [] if x is None else [x]

class ProgramState():
    def __init__(self, stack=None, queue=None, continuations=None, dict=None):
        self.stack=stack or []
        self.queue=queue or []
        self.continuations=continuations or []
        self.dict=dict or {
            '+': Plus,
        }

    #TODO: dict.get Symbol must be recursive
    # always returns a stack (list)
    def eval(self, x):
        if isinstance(x, Symbol):
            return [self.dict.get(x,x)]
        elif isinstance(x, Paren):
            ps = ProgramState(
                queue=list(x),
                continuations=self.continuations+[self],
                dict=self.dict
            )
            ps.finish()
            return ps.stack
        else:
            return asstack(x)

    # def stage2(self, x): #exec
    #     return x.run(self) # i.e. everything is a program that runs. Number just pushes itself to stack. + requests Number from queue 

    def next(self):
        x, *xs = self.queuepopeval() #TODO: this will compute entire Paren, we would like .next to step in and do a step at a time
        self.queue = xs + self.queue
        if isinstance(x, (Symbol, Number, String, Function)):
            x.run(self)
        else: print(f"Exception(next {type(x)})")

    def finish(self):
        while self.queue: self.next()
    def queuepopeval(self):return self.eval(self.queuepop())
    def queuepop(self):return self.queue.pop(0) if self.queue else None
    def stackpop(self):return self.stack.pop()  if self.stack else None
    def getx(self) -> list:
        x= asstack(self.stackpop())
        x= x or self.queuepopeval()
        return x
    def getxy(self): # shouldn't we exhaust stack first?
        x= asstack(self.stackpop())
        y= self.queuepopeval()
        x= x or self.queuepopeval()
        y= y or asstack(self.stackpop())
        return x,y



code = r"{dom : acl# ( + 5 3)}"
# code = r"(+ 5 3)"
tree = parser.parse(code)
# pi=parser.parse_interactive(code)
# for tok in pi.iter_parse():
#     print(pi.parser_state.state_stack)
#     print(pi.parser_state.value_stack)
#     print(pi.pretty())
#     print(tok)

# for x in CollapseAmbiguities().transform(tree):
#     print(x.pretty())

print(tree)
print(tree.pretty())
tree = MyTransformer().transform(tree)
print(tree.children)
ps = ProgramState(queue=tree.children)
while ps.queue:
    ps.next()
    print(ps.stack)