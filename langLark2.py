from lark import Lark, Transformer, v_args, Token, Tree
from lark.visitors import CollapseAmbiguities

import inspect
import logging
from lark import logger
logger.setLevel(logging.DEBUG)

# obj keys variables 3 cases: symbol, existing variable, name arg

# maybe :; should be implemented as a builtin macro (parse as separator symbol)
# each ()[]{}"" define their own parser?
parser = Lark(
r"""
    start: _listof{expr}?
    ?expr: chunk | _atom
    _atom: SYM | NUMBER | STRING | paren | bracket | braces
    chunk: _atom _atom+
    chain: _listof{expr}
    _S: WS
    mydef: expr _S* ":" _S* chain
    _listof{x}: (_S* x (_S+ x)* _S*)
    _defch: chain|mydef
    _defex: expr|mydef
    paren: "(" _listof{_defch}? ")"
    bracket: "[" _listof{_defex}? "]"
    braces: "{" _listof{_defch}? "}"
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

    def paren   (self, *args):return Paren.new(args)
    def bracket (self, *args):return Bracket.new(args)
    def braces  (self, *args):return Braces.new(args)
    def chain   (self, *args):return Chain.new(args)
    def chunk   (self, *args):return Chunk.new(args)
    def mydef   (self, k,v):return Mydef(k,v)

class Base():
    def eval(self, ps):return [self]
    def run(self, ps):
        ps.stack.append(self)
    def __repr__(self):
        return f"{type(self).__name__}({hex(id(self))})"

class Symbol  (str   ,Base):
    def __init__(self, content):
        self.evaled=False

    def __repr__(self): return self

    def eval(self, ps):
        if self.evaled:
            return [self]
        else:
            self.evaled=True
            return [ps.dict.get(self,self)]

    def run(self, ps):
        if self in ps.dict: ps.dict[self].run(ps)
        else: ps.stack.append(self)
        
class Number  (float ,Base):pass
class String  (str   ,Base):pass

class Collection():
    @classmethod
    def new(klass, args):
        p=klass(args)
        for x in args: x.parent=p
        return p

class Mydef(Base):
    def __repr__(self): return f"{self.k}:{self.v}"
    def __init__(self,k,v):
        self.k=k
        self.v=v

    def eval(self, ps): return self.v.eval(ps)

    def run(self, ps):
        if stack:=self.eval(ps):
            ps.dict[self.k]=stack[-1]

class Chain   (Collection, Base):
    def __repr__(self): return f"({' '.join([repr(x) for x in self.list])})"
    def __init__(self, items):
        self.list=list(items)
    def eval(self, ps):
        inps = ps.sub(queue=self.list)
        inps.finish()
        return inps.stack
        
    def run(self,ps):
        for x in self.eval(ps): x.run(ps)

class Chunk   (Chain):
    def __repr__(self): return f"({''.join([repr(x) for x in self.list])})"

class Paren   (Collection, Base):
    def __repr__(self): return f"({' '.join([repr(x) for x in self.items])})"
    def __init__(self, items):
        # self.items=items
        self.list=[x for x in items if not isinstance(x,Mydef)]
        self.dict=[x for x in items if     isinstance(x,Mydef)]
        # TODO: this is the logic for inline Paren, which eval definitions first,
        # we also need to implement block Paren.
        self.items=self.dict+[Mydef(Number(i),x) for i,x in enumerate(self.list)]
    def eval(self, ps):
        if not self.items: return []
        inps = ps.sub(queue=list(self.items))
        inps.finish()
        v=inps.dict.get(self.items[-1].k)
        return [v] if v else []
        
    def run(self,ps):
        # TODO: handle parens with defs (leak or not?)
        # ps.queue = self.eval(ps) + ps.queue
        for x in self.eval(ps): x.run(ps)
                    
# Maybe?
class Block (list ,Base):pass
class Line  (list ,Base):pass
class Stack (list ,Base):
    def run(self, ps):
        ps.stack.extend(self)

class Function(Base):
    def __init__(self, f, signature, closure=None):
        self.f=f
        self.signature=signature
        self.closure=closure or dict()
    
    @property
    def arity(self):return len(self.signature)
        
    def pushrun(self,ps):
        if (ps.queue or ps.stack): self.run(ps)
        else: ps.stack.append(self)
        
    def run(self, ps):
        if self.arity==1:
            Sx=self.getx(ps)
            if Sx:
                x,*xs=Sx      
                inps=ps.sub(queue=xs,dict={**self.closure,'$':self,'x':x})
                self.f(x).run(inps)
                for x in inps.stack: x.run(ps)
            else: self.pushrun(ps)
        elif self.arity==2:
            Sx,Sy=self.getxy(ps)
            # need to capture in lambdas?
            if is2(Sx) or is2(Sx):    self.run(ps.sub(stack=Sx,queue=Sy))
            elif is1(Sx) and is1(Sy): self.f(Sx[0],Sy[0]).run(ps)
            elif is1(Sx):             Function(lambda y:self.f(Sx[0], y    ), [self.signature[1]]).run(ps)
            elif is1(Sy):             Function(lambda x:self.f(x    , Sy[0]), [self.signature[0]]).run(ps)
            else:                     self.pushrun(ps)
    
    def getx(self, ps):
        T=lambda x: isinstance(x, self.signature[0])
        return ps.getx(T)
        
    def getxy(self, ps):
        Tx=lambda x: isinstance(x, self.signature[0])
        Ty=lambda x: isinstance(x, self.signature[1])
        return ps.getxy(Tx,Ty)
      

class Bracket (Collection, Function):
    def __init__(self, items):
        self.evaled=False
        self.items=items
        self.list=[x for x in items if not isinstance(x,Mydef)]
        self.dict={x.k:x.v for x in items if isinstance(x,Mydef) }
        #self.patterns= #complex keys
        
        for i, item in enumerate(self.list):
            self.dict[i] = item
            
        def lookup(key):
            result = self.dict.get(key)
            if result is not None:
                return result # TODO: value is a Paren, it must be evaled first
            return self
        
        super().__init__(lookup, [object])

    def eval(self, ps):
        if not self.evaled:
            self.evaled=True
            self.closure=ps.dict;
        return [self]

    def run(self, ps): self.eval(ps); Function.run(self,ps)
                
class Braces  (Collection, list  ,Base):pass

# builtin ops
builtin = {'+': Function(lambda x,y:Number(x+y), [Number, Number])}

def asstack(x):return [] if x is None else [x]
def is1(x): return len(x)==1
def is2(x): return len(x)>1

class ProgramState():
    def __init__(self, stack=None, queue=None, continuations=None, dict=None):
        self.stack=stack or []
        self.queue=queue or []
        self.continuations=continuations or []
        self.dict=dict or {**builtin}

    def __repr__(self): return f'{hex(id(self))}| {self.code_repr()}'
    #TODO: dict.get Symbol must be recursive
    # always returns a stack (list)
    def eval(self, x): return [] if x is None else x.eval(self)
    def next(self): debug_hook(); self.queuepop().run(self)
    def finish(self):
        while self.queue: self.next()
        debug_hook('STATE RETURN')
    def queuepopeval(self,T=None):
        S = self.eval(self.queuepop())
        if T and is1(S) and not T(S[0]):
            self.queue=S+self.queue
            return []
        else:
            return S
    def queuepop(self):return self.queue.pop(0) if self.queue else None
    def stackpop(self,T=None):return [self.stack.pop()] if self.stack and (not T or T(self.stack[-1])) else []
    def getx (self,T =None        ):return self.stackpop(T ) or self.queuepopeval(T)
    def getxy(self,Tx=None,Ty=None):return self.stackpop(Tx) ,  self.queuepopeval(Ty)
    def sub(self,**kw):
        return ProgramState(**{
            'continuations':self.continuations+[self],
            'dict':dict(self.dict),
            **kw
        })
    def code_repr(self):
        neat=lambda x:' '.join(map(repr,x))
        return f'{neat(self.stack)} ⋄ {neat(self.queue)}'

def debugps(up=1):
    var=inspect.stack()[up].frame.f_locals
    for x in var.values():
        if isinstance(x,ProgramState):
            print({k:v for k,v in x.dict.items() if k not in builtin})
            for c in x.continuations+[x]: print(c)
            # print(x.code_repr())
    if ('self' in var) and not isinstance(var['self'],ProgramState):
        print(type(var['self']))

def debugger(msg='STATE'): print(msg); debugps(2); input('>')
debug_hook = lambda *x,**kw:None

if __name__=='__main__':
    debug_hook = debugger

    import readline
    readline.set_startup_hook(lambda: readline.insert_text("([myvar+3] myvar: 2) 0"))
    try:
        code=input('>')
    finally:
        # Clear the hook so it doesn't affect future inputs
        readline.set_startup_hook()

    # code = r"(myvar + 3 myvar: 2)" # 5
    # code = r"([myvar+3] myvar: 2) 0" # 5
    # r"[dom : acl# ( + 5 3)] dom"
    # [1 1 [x]:(x-1$ + x-2$)][5]  8
    # [1 1 _:(x-1$ + x-2$)] 5     8
    # [mymethod[x y]:x*y other[x y z]:x+y+z].mymethod[5 10]   50
    # [a←2+3 f→a].f    5
    # (x)→x+2 3     5


    tree = parser.parse(code)
    # pi=parser.parse_interactive(code)
    # for tok in pi.iter_parse():
    #     print(pi.parser_state.state_stack)
    #     print(pi.parser_state.value_stack)
    #     print(pi.pretty())
    #     print(tok)

    # for x in CollapseAmbiguities().transform(tree):
    #     print(x.pretty())

    print(tree.pretty())
    tree = MyTransformer().transform(tree)
    print(tree.children)
    ps = ProgramState(queue=tree.children)
    ps.finish()
    #     print(ps.stack)