#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import numpy as np
from abc import ABC, abstractmethod
import math


# In[2]:


# %%capture
# %run symbolic_expressions.ipynb


# In[3]:


eq_str = "-2*x**9.8 - x**8.6 + 6*x**2 + 10*x**5 - 3.1*x**8 + 5 + 3.2*x**2.1 + x**8 + 4.5*x**1.4 + x**8 + 10.2*x**4.2 + 9.1*x**1.4 + 3*x**2 + 6*x + 4*x + 9*x + 10*x**3 +5 + 2*x**3 + 7  + 3*x**3 + 6 + 8*x**2 + 5*x**5 + 6 + 5*x**2 + x**8 + 4.4*x**2.3 + x**8"
# eq_str = "-2*x**9.8 - 3*x**8.6"
# eq_str = "x**8 + 6*x**2 + 10*x**5 + x**8 - 5 + 3.2*x**2.5 + x**8"
# eq_str = "-x**3 + 4*x + 2.0"
# eq_str = "-6*x**3 + 4*x**2 - 2*x**3 + 3*x - 5.5"
# eq_str = "4*x**2 - 5.5"
eq_str_2 = eq_str
print(eq_str_2)
# from math import sin
def f(x):
    return eq_str


# In[4]:


class Power():
    def __init__(self,base,exponent):
        self.base = base
        self.exponent = exponent


# In[7152]:


class Number():
    def __init__(self,number):
        self.number = number

class Variable():
    def __init__(self,symbol):
        self.symbol = symbol


# In[7153]:


Power(Variable("x"),Number(2))
# In[7154]:


class Product():
    def __init__(self, exp1, exp2):
        self.exp1 = exp1
        self.exp2 = exp2


# In[7155]:


Product(Number(3),Power(Variable("x"),Number(2)))


# In[7156]:


class Sum():
    def __init__(self, *exps): #<1>
        self.exps = exps

class Function(): #<2>
    def __init__(self,name):
        self.name = name

class Apply(): #<3>
    def __init__(self,function,argument):
        self.function = function
        self.argument = argument

f_expression = Product( #<4>
                Sum(
                    Product(
                        Number(3),
                        Power(
                            Variable("x"),
                            Number(2))), 
                    Variable("x")), 
                Apply(
                    Function("sin"),
                    Variable("x")))


# In[7157]:


Apply(Function("cos"),Sum(Power(Variable("x"),Number("3")), Number(-5)))


# In[7158]:


from math import log
def f(y,z):
    return log(y**z)


# In[7159]:


Apply(Function("ln"), Power(Variable("y"), Variable("z")))


# In[7160]:


class Quotient():
    def __init__(self,numerator,denominator):
        self.numerator = numerator
        self.denominator = denominator


# In[7161]:


Quotient(Sum(Variable("a"),Variable("b")),Number(2))


# In[7162]:


class Difference():
    def __init__(self,exp1,exp2):
        self.exp1 = exp1
        self.exp2 = exp2


# In[7163]:


Difference(
    Power(Variable('b'),Number(2)),
    Product(Number(4),Product(Variable('a'), Variable('c'))))


# In[7164]:


class Negative():
    def __init__(self,exp):
        self.exp = exp


# In[7165]:


Negative(Sum(Power(Variable("x"),Number(2)),Variable("y")))


# In[7166]:


A = Variable('a')
B = Variable('b')
C = Variable('c')
Sqrt = Function('sqrt')


# In[7167]:


Quotient(
    Sum(
        Negative(B),
        Apply(
            Sqrt, 
            Difference(
                Power(B,Number(2)),
                Product(Number(4), Product(A,C))))),
    Product(Number(2), A))


# In[7168]:


def distinct_variables(exp):
    if isinstance(exp, Variable):
        return set(exp.symbol)
    elif isinstance(exp, Number):
        return set()
    elif isinstance(exp, Sum):
        return set().union(*[distinct_variables(exp) for exp in exp.exps])
    elif isinstance(exp, Product):
        return distinct_variables(exp.exp1).union(distinct_variables(exp.exp2))
    elif isinstance(exp, Power):
        return distinct_variables(exp.base).union(distinct_variables(exp.exponent))
    elif isinstance(exp, Apply):
        return distinct_variables(exp.argument)
    else:
        raise TypeError("Not a valid expression.")


# In[7169]:


distinct_variables(Variable("z"))


# In[7170]:


distinct_variables(Number(3))


# In[7171]:


distinct_variables(f_expression)


# In[7172]:


from abc import ABC, abstractmethod

class Expression(ABC):
    @abstractmethod
    def evaluate(self, **bindings):
        pass


# In[7173]:


class Number(Expression):
    def __init__(self,number):
        self.number = number
    def evaluate(self, **bindings):
        return self.number
    
class Variable(Expression):
    def __init__(self,symbol):
        self.symbol = symbol
    def evaluate(self, **bindings):
        try:
            return bindings[self.symbol]
        except:
            raise KeyError("Variable '{}' is not bound.".format(self.symbol))
            
class Product(Expression):
    def __init__(self, exp1, exp2):
        self.exp1 = exp1
        self.exp2 = exp2
    def evaluate(self, **bindings):
        return self.exp1.evaluate(**bindings) * self.exp2.evaluate(**bindings)


# In[7174]:


Product(Variable("x"), Variable("y")).evaluate(x=2,y=5)


# In[7175]:


import math
from math import sin, cos, log

_function_bindings = {
    "sin": math.sin,
    "cos": math.cos,
    "ln": math.log    
}

class Apply(Expression):
    def __init__(self,function,argument):
        self.function = function
        self.argument = argument
    def evaluate(self, **bindings):
        return _function_bindings[self.function.name](self.argument.evaluate(**bindings))


# In[7176]:


class Sum(Expression):
    def __init__(self, *exps):
        self.exps = exps
    def evaluate(self, **bindings):
        return sum([exp.evaluate(**bindings) for exp in self.exps])
    
class Power(Expression):
    def __init__(self,base,exponent):
        self.base = base
        self.exponent = exponent
    def evaluate(self, **bindings):
        return self.base.evaluate(**bindings) ** self.exponent.evaluate(**bindings)
    
class Difference(Expression):
    def __init__(self,exp1,exp2):
        self.exp1 = exp1
        self.exp2 = exp2
    def evaluate(self, **bindings):
        return self.exp1.evaluate(**bindings) - self.exp2.evaluate(**bindings)
    
class Quotient(Expression):
    def __init__(self,numerator,denominator):
        self.numerator = numerator
        self.denominator = denominator
    def evaluate(self, **bindings):
        return self.numerator.evaluate(**bindings) / self.denominator.evaluate(**bindings)


# In[7177]:


f_expression = Product( #<4>
                Sum(
                    Product(
                        Number(3),
                        Power(
                            Variable("x"),
                            Number(2))), 
                    Variable("x")), 
                Apply(
                    Function("sin"),
                    Variable("x")))


# In[7178]:


f_expression.evaluate(x=5)


# In[7179]:


from math import sin
def f(x):
    return (3*x**2 + x) * sin(x)

f(5)


# In[7180]:


class Expression(ABC):
    @abstractmethod
    def evaluate(self, **bindings):
        pass
    @abstractmethod
    def expand(self):
        pass
    
    # Printing expressions legibly in REPL (See first mini project in 2.4)
    @abstractmethod
    def display(self):
        pass
    def __repr__(self):
        return self.display()


# In[7181]:


class Sum(Expression):
    def __init__(self, *exps):
        self.exps = exps
    def evaluate(self, **bindings):
        return sum([exp.evaluate(**bindings) for exp in self.exps])
    def expand(self):
        return Sum(*[exp.expand() for exp in self.exps])
    def display(self):
        return "Sum({})".format(",".join([e.display() for e in self.exps]))
    
class Product(Expression):
    def __init__(self, exp1, exp2):
        self.exp1 = exp1
        self.exp2 = exp2
    def evaluate(self, **bindings):
        return self.exp1.evaluate(**bindings) * self.exp2.evaluate(**bindings)
    def expand(self):
        expanded1 = self.exp1.expand()
        expanded2 = self.exp2.expand()
        if isinstance(expanded1, Sum):
            return Sum(*[Product(e,expanded2).expand() for e in expanded1.exps])
        elif isinstance(expanded2, Sum):
            return Sum(*[Product(expanded1,e) for e in expanded2.exps])
        else:
            return Product(expanded1,expanded2)
    def display(self):
        return "Product({},{})".format(self.exp1.display(),self.exp2.display())
        
class Difference(Expression):
    def __init__(self,exp1,exp2):
        self.exp1 = exp1
        self.exp2 = exp2
    def evaluate(self, **bindings):
        return self.exp1.evaluate(**bindings) - self.exp2.evaluate(**bindings)
    def expand(self):
        return self
    def display(self):
        return "Difference({},{})".format(self.exp1.display(), self.exp2.display())
    
class Quotient(Expression):
    def __init__(self,numerator,denominator):
        self.numerator = numerator
        self.denominator = denominator
    def evaluate(self, **bindings):
        return self.numerator.evaluate(**bindings) / self.denominator.evaluate(**bindings)
    def expand(self):
        return self
    def display(self):
        return "Quotient({},{})".format(self.numerator.display(),self.denominator.display())
    
class Negative(Expression):
    def __init__(self,exp):
        self.exp = exp
    def evaluate(self, **bindings):
        return - self.exp.evaluate(**bindings)
    def expand(self):
        return self
    def display(self):
        return "Negative({})".format(self.exp.display())
    
class Number(Expression):
    def __init__(self,number):
        self.number = number
    def evaluate(self, **bindings):
        return self.number
    def expand(self):
        return self
    def display(self):
        return "Number({})".format(self.number)
    
class Power(Expression):
    def __init__(self,base,exponent):
        self.base = base
        self.exponent = exponent
    def evaluate(self, **bindings):
        return self.base.evaluate(**bindings) ** self.exponent.evaluate(**bindings)
    def expand(self):
        return self
    def display(self):
        return "Power({},{})".format(self.base.display(),self.exponent.display())
    
class Variable(Expression):
    def __init__(self,symbol):
        self.symbol = symbol
    def evaluate(self, **bindings):
        return bindings[self.symbol]
    def expand(self):
        return self
    def display(self):
        return "Variable(\"{}\")".format(self.symbol)
    
class Function():
    def __init__(self,name,make_latex=None):
        self.name = name
        self.make_latex = make_latex
    def latex(self,arg_latex):
        if self.make_latex:
            return self.make_latex(arg_latex)
        else:
            return " \\operatorname{{ {} }} \\left( {} \\right)".format(self.name, arg_latex)
  
class Apply(Expression):
    def __init__(self,function,argument):
        self.function = function
        self.argument = argument
    def evaluate(self, **bindings):
        return _function_bindings[self.function.name](self.argument.evaluate(**bindings))
    def expand(self):
        return Apply(self.function, self.argument.expand())
    def display(self):
        return "Apply(Function(\"{}\"),{})".format(self.function.name, self.argument.display())


# In[7182]:


Y = Variable('y')
Z = Variable('z')
A = Variable('a')
B = Variable('b')
Product(Sum(A,B),Sum(Y,Z))


# In[7183]:


Product(Sum(A,B),Sum(Y,Z)).expand()


# In[7184]:


f_expression = Product( #<4>
                Sum(
                    Product(
                        Number(3),
                        Power(
                            Variable("x"),
                            Number(2))), 
                    Variable("x")), 
                Apply(
                    Function("sin"),
                    Variable("x")))


# In[7185]:


f_expression.expand()


# In[7186]:


def contains(exp, var):
    if isinstance(exp, Variable):
        return exp.symbol == var.symbol
    elif isinstance(exp, Number):
        return False
    elif isinstance(exp, Sum):
        return any([contains(e,var) for e in exp.exps])
    elif isinstance(exp, Product):
        return contains(exp.exp1,var) or contains(exp.exp2,var)
    elif isinstance(exp, Power):
        return contains(exp.base, var) or contains(exp.exponent, var)
    elif isinstance(exp, Apply):
        return contains(exp.argument, var)
    else:
        raise TypeError("Not a valid expression.")


# In[7187]:


def distinct_functions(exp):
    if isinstance(exp, Variable):
        return set()
    elif isinstance(exp, Number):
        return set()
    elif isinstance(exp, Sum):
        return set().union(*[distinct_functions(exp) for exp in exp.exps])
    elif isinstance(exp, Product):
        return distinct_functions(exp.exp1).union(distinct_functions(exp.exp2))
    elif isinstance(exp, Power):
        return distinct_functions(exp.base).union(distinct_functions(exp.exponent))
    elif isinstance(exp, Apply):
        return set([exp.function.name]).union(distinct_functions(exp.argument))
    else:
        raise TypeError("Not a valid expression.")


# In[7188]:


def contains_sum(exp):
    if isinstance(exp, Variable):
        return False
    elif isinstance(exp, Number):
        return False
    elif isinstance(exp, Sum):
        return True
    elif isinstance(exp, Product):
        return contains_sum(exp.exp1) or contains_sum(exp.exp2)
    elif isinstance(exp, Power):
        return contains_sum(exp.base) or contains_sum(exp.exponent)
    elif isinstance(exp, Apply):
        return contains_sum(exp.argument)
    else:
        raise TypeError("Not a valid expression.")


# In[7189]:


from abc import ABC, abstractmethod
import math

def paren_if_instance(exp,*args):
    for typ in args:
        if isinstance(exp,typ):
            return "\\left( {} \\right)".format(exp.latex())
    return exp.latex()

def package(maybe_expression):
    if isinstance(maybe_expression,Expression):
        return maybe_expression
    elif isinstance(maybe_expression,int) or isinstance(maybe_expression,float):
        return Number(maybe_expression)
    else:
        raise ValueError("can't convert {} to expression.".format(maybe_expression))
def dot_if_necessary(latex):
    if latex[0] in '-1234567890':
        return '\\cdot {}'.format(latex)
    else:
        return latex
        
class Expression(ABC):
    @abstractmethod
    def latex(self):
        pass
    def _repr_latex_(self):
        return "$$" + self.latex() + "$$"
    @abstractmethod
    def evaluate(self, **bindings):
        pass
    @abstractmethod
    def substitute(self, var, expression):
        pass
    @abstractmethod
    def expand(self):
        pass
    @abstractmethod
    def display(self):
        pass
    def __repr__(self):
        return self.display()
    @abstractmethod
    def derivative(self,var):
        pass
    
    def __call__(self, *inputs):
        var_list = sorted(distinct_variables(self))
        return self.evaluate(**dict(zip(var_list, inputs)))
    
    def __add__(self, other):
        return Sum(self,package(other))
    
    def __sub__(self,other):
        return Difference(self,package(other))
    
    def __mul__(self,other):
        return Product(self,package(other))
    
    def __rmul__(self,other):
        return Product(package(other),self)
    
    def __truediv__(self,other):
        return Quotient(self,package(other))
    
    def __pow__(self,other):
        return Power(self,package(other))
    
    @abstractmethod
    def _python_expr(self):
        pass
    
    def python_function(self,**bindings):
#         code = "lambda {}:{}".format(
#             ", ".join(sorted(distinct_variables(self))),
#             self._python_expr())
#         print(code)
        global_vars = {"math":math}
        return eval(self._python_expr(),global_vars,bindings)

class Sum(Expression):
    def __init__(self, *exps):
        self.exps = exps
    def latex(self):
        return " + ".join(exp.latex() for exp in self.exps)
    def evaluate(self, **bindings):
        return sum([exp.evaluate(**bindings) for exp in self.exps])
    def expand(self):
        return Sum(*[exp.expand() for exp in self.exps])
    def display(self):
        return "Sum({})".format(",".join([e.display() for e in self.exps]))
    def derivative(self, var):
        return Sum(*[exp.derivative(var) for exp in self.exps])
    def substitute(self, var, new):
        return Sum(*[exp.substitute(var,new) for exp in self.exps])
    def _python_expr(self):
        return "+".join("({})".format(exp._python_expr()) for exp in self.exps)
    
class Product(Expression):
    def __init__(self, exp1, exp2):
        self.exp1 = exp1
        self.exp2 = exp2
    def latex(self):
        return "{}{}".format(
            paren_if_instance(self.exp1,Sum,Negative,Difference),
            dot_if_necessary(paren_if_instance(self.exp2,Sum,Negative,Difference)))
    def evaluate(self, **bindings):
        return self.exp1.evaluate(**bindings) * self.exp2.evaluate(**bindings)
    def expand(self):
        expanded1 = self.exp1.expand()
        expanded2 = self.exp2.expand()
        if isinstance(expanded1, Sum):
            return Sum(*[Product(e,expanded2).expand() for e in expanded1.exps])
        elif isinstance(expanded2, Sum):
            return Sum(*[Product(expanded1,e) for e in expanded2.exps])
        else:
            return Product(expanded1,expanded2)
        
    def display(self):
        return "Product({},{})".format(self.exp1.display(),self.exp2.display())
    
    def derivative(self,var):
        if not contains(self.exp1, var):
            return Product(self.exp1, self.exp2.derivative(var))
        elif not contains(self.exp2, var):
            return Product(self.exp1.derivative(var), self.exp2)
        else:
            return Sum(
                Product(self.exp1.derivative(var), self.exp2),
                Product(self.exp1, self.exp2.derivative(var)))

    def substitute(self, var, exp):
        return Product(self.exp1.substitute(var,exp), self.exp2.substitute(var,exp))
    
    def _python_expr(self):
        return "({})*({})".format(self.exp1._python_expr(), self.exp2._python_expr())
    
class Difference(Expression):
    def __init__(self,exp1,exp2):
        self.exp1 = exp1
        self.exp2 = exp2
    def latex(self):
        return "{} - {}".format(
            self.exp1.latex(),
            paren_if_instance(self.exp2,Sum,Difference,Negative))
    def evaluate(self, **bindings):
        return self.exp1.evaluate(**bindings) - self.exp2.evaluate(**bindings)
    def expand(self):
        return self
    def display(self):
        return "Difference({},{})".format(self.exp1.display(), self.exp2.display())
    def derivative(self,var):
        return Difference(self.exp1.derivative(var),self.exp2.derivative(var))
    def substitute(self, var, exp):
        return Difference(self.exp1.substitute(var,exp), self.exp2.substitute(var,exp))   
    def _python_expr(self):
        return "({}) - ({})".format(self.exp1._python_expr(), self.exp2._python_expr())
    
class Quotient(Expression):
    def __init__(self,numerator,denominator):
        self.numerator = numerator
        self.denominator = denominator
    def latex(self):
        return "\\frac{{ {} }}{{ {} }}".format(self.numerator.latex(),self.denominator.latex())
    def evaluate(self, **bindings):
        return self.numerator.evaluate(**bindings) / self.denominator.evaluate(**bindings)
    def expand(self):
        return self
    def display(self):
        return "Quotient({},{})".format(self.numerator.display(),self.denominator.display())
    def substitute(self, var, exp):
        return Quotient(self.numerator.substitute(var,exp), self.denominator.substitute(var,exp))
    def derivative(self, var):
        return Quotient(
            Difference(
                Product(self.denominator, self.numerator.derivative(var)),
                Product(self.numerator, self.denominator.derivative(var))
            ),
            Power(self.denominator,Number(2)))
    def _python_expr(self):
        return "({}) / ({})".format(self.exp1._python_expr(), self.exp2._python_expr())
    
class Negative(Expression):
    def __init__(self,exp):
        self.exp = exp
    def latex(self):
        return "- {}".format(
            paren_if_instance(self.exp,Sum,Difference,Negative))
    def evaluate(self, **bindings):
        return - self.exp.evaluate(**bindings)
    def expand(self):
        return self
    def derivative(self,var):
        return Negative(self.exp.derivative(var))
    def substitute(self,var,exp):
        return Negative(self.exp.substitute(var,exp))
    def _python_expr(self):
        return "- ({})".format(self.exp._python_expr())
    def display(self):
        return "Negative({})".format(self.exp.display())
    
class Number(Expression):
    def __init__(self,number):
        self.number = number
    def latex(self):
        return str(self.number)
    def evaluate(self, **bindings):
        return self.number
    def expand(self):
        return self
    def display(self):
        return "Number({})".format(self.number)
    def derivative(self,var):
        return Number(0)
    def substitute(self,var,exp):
        return self
    def _python_expr(self):
        return str(self.number)
    
class Power(Expression):
    def __init__(self,base,exponent):
        self.base = base
        self.exponent = exponent
    def latex(self):
        return "{} ^ {{ {} }}".format(
            paren_if_instance(self.base, Sum, Negative, Difference, Quotient, Product),
            self.exponent.latex())
    def evaluate(self, **bindings):
        return self.base.evaluate(**bindings) ** self.exponent.evaluate(**bindings)
    def expand(self):
        return self
#         expanded_exponent = self.exponent.expand()
#         print (expanded_exponent)
#         if isinstance(expanded_exponent, Number)\
#             and (expanded_exponent.number % 1 == 0)\
#             and (expanded_exponent.number > 0):
#                 power = int(expanded_exponent.number)
#                 if power == 1:
#                     return self.base.expand()
#                 else:
#                     return Product(self.base.expand(), Power(self.base,Number(power-1)).expand()).expand()
#         else:
#             return Power(self.base.expand, expanded_exponent)
    def display(self):
        return "Power({},{})".format(self.base.display(),self.exponent.display())
    def derivative(self,var):
        if isinstance(self.exponent, Number):
            power_rule = Product(
                    Number(self.exponent.number), 
                    Power(self.base, Number(self.exponent.number - 1)))
            return Product(self.base.derivative(var),power_rule)
        elif isinstance(self.base, Number):
            exponential_rule = Product(Apply(Function("ln"),Number(self.base.number)), self)
            return Product(self.exponent.derivative(var), exponential_rule)
        else:
            raise Exception("couldn't take derivative of power {}".format(self.display()))
    def substitute(self,var,exp):
        return Power(self.base.substitute(var,exp), self.exponent.substitute(var,exp))
    
    def _python_expr(self):
        return "({}) ** ({})".format(self.base._python_expr(), self.exponent._python_expr())
    
class Variable(Expression):
    def __init__(self,symbol):
        self.symbol = symbol
    def latex(self):
        return self.symbol
    def evaluate(self, **bindings):
        return bindings[self.symbol]
    def expand(self):
        return self
    def display(self):
        return "Variable(\"{}\")".format(self.symbol)
    def derivative(self, var):
        if self.symbol == var.symbol:
            return Number(1)
        else:
            return Number(0)
    def substitute(self, var, exp):
        if self.symbol == var.symbol:
            return exp
        else:
            return self
        
    def _python_expr(self):
        return self.symbol
        
class Function():
    def __init__(self,name,make_latex=None):
        self.name = name
        self.make_latex = make_latex
    def latex(self,arg_latex):
        if self.make_latex:
            return self.make_latex(arg_latex)
        else:
            return " \\operatorname{{ {} }} \\left( {} \\right)".format(self.name, arg_latex)
  
class Apply(Expression):
    def __init__(self,function,argument):
        self.function = function
        self.argument = argument
    def latex(self):
        return self.function.latex(self.argument.latex())
#         return "\\operatorname{{ {} }} \\left( {} \\right)".format(self.function.name, self.argument.latex())
    def evaluate(self, **bindings):
        return _function_bindings[self.function.name](self.argument.evaluate(**bindings))
    def expand(self):
        return Apply(self.function, self.argument.expand())
    def display(self):
        return "Apply(Function(\"{}\"),{})".format(self.function.name, self.argument.display())
    def derivative(self, var):
        return Product(
                self.argument.derivative(var), 
                _derivatives[self.function.name].substitute(_var, self.argument))
    def substitute(self,var,exp):
        return Apply(self.function, self.argument.substitute(var,exp))
    
    def _python_expr(self):
        return _function_python[self.function.name].format(self.argument._python_expr())

_function_bindings = {
    "sin": math.sin,
    "cos": math.cos,
    "ln": math.log,
    "sqrt": math.sqrt
}

_function_python = {
    "sin": "math.sin({})",
    "cos": "math.cos({})",
    "ln": "math.log({})",
    "sqrt": "math.sqrt({})"
}

_var = Variable('placeholder variable')

_derivatives = {
    "sin": Apply(Function("cos"), _var),
    "cos": Product(Number(-1), Apply(Function("sin"), _var)),
    "ln": Quotient(Number(1), _var),
    "sqrt": Quotient(Number(1), Product(Number(2), Apply(Function("sqrt"), _var)))
}
    
x = Variable('x')
y = Variable('y')
z = Variable('z')
a = Variable('a')
b = Variable('b')

def _apply(func_name):
    return (lambda x: Apply(Function(func_name), x)) 

Sin = _apply("sin")
Cos = _apply("cos")
Sqrt = lambda exp: Apply(Function('sqrt', lambda s: "\\sqrt{{ {} }}".format(s)), exp)

def distinct_variables(exp):
    if isinstance(exp, Variable):
        return set(exp.symbol)
    elif isinstance(exp, Number):
        return set()
    elif isinstance(exp, Sum):
        return set().union(*[distinct_variables(exp) for exp in exp.exps])
    elif isinstance(exp, Product):
        return distinct_variables(exp.exp1).union(distinct_variables(exp.exp2))
    elif isinstance(exp, Power):
        return distinct_variables(exp.base).union(distinct_variables(exp.exponent))
    elif isinstance(exp, Apply):
        return distinct_variables(exp.argument)
    else:
        raise TypeError("Not a valid expression.")
        
def contains(exp, var):
    if isinstance(exp, Variable):
        return exp.symbol == var.symbol
    elif isinstance(exp, Number):
        return False
    elif isinstance(exp, Sum):
        return any([contains(e,var) for e in exp.exps])
    elif isinstance(exp, Product):
        return contains(exp.exp1,var) or contains(exp.exp2,var)
    elif isinstance(exp, Power):
        return contains(exp.base, var) or contains(exp.exponent, var)
    elif isinstance(exp, Apply):
        return contains(exp.argument, var)
    else:
        raise TypeError("Not a valid expression.")

# TODO: equality
# TODO: evalb
# TODO: substitution
# TODO: derivative


# In[7190]:


Product(Power(Variable("x"),Number(2)),Apply(Function("sin"),Variable("y")))


# In[7191]:


Sum(Variable("x"),Variable("c"),Number(1)).derivative(Variable("x"))


# In[7192]:


Product(Variable("c"),Variable("x")).derivative(Variable("x"))


# In[7193]:


Apply(Function("sin"),Power(Variable("x"),Number(2))).derivative(x)


# In[7194]:


Product(Power(Variable("x"),Number(2)),Number(4))


# In[7195]:


f_expression = Product( #<4>
                Sum(
                    Product(
                        Number(3),
                        Power(
                            Variable("x"),
                            Number(2))), 
                    Variable("x")), 
                Apply(
                    Function("sin"),
                    Variable("x")))


# In[7196]:


f_expression = Product(Sum(Product(Number(3),Power(Variable("x"),Number(2))),Variable("x")), 
                Apply(
                    Function("sin"),
                    Variable("x")))

# f_expression = Power(Variable("x"),Number(2))

# In[7197]:


# f_expression = Sum(
#                 Product(
#                     Number(3),
#                     Power(
#                         Variable("x"),
#                         Number(2))), 
#                     Variable("x")), 
#                 Apply(
#                     Function("sin"),
#                     Variable("x"))


# In[7198]:


# f_expression = Sum(Product(Number(3),Power(Variable("x"),Number(2))),Variable("x")),Apply(Function("sin"),Variable("x"))
# print(f_expression)


# In[7199]:


# f_expression.derivative(x)
f_expression


# In[5]:


def int_float(val):
    
    # Copy value
#     val = value.copy()
    
    try:
        # Convert integers
        c  = int(float(val))
#         print("c:",c)
        d = float(value)
#         print("d:",d)

        # Check for integer
        if c == val:
            return val
        elif float(c) == d:
            return c
        else:
            return d
    except:
        return val


# In[6]:


math_operators = ['sin','cos','tan','exp','log','log10','atan','asin','acos','sinh','cosh','tanh']


# In[7]:


def sym_algebra(part_2a):
    
    try:
        f_index = full_eq_str_edit_2.index(part_2a)
    except:
        f_index = full_eq_str_edit_2.index(full_eq_str_edit_2[part_2a])

    print("P2:",full_eq_str_edit_2[f_index])
    
    i = f_index

    try:
        f_exp = var_type(full_eq_str_edit_2[f_index+2])
    except:
        f_exp = var_type(full_eq_str_edit_2[f_index])
    
    f_lst = []
    algebra_list = []
    
    if not full_eq_str_edit_2[f_index] == part_2a:
        print("C2")

    try:
        part_1 = var_type(full_eq_str_edit_2[f_index+2])
    except:
        part_1 = var_type(full_eq_str_edit_2[f_index])

    try:
        part_2 = var_type(full_eq_str_edit_2[f_index+4])
    except:
        part_1 = var_type(full_eq_str_edit_2[f_index])

    if full_eq_str_edit_2[f_index-1] == '**':
    #         print("eq2:",eq_lst_sorted[i])
        part_1 = var_type(full_eq_str_edit_2[f_index+4])
        print("part 1:",part_1)
        part_1a = full_eq_str_edit_2[f_index+4]
        print("part 1a:",part_1a)
        part_2 = var_type(full_eq_str_edit_2[f_index+6])
        print("part 2:", part_2)        
        part_2a = full_eq_str_edit_2[f_index+6]
        print("part 2a:",part_2a)

        if full_eq_str_edit_2[f_index-1]  == '**':
            print("Hi A")
#             f_exp = Product(f_exp,Power(part_1,part_2))
            if part_2a == full_eq_str_edit_2[-1]:
                print("Hi A1")
                f_exp = Product(f_exp,part_1)
            else:
                print("Hi A2")
                f_exp = Product(f_exp,Power(part_1,part_2))                
                
        elif full_eq_str_edit_2[eq_lst_sorted_2[i]-2] == '*':
            print("Hi B")
            f_exp = Power(part_1,part_2)
        elif full_eq_str_edit_2[f_index+3] == '*':
            print("HI")
            f_exp = Product(part_1,part_2)
        elif full_eq_str_edit_2[f_index] == '*':
            print("HI")
            f_exp = Product(part_1,part_2)
        
    if full_eq_str_edit_2[f_index] == part_2a:
        print("A")
        f_exp = var_type(full_eq_str_edit_2[f_index])
        f_lst.append(f_exp)

    f_lst.append(f_exp)
    #     display(f_exp)
    print(f_exp)
    
    return f_lst,part_2a

    


# In[8]:


def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]


# In[9]:


eq_str_split = re.findall('\W+|\w+',eq_str)
eq1 = []
delete_empty = [ele for ele in eq_str_split if ele.strip()]

del_1 = []

for d in delete_empty:
    de = re.split(r'([*,/,+,-])', d)
    de_empty = [ele for ele in de if ele.strip()]
    if  len(de_empty) > 1:
        for d in de_empty:
            del_1.append(d)
    else:
        del_1.append(de_empty[0])

print("del_1:",del_1)
# Update array
eq_str_split_0 = del_1.copy()

decimal_place = []
# Combine decimal places
for i in range(len(eq_str_split_0)):
    if eq_str_split_0[i] == '.':
        dec_flag = i
        dec_0 = eq_str_split_0[i]
#         eq_str_split.remove(dec_0)
        dec_1 = eq_str_split_0[i-1]
        dec_2 = eq_str_split_0[i+1]
        dec = dec_1 + dec_0 + dec_2
        decimal_place.append(dec)

i = 0
dec_places = []
while i < len(eq_str_split_0):
    if eq_str_split[i] == '.':
#         eq_str_split.pop(i)
        dec_places.append(i)
        eq_str_split_0.pop(i+1)
        eq_str_split_0.pop(i)
        eq_str_split_0.pop(i-1)
    i += 1
      
eq_str_split = del_1.copy()  
print(eq_str_split)


# In[10]:


def same_operators(eq_str_split):
    # eq_str_split = del_1
    del_index = []
    eq_power = []
    join_lst = []
    for eq in range(1,len(eq_str_split)):
    #     print(eq)
        if eq_str_split[eq-1] == eq_str_split[eq]:
            eq_pow = eq_str_split[eq-1] + eq_str_split[eq]
            eq_power.append(eq_pow)
            # Save indices
            del_index.append(eq-1)
            del_index.append(eq)
            join_lst.append([eq-1,eq])
    #         break
    # print(del_index
#         print(join_lst)
    return join_lst,eq_power

def indices(index,eq_str_split):
    del_index = []
    eq_power = []
    for eq in range(index,len(eq_str_split)):
    #     print(eq)
        if eq_str_split[eq-1] == eq_str_split[eq]:
            eq_pow = eq_str_split[eq-1] + eq_str_split[eq]
            eq_power.append(eq_pow)
            # Save indices
            del_index.append(eq-1)
            del_index.append(eq)
            break
    return del_index


# In[11]:


del_1 = []

for d in eq_str_split:
    de = re.split(r'([*,/,+,-])', d)
    de_empty = [ele for ele in de if ele.strip()]
    if len(de_empty) > 1:
        B, C = split_list(de_empty)
        del_1.append(B[0])
        del_1.append(C[0])
    else:
        del_1.append(de_empty[0])
# Update array
eq_str_split = del_1.copy()


# In[12]:


def split_operators(eq_str_split):
    
    join_lst,eq_power = same_operators(eq_str_split)
    print(join_lst)

    index = 1
    for i in range(len(join_lst)):    

        del_index = indices(index,eq_str_split)

        print(del_index)
        insert_ind = del_index[-1]
        eq_str_split.insert(insert_ind,eq_power[i])
        for de in del_index:
            eq_str_split.pop(de)
        print(eq_str_split)

        index = insert_ind
        
    return eq_str_split

eq_str_split_2 = split_operators(eq_str_split)
    
print(eq_str_split_2)
    


# In[13]:


index_lst = []
for i in eq_str_split_2:
    if i in math_operators:
        index_lst.append(eq_str_split_2.index(i))
print("C11")


# In[14]:


func = []
for j in index_lst:
    mth_func = eq_str_split[j]
    if mth_func in math_operators:
#         print(mth_func) 
        for k in range(j+1,len(eq_str_split)):
            mth_func += eq_str_split[k]
            if eq_str_split[k] == ')':
                func.append(mth_func)
                print("Hi")
                break
        print(mth_func)

print(func)


# In[15]:


def eq_ind(eq_str_split):
    for eq in eq_str_split:
        if eq in math_operators:
            eq_ind = eq_str_split.index(eq)
            break
    return eq_ind


# In[16]:


for i in range(len(func)):
    
    eq_index = eq_ind(eq_str_split)
    
    cnt = 0            
    for e in range(eq_index, len(eq_str_split)):
        cnt += 1
        if eq_str_split[e] == ')':
    #         flag = e+2
            print("e:",e)
            print("Counter:", cnt)
    #         print("Flag:",flag)
            eq_str_split.insert(eq_index,func[i])
            break
            
    del eq_str_split[eq_index+1:eq_index+cnt+1]


# In[17]:


equation = eq_str_split
eq_ind = []
eq_indices = []
for e in equation:
    if e == '(':
        eq_ind.append(equation.index('('))
    if e == ')':
        eq_ind.append(equation.index(')'))
eq_indices.append(eq_ind)
    


# In[18]:


operators = ['**','*','/','+','-']
operator_lst = []
eq_dict = {}
eq_val = []

# Extract Operators
op = []
for j in eq_str_split:
    if not j in op and j in operators:
        op.append(j)

# # Initialize array
# eq_arr = []
for k in op:
#     print(k)
    # Initialize array
    eq_arr = []
    cnt = 0
    for i in eq_str_split:
        if i == k:
            eq_arr.append(cnt)
        cnt += 1
    eq_dict[k] = eq_arr


# In[19]:


operator_dict = {}
cnt = 1
for i in operators:
    operator_dict[i] = cnt
    cnt += 1
print(operator_dict)


# In[20]:


# eq_para_sort = np.zeros((1,3))
eq_para_sort = []
for i in operator_lst:
    if eq_para[i] == '**':
        eq_para_sort.append(i)
    elif eq_para[i] == '*' :
        eq_para_sort.append(i)
   


# In[21]:


eq_para_sort = []
cnt = 0
for key in eq_dict.keys():
    if key == '**':
        eq_para_sort.insert(0,eq_dict[key])
    if key == '*':
        eq_para_sort.insert(1,eq_dict[key])
    if key == '/':
        eq_para_sort.insert(2,eq_dict[key])
    if key == '+':
        eq_para_sort.insert(3,eq_dict[key])
    if key == '-':
        eq_para_sort.insert(4,eq_dict[key])
        


# In[22]:


# Copy
eq_str_split_1 = eq_str_split.copy()


# In[23]:


# Check for parantheses
parantheses = []
para = []
for i in range(len(eq_str_split)):
#     print(i)
    if eq_str_split[i] == "(" or  eq_str_split[i] == ")":
#         print(eq_str_split.index(i))
#         para.append(eq_str_split.index(eq_str_split[i]))
#         print(i)
        para.append(i)
print(para)

para_len = int(len(para)/2)
for i in range(para_len):
    para_0 = para[i*2:i*2+2]
    parantheses.append(para_0)
    


# In[24]:


def var_type(val):
    if isinstance(val,float):
        return Number(val)
    elif isinstance(val,int):
        return Number(val)
    else:
        return Variable(val)


# In[25]:


# # Create dictionary
operators_dict = {}
operators_dict[operators[0]] = "Power"
operators_dict[operators[1]] = "Multiply"
operators_dict[operators[2]] = "Quotient"
operators_dict[operators[3]] = 'Sum'
operators_dict[operators[4]] = 'Difference'


# In[26]:


# Initialize list
eq_list = []
eq_list = ['(',')']
print("C10")


# In[27]:


powers = []
for i in range(len(eq_str_split_1)):
    if eq_str_split_1[i] == '**':
        powers.append(float(eq_str_split_1[i+1]))


# In[28]:


for i in range(len(eq_str_split_1)):
    if eq_str_split_1[i] == '.':
        eq_str_split_1[i-1] = eq_str_split_1[i-1] + eq_str_split_1[i] + eq_str_split_1[i+1]


leng = len(eq_str_split_1)
print(leng)

i = 0
while i < len(eq_str_split_1):
    if eq_str_split_1[i] == '.':
        del eq_str_split_1[i:i+2]
        leng = leng - 2

    i += 1
    if i == leng:
        break


# In[29]:


eq_str_split_3 = eq_str_split_1.copy() 


# In[30]:



power = []
for i in range(len(eq_str_split_1)):
    if eq_str_split_1[i] == '**':
        power.append(float(eq_str_split_1[i+1]))

powers = []
for i in power:
    if i not in powers:
        powers.append(i)
        
print("C10")


# In[31]:



for k in range(len(eq_str_split_1)):
    if eq_str_split_1[k] == 'x':
        if eq_str_split_1[k-1] == '+':
            eq_str_split_1.insert(k,'1')
            eq_str_split_1.insert(k+1,'*')
        elif eq_str_split_1[k-1] == '-':
            eq_str_split_1.insert(k,'-1')
            eq_str_split_1.insert(k+1,'*')
 
if eq_str_split_1[0] == '-' and eq_str_split_1[1] == 'x':
    eq_str_split_1.insert(0,'-1')
    eq_str_split_1.insert(1,'*')
elif eq_str_split_1[0] == 'x':
    eq_str_split_1.insert(0,'1')
    eq_str_split_1.insert(1,'*')
    
    

  


# In[32]:


for i in range(len(eq_str_split_1)):
    if eq_str_split_1[i] == '-' and float(eq_str_split_1[i+1]) > 0.0:
        eq_str_split_1[i+1] = str(-1*float(eq_str_split_1[i+1]))


# In[33]:


powz = []
powz_index = []
exp = []
exp2 = []
exp_integer = []
signs = []

if powers == []:
    if eq_str_split_1[0] == "-":
        exp_integer.append(eq_str_split_1[1])
    for eq in range(len(eq_str_split_1)):
        if eq_str_split_1[eq] == "*" or eq_str_split_1[eq] == "**":
            exp.append(eq_str_split_1[eq-1])
        elif eq_str_split_1[eq] == eq_str_split_1[-1]:
            exp_integer.append(eq_str_split_1[-1])
        elif eq_str_split_1[eq] == eq_str_split_1[0]:
            exp_integer.append(eq_str_split_1[0])
    
   
for m in powers:
    eq_pow = []
    eq_pow_val = []
    f_lst = []
    exp = []
    exp_integer = []
    if eq_str_split_1[1] == '+' or eq_str_split_1[1] == '-':
#         print("E2")
        exp_integer.append(float(eq_str_split_1[0]))
    if eq_str_split_1[-1] == '+' or eq_str_split_1[1] == '-':
#         print("E3")
#         print("Hi4",eq_str_split_1[-2])
        exp_integer.append(float(eq_str_split_1[-1]))
    if eq_str_split_1[-2] == '+' or eq_str_split_1[-2] == '-':
#         print("E4")
#         print("Hi4",eq_str_split_1[-2])
        exp_integer.append(float(eq_str_split_1[-1]))
    
    for i in range(len(eq_str_split_1)-1):

        f_lst = []
        if eq_str_split_1[i] == '+':
            signs.append(0)
        elif eq_str_split_1[i] == '-':
            signs.append(1)

        if not eq_str_split_1[i] == 'x':
            if (eq_str_split_1[i-1] == '+' and eq_str_split_1[i+1] == '+') or (eq_str_split_1[i-1] == '-' and eq_str_split_1[i+1] == '-') :
#                 print("Hi4a",eq_str_split_1[i])
                exp_integer.append(float(eq_str_split_1[i]))
                i += 2                

            elif eq_str_split_1[i-1] == '-' and eq_str_split_1[i+1] == '+' :
#                 print("Hi4c",eq_str_split_1[1])
                exp_integer.append(float(eq_str_split_1[i]))
            elif eq_str_split_1[i-1] == '-' and eq_str_split_1[i+1] == '-' :
#                 print("Hi4d",eq_str_split_1[1])
                exp_integer.append(float(eq_str_split_1[i]))   

        if eq_str_split_1[i] == 'x':
            if eq_str_split_1[i-1] == '+':
                f_lst.extend([eq_str_split_1[i-2]])
            elif eq_str_split_1[i-1] == '-':
                f_lst.extend([-1*float(eq_str_split_1[i-2])])
                
            if eq_str_split_1[i+1] == '**' and eq_str_split_1[i-1] == '*':
                flag = eq_str_split_1[i+2] 
                flag_2 = flag

            elif eq_str_split_1[i-1] == '*':
                f_lst.append(eq_str_split_1[i-2])
                exp.append(float(f_lst[0]))

            if float(eq_str_split_1[i+2]) == m:
                eq_pow.extend([float(eq_str_split_1[i-2])])
                eq_pow_val.extend([float(m)])
    powz_index.extend(eq_pow_val)
    powz.extend([eq_pow])
             
# Power Exponent
pow_exp = []
for i in powz_index:
    if i not in pow_exp:
        pow_exp.append(i)


# In[34]:


# float(exp_integer[1])
for i in powz:
    print(sum(i))


# In[35]:


exp_int = []
for i in exp_integer:
    try:
        if float(i) or int(i):
            exp_int.append(i)
    except Exception as err:
        pass


# In[36]:


for i in range(len(exp)):
    exp[i] = int_float(exp[i])

for i in range(len(exp_int)):
    exp_int[i] = int_float(exp_int[i])


# In[37]:


term = 'x'
if not powz == []:
    # First value
    pow_1 = str(sum(powz[0])) + '*x**' + str(pow_exp[0])
    if sum(powz[0]) == -1.0:
        pow_1 = '-x**' + str(pow_exp[0])
    for i in range(1,len(powz)):
        if  sum(powz[i]) < 0:
            pow_1a = pow_1 + str(sum(powz[i])) + '*x**' + str(pow_exp[i])
            pow_1 = pow_1a
        else:
            pow_1a = pow_1 + '+' + str(sum(powz[i])) + '*x**' + str(pow_exp[i])
            pow_1 = pow_1a

#     # x power of 1
    power_1 = str(sum(exp)) + '*x'

    # Integers
    integers = str(sum(exp_integer)) 

    if sum(exp) > 0:
        if float(integers) > 0:
            full_eq = pow_1 + '+' + power_1 + '+' + integers
        else:
            full_eq = pow_1 + '+' + power_1 + integers
    elif not exp == []: 
        if eq_str_split_1[-2] == '-' :
            full_eq = pow_1 + integers
        elif eq_str_split_1[-2] == '+':
            full_eq = pow_1 + '+' + integers
    elif sum(exp_integer) > 0:    
        full_eq = pow_1 + '+' + integers
    elif sum(exp_integer) < 0:    
        full_eq = pow_1 + integers
    


# In[38]:


eq_str = full_eq
eq_str_split = re.findall('\W+|\w+',eq_str)
eq1 = []
delete_empty = [ele for ele in eq_str_split if ele.strip()]

del_1 = []

for d in delete_empty:
    de = re.split(r'([*,/,+,-])', d)
    de_empty = [ele for ele in de if ele.strip()]
    if  len(de_empty) > 1:
        for d in de_empty:
            del_1.append(d)
    else:
        del_1.append(de_empty[0])

print("del_1:",del_1)
# Update array
eq_str_split_0 = del_1.copy()

decimal_place = []
# Combine decimal places
for i in range(len(eq_str_split_0)):
    if eq_str_split_0[i] == '.':
        dec_flag = i
        dec_0 = eq_str_split_0[i]
#         eq_str_split.remove(dec_0)
        dec_1 = eq_str_split_0[i-1]
        dec_2 = eq_str_split_0[i+1]
        dec = dec_1 + dec_0 + dec_2
        decimal_place.append(dec)
        
print(dec_flag)
print(decimal_place)

i = 0
dec_places = []
while i < len(eq_str_split_0):
    if eq_str_split[i] == '.':
#         eq_str_split.pop(i)
        dec_places.append(i)
        eq_str_split_0.pop(i+1)
        eq_str_split_0.pop(i)
        eq_str_split_0.pop(i-1)
    i += 1      
  
eq_str_split = del_1.copy()  


# In[39]:


eq_str_split_1 = eq_str_split[:]

for i in range(len(eq_str_split_1)):
    if eq_str_split_1[i] == '.':
        eq_str_split_1[i-1] = eq_str_split_1[i-1] + eq_str_split_1[i] + eq_str_split_1[i+1]

leng = len(eq_str_split_1)

i = 0
while i < len(eq_str_split_1):
    if eq_str_split_1[i] == '.':
#         eq_str_split_3.remove(eq_str_split_3[i-1])
        del eq_str_split_1[i:i+2]
#         eq_str_split_3.remove(eq_str_split_3[i+1])
#         print(eq_str_split_3)
        leng = leng - 2

    i += 1
    if i == leng:
        break
        


# In[40]:


# eq_str_split_1 = eq_str_split
eq_str_split_3 = []
for i in range(0,len(eq_str_split_1)-1):
    if eq_str_split_1[i+1] == '.' :
        dec_value = eq_str_split_1[i] + eq_str_split_1[i+1] + eq_str_split_1[i+2] 
        eq_str_split_3.append(dec_value)
    else:
        eq_str_split_3.append(eq_str_split_1[i])
        

eq_str_split_3a = []
leng = len(eq_str_split_3)   
full_eq_str_edit = eq_str_split_1[:]   


# In[41]:


def int_float(value):
    
    # Copy value
    val = value[:]
    
    try:
        # Convert integers
        c  = int(float(val))
#         print("c:",c)
        d = float(value)
#         print("d:",d)

        # Check for integer
        if c == val:
            return val
        elif float(c) == d:
            return c
        else:
            return d
    except:
        return val


# In[42]:


for i in range(len(full_eq_str_edit)):
    full_eq_str_edit[i] = int_float(full_eq_str_edit[i])


# In[43]:


def split_operators(eq_str_split):
    
    join_lst,eq_power = same_operators(eq_str_split)
    print(join_lst)

    index = 1
    for i in range(len(join_lst)):    

    #     index = 1
        del_index = indices(index,eq_str_split)
#         print(del_index)

        print(del_index)
        # del_index = join_lst[0]
        insert_ind = del_index[-1]
        eq_str_split.insert(insert_ind,eq_power[i])
        for de in del_index:
            eq_str_split.pop(de)
        print(eq_str_split)

        index = insert_ind
        
    return eq_str_split

eq_str_split_2 = split_operators(full_eq_str_edit)
    
print(eq_str_split_2)


# In[44]:


full_eq_str_edit_2 = full_eq_str_edit[:]
full_eq_str_edit_2a = full_eq_str_edit_2[:] 
print(full_eq_str_edit_2)
print("C1")


# In[45]:


i = i + 1
val_list = []
val_list.append(full_eq_str_edit_2a[0])
for i in range(len(full_eq_str_edit_2a)):
    if full_eq_str_edit_2a[i] == '*':
        part_3a = full_eq_str_edit_2a[i-3]
#         print("part_3a:", part_3a)
        part_3 = var_type(full_eq_str_edit_2a[i-3])
#         print("part_3:", part_3)
        part_1 = var_type(full_eq_str_edit_2a[i-1])
#         print("part 1:",part_1)
        part_1a = full_eq_str_edit_2a[i-1]
#         print("part 1a:",part_1a)
        part_2 = var_type(full_eq_str_edit_2a[i+1])
#         print("part 2:",part_2)
        part_2a = full_eq_str_edit_2a[i+1]
#         print("part 2a:",part_2a)
        
        


# In[46]:


# for i in range(2,len(full_eq_str_edit_2a)):
# #     f_exp = var_type(full_eq_str_edit_2[i])
# #     print("full:",f_exp)
#     if full_eq_str_edit_2a[i] == '**':
#         part_3a = full_eq_str_edit_2a[i-3]
#         part_3 = var_type(full_eq_str_edit_2a[i-3])
#         print("full_1:",type(full_eq_str_edit_2a[i-3]))
#         print("full_2:",full_eq_str_edit_2a[i-3])
#         print("full_2a:",full_eq_str_edit_2a[i-2])
#         if full_eq_str_edit_2a[i-1] == full_eq_str_edit_2a[0]:
#             if full_eq_str_edit_2a[i-2] == '-':
#                 part_1 = var_type(full_eq_str_edit_2a[i-1])
#                 print("part 1:",part_1)
#                 part_1a = full_eq_str_edit_2a[i-1]
#                 print("part 1a:",part_1a)
#                 part_2 = var_type(full_eq_str_edit_2a[i+1])
#                 print("part 2:",part_2)
#                 part_2a = full_eq_str_edit_2a[i+1]
#                 print("part 2a:",part_2a)
#                 val = Negative(Power(part_1,part_2))
#                 display(val)
#                 val_list.append(val)
#                 part_i = i
#                 print("part_i:", i)       
#             else:              
#                 part_1 = var_type(full_eq_str_edit_2a[i-1])
#                 print("part 1:",part_1)
#                 part_1a = full_eq_str_edit_2a[i-1]
#                 print("part 1a:",part_1a)
#                 part_2 = var_type(full_eq_str_edit_2a[i+1])
#                 print("part 2:",part_2)
#                 part_2a = full_eq_str_edit_2a[i+1]
#                 print("part 2a:",part_2a)
#                 val = Power(part_1,part_2)
#                 display(val)
#                 val_list.append(val)
#                 part_i = i
#                 print("part_i:", i)
#         break


# In[47]:


i = i + 1
val_list = []

if full_eq_str_edit_2a[0] == '-':
    val_list.append('-')

for i in range(1,len(full_eq_str_edit_2a)):     
  
    if full_eq_str_edit_2a[i] == '*' and full_eq_str_edit_2a[i+2] == '**':
#         print("Terms")
        part_3a = full_eq_str_edit_2a[i-3]
#         print("part_3a:", part_3a)
        part_3 = var_type(full_eq_str_edit_2a[i-3])
#         print("part_3:", part_3)
        part_1 = var_type(full_eq_str_edit_2a[i-1])
#         print("part 1:",part_1)
        part_1a = full_eq_str_edit_2a[i-1]
#         print("part 1a:",part_1a)
        part_2 = var_type(full_eq_str_edit_2a[i+1])
#         print("part 2:",part_2)
        part_2a = full_eq_str_edit_2a[i+1]
        
        if full_eq_str_edit_2a[i+4] == '-' or full_eq_str_edit_2a[i+4] == '+':
            if full_eq_str_edit_2a[i-2]:
                val = Product(part_1,Power(part_2,var_type(full_eq_str_edit_2a[i+3])))
            elif full_eq_str_edit_2a[0] == full_eq_str_edit_2a[i-2]:
                val = Negative(Product(part_1,Power(part_2,var_type(full_eq_str_edit_2a[i+3]))))
            else:
                val = Product(part_1,Power(part_2,var_type(full_eq_str_edit_2a[i+3])))
            val_list.append(val)
#             i += 1    
        elif full_eq_str_edit_2a[i-2] == '-':
            val = Power(part_1,part_2)
            val_list.append(val)
            i += 1            
        elif isinstance(part_3a, int) or isinstance(part_3a, float):
            val_2 = Product(part_3,Power(part_1,part_2))
            val_list.append(val_2)
            i += 1
    
    elif full_eq_str_edit_2a[i] == '*':
        part_3a = full_eq_str_edit_2a[i-3]
        part_3 = var_type(full_eq_str_edit_2a[i-3])
#         print("part_3:", part_3)
        part_1 = var_type(full_eq_str_edit_2a[i-1])
#         print("part 1:",part_1)
        part_1a = full_eq_str_edit_2a[i-1]
#         print("part 1a:",part_1a)
        part_2 = var_type(full_eq_str_edit_2a[i+1])
#         print("part 2:",part_2)
        part_2a = full_eq_str_edit_2a[i+1]
#         print("part 2a:",part_2a)
        part_4 = var_type(full_eq_str_edit_2a[i+2])
#         print("part 4:",part_4)
        part_4a = full_eq_str_edit_2a[i+2]
#         print("part 4a:",part_4a)
        
        if full_eq_str_edit_2a[i-2] == '+':
            val = Product(part_1,part_2)
            val_list.append(val)
            i += 1
        elif full_eq_str_edit_2a[i-2] == '-':
            val = Product(part_1,part_2)
            val_list.append(val)
            i += 1            
        elif isinstance(part_3a, int) or isinstance(part_3a, float):
            val_2 = Product(part_3,Power(part_1,part_2))
            val_list.append(val_2)
            val_list.append(part_4a)
            i += 1
            
    elif full_eq_str_edit_2a[i] == '**':
        part_3a = full_eq_str_edit_2a[i-3]
        part_3 = var_type(full_eq_str_edit_2a[i-3])
#         print("part_3:", part_3)
        part_1 = var_type(full_eq_str_edit_2a[i-1])
#         print("part 1:",part_1)
        part_1a = full_eq_str_edit_2a[i-1]
#         print("part 1a:",part_1a)
        part_2 = var_type(full_eq_str_edit_2a[i+1])
#         print("part 2:",part_2)
        part_2a = full_eq_str_edit_2a[i+1]
#         print("part 2a:",part_2a)
        part_4 = var_type(full_eq_str_edit_2a[i+2])
#         print("part 4:",part_4)
        part_4a = full_eq_str_edit_2a[i+2]
#         print("part 4a:",part_4a)
        
        if full_eq_str_edit_2a[i-1] == 'x':
            val = Power(var_type(full_eq_str_edit_2a[i-1]),var_type(full_eq_str_edit_2a[i+1]))
            val_list.append(val)
            val_list.append(full_eq_str_edit_2a[i+2])
            i += 1
        elif full_eq_str_edit_2a[i-2] == '+':
            val = Power(part_1,part_2)
            val_list.append(val)
            i += 1
        elif full_eq_str_edit_2a[i-2] == '-':
            val = Power(part_1,part_2)
            val_list.append(val)
            i += 1   
        elif i == 3:
            val_2 = Negative(Product(part_3,Power(part_1,part_2)))
            val_list.append(val_2)
            val_list.append(part_4a)
            i += 1
            
full_index = full_eq_str_edit_2a.index(part_2a)

# End. Integers
integer = var_type(full_eq_str_edit_2a[-1])
val_list.append(full_eq_str_edit_2a[-2])
val_list.append(integer)

val_list_2 = []
for i in range(len(val_list)):
    if not val_list[i] == val_list[i-1]:
        val_list_2.append(val_list[i])

val_list = val_list_2[:]



 

        
        


# In[48]:


# Equation
val_list_0 = val_list[:]


# In[49]:


if val_list[0] == '-' or val_list[0] == '+':
    val_index = 1
    val_list[1] = Negative(val_list[1])
else:
    val_index = 0
term_1 = val_list[val_index]
for m in range(1,len(val_list)):
    if val_list[m] == "+":
        summation = Sum(term_1,val_list[m+1])
        term_1 = summation
    if val_list[m] == "-":
        summation = Difference(term_1,val_list[m+1])
        term_1 = summation
full_eq_1 = term_1
display(full_eq_1)


# In[50]:


def derivative(var_1):
    
    i = -1
    num = []
    bracket = []
    while not var_1[i] == "(":
        print("I:", i)
        bracket.append(")")
        if not var_1[i] == ")":
            num.append(var_1[i])
    #         Index
            var_1_index = var_1.index(var_1[i])

        i -= 1
    print(num)

    deriv_num = num[::-1]
    deriv_num_0 = deriv_num[0]
    for i in range(1,len(deriv_num)):
        first_num = deriv_num_0 + deriv_num[i]
        deriv_num_0 = first_num

    print("Derivative:", first_num)

    print(var_1[0:var_1_index])
    brac_cnt = 0
    for i in var_1[0:var_1_index]:
        if i == ')':
            brac_cnt += 1

    i = 0
    num = []
    bracket = []
    while not var_1[i] == ")":
        print("I:", i)
        bracket.append("(")
        if not var_1[i] == "(":
            num.append(var_1[i])
    #         Index
            var_1a_index = var_1.index(var_1[i])

        i += 1

    part_1 = var_1[0:var_1a_index]
    print(part_1)
    part_2 = str(float(var_1_0))
    print(part_2)
    part_3 = var_1[var_1a_index+1:var_1_index]
    print(part_3)
    part_4 = str(float(first_num) - 1)
    print(part_4)
    part_5 = var_1[var_1_index+3:len(var_1)]
    print(part_5)
    # part_4 = var[0]
    parts = part_1 + part_2 + part_3 + part_4 + part_5
    print(parts)
#     display(eval(parts))
    
    return eval(parts)


# In[51]:


def integration(var_2):

    if var_2 == '+' or var_2 == '-':
        return print("Error. Please do not input operators '+' or '-'.")

    i = -1
    num = []
    bracket = []
    while not var_2[i] == "(":
        bracket.append(")")
        if not var_2[i] == ")":
            num.append(var_2[i])
    #         Index
            var_1_index = var_2.index(var_2[i])

        i -= 1

    deriv_num = num[::-1]
    deriv_num_0 = deriv_num[0]
    for i in range(1,len(deriv_num)):
        first_num = deriv_num_0 + deriv_num[i]
        deriv_num_0 = first_num

    brac_cnt = 0
    for i in var_2[0:var_1_index]:
        if i == ')':
            brac_cnt += 1

    i = 0
    num = []
    bracket = []
    while not var_2[i] == ")":
        bracket.append("(")
        if not var_2[i] == "(":
            num.append(var_2[i])
    #         Index
            var_1a_index = var_2.index(var_2[i])
        i += 1

    i = 0
    num = []
    bracket = []
    while not var_2[i] == ")":
        bracket.append("(")
        if not var_2[i] == ")":
            num.append(var_2[i])
    #         Index
            var_1_index = var_2.index(var_2[i])

        i += 1

    var_1_1 = var_2[var_1_index+1:]
    var_1_0 = var_2[0:var_1_index]
    var_1_0 = float(var_1[var_1_index]) * float(first_num)

    # Integral
    integral = float(var_1[var_1_index])/(float(first_num)+1)

    i = -1
    num = []
    bracket = []
    while not var_2[i] == "(":
        bracket.append(")")
        if not var_2[i] == ")":
            num.append(var_2[i])
    #         Index
            var_1_index = var_1.index(var_2[i])

        i -= 1

    deriv_num = num[::-1]
    deriv_num_0 = deriv_num[0]
    for i in range(1,len(deriv_num)):
        first_num = deriv_num_0 + deriv_num[i]
        deriv_num_0 = first_num

    brac_cnt = 0
    for i in var_2[0:var_1_index]:
        if i == ')':
            brac_cnt += 1

    i = 0
    num = []
    bracket = []
    while not var_2[i] == ")":
        bracket.append("(")
        if not var_2[i] == "(":
            num.append(var_2[i])
    #         Index
            var_1a_index = var_2.index(var_2[i])

        i += 1
    if str(var_2[0:5]) == "Power":
        # Derivative
        deri_num = float(first_num) - 1
        pows = var_1[0:var_1_index] + str(deri_num) + var_2[var_1_index+3:len(var_2)]
        parts = Product(var_type(first_num),eval(pows))
    elif str(var_2[0:7]) == "Product":

        try:
            part_1 = var_2[0:var_1a_index]
#             print("part 1:",part_1)
            part_2 = str(float(integral))
#             print("part 2:",part_2)
            part_3 = var_2[var_1a_index+1:var_1_index]
#             print("part 3:",part_3)
            part_4 = str(float(first_num) + 1)
#             print("part 4:",part_4)
            part_5 = var_2[var_1_index+3:len(var_2)]
#             print("part_5:",part_5)
            # part_4 = var[0]
            parts = part_1 + part_2 + part_3 + part_4 + part_5
        except Exception as err:
            I = 8
            num_1 = []
            bracket = []
            while not var_2[i] == ")":
                bracket.append("(")
                if not var_2[i] == "(":
                    num_1.append(var_2[i])
        #               Index
                    var_1b_index = var_2.index(var_2[i])
                i += 1  
            parts = var_2[I:var_1b_index+2]
    
    if var_2[0:8] == 'Negative':
        return Negative(eval(parts)) 
    else:
        try:
            return eval(parts)
        except Exception as err:
            return parts





# In[52]:



def derivative(index,val_list_0):

    final_product = []
    var_1 = str(val_list_0[index])
    starting_index = 0

    if var_1[0:5] == "Power":
        starting_index = 0
        # Derivative
        i = 0 
        variable_1 = []
        while not var_1[i] == ')':
            if var_1[i] == '(':
                var_index = str(var_1[starting_index:len(var_1)]).index(var_1[i])
                variable_1.append(var_index)
                starting_index = var_index + 1
            elif var_1[i+1] == ')':
                var_index = str(var_1).index(var_1[i])
                variable_1.append(var_index)
            i += 1
        starting_index = i
        str_var_1 = var_1[i+2:len(var_1)-1]
        var_1 = str(val_list_0[index])
        starting_index = 20
        
        if var_1[0:5] == "Power":
            # Derivative
            i = 0
            variable_1 = []
            while not var_1[i] == ')':
                if var_1[i] == '(':
                    var_index = str(var_1[starting_index:len(var_1)]).index(var_1[i])
                    variable_1.append(var_index)
                    starting_index = var_index + 1
                elif var_1[i+1] == ')':
                    var_index = str(var_1).index(var_1[i])
                    variable_1.append(var_index)
                i += 1
                
        starting_index = i

        pow_ind = float(str_var_1[7:len(str_var_1)-1])-1
        factor  = var_type(str_var_1[7:len(str_var_1)-1])
        powz = Power(Variable("x"),Number(pow_ind))
        final_product = Product(factor,powz)
        
    elif var_1[0:7] == "Product":
        # Derivative
        i = 0
        variable_1 = []
        while not var_1[i] == ')':
            if var_1[i] == '(':
                var_index = str(var_1[starting_index:len(var_1)]).index(var_1[i])
                variable_1.append(var_index)
                starting_index = var_index + 1
            elif var_1[i+1] == ')':
                var_index = str(var_1).index(var_1[i])
                brac_index = var_index
            i += 1
        starting_index = i
        prod = var_1[brac_index:starting_index]
        str_var_1 = var_1[i+2:len(var_1)-1]

        brac_index = starting_index
        while not var_1[brac_index] == '(':
            brac_index -= 1

        prod = var_1[brac_index+1:starting_index]

        if str_var_1 == str(Variable("x")):
            final_product =  var_type(prod)
        else:
            var_1 = str_var_1
            starting_index = 20
            if var_1[0:5] == "Power":
                # Derivative
                i = 0
                variable_1 = []
                while not var_1[i] == ')':
                    if var_1[i] == '(':
                        var_index = str(var_1[starting_index:len(var_1)]).index(var_1[i])
                        variable_1.append(var_index)
                        starting_index = var_index + 1
                    elif var_1[i+1] == ')':
                        var_index = str(var_1).index(var_1[i])
                        variable_1.append(var_index)
                    i += 1

            starting_index = i
            str_var_1 = var_1[i+2:len(var_1)-1]
            pow_ind = float(str_var_1[7:len(str_var_1)-1])-1
            factor = str_var_1[7:len(str_var_1)-1]
            var_factor  = var_type(str_var_1[7:len(str_var_1)-1])
            powz = Power(Variable("x"),Number(pow_ind))

            # Product
            prod_val = float(prod)*float(factor)
            if pow_ind == float(1):
                final_product =  Product(var_type(prod_val),Variable("x"))
            else:
                final_product = Product(var_type(prod_val),powz)
                
    return final_product


# In[53]:


def derivative_eq(val_list_1):
    
    # Differential Equation
    val_list_1 = []
    for v in range(len(val_list_0)):
        if not str(val_list_0[v])[0:6] == "Number":
            val_list_1.append(val_list_0[v])
            num_index = v
    del val_list_1[num_index]
      
    diff = []
    for val in range(len(val_list_1)):
    #     print(val)
        if val_list_1[val] == '-' or val_list_1[val] == '+':
            diff.append(val_list_1[val])
        else:
            diff.append(derivative(val,val_list_1))

    val_list = diff
    if val_list[0] == '-' or val_list[0] == '+':
        val_index = 1
        val_list[1] = Negative(val_list[1])
    else:
        val_index = 0
    term_1 = val_list[val_index]
    for m in range(1,len(val_list)):
        if val_list[m] == "+":
            summation = Sum(term_1,val_list[m+1])
            term_1 = summation
        if val_list[m] == "-":
            summation = Difference(term_1,val_list[m+1])
            term_1 = summation
    full_eq_1 = term_1
    display(full_eq_1)
    return


# In[54]:


derivative_eq(val_list_0)


# In[55]:


def integration(index,val_list_0):

    var_1 = str(val_list_0[index])
    starting_index = 0

    if var_1[0:6] == "Number":
        starting_index = 7
        i = 7
        while not var_1[i] == ')':
            i += 1
        num = var_1[starting_index:i]
        final_product = Product(var_type(num),var_type("x"))

    elif var_1[0:5] == "Power":
        starting_index = 0
        # Derivative
        i = 0
        variable_1 = []
        while not var_1[i] == ')':
            if var_1[i] == '(':
                var_index = str(var_1[starting_index:len(var_1)]).index(var_1[i])
                variable_1.append(var_index)
                starting_index = var_index + 1
            elif var_1[i+1] == ')':
                var_index = str(var_1).index(var_1[i])
                variable_1.append(var_index)
            i += 1
        starting_index = i
        str_var_1 = var_1[i+2:len(var_1)-1]
        var_1 = str(val_list_0[index])
        starting_index = 20

        if var_1[0:5] == "Power":
            # Derivative
            i = 0
            variable_1 = []
            while not var_1[i] == ')':
                if var_1[i] == '(':
                    var_index = str(var_1[starting_index:len(var_1)]).index(var_1[i])
                    variable_1.append(var_index)
                    starting_index = var_index + 1
                elif var_1[i+1] == ')':
                    var_index = str(var_1).index(var_1[i])
                    variable_1.append(var_index)
                i += 1

        starting_index = i
        pow_ind = float(str_var_1[7:len(str_var_1)-1])+1
        factor_0  = var_type(str_var_1[7:len(str_var_1)-1])
        factor_1 = str_var_1[7:len(str_var_1)-1]
        powz = Power(Variable("x"),Number(pow_ind))
        integral= 1/pow_ind
        factor = var_type(integral)
        final_product = Product(factor,powz)
    elif var_1[0:7] == "Product":
        # Derivative
        i = 0
        variable_1 = []
        while not var_1[i] == ')':
            if var_1[i] == '(':
                var_index = str(var_1[starting_index:len(var_1)]).index(var_1[i])
                variable_1.append(var_index)
                starting_index = var_index + 1
            elif var_1[i+1] == ')':
                var_index = str(var_1).index(var_1[i])
                brac_index = var_index
            i += 1

        starting_index = i
        prod = var_1[brac_index:starting_index]
        str_var_1 = var_1[i+2:len(var_1)-1]
        brac_index = starting_index
        while not var_1[brac_index] == '(':
            brac_index -= 1
        prod = var_1[brac_index+1:starting_index]
        
        if str_var_1 == str(Variable("x")):
            prod =  var_type(float(prod)/float(2))
            var_x = Power(var_type("x"),var_type(2))
            final_product = Product(prod,var_x)
        else:
            var_1 = str_var_1
            starting_index = 20
            if var_1[0:5] == "Power":
                # Derivative
                i = 0
                variable_1 = []
                while not var_1[i] == ')':
                    if var_1[i] == '(':
                        var_index = str(var_1[starting_index:len(var_1)]).index(var_1[i])
                        variable_1.append(var_index)
                        starting_index = var_index + 1
                    elif var_1[i+1] == ')':
                        var_index = str(var_1).index(var_1[i])
                        variable_1.append(var_index)
                    i += 1
            
            starting_index = i
            str_var_1 = var_1[i+2:len(var_1)-1]
            pow_ind = float(str_var_1[7:len(str_var_1)-1])-1
            factor = str_var_1[7:len(str_var_1)-1]
            var_factor  = var_type(str_var_1[7:len(str_var_1)-1])
            powz = Power(Variable("x"),Number(float(factor) + float(1)))
            
            # Product
            prod_val = float(prod)/(float(factor)+float(1))
            if pow_ind == float(1):
                final_product = Product(var_type(prod_val),Variable("x"))
            else:
                final_product = Product(var_type(prod_val),powz)
                
    return final_product


# In[56]:


def integration_eq(val_list_0):
    integral = []
    for val in range(len(val_list_0)):
        if val_list_0[val] == '-' or val_list_0[val] == '+':
            integral.append(val_list_0[val])
        else:
            integral.append(integration(val,val_list_0))

    val_list = integral
    if val_list[0] == '-' or val_list[0] == '+':
        val_index = 1
        val_list[1] = Negative(val_list[1])
    else:
        val_index = 0
    term_1 = val_list[val_index]
    for m in range(1,len(val_list)):
        if val_list[m] == "+":
            summation = Sum(term_1,val_list[m+1])
            term_1 = summation
        if val_list[m] == "-":
            summation = Difference(term_1,val_list[m+1])
            term_1 = summation
    full_eq_1 = term_1
    display(full_eq_1)
    return


# In[57]:


integration_eq(val_list_0)

