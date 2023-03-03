#!/usr/bin/env python
# coding: utf-8

# In[75]:


import sympy as sy
import numpy as np
from sympy.functions import sin,cos,exp
import matplotlib.pyplot as plt
plt.style.use("ggplot")


# In[76]:


# Define the variable and the function to approximate
x = sy.Symbol('x')
f = 3*exp(x) / (x**2 + x + 1)


# In[77]:


# Factorial function
def factorial(n):
    if n <= 0:
        return 1
    else:
        return n*factorial(n-1)


# In[78]:


# Taylor approximation at x0 of the function 'function'
def taylor(function,x0,n):
    i = 0
    p = 0
    while i <= n:
        p = p + (function.diff(x,i).subs(x,x0))/(factorial(i))*(x-x0)**i
        i += 1
    return p


# In[79]:


# Plot results
def plot():
    x_lims = [-3,3]
    x1 = np.linspace(x_lims[0],x_lims[1],100)
    y1 = []
    y2 = []
    # Approximate up until 10 starting from 1 and using steps of 2
    for j in range(1,10,1):
        func = taylor(f,0,j)
#         print("J:", func)
        print('Taylor expansion at n='+str(j),func)
        y1_val = 0
        for k in x1:
            y1_val = func.subs(x,k)
            y1.append(func.subs(x,k))
#             print('y1:',y1)
#         print('y1:',y1)
        plt.plot(x1,y1,label='order '+str(j))
        print("Values:",y1_val)
        y1 = []
    # Plot the function to approximate (sine, in this case)
    plt.plot(x1,np.sin(x1),label='sin of x')
    plt.xlim(x_lims)
    plt.ylim([-5,5])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.title('Taylor series approximation')
    plt.show()
    
    return y2

y2 = plot()
print(y2)


# In[80]:


# # loop through tayloer terms
# for term in range(len(y1)):
#     # build up polynomial on each iteration
#     _poly = _terms[term] if _poly is None else _poly + _terms[term]

#     # store current taylor polynomial
#     polynomials.append(_poly)


# In[81]:


import math
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

t, a = sp.symbols('t a')


def taylor_terms(func, order, point, derivatives=None):
    """
    find symbolic derivatives and Taylor terms
    func: symbolic function to approximate
    order: highest order derivative for Taylor polynomial (interger)
    point: point about which the function is approximated
    derivatives: list of symbolic derivatives
    """

    # initialize list of derivatives
    if derivatives is None:
        derivatives = [func.subs({t: a})]

    # check if highest order derivative is reached
    if len(derivatives) > order:
        # return list of taylor terms evaluated using substitution
        return derivatives, [derivatives[i].subs({a: point}) / math.factorial(i) * (t - point) ** i for i in range(len(derivatives))]

    # differentiate function with respect to t
    derivative = func.diff(t)

    # append to list of symbolic derivatives ** substitute t with a **
    derivatives.append(derivative.subs({t: a}))

    # recursive call to find next term in Taylor polynomial
    return taylor_terms(derivative, order, point, derivatives)


def taylor_polynomials(_terms):
    """
    find Taylor polynomials
    func: symbolic function to approximate
    order: highest order derivative for Taylor polynomial (interger)
    point: point about which the function is approximated
    derivatives: list of Taylor terms
    """

    # initialize list
    polynomials = []

    # initialize taylor polynomial
    _poly = None

    # loop through tayloer terms
    for term in range(len(_terms)):
        # build up polynomial on each iteration
        _poly = _terms[term] if _poly is None else _poly + _terms[term]

        # store current taylor polynomial
        polynomials.append(_poly)

    # return taylor polynomials
    return polynomials


if __name__ == '__main__':

    # analysis label
    label = 'ln(t)'

    # symbolic function to approximate
    f = sp.log(t)

    # point about which to approximate
    approximation_point = np.pi

    # definte time start and stop
    start = 0.01
    stop = 2 * sp.pi
    time = np.arange(start, stop, 0.1)

    # find taylor polynomial terms describing function f(t)
    symbolic_derivatives, terms = taylor_terms(func=f, order=4, point=approximation_point)
    print("Terms:", terms)
    polys = taylor_polynomials(terms)
    print('Polys:', polys)

    # initialize plot
    fig, ax = plt.subplots()
    ax.set(xlabel='t', ylabel='y', title=f'Taylor Polynomial Approximation: {label}')
    legend = []

    for p, poly in enumerate(polys):
        # plot current polynomial approximation
        ax.plot(time, [poly.subs({t: point}) for point in time])

        # append item to legend
        legend.append(f'P{p}')

    # plot actual function for comparison
    ax.plot(time, [f.subs({t: point}) for point in time])
    legend.append(f'f(t)')

    # create dataframe
    df = pd.DataFrame({'symbolic_derivatives': symbolic_derivatives,
                       'taylor_terms': terms,
                       'polynomials': polys
                       })

    # save and show results
    ax.legend(legend)
    ax.grid()
    plt.savefig(f'taylor_{label}.png')
    plt.show()
    
    df.to_csv(f'taylor_{label}.csv', encoding='utf-8')
    print(df.head(10))


# In[ ]:




