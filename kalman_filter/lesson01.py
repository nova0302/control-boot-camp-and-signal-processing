'''
Xk = (x1+x2+...+xk)/k -> batch expression
Xk-1 = (x1+x2+...+xk-1)/(k-1) -> batch expression
k*Xk = (x1+x2+...+xk) 
     = Xk-1 * (k-1)
'''

xList = [10, 20, 30, 40, 50]
k = 1
#alpha = (1-1/k)
alpha = (k-1)/k
X=0
for x in xList:
    X = alpha*X + (1-alpha)*x
    print(f'k: {k}, X: {X}')
    k = k + 1
    alpha = (1-1/k)

def AvgFilter(x):
    
