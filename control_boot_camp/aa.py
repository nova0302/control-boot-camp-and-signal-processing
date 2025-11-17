import numpy as np
import control as ct
#import networkx as nx
import matplotlib.pyplot as plt

## Create a graph
#G = nx.Graph()
#
## Add nodes
#G.add_nodes_from([1, 2, 3, 4, 5])
#
## Add edges
#G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 5), (4, 5)])
#
## Draw the graph
#nx.draw(G, with_labels=True, node_color='skyblue', node_size=700, font_size=10, font_weight='bold')
#plt.title("Simple Graph Visualization")
#plt.show()
import numpy as np

d = -0.1

###################################################3333
# Define a square matrix
Ad = np.array([[ 0, 1],
               [-1, d]])
print(f'Ad = {Ad}')

# Compute eigenvalues and eigenvectors
Dd, Td = np.linalg.eig(Ad)

print("Eigenvalues:", Dd)
print("Eigenvectors:\n", Td)

###################################################3333
# Define a square matrix
Au = np.array([[0,     1],
               [1, d]])

print(f'Au = {Au}')
# Compute eigenvalues and eigenvectors
Du, Tu = np.linalg.eig(Au)

print("Eigenvalues:", Du)
print("Eigenvectors:\n", Tu)

A = np.array([[1,0],
              [0,2]])
B=[0,1]

ctrb_matrix = ct.ctrb(A,B)
print(ctrb_matrix)

ctrb_rank = np.linalg.matrix_rank(ctrb_matrix)
print(ctrb_rank)

A = np.array([[1,2],
              [0,2]])
B=[0,1]

ctrb_matrix = ct.ctrb(A,B)
print(ctrb_matrix)

ctrb_rank = np.linalg.matrix_rank(ctrb_matrix)
print(ctrb_rank)


