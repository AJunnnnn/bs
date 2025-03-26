# 伊辛模型的能量函数
def energyFunction(node_states, adj_matrix, J, h):
    N = len(node_states)
    energy = 0
    for i in range(N):
        for j in range(i + 1, N):
            if adj_matrix[i, j]:
                energy = energy -J[i][j] * node_states[i] * node_states[j]
        energy = energy -h * node_states[i]
    return energy

