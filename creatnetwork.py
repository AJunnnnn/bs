import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import random

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

file_path = 'data.xlsx'
sheet_name = 'Sheet1'
# 创建一个字典存储边权重总和
edge_weights = {}
# 权重的阈值，小于该值则连边不构建
n = 5
# 伊辛模型参数
T_list = [100,300,500,700,900]  # 温度参数列表
h = 0.0      # 外部磁场强度
# T = 1000.0      # 温度参数
num_steps = 10000


# 存放结果
results = {}  # 存储每个温度的结果
all_subgraphs = {}  # 存储各温度子图数据
negative_state_ratios = {T: [] for T in T_list} # 用于存储每个温度下 -1 状态节点占比的变化



# 计算中心性指标
def analyze_centralities(G):
    # 介数中心性（考虑权重）
    betweenness = nx.betweenness_centrality(G, weight='distance', normalized=True)
    # 接近中心性（考虑权重）
    closeness = nx.closeness_centrality(G, distance='distance')  # 注意这里使用distance参数
    # 转换为DataFrame
    df_centrality = pd.DataFrame({
        'Betweenness': betweenness,
        'Closeness': closeness
    })
    # print(betweenness)
    # print(closeness)
    # print(df_centrality)
    return df_centrality

# 可视化函数
def plot_centrality_distribution(df, centrality_type, bins=20):
    plt.figure(figsize=(10, 6))
    plt.hist(df[centrality_type], bins=bins, edgecolor='k', alpha=0.7)
    plt.title(f'{centrality_type} Distribution')
    plt.xlabel(centrality_type)
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.show()

# 计算初始磁化强度和能量
def calculate_energy(G):
    energy = 0.0
    for u, v in G.edges():
        J = G.edges[u, v]['weight']
        energy -= J * G.nodes[u]['state'] * G.nodes[v]['state']
    # 外部磁场项
    for node in G.nodes():
        energy -= h * G.nodes[node]['state']
    return energy

# 定义单个节点状态翻转的接受概率
def acceptance_probability(dE, T):
    if dE < 0:
        return 1.0
    else:
        return np.exp(-dE / T)

# 读取Excel文件中的Sheet1
df = pd.read_excel(file_path, sheet_name=sheet_name)
for _, row in df.iterrows():
    s_city = row['s_city']
    e_city = row['e_city']
    s_prov = row['s_prov']
    e_prov = row['e_prov']
    weight = row['counts']

    # 组合城市名称和省份名称，确保唯一性
    s_unique = f"{s_city}_{s_prov}"
    e_unique = f"{e_city}_{e_prov}"

    # 对城市名称排序，确保A-B和B-A视为同一无向边
    sorted_edge = tuple(sorted([s_unique, e_unique]))
    # print(sorted_edge)

    # 累加权重
    if sorted_edge in edge_weights:
        edge_weights[sorted_edge] += weight
    else:
        edge_weights[sorted_edge] = weight

# 准备写入新的xlsx文件的数据
# edges_data = [(n1, n2, total_weight) for (n1, n2), total_weight in edge_weights.items()]
# df_edges = pd.DataFrame(edges_data, columns=['Node1', 'Node2', 'Weight'])
# output_file_path = 'output_edges.xlsx'
# df_edges.to_excel(output_file_path, index=False)
# print(f"数据已成功写入到 {output_file_path}")

G = nx.Graph()
for (node1, node2), total_weight in edge_weights.items():
    # 需求1：去除自环（当node1等于node2时跳过）
    if node1 == node2:
        continue
    # 需求2：仅保留权重≥n的边
    if total_weight >= n:
        # 核心修改：将权重转换为距离（权重越大，距离越小）
        distance = 1 / total_weight  # 取倒数
        G.add_edge(node1, node2, weight=total_weight, distance=distance)

# 生成邻接矩阵（节点顺序保持一致）
nodes = list(G.nodes())
adj_matrix = nx.to_numpy_array(G, nodelist=nodes, weight='weight')
print(f"邻接矩阵为:\n{adj_matrix}")
# # 验证邻接矩阵
# print("邻接矩阵维度:", adj_matrix.shape)
# print("示例元素（节点0-1的J值）:", adj_matrix[0][1])

# 验证自环是否已去除
print("图中自环数量:", nx.number_of_selfloops(G))  # 应该输出0

# 验证边权重过滤
min_weight = min(dict(G.edges).values(), key=lambda x: x['weight'])['weight']
print("图中最小权重:", min_weight)  # 应该≥n


'''
对网络进行结构分析
'''
print("节点数量:", G.number_of_nodes())
print("边数量:", G.number_of_edges())

# 平均度（无向图）
avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
print("平均度:", avg_degree)

# 网络密度（实际边数 / 最大可能边数）
density = nx.density(G)
print("密度:", density)

# 连通性
is_connected = nx.is_connected(G)
print("无向图是否连通:", is_connected)

# 绘制度分布直方图
degrees = nx.degree_histogram(G)
step = 30
max_degree = len(degrees) - 1  # 最大度数
bins = list(range(0, max_degree + step, step))  # 生成区间边界
# 计算每个区间的节点数总和
hist_values = []
for i in range(len(bins)-1):
    start = bins[i]
    end = bins[i+1]
    hist_values.append(sum(degrees[start:end]))
# 生成区间标签（左闭右开）
labels = [f'{bins[i]}-{bins[i+1]-1}' for i in range(len(bins)-1)]
# 绘制直方图
plt.figure(figsize=(12, 6))
bars = plt.bar(labels, hist_values)
# 添加数值标签
for bar in bars:
    height = bar.get_height()
    if height > 0:
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height}',
                 ha='center', va='bottom')
plt.xlabel('Degree Range')
plt.ylabel('Number of Nodes')
plt.title('度分布直方图')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# 执行分析
df_centrality = analyze_centralities(G)

# 查看前10个节点的中心性值
print("中心性指标示例：")
print(df_centrality.head(10))

# 可视化分布
plot_centrality_distribution(df_centrality, 'Betweenness')
plot_centrality_distribution(df_centrality, 'Closeness')

# 保存结果到Excel
# df_centrality.to_excel('network_centrality_analysis.xlsx')
# print("分析结果已保存到 network_centrality_analysis.xlsx")


'''
完整图太大了 绘制子图
'''
# plt.figure(figsize=(10, 8))
# selected_nodes = random.sample(list(G.nodes()), 10)
# # 提取与选定节点相关的边以形成子图
# subgraph = G.subgraph(selected_nodes)
# pos = nx.spring_layout(subgraph)
#
# # 提取城市名作为节点标签，只考虑子图中的节点
# city_labels = {node: node.split('_')[0] for node in subgraph.nodes()}
#
# nx.draw(subgraph, pos=pos, with_labels=True, labels=city_labels, edge_color='black', node_color='lightblue', node_size=100, font_size=10)
# labels = nx.get_edge_attributes(subgraph, 'weight')
# nx.draw_networkx_edge_labels(subgraph, pos=pos, edge_labels=labels, font_size=8)
# plt.show()

# 固定随机种子以确保可重复性
random.seed(42)
# 预先选择固定节点（示例选择50个节点）
sample_size = min(50, G.number_of_nodes())
fixed_nodes = random.sample(list(G.nodes()), sample_size)
# 生成基准布局（使用第一个温度模拟后的状态）
nx.set_node_attributes(G, 1, "state")  # 临时初始化状态
base_subgraph = G.subgraph(fixed_nodes)
# pos = nx.spring_layout(base_subgraph, seed=42, weight='distance')  # 保持布局一致
pos = nx.circular_layout(base_subgraph)
# pos = nx.random_layout(base_subgraph, seed=42)


for T in T_list:
    print(f"\n=== 开始模拟 T={T} ===")

    # 初始化节点状态（所有节点状态设为1）
    nx.set_node_attributes(G,1,"state")
    # 在仿真开始前随机翻转10%的节点状态
    num_nodes = len(nodes)
    num_flip_initial = max(1, int(0.2 * num_nodes))  # 随机翻转20%的节点
    random.seed(42)  # 设置种子
    selected_nodes = random.sample(nodes, num_flip_initial)

    for node in selected_nodes:
        G.nodes[node]['state'] *= -1  # 直接翻转状态
    print(f"初始翻转了 {num_flip_initial} 个节点状态")

    # 模拟过程（使用Metropolis算法）
    energy_history = []
    magnetization_history = []

    # 初始状态验证
    print("初始磁化强度:",
         sum(nx.get_node_attributes(G, 'state').values()) / num_nodes)
    print("初始能量:", calculate_energy(G))

    for step in range(0,num_steps):
        # 随机选择一个节点
        node = random.choice(nodes)
        # 计算当前能量（局部能量）
        current_energy = calculate_energy(G)

        # 尝试翻转状态
        G.nodes[node]['state'] *= -1
        new_energy = calculate_energy(G)

        # 计算能量变化
        delta_E = new_energy - current_energy

        # 决定是否接受状态翻转
        if random.random() < acceptance_probability(delta_E, T):
            # 接受翻转
            pass
        else:
            # 拒绝翻转，恢复原状态
            G.nodes[node]['state'] *= -1

        # 每100步记录系统状态
        if step % 500 == 0:
            print(f"T={T}，已完成 {step / num_steps * 100:.1f}%")
            # 计算总能量
            total_energy = calculate_energy(G)
            energy_history.append(total_energy)

            # 计算磁化强度（平均状态）
            magnetization = sum(nx.get_node_attributes(G, 'state').values()) / G.number_of_nodes()
            magnetization_history.append(magnetization)

            # 计算 -1 状态节点的占比
            states = list(nx.get_node_attributes(G, 'state').values())
            negative_ratio = states.count(-1) / len(states) * 100
            negative_state_ratios[T].append(negative_ratio)

    # 最终状态验证
    print("最终磁化强度:", magnetization_history[-1])
    print("最终能量:", energy_history[-1])
    # 保存结果
    results[T] = {
        'energy': energy_history,
        'magnetization': magnetization_history,
        'final_states': nx.get_node_attributes(G, 'state')
    }
    subgraph = G.subgraph(fixed_nodes)
    all_subgraphs[T] = {
        'subgraph': subgraph,
        'node_colors': [subgraph.nodes[n]['state'] for n in fixed_nodes],
        'edge_weights': [subgraph.edges[e]['weight'] for e in subgraph.edges]
    }
    print(f"=== 完成 T={T} 模拟 ===")


# 磁化强度变化和能量变化结果可视化
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
# plt.plot(range(0, num_steps, 100), magnetization_history)
for T in T_list:
    plt.plot(
        range(0, num_steps, 100),
        results[T]['magnetization'],
        marker='o',
        linestyle='--',
        label=f'T={T}'
    )
plt.xlabel('模拟步数（每100步记录）')
plt.ylabel('平均磁化强度')
plt.title('不同温度下磁化强度演化')
plt.legend()
plt.grid(True)


plt.subplot(1, 2, 2)
# plt.plot(range(0, num_steps, 100), energy_history)
for T in T_list:
    plt.plot(
        range(0, num_steps, 100),
        results[T]['energy'],
        marker='s',
        linestyle='-.',
        label=f'T={T}'
    )
plt.xlabel('模拟步数（每100步记录）')
plt.ylabel('系统能量')
plt.title('不同温度下系统能量演化')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

# 统计状态分布并计算百分比
# plt.figure(figsize=(15, 3))
# for idx, T in enumerate(T_list, 1):
#     states = list(results[T]['final_states'].values())
#     plus1 = states.count(1) / len(states) * 100
#     minus1 = states.count(-1) / len(states) * 100
#     plt.subplot(1, len(T_list), idx)
#     plt.pie([plus1, minus1],
#             explode=(0.1, 0),
#             labels=['状态 +1', '状态 -1'],
#             autopct='%1.1f%%',
#             colors=['#ff9999', '#66b3ff'],
#             shadow=True)
#     plt.title(f'T = {T}')
# plt.suptitle('不同温度下最终状态分布', y=1.1)
# plt.tight_layout()
# plt.show()

# 绘制 -1 状态节点占比的折线图
plt.figure(figsize=(12, 6))
for T in T_list:
    plt.plot(
        range(0, num_steps, 100),
        negative_state_ratios[T],
        marker='o',
        linestyle='--',
        label=f'T={T}'
    )
plt.xlabel('模拟步数（每100步记录）')
plt.ylabel('-1 状态节点占比 (%)')
plt.title('不同温度下 -1 状态节点占比变化')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# # 绘制最终状态网络图（随机采样部分节点）
# plt.figure(figsize=(10, 8))
# selected_nodes = random.sample(list(G.nodes()), min(50, len(G.nodes()))) #采样50个节点
# subgraph = G.subgraph(selected_nodes)
# # pos = nx.spring_layout(subgraph)
# # pos = nx.kamada_kawai_layout(subgraph, weight='distance', scale=2)
# # 方案2：弹簧布局（调整参数优化布局）
# pos = nx.spring_layout(subgraph,
#                        k=0.6,          # 节点间最优距离系数（增大可减少重叠）
#                        iterations=300, # 增加迭代次数确保收敛
#                        weight='distance', # 使用距离作为权重
#                        seed=42)        # 固定随机种子保证可重复性
# # 节点颜色映射
# node_colors = [subgraph.nodes[node]['state'] for node in selected_nodes]
# node_labels = {node: node.split('_')[0] for node in subgraph.nodes()}  # 简化节点标签
# # edge_weights = [subgraph.edges[edge]['weight']/10 for edge in subgraph.edges()]
#
# # nx.draw(subgraph, pos,
# #         node_color=node_colors,
# #         cmap=plt.cm.coolwarm,
# #         vmin=-1,
# #         vmax=1,
# #         edgelist=[],
# #         with_labels=True,
# #         node_size=200,
# #         font_size=8)
# # 绘制节点
# nx.draw_networkx_nodes(subgraph, pos,
#                       node_color=node_colors,
#                       cmap=plt.cm.coolwarm,
#                       vmin=-1,
#                       vmax=1,
#                       node_size=300,
#                       alpha=0.8)
# # 绘制标签（单独控制标签尺寸）
# nx.draw_networkx_labels(subgraph, pos,
#                        labels=node_labels,
#                        font_size=8,
#                        font_color='black')
#
# plt.title('节点状态分布(部分)')
# plt.axis('off')  # 关闭坐标轴
# patches = [
#     mpatches.Patch(color='red', label='状态 +1'),
#     mpatches.Patch(color='blue', label='状态 -1')
# ]
# plt.legend(handles=patches)
# plt.tight_layout()
# plt.show()


plt.figure(figsize=(15, 5 * len(T_list)))
for idx, T in enumerate(T_list, 1):
    sg_data = all_subgraphs[T]

    plt.subplot(len(T_list), 1, idx)
    nx.draw_networkx_nodes(
        sg_data['subgraph'], pos,
        node_color=sg_data['node_colors'],
        cmap=plt.cm.coolwarm,
        vmin=-1,
        vmax=1,
        node_size=150,
        edgecolors='black',
        linewidths=0.5
    )
    # nx.draw_networkx_edges(
    #     sg_data['subgraph'], pos,
    #     width=np.array(sg_data['edge_weights']) / 10,
    #     edge_color='gray'
    # )
    nx.draw_networkx_labels(
        sg_data['subgraph'], pos,
        labels={n: n.split('_')[0] for n in fixed_nodes},
        font_size=8,
        verticalalignment='center'
    )

    plt.legend(
        handles=[
            mpatches.Patch(color='#ff9999', label='状态 +1'),
            mpatches.Patch(color='#66b3ff', label='状态 -1')
        ],
        loc='upper right',
        bbox_to_anchor=(0.98, 0.95),  # 调整图例位置
        frameon=False
    )
    plt.title(f'T = {T} 网络状态分布（共{sample_size}个节点）')
    plt.axis('off')

# 添加统一图例
plt.tight_layout()
plt.subplots_adjust(right=0.85)
cax = plt.axes([0.88, 0.2, 0.02, 0.6])
plt.suptitle('不同温度下固定节点网络状态对比', y=0.98)
plt.show()

