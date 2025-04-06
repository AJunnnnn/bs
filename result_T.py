import copy
import networkx as nx
from concurrent.futures import ProcessPoolExecutor

import network_builder
from IsingModle import (
    normalization,
    select_fixed_nodes_and_layout,
    run_simulation,
    plot_result,
    plot_negative_ratio_vs_temperature,
)

class Config:
    N = 95 # 权重的阈值，小于该值则连边不构建
    T_LIST = [0.1, 0.4, 0.7, 1.0] # 温度参数列表
    h = 0.0 # 外部磁场强度
    NUM_STEPS = 10000 # 模拟步数
    RECORD_INTERVAL = 1000 # 记录间隔

def simulate_temperature(T, G_data):
    import networkx as nx
    from IsingModle import run_simulation

    G = nx.from_dict_of_dicts(G_data)
    energy_history, magnetization_history, negative_ratio_history = run_simulation(
        G, T, Config.NUM_STEPS, Config.h, Config.RECORD_INTERVAL
    )
    final_states = nx.get_node_attributes(G, 'state')
    return T, energy_history, magnetization_history, negative_ratio_history, final_states

if __name__ == '__main__':
    # 构建网络
    G = network_builder.build_network('data.xlsx', 'Sheet1', Config.N)
    G = normalization(G)
    G_data = nx.to_dict_of_dicts(G)  # 转换为可序列化格式

    # 固定节点和布局
    fixed_nodes, pos = select_fixed_nodes_and_layout(G, 50, 'circular')

    # 多进程模拟
    results = {}
    all_subgraphs = {}
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(simulate_temperature, T, G_data): T for T in Config.T_LIST}
        for future in futures:
            try:
                T, energy, magnetization, neg_ratio, final_states = future.result()
                results[T] = {
                    'energy': energy,
                    'magnetization': magnetization,
                    'negative_ratio': neg_ratio,
                }
                # 更新主进程中的 G
                G_temp = nx.from_dict_of_dicts(G_data)
                nx.set_node_attributes(G_temp, final_states, 'state')
                subgraph = G_temp.subgraph(fixed_nodes)
                all_subgraphs[T] = {
                    'subgraph': subgraph,
                    'node_colors': [subgraph.nodes[n]['state'] for n in fixed_nodes],
                    'edge_weights': [subgraph.edges[e]['weight'] for e in subgraph.edges],
                }
            except Exception as e:
                print(f"Error processing T={futures[future]}: {e}")
                raise

    # 可视化
    plot_result(results, Config.T_LIST, Config.RECORD_INTERVAL)
    plot_negative_ratio_vs_temperature(results, Config.T_LIST)