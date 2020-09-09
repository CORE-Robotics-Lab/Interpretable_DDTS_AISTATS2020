# Created by Andrew Silva on 3/28/19
import numpy as np


def ddt_init_from_dt(estimator):
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold
    num_feats = estimator.n_features_

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1, -1, False)]
    master_list = []
    while len(stack) > 0:
        node_id, parent_depth, parent_node_id, right_child = stack.pop()
        node_depth[node_id] = parent_depth + 1
        master_list.append([node_id, parent_node_id, right_child])
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1, node_id, False))
            stack.append((children_right[node_id], parent_depth + 1, node_id, True))
        else:
            is_leaves[node_id] = True

    init_weights = []
    init_comparators = []
    leaves = []
    weight_node_map = []
    init_leaves = []
    for i in range(n_nodes):
        if is_leaves[i]:
            probs = np.zeros(estimator.n_classes_)
            probs[np.argmax(estimator.tree_.value[i])] = 1.
            new_leaf = [[], [], probs]
            current_id = i
            while current_id != 0:
                for node in master_list:
                    if node[0] == current_id:
                        if node[2]:
                            new_leaf[1].append(node[1])
                        else:
                            new_leaf[0].append(node[1])
                        current_id = node[1]
                        break
            leaves.append(new_leaf)
        else:
            init_weight = np.zeros(num_feats)

            init_weight[feature[i]] = -1.
            init_weights.append(init_weight)
            init_comparators.append([-threshold[i]])

            weight_node_map.append(i)
    for leaf in leaves:
        new_left = []
        new_right = []
        for left_turn in leaf[0]:
            new_left.append(weight_node_map.index(left_turn))
        for right_turn in leaf[1]:
            new_right.append(weight_node_map.index(right_turn))
        init_leaves.append([new_left, new_right, leaf[2]])
    return init_weights, init_comparators, init_leaves
