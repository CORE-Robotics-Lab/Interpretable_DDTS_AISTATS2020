# Created by Andrew Silva on 3/29/19
import numpy as np
import torch
from interpretable_ddts.agents.ddt import DDT


def convert_to_discrete(fuzzy_model):
    new_weights = []
    new_comps = []

    weights = np.abs(fuzzy_model.layers.detach().numpy())
    most_used = np.argmax(weights, axis=1)
    for comp_ind, comparator in enumerate(fuzzy_model.comparators):
        comparator = comparator.item()
        divisor = abs(fuzzy_model.layers[comp_ind][most_used[comp_ind]].item())
        if divisor == 0:
            divisor = 1
        comparator /= divisor
        new_comps.append([comparator])
        max_ind = most_used[comp_ind]
        new_weight = np.zeros(len(fuzzy_model.layers[comp_ind].data))
        new_weight[max_ind] = fuzzy_model.layers[comp_ind][most_used[comp_ind]].item() / divisor
        new_weights.append(new_weight)

    new_input_dim = fuzzy_model.input_dim
    new_weights = np.array(new_weights)
    new_comps = np.array(new_comps)
    crispy_model = DDT(input_dim=new_input_dim,
                       output_dim=fuzzy_model.output_dim,
                       weights=new_weights,
                       comparators=new_comps,
                       leaves=fuzzy_model.leaf_init_information,
                       alpha=99999.,
                       is_value=fuzzy_model.is_value,
                       use_gpu=fuzzy_model.use_gpu)

    # For a ddt that preserves actions using softmaxes, use old action probs
    crispy_model.action_probs.data = fuzzy_model.action_probs.data
    #
    # For a set of discrete ddt parameters (0, 1) leaves, use the one-hot method:
    # max_inds = fuzzy_model.action_probs.data.argmax(dim=1)
    # new_action_probs = torch.zeros_like(fuzzy_model.action_probs.data)
    # new_action_probs[np.arange(len(new_action_probs)), max_inds] = 10
    # crispy_model.action_probs.data = new_action_probs

    if fuzzy_model.use_gpu:
        crispy_model = crispy_model.cuda()

    return crispy_model

