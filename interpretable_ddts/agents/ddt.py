# Created by Andrew Silva on 2/21/19
import torch.nn as nn
import torch
import numpy as np


class DDT(nn.Module):
    def __init__(self,
                 input_dim,
                 weights,
                 comparators,
                 leaves,
                 output_dim=None,
                 alpha=1.0,
                 is_value=False,
                 use_gpu=False):
        super(DDT, self).__init__()
        """
        Initialize the DDT, taking in premade weights for inputs to comparators and sigmoids
        initialized tree. 
        :param weights: input weights (for intelligent init or for loading)
        :param comparators: input comparators (for intelligent init or for loading)
        :param input_dim: int. always required for input dimensionality
        :param leaves: int, must be 2**N
        :param output_dim: None or int, must be an int if weights and comparators are None
        :param alpha: int. Strictness of the tree, default 1
        :param is_value: if False, outputs are passed through a Softmax final layer. Default: False
        :param use_gpu: is this a GPU-enabled network? Default: False
        """
        self.use_gpu = use_gpu
        self.leaf_init_information = leaves

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = None
        self.comparators = None
        self.selector = None

        self.init_comparators(comparators)
        self.init_weights(weights)
        self.init_alpha(alpha)
        self.init_paths()
        self.init_leaves()
        self.added_levels = nn.Sequential()

        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.is_value = is_value

    def init_comparators(self, comparators):
        if comparators is None:
            comparators = []
            if type(self.leaf_init_information) is int:
                depth = int(np.floor(np.log2(self.leaf_init_information)))
            else:
                depth = 4
            for level in range(depth):
                for node in range(2**level):
                    comparators.append(np.array([1.0/self.input_dim]))
        new_comps = torch.Tensor(comparators)
        new_comps.requires_grad = True
        if self.use_gpu:
            new_comps = new_comps.cuda()
        self.comparators = nn.Parameter(new_comps)

    def init_weights(self, weights):
        if weights is None:
            weights = []
            if type(self.leaf_init_information) is int:
                depth = int(np.floor(np.log2(self.leaf_init_information)))
            else:
                depth = 4
            for level in range(depth):
                for node in range(2**level):
                    weights.append(np.random.rand(self.input_dim))

        new_weights = torch.Tensor(weights)
        new_weights.requires_grad = True
        if self.use_gpu:
            new_weights = new_weights.cuda()
        self.layers = nn.Parameter(new_weights)

    def init_alpha(self, alpha):
        self.alpha = torch.Tensor([alpha])
        if self.use_gpu:
            self.alpha = self.alpha.cuda()
        self.alpha.requires_grad = True
        self.alpha = nn.Parameter(self.alpha)

    def init_paths(self):
        if type(self.leaf_init_information) is list:
            left_branches = torch.zeros((len(self.layers), len(self.leaf_init_information)))
            right_branches = torch.zeros((len(self.layers), len(self.leaf_init_information)))
            for n in range(0, len(self.leaf_init_information)):
                for i in self.leaf_init_information[n][0]:
                    left_branches[i][n] = 1.0
                for j in self.leaf_init_information[n][1]:
                    right_branches[j][n] = 1.0
        else:
            if type(self.leaf_init_information) is int:
                depth = int(np.floor(np.log2(self.leaf_init_information)))
            elif self.leaf_init_information is None:
                depth = 4
            left_branches = torch.zeros((2 ** depth - 1, 2 ** depth))
            for n in range(0, depth):
                row = 2 ** n - 1
                for i in range(0, 2 ** depth):
                    col = 2 ** (depth - n) * i
                    end_col = col + 2 ** (depth - 1 - n)
                    if row + i >= len(left_branches) or end_col >= len(left_branches[row]):
                        break
                    left_branches[row + i, col:end_col] = 1.0
            right_branches = torch.zeros((2 ** depth - 1, 2 ** depth))
            left_turns = np.where(left_branches == 1)
            for row in np.unique(left_turns[0]):
                cols = left_turns[1][left_turns[0] == row]
                start_pos = cols[-1] + 1
                end_pos = start_pos + len(cols)
                right_branches[row, start_pos:end_pos] = 1.0
        left_branches.requires_grad = False
        right_branches.requires_grad = False
        if self.use_gpu:
            left_branches = left_branches.cuda()
            right_branches = right_branches.cuda()
        self.left_path_sigs = left_branches
        self.right_path_sigs = right_branches

    def init_leaves(self):
        if type(self.leaf_init_information) is list:
            new_leaves = [leaf[-1] for leaf in self.leaf_init_information]
        else:
            new_leaves = []
            if type(self.leaf_init_information) is int:
                depth = int(np.floor(np.log2(self.leaf_init_information)))
            else:
                depth = 4

            last_level = np.arange(2**(depth-1)-1, 2**depth-1)
            going_left = True
            leaf_index = 0
            self.leaf_init_information = []
            for level in range(2**depth):
                curr_node = last_level[leaf_index]
                turn_left = going_left
                left_path = []
                right_path = []
                while curr_node >= 0:
                    if turn_left:
                        left_path.append(int(curr_node))
                    else:
                        right_path.append(int(curr_node))
                    prev_node = np.ceil(curr_node / 2) - 1
                    if curr_node // 2 > prev_node:
                        turn_left = False
                    else:
                        turn_left = True
                    curr_node = prev_node
                if going_left:
                    going_left = False
                else:
                    going_left = True
                    leaf_index += 1
                new_probs = np.random.uniform(0, 1, self.output_dim)  # *(1.0/self.output_dim)
                self.leaf_init_information.append([sorted(left_path), sorted(right_path), new_probs])
                new_leaves.append(new_probs)

        labels = torch.Tensor(new_leaves)
        if self.use_gpu:
            labels = labels.cuda()
        labels.requires_grad = True
        self.action_probs = nn.Parameter(labels)

    def forward(self, input_data, embedding_list=None):

        input_data = input_data.t().expand(self.layers.size(0), *input_data.t().size())

        input_data = input_data.permute(2, 0, 1)
        comp = self.layers.mul(input_data)
        comp = comp.sum(dim=2).unsqueeze(-1)
        comp = comp.sub(self.comparators.expand(input_data.size(0), *self.comparators.size()))
        comp = comp.mul(self.alpha)
        sig_vals = self.sig(comp)

        sig_vals = sig_vals.view(input_data.size(0), -1)

        one_minus_sig = torch.ones(sig_vals.size())
        if self.use_gpu:
            one_minus_sig = one_minus_sig.to('cuda')

        one_minus_sig = torch.sub(one_minus_sig, sig_vals)

        left_path_probs = self.left_path_sigs.t()
        right_path_probs = self.right_path_sigs.t()
        left_path_probs = left_path_probs.expand(input_data.size(0), *left_path_probs.size()) * sig_vals.unsqueeze(1)
        right_path_probs = right_path_probs.expand(input_data.size(0), *right_path_probs.size()) * one_minus_sig.unsqueeze(1)
        left_path_probs = left_path_probs.permute(0, 2, 1)
        right_path_probs = right_path_probs.permute(0, 2, 1)

        # We don't want 0s to ruin leaf probabilities, so replace them with 1s so they don't affect the product
        left_filler = torch.zeros(self.left_path_sigs.size())
        left_filler[self.left_path_sigs == 0] = 1
        right_filler = torch.zeros(self.right_path_sigs.size())
        if self.use_gpu:
            left_filler = left_filler.cuda()
            right_filler = right_filler.cuda()
        right_filler[self.right_path_sigs == 0] = 1

        left_path_probs = left_path_probs.add(left_filler)
        right_path_probs = right_path_probs.add(right_filler)

        probs = torch.cat((left_path_probs, right_path_probs), dim=1)
        probs = probs.prod(dim=1)
        actions = probs.mm(self.action_probs)

        if not self.is_value:
            return self.softmax(actions)
        else:
            return actions
