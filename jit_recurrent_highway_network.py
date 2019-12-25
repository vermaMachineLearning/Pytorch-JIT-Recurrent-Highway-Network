import torch
import torch.nn as nn
import torch.jit as jit
from typing import List
from torch import Tensor


class RHNCell(jit.ScriptModule):
    __constants__ = ['nb_rhn_layers', 'drop_prob', 'hidden_dim']

    def __init__(self, input_dim, hidden_dim, nb_rhn_layers, drop_prob):
        super(RHNCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nb_rhn_layers = nb_rhn_layers
        self.drop_prob = drop_prob

        self.drop_layer = nn.Dropout(drop_prob)
        self.input_fc = nn.Linear(input_dim, 2 * hidden_dim)
        self.first_fc_layer = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.fc_layers = nn.ModuleList([nn.Linear(hidden_dim, 2 * hidden_dim) for _ in range(nb_rhn_layers-1)])

    @jit.script_method
    def highwayGate(self, hidden_states, s_t_l_minus_1):
        h, t = torch.split(hidden_states, self.hidden_dim, 1)
        h, t = torch.tanh(h), torch.sigmoid(t)
        c = 1 - t
        t = self.drop_layer(t)
        s_t = h * t + s_t_l_minus_1 * c
        return s_t

    @jit.script_method
    def forward(self, x_t, s_t_l_0):

        hidden_states = self.input_fc(x_t) + self.first_fc_layer(s_t_l_0)
        s_t_l = self.highwayGate(hidden_states, s_t_l_0)

        s_t_l_minus_1 = s_t_l
        for fc_layer in self.fc_layers:
            hidden_states = fc_layer(s_t_l_minus_1)
            s_t_l = self.highwayGate(hidden_states, s_t_l_minus_1)
            s_t_l_minus_1 = s_t_l

        return s_t_l


class RHN(jit.ScriptModule):

    def __init__(self, input_dim, hidden_dim, nb_rhn_layers, drop_prob):
        super(RHN, self).__init__()

        self.rhncell = RHNCell(input_dim, hidden_dim, nb_rhn_layers, drop_prob)
        self.output_fc = nn.Linear(hidden_dim, hidden_dim)

    @jit.script_method
    def forward(self, input, s_t_0_l_0):

        inputs = input.unbind(1)
        s_t_minus_1_L = s_t_0_l_0
        outputs = torch.jit.annotate(List[Tensor], [])
        for t in range(len(inputs)):
            s_t_L = self.rhncell(inputs[t], s_t_minus_1_L)
            s_t_minus_1_L = s_t_L
            outputs += [s_t_L]

        x = torch.stack(outputs).transpose(0, 1)
        x = self.output_fc(x)
        return x

