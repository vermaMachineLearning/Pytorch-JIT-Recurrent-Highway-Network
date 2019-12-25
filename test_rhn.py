from jit_recurrent_highway_network import RHN

batch_size = 2
seq_len = 1000
input_dim = 10
hidden_dim = 20
nb_rhn_layers = 5


input = torch.randn(batch_size, seq_len, input_dim)
rhn = RHN(input_dim, hidden_dim, nb_rhn_layers, drop_prob=0.1)

rhn.train()
s_t_0_l_0 = torch.zeros(batch_size, hidden_dim)
output = rhn(input, s_t_0_l_0)
