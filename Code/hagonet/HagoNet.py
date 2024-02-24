import torch
from torch import nn
from torch.nn import Linear, Embedding
from torch_geometric.nn.acts import swish
# from tol_acts import swish
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn import radius_graph
from torch_scatter import scatter, scatter_min
from math import sqrt
from math import pi as PI
from geonet.features import dist_emb, angle_emb, torsion_emb, area_emb, volume_emb, torsion_emb_3
from torch_sparse import SparseTensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cutoff = 5.0

@torch.no_grad()
def compute_index(p_idx_a, en, pn, vn):
    batch_size = pn.size(0)
    # e_pointer = 0
    p_pointer = 0
    for i in range(batch_size):
        p_pointer += pn[i]
        p_idx_a[p_pointer:] += pn[i]
    return p_idx_a
    
@torch.no_grad()
def compute_index_2(idx_ji, idx_kj, p_idx_a, en, pn, vn):
    batch_size = pn.size(0)
    # e_pointer = 0
    p_pointer = 0
    for i in range(batch_size):
        p_pointer += pn[i]
        idx_ji[p_pointer:] += en[i]
        idx_kj[p_pointer:] += en[i]
        p_idx_a[p_pointer:] += pn[i]
    return idx_ji, idx_kj, p_idx_a

def compute_geometry(pos, edge_index, q_index):
    j, i = edge_index  # j->i

    # j1,i1 = edge_index1 # j1-> i1
    # k, j = edge_index

    # Calculate distances. # number of edges
    dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
    # dist1 = (pos[i1] - pos[j1]).pow(2).sum(dim=-1).sqrt()
    num_nodes = j.size(0)
    value = torch.arange(j.size(0), device=j.device)
    # value1 = torch.arange(k.size(0), device=k.device)
    adj_t = SparseTensor(row=i, col=j, value=value, sparse_sizes=(num_nodes, num_nodes))
    # adj_t1 = SparseTensor(row=j, col=k, value=value1, sparse_sizes=(num_nodes, num_nodes))
    # adj_t1 = SparseTensor(row=j, col=i, value=value, sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[j]

    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # num1_triplets = adj_t1_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (l->k->j->i->jn) for triplets.
    idx_i = i.repeat_interleave(num_triplets)
    idx_j = j.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()

    # idx_l = adj_t_row.storage.col()
    # mask = idx_j != idx_l
    mask = idx_i != idx_k

    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]
    
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]

    # Calculate angles. 0 to pi
    pos_ji = pos[idx_i] - pos[idx_j]
    pos_jk = pos[idx_k] - pos[idx_j]
    a = (pos_ji * pos_jk).sum(dim=-1)  # cos_angle * |pos_ji| * |pos_jk|
    b = torch.cross(pos_ji, pos_jk).norm(dim=-1)  # sin_angle * |pos_ji| * |pos_jk|
    angle = torch.atan2(b, a)

    idx_batch = torch.arange(len(idx_i), device=device)
    idx_k_n = adj_t[idx_j].storage.col()  # j邻居
    # idx_h_n = adj_t[idx_k].storage.col()

    repeat = num_triplets
    num_triplets_t = num_triplets.repeat_interleave(repeat)[mask]
    idx_i_t = idx_i.repeat_interleave(num_triplets_t)
    idx_j_t = idx_j.repeat_interleave(num_triplets_t)
    idx_k_t = idx_k.repeat_interleave(num_triplets_t)

    idx_batch_t = idx_batch.repeat_interleave(num_triplets_t)
    mask = (idx_i_t != idx_k_n) & (idx_k_t != idx_k_n)

    idx_i_t, idx_j_t, idx_k_t, idx_k_n, idx_batch_t = idx_i_t[mask], idx_j_t[mask], idx_k_t[mask], \
                                                      idx_k_n[mask], idx_batch_t[mask]
    # Calculate torsions.
    pos_j0 = pos[idx_k_t] - pos[idx_j_t]
    pos_ji = pos[idx_i_t] - pos[idx_j_t]
    pos_jk = pos[idx_k_n] - pos[idx_j_t]
    plane1 = torch.cross(pos_ji, pos_j0)  # ijk
    plane2 = torch.cross(pos_ji, pos_jk)  # ijk'

    dist_ji = (pos_ji.pow(2).sum(dim=-1)).sqrt()
    a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
    b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji

    torsion1 = torch.atan2(b, a)  # -pi to pi
    torsion1[torsion1 <= 0] += 2 * PI  # 0 to 2pi


    torsiona, argmin_a = scatter_min(torsion1, idx_batch_t, dim=-1)
    # torsionc, argmin_c = scatter_min(torsion3, idx_batch_t, dim=-1)

    plane_s = 0.5 * torch.sin(angle) * dist[idx_kj] * dist[idx_ji]

    q_index = torch.t(q_index)
    pos_q = pos[q_index]
    pos0 = pos_q[:, 0]
    pos1 = pos_q[:, 1]
    pos2 = pos_q[:, 2]
    pos3 = pos_q[:, 3]
    vec1 = pos1 - pos0
    vec2 = pos2 - pos0
    vec3 = pos3 - pos0
    volume = torch.abs(torch.sum(torch.cross(vec1, vec2) * vec3, dim=-1)) / 6

    return dist, angle, torsiona, torsiona, torsiona, plane_s, volume, idx_ji, idx_kj

class emb(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff, envelope_exponent):
        super(emb, self).__init__()
        self.dist_emb = dist_emb(num_radial, cutoff, envelope_exponent)
        self.angle_emb = angle_emb(num_spherical, num_radial, cutoff, envelope_exponent)
        self.torsiona_emb = torsion_emb_3(num_spherical, num_radial, cutoff, envelope_exponent)
        # self.torsionb_emb = torsion_emb(num_spherical, num_radial, cutoff, envelope_exponent)
        # self.torsionc_emb = torsion_emb(num_spherical, num_radial, cutoff, envelope_exponent)
        # self.area_emb = area_emb(num_radial, cutoff, envelope_exponent)
        # self.volume_emb = volume_emb(num_radial, cutoff, envelope_exponent)

        self.reset_parameters()

    def reset_parameters(self):
        self.dist_emb.reset_parameters()

    def forward(self, dist, angle, torsiona, torsionb, torsionc, area, volume, idx_ji, idx_kj, p_idx_a, hidden_channels):
        dist_emb = self.dist_emb(dist)
        angle_emb = self.angle_emb(dist, angle, idx_kj)
        angle_emb_2 = self.angle_emb(dist, angle, idx_ji)
        torsiona_emb = self.torsiona_emb(dist, angle, torsiona, idx_ji, p_idx_a)
        #torsionb_emb = self.torsionb_emb(dist, angle, torsionb, idx_kj)
        #torsionc_emb = self.torsionc_emb(dist, angle, torsionc, idx_kj)
        area_emb = torch.t(area.unsqueeze(0)).repeat(1, hidden_channels)
        volume_emb = torch.t(volume.unsqueeze(0)).repeat(1, hidden_channels)

        return dist_emb, angle_emb, torsiona_emb, torsiona_emb, torsiona_emb, area_emb, volume_emb, angle_emb_2

class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels, act=swish):
        super(ResidualLayer, self).__init__()
        self.act = act
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class feaembed(torch.nn.Module):
    def __init__(self, hidden_channels, use_node_features=True):
        super(feaembed, self).__init__()
        self.use_node_features = use_node_features
        if self.use_node_features:
            self.emb = Embedding(3, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_node_features:
            self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))

    def forward(self, chi):
        if self.use_node_features:
            # x = self.emb(x)
            chi = self.emb(chi)
            # ch = self.emb(ch)
        return chi

class init_v(torch.nn.Module):
    def __init__(self, num_radial, num_spherical, hidden_channels, out_channels, act=swish, use_node_features=True):
        super(init_v, self).__init__()
        self.act = act
        self.use_node_features = use_node_features
        if self.use_node_features:
            self.emb = Embedding(95, hidden_channels)
            # self.emb1 = Embedding(95, hidden_channels)
        else:  # option to use no node features and a learned embedding vector for each node instead
            self.node_embedding = nn.Parameter(torch.empty((hidden_channels,)))
            nn.init.normal_(self.node_embedding)

        self.lin = Linear(3 * hidden_channels, hidden_channels)

        self.lin1 = nn.Linear(hidden_channels, out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_node_features:
            self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        # self.lin_rbf_0.reset_parameters()
        self.lin.reset_parameters()
        self.lin1.reset_parameters()
        # glorot_orthogonal(self.lin_rbf_1.weight, scale=2.0)

    def forward(self, x, chi, b):
        if self.use_node_features:
            x = self.emb(x)
        else:
            x = self.node_embedding[None, :].expand(x.shape[0], -1)
        v = self.act(self.lin(torch.cat([x, chi, b], dim=-1)))
        v1 = self.lin1(v)
        return v, v1


class init_e(torch.nn.Module):
    def __init__(self, num_radial, num_spherical, hidden_channels, act=swish, use_node_features=True):
        super(init_e, self).__init__()
        self.act = act
        self.use_node_features = use_node_features
        if self.use_node_features:
            self.emb = Embedding(95, hidden_channels)

        else:  # option to use no node features and a learned embedding vector for each node instead
            self.node_embedding = nn.Parameter(torch.empty((hidden_channels,)))
            nn.init.normal_(self.node_embedding)
        self.lin_rbf_0 = Linear(num_radial, hidden_channels)

        self.lin = Linear(5 * hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_rbf_0.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, chi, emb, i, j):
        rbf, _, _, _, _, _, _, _ = emb
        if self.use_node_features:
            x = self.emb(x)
        else:
            x = self.node_embedding[None, :].expand(x.shape[0], -1)

        rbf0 = self.act(self.lin_rbf_0(rbf))
        e = self.act(self.lin(torch.cat([x[i], chi[i], x[j], chi[j], rbf0], dim=-1)))

        return e, 1


class init_p(torch.nn.Module):
    def __init__(self, num_radial, num_spherical, hidden_channels, act=swish, use_node_features=True):
        super(init_p, self).__init__()
        self.act = act
        
        self.lin_sbf_0 = Linear(num_spherical * num_radial, hidden_channels)

        self.weight1 = nn.Parameter(torch.tensor([1], dtype=torch.float32))
        self.bias1 = nn.Parameter(torch.tensor([0], dtype=torch.float32))
        self.lin = Linear(4 * hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.weight1 = nn.Parameter(torch.tensor([1], dtype=torch.float32))
        self.bias1 = nn.Parameter(torch.tensor([0], dtype=torch.float32))

    def forward(self, e, emb, idx_ji, idx_kj):
        _, _, _, _, _, area, _, sbf = emb
        e, _ = e
        area = self.act(area * self.weight1 + self.bias1)
        sbf0 = self.act(self.lin_sbf_0(sbf))
        p = self.act(self.lin(torch.cat([e[idx_ji], e[idx_kj], area, sbf0], dim=-1)))
        return p, 1


class init_b(torch.nn.Module):
    def __init__(self, num_radial, num_spherical, hidden_channels, act=swish, use_node_features=True):
        super(init_b, self).__init__()
        self.act = act
        self.weight1 = nn.Parameter(torch.tensor([1], dtype=torch.float32))
        self.bias1 = nn.Parameter(torch.tensor([0], dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        self.weight1 = nn.Parameter(torch.tensor([1], dtype=torch.float32))
        self.bias1 = nn.Parameter(torch.tensor([0], dtype=torch.float32))

    def forward(self, emb):
        _, _, _, _, _, _, volume, _ = emb
        rbf0 = self.act(volume * self.weight1 + self.bias1)
        return rbf0


class update_p(torch.nn.Module):
    def __init__(self, hidden_channels, int_emb_size, basis_emb_size_dist, basis_emb_size_angle, basis_emb_size_torsion,
                 num_spherical, num_radial,
                 num_before_skip, num_after_skip, act=swish):
        super(update_p, self).__init__()
        self.act = act
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size_dist, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size_dist, hidden_channels, bias=False)
        self.lin_sbf1 = nn.Linear(num_spherical * num_radial, basis_emb_size_angle, bias=False)
        self.lin_sbf2 = nn.Linear(basis_emb_size_angle, hidden_channels, bias=False)
        self.lin_t1 = nn.Linear(num_spherical * num_spherical * num_radial, basis_emb_size_torsion, bias=False)
        self.lin_t2 = nn.Linear(basis_emb_size_torsion, int_emb_size, bias=False)
        self.lin_sbf = nn.Linear(num_spherical * num_radial, hidden_channels, bias=False)

        self.lin_p2 = nn.Linear(hidden_channels, hidden_channels)
        self.lin_p4 = nn.Linear(hidden_channels, hidden_channels)

        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)

        self.layers_before_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_before_skip)
        ])
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_after_skip)
        ])

        self.reset_parameters()

    def reset_parameters(self):

        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_t1.weight, scale=2.0)
        glorot_orthogonal(self.lin_t2.weight, scale=2.0)

        glorot_orthogonal(self.lin_p2.weight, scale=2.0)
        self.lin_p2.bias.data.fill_(0)
        glorot_orthogonal(self.lin_p4.weight, scale=2.0)
        self.lin_p4.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

        glorot_orthogonal(self.lin_sbf.weight, scale=2.0)

    def forward(self, p, emb, p_idx_a, p_idx_c):

        rbf0, _, ta, tb, tc, _, _, sbf0 = emb
        x1, _ = p

        x_p2 = self.act(self.lin_p2(x1))
        x_p4 = self.act(self.lin_p4(x1))

        sbf = self.lin_sbf1(sbf0)
        sbf = self.lin_sbf2(sbf)

        x_p4 = x_p4 * sbf

        x_p4 = self.act(self.lin_down(x_p4))


        ta = self.lin_t1(ta)
        ta = self.lin_t2(ta)

        x_p4 = x_p4[p_idx_a] * ta

        x_p4 = self.act(self.lin_up(x_p4))

        p1 = x_p2 + x_p4

        for layer in self.layers_before_skip:
            p1 = layer(p1)
        p1 = self.act(self.lin(p1)) + x1
        for layer in self.layers_after_skip:
            p1 = layer(p1)

        p2 = self.lin_sbf(sbf0) * p1
        return p1, p2

class update_e(torch.nn.Module):
    def __init__(self, out_channels, hidden_channels, int_emb_size, basis_emb_size_dist, basis_emb_size_angle,
                 basis_emb_size_torsion,
                 num_spherical, num_radial,
                 num_before_skip, num_after_skip, act=swish):
        super(update_e, self).__init__()
        self.act = act
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size_dist, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size_dist, hidden_channels, bias=False)
        self.lin_sbf1 = nn.Linear(num_spherical * num_radial, basis_emb_size_angle, bias=False)
        self.lin_sbf2 = nn.Linear(basis_emb_size_angle, int_emb_size, bias=False)
        #self.lin_t1 = nn.Linear(num_spherical * num_spherical * num_radial, basis_emb_size_torsion, bias=False)
        #self.lin_t2 = nn.Linear(basis_emb_size_torsion, int_emb_size, bias=False)
        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)

        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)
        self.lin_connect = nn.Linear(hidden_channels, hidden_channels)

        self.lin_get_up = nn.Linear(hidden_channels, hidden_channels, bias=True)
        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=True)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=True)
        # self.lin1 = nn.Linear(out_channels, hidden_channels,bias=False)
        self.layers_before_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_before_skip)
        ])
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_after_skip)
        ])

        self.lins = torch.nn.ModuleList()

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)
        #glorot_orthogonal(self.lin_t1.weight, scale=2.0)
        #glorot_orthogonal(self.lin_t2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)
        glorot_orthogonal(self.lin_connect.weight, scale=2.0)
        self.lin_connect.bias.data.fill_(0)

        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        self.lin_up.bias.data.fill_(0)
        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        self.lin_down.bias.data.fill_(0)
        glorot_orthogonal(self.lin_get_up.weight, scale=2.0)
        self.lin_get_up.bias.data.fill_(0)


        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)

    def forward(self, p, e, emb, idx_kj, idx_ji):
        rbf0, sbf, ta, tb, tc, _, _, _ = emb
        _, x1 = p
        x_old, _ = e
        x_up = scatter(x1, idx_ji, dim=0, dim_size=x_old.size(0))
        x_up = self.act(self.lin_get_up(x_up))

        x_ji = self.act(self.lin_ji(x_old))
        x_kj = self.act(self.lin_kj(x_old))

        rbf = self.lin_rbf1(rbf0)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        x_kj = self.act(self.lin_down(x_kj))

        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf
        
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x_old.size(0))
        x_kj = self.act(self.lin_up(x_kj))
        e1 = x_ji + x_kj

        e1 = self.act(self.lin_connect(e1)) + x_up

        for layer in self.layers_before_skip:
            e1 = layer(e1)
        e1 = self.act(self.lin(e1)) + x_old

        for layer in self.layers_after_skip:
            e1 = layer(e1)

        e2 = self.lin_rbf(rbf0) * e1

        return e1, e2


class update_v(torch.nn.Module):
    def __init__(self, hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init,
                 num_before_skip, num_after_skip, num_radial, basis_emb_size_dist):
        super(update_v, self).__init__()
        self.act = act
        self.output_init = output_init
        self.lin_up = nn.Linear(hidden_channels, hidden_channels, bias=True)
        self.lin_down = nn.Linear(hidden_channels, hidden_channels, bias=True)
        self.lin_get_up = nn.Linear(hidden_channels, hidden_channels, bias=True)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_output_layers):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.lin_connect = nn.Linear(hidden_channels, hidden_channels)
        self.lin0 = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.lin1 = nn.Linear(hidden_channels, out_channels, bias=False)
        self.lin_i = nn.Linear(hidden_channels, hidden_channels)
        self.lin_j = nn.Linear(hidden_channels, hidden_channels)
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size_dist, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size_dist, hidden_channels, bias=False)
        self.layers_before_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_before_skip)
        ])
        self.layers_after_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_after_skip)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)

        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        self.lin_up.bias.data.fill_(0)
        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        self.lin_down.bias.data.fill_(0)
        glorot_orthogonal(self.lin_get_up.weight, scale=2.0)
        self.lin_get_up.bias.data.fill_(0)

        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        glorot_orthogonal(self.lin_connect.weight, scale=2.0)
        self.lin_connect.bias.data.fill_(0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        glorot_orthogonal(self.lin_i.weight, scale=2.0)
        self.lin_i.bias.data.fill_(0)
        glorot_orthogonal(self.lin_j.weight, scale=2.0)
        self.lin_j.bias.data.fill_(0)
        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

        if self.output_init == 'zeros':
            self.lin0.weight.data.fill_(0)
            self.lin1.weight.data.fill_(0)
        if self.output_init == 'GlorotOrthogonal':
            glorot_orthogonal(self.lin0.weight, scale=2.0)
            glorot_orthogonal(self.lin1.weight, scale=2.0)

    def forward(self, v, emb, e, i, j):
        rbf0, _, _, _, _, _, _, _ = emb
        _, e2 = e
        v_up = scatter(e2, i, dim=0)
        v_up = self.act(self.lin_get_up(v_up))

        v_old = v
        x_i = self.act(self.lin_i(v_old))
        x_j = self.act(self.lin_j(v_old))

        rbf = self.lin_rbf1(rbf0)
        rbf = self.lin_rbf2(rbf)

        x_j = x_j[j] * rbf

        x_j = self.act(self.lin_down(x_j))

        x_j = scatter(x_j, j, dim=0, dim_size=v_old.size(0))
        x_j = self.act(self.lin_up(x_j))

        v2 = x_j + x_i

        v2 = self.act(self.lin_connect(v2)) + v_up

        for layer in self.layers_before_skip:
            v2 = layer(v2)
        v2 = self.act(self.lin(v2)) + v_old
        for layer in self.layers_after_skip:
            v2 = layer(v2)

        v = v2
        v1 = self.lin1(v)

        return v, v1


class update_u(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(update_u, self).__init__()
        self.lin = nn.Linear(hidden_channels, out_channels, bias=False)

    def reset_parameters(self):
        glorot_orthogonal(self.lin.weight, scale=2.0)

    def forward(self, u, v, batch):
        u += scatter(v, batch, dim=0)
        return u

class pgeoNet(torch.nn.Module):

    def __init__(
            self, energy_and_force=False, cutoff=5.0, num_layers=4,
            hidden_channels=128, out_channels=1, int_emb_size=64,
            basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
            num_spherical=7, num_radial=6, envelope_exponent=5,
            num_before_skip=1, num_after_skip=2, num_output_layers=3,
            act=swish, output_init='GlorotOrthogonal', use_node_features=True):
        super(pgeoNet, self).__init__()

        self.cutoff = cutoff
        self.energy_and_force = energy_and_force
        self.hidden_channels = hidden_channels

        self.feaemb = feaembed(hidden_channels, use_node_features=use_node_features)
        self.init_p = init_p(num_radial, num_spherical, hidden_channels, act=swish, use_node_features=use_node_features)
        self.init_e = init_e(num_radial, num_spherical, hidden_channels, act=swish, use_node_features=use_node_features)
        self.init_v = init_v(num_radial, num_spherical, hidden_channels, out_channels, act=swish,
                             use_node_features=use_node_features)
        self.init_b = init_b(num_radial, num_spherical, hidden_channels, act=swish, use_node_features=use_node_features)
        self.init_u = update_u(hidden_channels, out_channels)
        self.emb = emb(num_spherical, num_radial, self.cutoff, envelope_exponent)

        self.update_vs = torch.nn.ModuleList([
            update_v(hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init,
                     num_before_skip, num_after_skip, num_radial, basis_emb_size_dist) for _ in
            range(num_layers)])

        self.update_es = torch.nn.ModuleList([
            update_e(out_channels, hidden_channels, int_emb_size, basis_emb_size_dist, basis_emb_size_angle,
                     basis_emb_size_torsion,
                     num_spherical, num_radial, num_before_skip, num_after_skip, act) for _ in range(num_layers)])

        self.update_ps = torch.nn.ModuleList([
            update_p(hidden_channels, int_emb_size, basis_emb_size_dist, basis_emb_size_angle, basis_emb_size_torsion,
                     num_spherical, num_radial, num_before_skip, num_after_skip, act) for _ in range(num_layers)])

        self.update_us = torch.nn.ModuleList([update_u(hidden_channels, out_channels) for _ in range(num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        self.feaemb.reset_parameters()
        self.init_p.reset_parameters()
        self.init_e.reset_parameters()
        self.init_v.reset_parameters()
        self.init_b.reset_parameters()
        self.emb.reset_parameters()
        for update_p in self.update_ps:
            update_p.reset_parameters()
        for update_e in self.update_es:
            update_e.reset_parameters()
        for update_v in self.update_vs:
            update_v.reset_parameters()

    def forward(self, batch_data):
        z, chi, pos, batch = batch_data.z, batch_data.chi, batch_data.pos, batch_data.batch
        p_idx_a = batch_data.p_idx_a
        edge_index = batch_data.edge_index
        
        vn, en, pn = batch_data.vn, batch_data.en, batch_data.pn
        
        j, i = edge_index
        
        if self.energy_and_force:
            q_index = batch_data.quadra_index
            p_idx_a = compute_index(p_idx_a, en, pn, vn)
            pos.requires_grad_()
            dist, angle, torsiona, torsionb, torsionc, area, volume, idx_ji, idx_kj = compute_geometry(pos, edge_index, q_index)
        else:
            area, volume = batch_data.plane_s, batch_data.v
            dist, angle, torsiona = batch_data.dist, batch_data.angle, batch_data.torsiona
            idx_kj, idx_ji = batch_data.idx_kj, batch_data.idx_ji
            idx_ji, idx_kj, p_idx_a = compute_index_2(idx_ji, idx_kj, p_idx_a, en, pn, vn)
        
        
        emb = self.emb(dist, angle, torsiona, torsiona, torsiona, area, volume, idx_ji, idx_kj, p_idx_a, self.hidden_channels)
        
        # Initialize edge, node, graph features
        chi = self.feaemb(chi)
        b = self.init_b(emb)
        v, v1 = self.init_v(z, chi, b)
        e = self.init_e(z, chi, emb, i, j)
        p = self.init_p(e, emb, idx_ji, idx_kj)
        u = self.init_u(torch.zeros_like(scatter(v1, batch, dim=0)), v1, batch)  # scatter(v, batch, dim=0)
        for update_p, update_e, update_v, update_u in zip(self.update_ps, self.update_es, self.update_vs,
                                                          self.update_us):
            p = update_p(p, emb, p_idx_a, p_idx_a)
            e = update_e(p, e, emb, idx_kj, idx_ji)
            v, v1 = update_v(v, emb, e, i, j)
            u = update_u(u, v1, batch)
        return u
