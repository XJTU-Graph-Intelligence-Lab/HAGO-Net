import os.path as osp
import numpy as np
from tqdm import tqdm
import torch
from sklearn.utils import shuffle

from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data, DataLoader
import os.path as osp
import numpy as np
from tqdm import tqdm
import torch
from sklearn.utils import shuffle
import torch
from torch_scatter import scatter, scatter_min
from torch_sparse import SparseTensor
from math import pi as PI
from torch_geometric.nn.acts import swish
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn import radius_graph
from math import sqrt
from dig.threedgraph.dataset import QM93D

from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data, DataLoader
import networkx as nx
import scipy
import sys

import numpy as np
import copy
from scipy.spatial.transform import Rotation as R
import scipy

cutoff = 5.0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_chirality(z, pos, z_center, pos_center):
    # phase one
    # n = 4 # four atoms
    pos = pos.cpu().numpy()
    z = z.cpu().numpy()
    atm_num_list = z
    atm_pst_matrix = pos
    pos_center = pos_center.cpu().numpy()
    z_center = z_center.cpu().numpy()
    
    n = 4
    
    epsilon_dist = 2.0
    epsilon = 0.001

    dist = np.sqrt(np.sum((pos - pos_center) ** 2, axis=1))
    
    flag = np.all(dist < epsilon_dist)
    if not flag:
        return 0
    
    # print(dist)
    dist_of_dist = scipy.spatial.distance.pdist(np.expand_dims(dist,axis=0).T, 'cityblock')
    # print(dist_of_dist)

    flag = np.all(dist_of_dist > epsilon)
    
    if not flag:
        return 0

    mol = dict.fromkeys(range(n))
    for idx in range(n):
        mol[idx] = {'atm_num': atm_num_list[idx], 'atm_pst': atm_pst_matrix[idx]}
    mol_center = {'atm_num': z_center, 'atm_pst': pos_center}
    v_1 = mol[0]['atm_pst'] - mol_center['atm_pst']
    v_2 = mol[1]['atm_pst'] - mol_center['atm_pst']
    v_ref = mol[2]['atm_pst'] - mol_center['atm_pst']

    v_right_hand = np.cross(v_1, v_2)

    syn = np.sign(np.inner(v_ref, v_right_hand))
    return syn

# idx2 -> idx1, idx3 -> idx2
def get_torsion_idx(arg, idx1, idx2, idx3, adj_t, adj_e):
    idx_v1 = idx1[arg]
    idx_v2 = idx2[arg]
    idx_v3 = idx3[arg]

    idx_e1 = adj_t[idx_v1, idx_v2]
    idx_e2 = adj_t[idx_v2, idx_v3]

    idx_p = adj_e[idx_e1, idx_e2]
    
    return idx_p


def xyz_to_dat(pos, edge_index, num_nodes, use_torsion=False):
    """
    Compute the diatance, angle, and torsion from geometric information.
    Args:
        pos: Geometric information for every node in the graph.
        edgee_index: Edge index of the graph.
        number_nodes: Number of nodes in the graph.
        use_torsion: If set to :obj:`True`, will return distance, angle and torsion, otherwise only return distance and angle (also retrun some useful index). (default: :obj:`False`)
    """
    j, i = edge_index  # j->i

    # j1,i1 = edge_index1 # j1-> i1
    # k, j = edge_index

    # Calculate distances. # number of edges
    dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
    # dist1 = (pos[i1] - pos[j1]).pow(2).sum(dim=-1).sqrt()

    value = torch.arange(j.size(0), device=j.device)
    # value1 = torch.arange(k.size(0), device=k.device)
    adj_t = SparseTensor(row=i, col=j, value=value, sparse_sizes=(num_nodes, num_nodes))
    # adj_t1 = SparseTensor(row=j, col=k, value=value1, sparse_sizes=(num_nodes, num_nodes))
    # adj_t1 = SparseTensor(row=j, col=i, value=value, sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[j]

    adj_t_dense = adj_t.to_dense()
    # adj_t1_row = adj_t1[k]
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

    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]
    num_edges = idx_kj.shape[0]
    value = torch.arange(num_edges, device=j.device)
    # adj_e = SparseTensor(row=torch.cat([idx_ji, idx_kj],dim=-1), col=torch.cat([idx_kj, idx_ji], dim=-1), value=torch.cat([value,value], dim=-1), sparse_sizes=(2*num_edges, 2*num_edges))
    adj_e = SparseTensor(row=idx_ji, col=idx_kj, value=value, sparse_sizes=(num_edges, num_edges))
    adj_e_dense = adj_e.to_dense()

    # Calculate angles. 0 to pi
    pos_ji = pos[idx_i] - pos[idx_j]
    pos_jk = pos[idx_k] - pos[idx_j]
    a = (pos_ji * pos_jk).sum(dim=-1)  # cos_angle * |pos_ji| * |pos_jk|
    b = torch.cross(pos_ji, pos_jk).norm(dim=-1)  # sin_angle * |pos_ji| * |pos_jk|
    angle = torch.atan2(b, a)

    idx_batch = torch.arange(len(idx_i), device=device)
    idx_k_n = adj_t[idx_j].storage.col()  # j邻居
    # idx_j_n = adj_t[idx_i].storage.col()  # i邻居
    # idx_h_n = adj_t[idx_k].storage.col()
    
    repeat = num_triplets
    num_triplets_t = num_triplets.repeat_interleave(repeat)[mask]
    idx_i_t = idx_i.repeat_interleave(num_triplets_t)
    idx_j_t = idx_j.repeat_interleave(num_triplets_t)
    idx_k_t = idx_k.repeat_interleave(num_triplets_t)
    
    # idx_h_t = idx_h.repeat_interleave(num_triplets_t)
    # idx_j_n = idx_j.re
    idx_batch_t = idx_batch.repeat_interleave(num_triplets_t)
    if num_nodes < 4:
        mask = (idx_i_t != idx_k_n)
    else:
        mask = (idx_i_t != idx_k_n) & (idx_k_t != idx_k_n)
    # mask = (idx_i_t != idx_k_n)
    # print(mask)
    # mask = idx_j_t != idx_j_n
    # print (mask)
    '''idx_i_t, idx_j_t , idx_k_t, idx_k_n,idx_j_n, idx_batch_t = idx_i_t[mask], idx_j_t[mask], idx_k_t[mask],\
                                                               idx_k_n[mask],idx_j_n[mask], idx_batch_t[mask]'''

    idx_i_t, idx_j_t, idx_k_t, idx_k_n, idx_batch_t = idx_i_t[mask], idx_j_t[mask], idx_k_t[mask], \
                                                      idx_k_n[mask], idx_batch_t[mask]
    # Calculate torsions.
    if use_torsion:
        pos_j0 = pos[idx_k_t] - pos[idx_j_t]
        pos_ji = pos[idx_i_t] - pos[idx_j_t]
        pos_jk = pos[idx_k_n] - pos[idx_j_t]
        plane1 = torch.cross(pos_ji, pos_j0)  # ijk
        plane2 = torch.cross(pos_ji, pos_jk)  # ijk'
        dist_ji = (pos_ji.pow(2).sum(dim=-1)).sqrt()
        a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji

        torsion1 = torch.atan2(b, a)
        torsion1[torsion1 <= 0] += 2 * PI
        torsiona, argmin_a = scatter_min(torsion1, idx_batch_t, dim=-1)
     
        p_idx_a = get_torsion_idx(argmin_a, idx_i_t, idx_j_t, idx_k_n, adj_t_dense, adj_e_dense)

        return dist, angle, torsiona, torsiona, torsiona, i, j, adj_t_row, idx_kj, idx_ji, p_idx_a, p_idx_a

    else:
        return dist, angle, i, j, idx_kj, idx_ji

class QM9new(InMemoryDataset):
    r"""
        A `Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/index.html>`_ data interface for :obj:`QM9` dataset 
        which is from `"Quantum chemistry structures and properties of 134 kilo molecules" <https://www.nature.com/articles/sdata201422>`_ paper.
        It connsists of about 130,000 equilibrium molecules with 12 regression targets: 
        :obj:`mu`, :obj:`alpha`, :obj:`homo`, :obj:`lumo`, :obj:`gap`, :obj:`r2`, :obj:`zpve`, :obj:`U0`, :obj:`U`, :obj:`H`, :obj:`G`, :obj:`Cv`.
        Each molecule includes complete spatial information for the single low energy conformation of the atoms in the molecule.
        .. note::
            We used the processed data in `DimeNet <https://github.com/klicperajo/dimenet/tree/master/data>`_, wihch includes spatial information and type for each atom.
            You can also use `QM9 in Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/qm9.html#QM9>`_.
    
        Args:
            root (string): the dataset folder will be located at root/qm9.
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
                (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
            pre_filter (callable, optional): A function that takes in an
                :obj:`torch_geometric.data.Data` object and returns a boolean
                value, indicating whether the data object should be included in the
                final dataset. (default: :obj:`None`)
        Example:
        --------
        >>> dataset = QM93D()
        >>> target = 'mu'
        >>> dataset.data.y = dataset.data[target]
        >>> split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=110000, valid_size=10000, seed=42)
        >>> train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
        >>> train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        >>> data = next(iter(train_loader))
        >>> data
        Batch(Cv=[32], G=[32], H=[32], U=[32], U0=[32], alpha=[32], batch=[579], gap=[32], homo=[32], lumo=[32], mu=[32], pos=[579, 3], ptr=[33], r2=[32], y=[32], z=[579], zpve=[32])
        Where the attributes of the output data indicates:
    
        * :obj:`z`: The atom type.
        * :obj:`pos`: The 3D position for atoms.
        * :obj:`y`: The target property for the graph (molecule).
        * :obj:`batch`: The assignment vector which maps each node to its respective graph identifier and can help reconstructe single graphs
    """
    def __init__(self, root = 'dataset/qm9/', transform = None, pre_transform = None, pre_filter = None):

        self.url = 'https://github.com/klicperajo/dimenet/raw/master/data/qm9_eV.npz'
        self.folder = osp.join(root)

        super(QM9new, self).__init__(self.folder, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'qm9_eV.npz'

    @property
    def processed_file_names(self):
        return 'qm9_pyg.pt'

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        
        data = np.load(osp.join(self.raw_dir, self.raw_file_names))

        R = data['R']
        Z = data['Z']
        N= data['N']
        split = np.cumsum(N)
        R_qm9 = np.split(R, split)
        Z_qm9 = np.split(Z,split)
        target = {}
        for name in ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve','U0', 'U', 'H', 'G', 'Cv']:
            target[name] = np.expand_dims(data[name],axis=-1)
        # y = np.expand_dims([data[name] for name in ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve','U0', 'U', 'H', 'G', 'Cv']], axis=-1)

        data_list = []
        for i in tqdm(range(len(N))):
            R_i = torch.tensor(R_qm9[i],dtype=torch.float32)
            z_i = torch.tensor(Z_qm9[i],dtype=torch.int64)
            y_i = [torch.tensor(target[name][i],dtype=torch.float32) for name in ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve','U0', 'U', 'H', 'G', 'Cv']]
            R_i = R_i.to(device)
            # Eidx_i = Eidx_i.to(device)
            edge_index = radius_graph(R_i, r=cutoff)
            edge_index = edge_index.to(device)
            num_nodes = z_i.size(0)
            dist, angle, torsiona, torsionb, torsionc, _, _, _, idx_kj, idx_ji, p_idx_a, p_idx_c = xyz_to_dat(R_i,
                                                                                                                  edge_index,
                                                                                                                  num_nodes,
                                                                                                                  use_torsion=True)
            dist = dist.to(torch.device('cpu'))
            angle = angle.to(torch.device('cpu'))
            torsiona = torsiona.to(torch.device('cpu'))
            torsionb = torsionb.to(torch.device('cpu'))
            torsionc = torsionc.to(torch.device('cpu'))
            idx_kj = idx_kj.to(torch.device('cpu'))
            idx_ji = idx_ji.to(torch.device('cpu'))
            R_i = R_i.to(torch.device('cpu'))
            edge_index = edge_index.to(torch.device('cpu'))
            p_idx_a = p_idx_a.to(torch.device('cpu'))
            p_idx_c = p_idx_c.to(torch.device('cpu'))


            w1 = torch.norm(R_i[:, None] - R_i, dim=2, p=2)
            w1 = w1.cpu().numpy()
            w1[w1 == 0] = 1000
            # print(w1)

            idx = np.argsort(w1)
            q_list = idx[:, 0:4]
            q_index = torch.tensor(q_list, dtype=torch.int64)
            chi = torch.zeros(num_nodes, dtype=torch.int64)

            carbon_index = torch.nonzero(z_i == 6)

            for count, carbon_idx in enumerate(carbon_index):
                q = q_index[carbon_idx]
                z_q = z_i[q].squeeze(0)
                pos = R_i[q].squeeze(0)
                pos_center = R_i[carbon_idx]
                z_center = z_i[carbon_idx]
                chi[carbon_idx] = torch.tensor(compute_chirality(z_q, pos, z_center, pos_center), dtype=torch.int64)
            chi = chi + 1
            
            plane_s = 0.5 * torch.sin(angle) * dist[idx_kj] * dist[idx_ji]
            
            volume_list = []
            for q in q_list:
                pos0 = R_i[q[0]]
                pos1 = R_i[q[1]]
                pos2 = R_i[q[2]]
                pos3 = R_i[q[3]]
                vec1 = pos1 - pos0
                vec2 = pos2 - pos0
                vec3 = pos3 - pos0
                volume = torch.abs(torch.dot(torch.cross(vec1, vec2), vec3)) / 6
                volume_list.append(volume)

            v = torch.tensor(volume_list, dtype=torch.float32)
            
            vn = torch.tensor(z_i.size(0), dtype=torch.int64)
            
            en = torch.tensor(edge_index.size(1), dtype=torch.int64)

            pn = torch.tensor(p_idx_a.size(0), dtype=torch.int64)
            
            data = Data(pos=R_i, z=z_i, chi=chi, y=y_i[0], mu=y_i[0], alpha=y_i[1], homo=y_i[2],
                         lumo=y_i[3], gap=y_i[4],
                         r2=y_i[5], zpve=y_i[6], U0=y_i[7], U=y_i[8], H=y_i[9], G=y_i[10], Cv=y_i[11],
                         edge_index=edge_index, dist=dist, angle=angle, torsiona=torsiona, idx_kj=idx_kj, idx_ji=idx_ji, plane_s=plane_s, v=v, p_idx_a=p_idx_a, vn=vn, en=en, pn=pn)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train':train_idx, 'valid':val_idx, 'test':test_idx}
        return split_dict

if __name__ == '__main__':
    dataset = QM9new()
    print(dataset)
    print(dataset.data.z.shape)
    print(dataset.data.pos.shape)
    target = 'mu'
    dataset.data.y = dataset.data[target]
    print(dataset.data.y.shape)
    print(dataset.data.y)
    print(dataset.data.mu)
    split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=110000, valid_size=10000, seed=42)
    print(split_idx)
    print(dataset[split_idx['train']])
    train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    data = next(iter(train_loader))
    print(data)