import torch
from qm9ev import QM9new
from hagonet.HagoNet import pgeoNet
from dig.threedgraph.method import run
import argparse
from evaluation.eval import ThreeDEvaluator

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

parser = argparse.ArgumentParser(description='MD17 Example')
parser.add_argument('--property', type=str, default='', metavar='N',
                    help='Property to predict.')
args = parser.parse_args()

dataset = QM9new(root='data/')
target = args.property
sl_dir = "./results/" + args.property

dataset.data.y = dataset.data[target]

split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=110000, valid_size=10000, seed=42)

train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]

print('train, validaion, test:', len(train_dataset), len(valid_dataset), len(test_dataset))

model = pgeoNet(energy_and_force=False, cutoff=5.0, num_layers=4,
        hidden_channels=128, out_channels=1, int_emb_size=64,
        basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
        num_spherical=3, num_radial=6, envelope_exponent=5,
        num_before_skip=1, num_after_skip=2, num_output_layers=3, use_node_features=True, output_init='zeros'
        )


loss_func = torch.nn.L1Loss()
evaluation = ThreeDEvaluator()
run3d = run()

run3d.run(device, train_dataset, valid_dataset, test_dataset, model, loss_func, evaluation, epochs=1000, batch_size=32, vt_batch_size=64, lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=150, save_dir=sl_dir, log_dir=sl_dir, weight_decay = 0.0000025)