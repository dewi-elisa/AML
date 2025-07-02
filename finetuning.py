from core.models.model_factory import Model
import argparse
import torch
from core.data_provider import datasets_factory
from core.models.model_factory import Model
from core.utils import preprocess
import core.trainer as trainer
from run import schedule_sampling, reserve_schedule_sampling_exp
import sys

parser = argparse.ArgumentParser(description='PyTorch video prediction model - PredRNN')

# training/test
parser.add_argument('--is_training', type=int, default=1)
parser.add_argument('--layer', type=str, default=None)
parser.add_argument('--device', type=str, default='cpu:0')

# data
parser.add_argument('--dataset_name', type=str, default='mnist')
parser.add_argument('--train_data_paths', type=str, default='data/moving-mnist-example/moving-mnist-train.npz')
parser.add_argument('--valid_data_paths', type=str, default='data/moving-mnist-example/moving-mnist-valid.npz')
parser.add_argument('--save_dir', type=str, default='checkpoints/mnist_predrnn')
parser.add_argument('--gen_frm_dir', type=str, default='results/mnist_predrnn')
parser.add_argument('--input_length', type=int, default=10)
parser.add_argument('--total_length', type=int, default=20)
parser.add_argument('--img_width', type=int, default=64)
parser.add_argument('--img_channel', type=int, default=1)
parser.add_argument('--noise', type=str, default=None)
parser.add_argument('--noise_ratio', type=float, default=0.1)
parser.add_argument('--noise_size', type=int, default=10)

# model
parser.add_argument('--model_name', type=str, default='predrnn')
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('--num_hidden', type=str, default='64,64,64,64')
parser.add_argument('--filter_size', type=int, default=5)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--patch_size', type=int, default=4)
parser.add_argument('--layer_norm', type=int, default=1)
parser.add_argument('--decouple_beta', type=float, default=0.1)

# reverse scheduled sampling
parser.add_argument('--reverse_scheduled_sampling', type=int, default=0)
parser.add_argument('--r_sampling_step_1', type=float, default=25000)
parser.add_argument('--r_sampling_step_2', type=int, default=50000)
parser.add_argument('--r_exp_alpha', type=int, default=5000)
# scheduled sampling
parser.add_argument('--scheduled_sampling', type=int, default=1)
parser.add_argument('--sampling_stop_iter', type=int, default=50000)
parser.add_argument('--sampling_start_value', type=float, default=1.0)
parser.add_argument('--sampling_changing_rate', type=float, default=0.00002)

# optimization
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--reverse_input', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--max_iterations', type=int, default=80000)
parser.add_argument('--display_interval', type=int, default=100)
parser.add_argument('--test_interval', type=int, default=5000)
parser.add_argument('--snapshot_interval', type=int, default=5000)
parser.add_argument('--num_save_samples', type=int, default=10)
parser.add_argument('--n_gpu', type=int, default=1)

# visualization of memory decoupling
parser.add_argument('--visual', type=int, default=0)
parser.add_argument('--visual_path', type=str, default='./decoupling_visual')

# action-based predrnn
parser.add_argument('--injection_action', type=str, default='concat')
parser.add_argument('--conv_on_input', type=int, default=0, help='conv on input')
parser.add_argument('--res_on_conv', type=int, default=0, help='res on conv')
parser.add_argument('--num_action_ch', type=int, default=4, help='num action ch')

args = parser.parse_args()
print(args)

if args.layer is None:
    print('No layer given to fine-tune.')
    sys.exit()

# load data
train_input_handle, test_input_handle = datasets_factory.data_provider(
    args.dataset_name, args.train_data_paths, args.valid_data_paths, args.batch_size, args.img_width,
    seq_length=args.total_length, injection_action=args.injection_action, noise=args.noise,
    noise_ratio=args.noise_ratio, noise_size=args.noise_size, input_length=args.input_length, is_training=True)

# load model
model = Model(args)
model.load(args.pretrained_model)

# print model
print(model.network)

# print model parameters
print('model parameters:')
for name, param in model.network.named_parameters():
    if param.requires_grad:
        print(name)#)

# only choose parameters from one layer
for name, p in model.network.named_parameters():
    if args.layer in name:          # keep learning
        p.requires_grad = True
    else:                            # freeze
        p.requires_grad = False

trainable = [p for p in model.network.parameters() if p.requires_grad]

if trainable == []:
    print('No existing layer found to fine-tune')
    sys.exit()
else:
    print(f'Updating the following parameters: {trainable}')

# train the layer
model.optimizer = torch.optim.Adam(trainable, lr=args.lr)

eta = args.sampling_start_value

for itr in range(1, args.max_iterations + 1):
    if train_input_handle.no_batch_left():
        train_input_handle.begin(do_shuffle=True)
    ims = train_input_handle.get_batch()
    ims = preprocess.reshape_patch(ims, args.patch_size)

    if args.reverse_scheduled_sampling == 1:
        real_input_flag = reserve_schedule_sampling_exp(itr)
    else:
        eta, real_input_flag = schedule_sampling(eta, itr)

    trainer.train(model, ims, real_input_flag, args, itr)

    if itr % args.snapshot_interval == 0:
        model.save(itr)

    if itr % args.test_interval == 0:
        trainer.test(model, test_input_handle, args, itr)

    train_input_handle.next()

# save model
model.save(itr)