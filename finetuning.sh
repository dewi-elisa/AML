# python -u run.py \
#     --is_training 0 \
#     --device cuda \
#     --dataset_name mnist \
#     --train_data_paths C:/Users/twank/Documents/Dewi/predrnn-pytorch-master/predrnn-pytorch-master/moving-mnist-example/moving-mnist-train.npz \
#     --valid_data_paths C:/Users/twank/Documents/Dewi/predrnn-pytorch-master/predrnn-pytorch-master/moving-mnist-example/moving-mnist-test.npz \
#     --save_dir checkpoints/mnist_predrnn_v2 \
#     --gen_frm_dir results_finetuning/test \
#     --model_name predrnn_v2 \
#     --reverse_input 1 \
#     --img_width 64 \
#     --img_channel 1 \
#     --input_length 10 \
#     --total_length 20 \
#     --num_hidden 128,128,128,128 \
#     --filter_size 5 \
#     --stride 1 \
#     --patch_size 4 \
#     --layer_norm 0 \
#     --decouple_beta 0.1 \
#     --reverse_scheduled_sampling 1 \
#     --r_sampling_step_1 25000 \
#     --r_sampling_step_2 50000 \
#     --r_exp_alpha 2500 \
#     --lr 0.0001 \
#     --batch_size 8 \
#     --max_iterations 80000 \
#     --display_interval 100 \
#     --test_interval 5000 \
#     --snapshot_interval 5000 \
#     --pretrained_model ./checkpointss/mnist_predrnn_v2/mnist_model.ckpt

# python -u finetuning.py \
#     --is_training 1 \
#     --device cuda \
#     --dataset_name mnist \
#     --train_data_paths C:/Users/twank/Documents/Dewi/predrnn-pytorch-master/predrnn-pytorch-master/moving-mnist-example/moving-mnist-train.npz \
#     --valid_data_paths C:/Users/twank/Documents/Dewi/predrnn-pytorch-master/predrnn-pytorch-master/moving-mnist-example/moving-mnist-test.npz \
#     --save_dir checkpoints_finetuning/mnist_predrnn_v2 \
#     --gen_frm_dir results_finetuning/random \
#     --model_name predrnn_v2 \
#     --reverse_input 1 \
#     --img_width 64 \
#     --img_channel 1 \
#     --noise random \
#     --parameters conv_last.weight \
#     --input_length 10 \
#     --total_length 20 \
#     --num_hidden 128,128,128,128 \
#     --filter_size 5 \
#     --stride 1 \
#     --patch_size 4 \
#     --layer_norm 0 \
#     --decouple_beta 0.1 \
#     --reverse_scheduled_sampling 1 \
#     --r_sampling_step_1 25000 \
#     --r_sampling_step_2 50000 \
#     --r_exp_alpha 2500 \
#     --lr 0.0001 \
#     --batch_size 8 \
#     --max_iterations 8000 \
#     --display_interval 100 \
#     --test_interval 10000 \
#     --snapshot_interval 10000 \
#     --pretrained_model ./checkpointss/mnist_predrnn_v2/mnist_model.ckpt

# python -u finetuning.py \
#     --is_training 1 \
#     --device cuda \
#     --dataset_name mnist \
#     --train_data_paths C:/Users/twank/Documents/Dewi/predrnn-pytorch-master/predrnn-pytorch-master/moving-mnist-example/moving-mnist-train.npz \
#     --valid_data_paths C:/Users/twank/Documents/Dewi/predrnn-pytorch-master/predrnn-pytorch-master/moving-mnist-example/moving-mnist-test.npz \
#     --save_dir checkpoints_finetuning/mnist_predrnn_v2 \
#     --gen_frm_dir results_finetuning/blocks_22 \
#     --model_name predrnn_v2 \
#     --reverse_input 1 \
#     --img_width 64 \
#     --img_channel 1 \
#     --noise blocks \
#     --noise_size 2 \
#     --parameters conv_last.weight \
#     --input_length 10 \
#     --total_length 20 \
#     --num_hidden 128,128,128,128 \
#     --filter_size 5 \
#     --stride 1 \
#     --patch_size 4 \
#     --layer_norm 0 \
#     --decouple_beta 0.1 \
#     --reverse_scheduled_sampling 1 \
#     --r_sampling_step_1 25000 \
#     --r_sampling_step_2 50000 \
#     --r_exp_alpha 2500 \
#     --lr 0.0001 \
#     --batch_size 8 \
#     --max_iterations 8000 \
#     --display_interval 100 \
#     --test_interval 10000 \
#     --snapshot_interval 10000 \
#     --pretrained_model ./checkpoints_finetuning/mnist_predrnn_v2/model.ckpt-8000-random

# python -u finetuning.py \
#     --is_training 1 \
#     --device cuda \
#     --dataset_name mnist \
#     --train_data_paths C:/Users/twank/Documents/Dewi/predrnn-pytorch-master/predrnn-pytorch-master/moving-mnist-example/moving-mnist-train.npz \
#     --valid_data_paths C:/Users/twank/Documents/Dewi/predrnn-pytorch-master/predrnn-pytorch-master/moving-mnist-example/moving-mnist-test.npz \
#     --save_dir checkpoints_finetuning/mnist_predrnn_v2 \
#     --gen_frm_dir results_finetuning/rows \
#     --model_name predrnn_v2 \
#     --reverse_input 1 \
#     --img_width 64 \
#     --img_channel 1 \
#     --noise rows \
#     --parameters conv_last.weight \
#     --input_length 10 \
#     --total_length 20 \
#     --num_hidden 128,128,128,128 \
#     --filter_size 5 \
#     --stride 1 \
#     --patch_size 4 \
#     --layer_norm 0 \
#     --decouple_beta 0.1 \
#     --reverse_scheduled_sampling 1 \
#     --r_sampling_step_1 25000 \
#     --r_sampling_step_2 50000 \
#     --r_exp_alpha 2500 \
#     --lr 0.0001 \
#     --batch_size 8 \
#     --max_iterations 8000 \
#     --display_interval 100 \
#     --test_interval 10000 \
#     --snapshot_interval 10000 \
#     --pretrained_model ./checkpoints_finetuning/mnist_predrnn_v2/model.ckpt-8000-blocks2

# python -u finetuning.py \
#     --is_training 1 \
#     --device cuda \
#     --dataset_name mnist \
#     --train_data_paths C:/Users/twank/Documents/Dewi/predrnn-pytorch-master/predrnn-pytorch-master/moving-mnist-example/moving-mnist-train.npz \
#     --valid_data_paths C:/Users/twank/Documents/Dewi/predrnn-pytorch-master/predrnn-pytorch-master/moving-mnist-example/moving-mnist-test.npz \
#     --save_dir checkpoints_finetuning/mnist_predrnn_v2 \
#     --gen_frm_dir results_finetuning/blocks_55 \
#     --model_name predrnn_v2 \
#     --reverse_input 1 \
#     --img_width 64 \
#     --img_channel 1 \
#     --noise blocks \
#     --noise_size 5 \
#     --parameters conv_last.weight \
#     --input_length 10 \
#     --total_length 20 \
#     --num_hidden 128,128,128,128 \
#     --filter_size 5 \
#     --stride 1 \
#     --patch_size 4 \
#     --layer_norm 0 \
#     --decouple_beta 0.1 \
#     --reverse_scheduled_sampling 1 \
#     --r_sampling_step_1 25000 \
#     --r_sampling_step_2 50000 \
#     --r_exp_alpha 2500 \
#     --lr 0.0001 \
#     --batch_size 8 \
#     --max_iterations 8000 \
#     --display_interval 100 \
#     --test_interval 10000 \
#     --snapshot_interval 10000 \
#     --pretrained_model ./checkpoints_finetuning/mnist_predrnn_v2/model.ckpt-8000-rows

python -u finetuning.py \
    --is_training 1 \
    --device cuda \
    --dataset_name mnist \
    --train_data_paths C:/Users/twank/Documents/Dewi/predrnn-pytorch-master/predrnn-pytorch-master/moving-mnist-example/moving-mnist-train.npz \
    --valid_data_paths C:/Users/twank/Documents/Dewi/predrnn-pytorch-master/predrnn-pytorch-master/moving-mnist-example/moving-mnist-test.npz \
    --save_dir checkpoints_finetuning/mnist_predrnn_v2 \
    --gen_frm_dir results_finetuning/blocks_1010 \
    --model_name predrnn_v2 \
    --reverse_input 1 \
    --img_width 64 \
    --img_channel 1 \
    --noise blocks \
    --noise_size 10 \
    --parameters conv_last.weight \
    --input_length 10 \
    --total_length 20 \
    --num_hidden 128,128,128,128 \
    --filter_size 5 \
    --stride 1 \
    --patch_size 4 \
    --layer_norm 0 \
    --decouple_beta 0.1 \
    --reverse_scheduled_sampling 1 \
    --r_sampling_step_1 25000 \
    --r_sampling_step_2 50000 \
    --r_exp_alpha 2500 \
    --lr 0.0001 \
    --batch_size 8 \
    --max_iterations 8000 \
    --display_interval 100 \
    --test_interval 10000 \
    --snapshot_interval 10000 \
    --pretrained_model ./checkpoints_finetuning/mnist_predrnn_v2/model.ckpt-8000-blocks5

python -u run.py \
    --is_training 0 \
    --device cuda \
    --dataset_name mnist \
    --train_data_paths C:/Users/twank/Documents/Dewi/predrnn-pytorch-master/predrnn-pytorch-master/moving-mnist-example/moving-mnist-train.npz \
    --valid_data_paths C:/Users/twank/Documents/Dewi/predrnn-pytorch-master/predrnn-pytorch-master/moving-mnist-example/moving-mnist-test.npz \
    --save_dir checkpoints/mnist_predrnn_v2 \
    --gen_frm_dir results_finetuning/test_end \
    --model_name predrnn_v2 \
    --reverse_input 1 \
    --img_width 64 \
    --img_channel 1 \
    --input_length 10 \
    --total_length 20 \
    --num_hidden 128,128,128,128 \
    --filter_size 5 \
    --stride 1 \
    --patch_size 4 \
    --layer_norm 0 \
    --decouple_beta 0.1 \
    --reverse_scheduled_sampling 1 \
    --r_sampling_step_1 25000 \
    --r_sampling_step_2 50000 \
    --r_exp_alpha 2500 \
    --lr 0.0001 \
    --batch_size 8 \
    --max_iterations 80000 \
    --display_interval 100 \
    --test_interval 5000 \
    --snapshot_interval 5000 \
    --pretrained_model ./checkpoints_finetuning/mnist_predrnn_v2/model.ckpt-8000-blocks10

python -u run.py \
    --is_training 0 \
    --device cuda \
    --dataset_name mnist \
    --train_data_paths C:/Users/twank/Documents/Dewi/predrnn-pytorch-master/predrnn-pytorch-master/moving-mnist-example/moving-mnist-train.npz \
    --valid_data_paths C:/Users/twank/Documents/Dewi/predrnn-pytorch-master/predrnn-pytorch-master/moving-mnist-example/moving-mnist-test.npz \
    --save_dir checkpoints/mnist_predrnn_v2 \
    --gen_frm_dir results_finetuning/random_end \
    --model_name predrnn_v2 \
    --reverse_input 1 \
    --img_width 64 \
    --img_channel 1 \
    --noise random \
    --input_length 10 \
    --total_length 20 \
    --num_hidden 128,128,128,128 \
    --filter_size 5 \
    --stride 1 \
    --patch_size 4 \
    --layer_norm 0 \
    --decouple_beta 0.1 \
    --reverse_scheduled_sampling 1 \
    --r_sampling_step_1 25000 \
    --r_sampling_step_2 50000 \
    --r_exp_alpha 2500 \
    --lr 0.0001 \
    --batch_size 8 \
    --max_iterations 80000 \
    --display_interval 100 \
    --test_interval 5000 \
    --snapshot_interval 5000 \
    --pretrained_model ./checkpoints_finetuning/mnist_predrnn_v2/model.ckpt-8000-blocks10

python -u run.py \
    --is_training 0 \
    --device cuda \
    --dataset_name mnist \
    --train_data_paths C:/Users/twank/Documents/Dewi/predrnn-pytorch-master/predrnn-pytorch-master/moving-mnist-example/moving-mnist-train.npz \
    --valid_data_paths C:/Users/twank/Documents/Dewi/predrnn-pytorch-master/predrnn-pytorch-master/moving-mnist-example/moving-mnist-test.npz \
    --save_dir checkpoints/mnist_predrnn_v2 \
    --gen_frm_dir results_finetuning/rows_end \
    --model_name predrnn_v2 \
    --reverse_input 1 \
    --img_width 64 \
    --img_channel 1 \
    --noise rows \
    --input_length 10 \
    --total_length 20 \
    --num_hidden 128,128,128,128 \
    --filter_size 5 \
    --stride 1 \
    --patch_size 4 \
    --layer_norm 0 \
    --decouple_beta 0.1 \
    --reverse_scheduled_sampling 1 \
    --r_sampling_step_1 25000 \
    --r_sampling_step_2 50000 \
    --r_exp_alpha 2500 \
    --lr 0.0001 \
    --batch_size 8 \
    --max_iterations 80000 \
    --display_interval 100 \
    --test_interval 5000 \
    --snapshot_interval 5000 \
    --pretrained_model ./checkpoints_finetuning/mnist_predrnn_v2/model.ckpt-8000-blocks10

python -u run.py \
    --is_training 0 \
    --device cuda \
    --dataset_name mnist \
    --train_data_paths C:/Users/twank/Documents/Dewi/predrnn-pytorch-master/predrnn-pytorch-master/moving-mnist-example/moving-mnist-train.npz \
    --valid_data_paths C:/Users/twank/Documents/Dewi/predrnn-pytorch-master/predrnn-pytorch-master/moving-mnist-example/moving-mnist-test.npz \
    --save_dir checkpoints/mnist_predrnn_v2 \
    --gen_frm_dir results_finetuning/blocks_22_end \
    --model_name predrnn_v2 \
    --reverse_input 1 \
    --img_width 64 \
    --img_channel 1 \
    --noise blocks \
    --noise_size 2 \
    --input_length 10 \
    --total_length 20 \
    --num_hidden 128,128,128,128 \
    --filter_size 5 \
    --stride 1 \
    --patch_size 4 \
    --layer_norm 0 \
    --decouple_beta 0.1 \
    --reverse_scheduled_sampling 1 \
    --r_sampling_step_1 25000 \
    --r_sampling_step_2 50000 \
    --r_exp_alpha 2500 \
    --lr 0.0001 \
    --batch_size 8 \
    --max_iterations 80000 \
    --display_interval 100 \
    --test_interval 5000 \
    --snapshot_interval 5000 \
    --pretrained_model ./checkpoints_finetuning/mnist_predrnn_v2/model.ckpt-8000-blocks10

python -u run.py \
    --is_training 0 \
    --device cuda \
    --dataset_name mnist \
    --train_data_paths C:/Users/twank/Documents/Dewi/predrnn-pytorch-master/predrnn-pytorch-master/moving-mnist-example/moving-mnist-train.npz \
    --valid_data_paths C:/Users/twank/Documents/Dewi/predrnn-pytorch-master/predrnn-pytorch-master/moving-mnist-example/moving-mnist-test.npz \
    --save_dir checkpoints/mnist_predrnn_v2 \
    --gen_frm_dir results_finetuning/blocks_55_end \
    --model_name predrnn_v2 \
    --reverse_input 1 \
    --img_width 64 \
    --img_channel 1 \
    --noise blocks \
    --noise_size 5 \
    --input_length 10 \
    --total_length 20 \
    --num_hidden 128,128,128,128 \
    --filter_size 5 \
    --stride 1 \
    --patch_size 4 \
    --layer_norm 0 \
    --decouple_beta 0.1 \
    --reverse_scheduled_sampling 1 \
    --r_sampling_step_1 25000 \
    --r_sampling_step_2 50000 \
    --r_exp_alpha 2500 \
    --lr 0.0001 \
    --batch_size 8 \
    --max_iterations 80000 \
    --display_interval 100 \
    --test_interval 5000 \
    --snapshot_interval 5000 \
    --pretrained_model ./checkpoints_finetuning/mnist_predrnn_v2/model.ckpt-8000-blocks10

python -u run.py \
    --is_training 0 \
    --device cuda \
    --dataset_name mnist \
    --train_data_paths C:/Users/twank/Documents/Dewi/predrnn-pytorch-master/predrnn-pytorch-master/moving-mnist-example/moving-mnist-train.npz \
    --valid_data_paths C:/Users/twank/Documents/Dewi/predrnn-pytorch-master/predrnn-pytorch-master/moving-mnist-example/moving-mnist-test.npz \
    --save_dir checkpoints/mnist_predrnn_v2 \
    --gen_frm_dir results_finetuning/blocks_1010_end \
    --model_name predrnn_v2 \
    --reverse_input 1 \
    --img_width 64 \
    --img_channel 1 \
    --noise blocks \
    --noise_size 10 \
    --input_length 10 \
    --total_length 20 \
    --num_hidden 128,128,128,128 \
    --filter_size 5 \
    --stride 1 \
    --patch_size 4 \
    --layer_norm 0 \
    --decouple_beta 0.1 \
    --reverse_scheduled_sampling 1 \
    --r_sampling_step_1 25000 \
    --r_sampling_step_2 50000 \
    --r_exp_alpha 2500 \
    --lr 0.0001 \
    --batch_size 8 \
    --max_iterations 80000 \
    --display_interval 100 \
    --test_interval 5000 \
    --snapshot_interval 5000 \
    --pretrained_model ./checkpoints_finetuning/mnist_predrnn_v2/model.ckpt-8000-blocks10