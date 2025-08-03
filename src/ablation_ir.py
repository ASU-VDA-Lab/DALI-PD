import numpy as np
import torch.nn as nn
import os
from tqdm.auto import tqdm
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import torch
from unet import TestUnet
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--load_weight_path", type=str, default="")
    parser.add_argument("--save_weight_path", type=str, default="../models/")
    parser.add_argument("--CircuitNet_test_path", type=str, default="../CircuitNet_N28_test/")
    parser.add_argument("--CircuitNet_train_path", type=str, default="../CircuitNet_N28_train/")
    parser.add_argument("--synthetic_path", type=str, default="../synthetic_benchmark/")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-6)
    parser.add_argument("--gradient_accum_steps", type=int, default=64)
    parser.add_argument("--steps", type=int, default=125)
    parser.add_argument("--train_with_synthetic", action="store_true")
    parser.set_defaults(train_with_synthetic=False)
    parser.add_argument("--num_of_heatmap", type=int, default=2000)
    args = parser.parse_args()

    # Prepare the arguments
    device = args.device
    load_weight_path = args.load_weight_path
    save_weight_path = args.save_weight_path
    CircuitNet_test_path = args.CircuitNet_test_path
    CircuitNet_train_path = args.CircuitNet_train_path
    synthetic_path = args.synthetic_path
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    gradient_accum_steps = args.gradient_accum_steps
    steps = args.steps
    train_with_synthetic = args.train_with_synthetic
    num_of_heatmap = args.num_of_heatmap
    train_step = steps * gradient_accum_steps

    # Load the test data
    test_dataset = list()
    test_data_dict = dict()
    for file_name in os.listdir(CircuitNet_test_path + "cell_density/"):
        data_id = file_name.split(".npy")[0]
        test_data_cell_density = np.load(CircuitNet_test_path + "cell_density/" + file_name)
        test_data_power_all = np.load(CircuitNet_test_path + "power_all/" + file_name)
        test_data_power_sca = np.load(CircuitNet_test_path + "power_sca/" + file_name)
        test_data_IR_drop = np.load(CircuitNet_test_path + "IR_drop/" + file_name)
        
        height = test_data_cell_density.shape[0]
        width = test_data_cell_density.shape[1]
        if height % 8 != 0:
            test_data_cell_density = np.pad(test_data_cell_density, ((0, 8 - height % 8), (0, 0), (0, 0)), mode="constant")
            test_data_power_all = np.pad(test_data_power_all, ((0, 8 - height % 8), (0, 0), (0, 0)), mode="constant")
            test_data_power_sca = np.pad(test_data_power_sca, ((0, 8 - height % 8), (0, 0), (0, 0)), mode="constant")
            test_data_IR_drop = np.pad(test_data_IR_drop, ((0, 8 - height % 8), (0, 0), (0, 0)), mode="constant")
        if width % 8 != 0:
            test_data_cell_density = np.pad(test_data_cell_density, ((0, 0), (0, 8 - width % 8), (0, 0)), mode="constant")
            test_data_power_all = np.pad(test_data_power_all, ((0, 0), (0, 8 - width % 8), (0, 0)), mode="constant")
            test_data_power_sca = np.pad(test_data_power_sca, ((0, 0), (0, 8 - width % 8), (0, 0)), mode="constant")
            test_data_IR_drop = np.pad(test_data_IR_drop, ((0, 0), (0, 8 - width % 8), (0, 0)), mode="constant")
        test_data_dict[data_id] = np.dstack((test_data_cell_density, test_data_power_all, test_data_power_sca, test_data_IR_drop)).transpose(2, 0, 1)[None, ...]
        test_dataset.append(test_data_dict[data_id])
    

    # Load the train data
    if train_with_synthetic:
        train_dataset = list()
        train_data_dict = dict()
        for file_name in os.listdir(synthetic_path + "cell_density/")[:num_of_heatmap]:
            data_id = file_name.split(".npy")[0]
            train_data_cell_density = np.load(synthetic_path + "cell_density/" + file_name)
            train_data_power_all = np.load(synthetic_path + "power_all/" + file_name)
            train_data_power_sca = np.load(synthetic_path + "power_sca/" + file_name)
            train_data_IR_drop = np.load(synthetic_path + "IR_drop/" + file_name)
            
            height = train_data_cell_density.shape[0]
            width = train_data_cell_density.shape[1]
            if height % 8 != 0:
                train_data_cell_density = np.pad(train_data_cell_density, ((0, 8 - height % 8), (0, 0), (0, 0)), mode="constant")
                train_data_power_all = np.pad(train_data_power_all, ((0, 8 - height % 8), (0, 0), (0, 0)), mode="constant")
                train_data_power_sca = np.pad(train_data_power_sca, ((0, 8 - height % 8), (0, 0), (0, 0)), mode="constant")
                train_data_IR_drop = np.pad(train_data_IR_drop, ((0, 8 - height % 8), (0, 0), (0, 0)), mode="constant")
            if width % 8 != 0:
                train_data_cell_density = np.pad(train_data_cell_density, ((0, 0), (0, 8 - width % 8), (0, 0)), mode="constant")
                train_data_power_all = np.pad(train_data_power_all, ((0, 0), (0, 8 - width % 8), (0, 0)), mode="constant")
                train_data_power_sca = np.pad(train_data_power_sca, ((0, 0), (0, 8 - width % 8), (0, 0)), mode="constant")
                train_data_IR_drop = np.pad(train_data_IR_drop, ((0, 0), (0, 8 - width % 8), (0, 0)), mode="constant")
            train_data_dict[data_id] = np.dstack((train_data_cell_density, train_data_power_all, train_data_power_sca, train_data_IR_drop)).transpose(2, 0, 1)[None, ...]
            train_dataset.append(train_data_dict[data_id]) 
    else:
        train_dataset = list()
        train_data_dict = dict()
        for file_name in os.listdir(CircuitNet_train_path + "cell_density/")[:num_of_heatmap]:
            data_id = file_name.split(".npy")[0]
            train_data_cell_density = np.load(CircuitNet_train_path + "cell_density/" + file_name)
            train_data_power_all = np.load(CircuitNet_train_path + "power_all/" + file_name)
            train_data_power_sca = np.load(CircuitNet_train_path + "power_sca/" + file_name)
            train_data_IR_drop = np.load(CircuitNet_train_path + "IR_drop/" + file_name)
            
            height = train_data_cell_density.shape[0]
            width = train_data_cell_density.shape[1]
            if height % 8 != 0:
                train_data_cell_density = np.pad(train_data_cell_density, ((0, 8 - height % 8), (0, 0), (0, 0)), mode="constant")
                train_data_power_all = np.pad(train_data_power_all, ((0, 8 - height % 8), (0, 0), (0, 0)), mode="constant")
                train_data_power_sca = np.pad(train_data_power_sca, ((0, 8 - height % 8), (0, 0), (0, 0)), mode="constant")
                train_data_IR_drop = np.pad(train_data_IR_drop, ((0, 8 - height % 8), (0, 0), (0, 0)), mode="constant")
            if width % 8 != 0:
                train_data_cell_density = np.pad(train_data_cell_density, ((0, 0), (0, 8 - width % 8), (0, 0)), mode="constant")
                train_data_power_all = np.pad(train_data_power_all, ((0, 0), (0, 8 - width % 8), (0, 0)), mode="constant")
                train_data_power_sca = np.pad(train_data_power_sca, ((0, 0), (0, 8 - width % 8), (0, 0)), mode="constant")
                train_data_IR_drop = np.pad(train_data_IR_drop, ((0, 0), (0, 8 - width % 8), (0, 0)), mode="constant")
            train_data_dict[data_id] = np.dstack((train_data_cell_density, train_data_power_all, train_data_power_sca, train_data_IR_drop)).transpose(2, 0, 1)[None, ...]
            train_dataset.append(train_data_dict[data_id])         

    np.random.shuffle(train_dataset)    
    progress_bar = tqdm(total=train_step, desc="Training", ncols=90)
    batch_loss = list()

    # Load the model
    unet = TestUnet(input_channels=2, output_channels=1)
    unet.to(device)
    unet.train()
    optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)
    if load_weight_path != "":
        unet.load_state_dict(torch.load(load_weight_path))
    
    length = len(train_dataset)
    optimizer.zero_grad()
    final_test_loss = list()
    for i in range(train_step):
        batch = train_dataset[int(i%length)][:, :3, :, :]
        target = train_dataset[int(i%length)][:, [3], :, :]
        batch = torch.from_numpy(batch).float().to(device)
        target = torch.from_numpy(target).float().to(device)
        # forward pass
        output = unet(batch)
        # compute the loss
        loss = nn.L1Loss()(output, target) / gradient_accum_steps
        # include l2 weight decay
        l2_loss = 0
        for param in unet.parameters():
            l2_loss += torch.norm(param, 2)
        loss += l2_loss * weight_decay / gradient_accum_steps
        batch_loss.append(loss.item())
        # backpropagate the loss
        loss.backward()
        # Gradient accumulation
        if ((i + 1) % gradient_accum_steps == 0 or i == train_step - 1) and i != 0:
            optimizer.step()
            optimizer.zero_grad()
            batch_loss = list()
            # test the unet
            unet.eval()
            with torch.no_grad():
                test_loss = list()
                for j in range(len(test_dataset)):
                    test_batch = test_dataset[j][:, :3, :, :]
                    test_target = test_dataset[j][:, [3], :, :]
                    test_batch = torch.from_numpy(test_batch).float().to(device)
                    test_target = torch.from_numpy(test_target).float().to(device)
                    test_output = unet(test_batch)
                    test_loss.append(nn.L1Loss()(test_output, test_target).item())
                final_test_loss.append(np.mean(test_loss))
            unet.train()
        # use tqdm to print the loss
        progress_bar.update(1)
        train_loss = loss.item() * gradient_accum_steps
        test_loss = final_test_loss[-1] if len(final_test_loss) > 0 else train_loss
        progress_bar.set_postfix(loss=train_loss, lr=optimizer.param_groups[0]['lr'], test=test_loss)
        
    torch.save(unet.state_dict(), save_weight_path)