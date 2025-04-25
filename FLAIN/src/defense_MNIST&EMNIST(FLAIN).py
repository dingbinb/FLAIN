import torch
import utils
import copy
import torch.nn as nn
from options import args_parser
from torch.utils.data import DataLoader, Subset
torch.manual_seed(3407)

class CNN_EMNIST(nn.Module):
    def __init__(self):
        super(CNN_EMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 62)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.drop1(x)
        if len(class_fc1_inputs) < args.num_classes:
            class_fc1_inputs.append(x)
        x = self.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        return x

#---------------------------------------------------------------------------------

args = args_parser()
device = args.device
criterion = nn.CrossEntropyLoss().to(device)

init_glo_model = CNN_EMNIST().to(device)
_, val_dataset = utils.get_datasets(args.data)

model_path = 'E:\FLAINmodels\mnist.pth'
parameters_dict = torch.load(model_path, map_location = device)
fin_glo_model= CNN_EMNIST().to(device)
fin_glo_model.load_state_dict(parameters_dict)

update = fin_glo_model.fc1.weight.data - init_glo_model.fc1.weight.data
glo_fc1_norm0 = torch.norm(fin_glo_model.fc1.weight.data)

#---------------------------------------------------------------------------------

val_data = copy.deepcopy(val_dataset)
val_loader = DataLoader(val_data, batch_size = args.bs, shuffle=False, num_workers=args.num_workers,
                        pin_memory=False)

idxs = (val_dataset.targets == args.base_class).nonzero().flatten().tolist()
poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
utils.poison_dataset(poisoned_val_set.dataset, args, idxs, poison_all=True)
poisoned_val_loader = DataLoader(poisoned_val_set, batch_size=len(poisoned_val_set), shuffle=False, num_workers=args.num_workers,
                                 pin_memory=False)

#---------------------------------------------------------------------------------

aux_class_loaders = {}
aux_idxs = []
aux_samples = 0
for i in range(args.num_classes):
    idxs = (val_dataset.targets == i).nonzero().flatten().tolist()[:args.aux_samples]
    class_val_set = Subset(copy.deepcopy(val_dataset), idxs)
    class_val_loader = DataLoader(class_val_set, batch_size=len(class_val_set), shuffle=False, num_workers=args.num_workers, pin_memory=False)
    aux_class_loaders[i] = class_val_loader
    aux_idxs.extend(idxs)
    aux_samples += args.aux_samples

aux_val_set = Subset(copy.deepcopy(val_dataset), aux_idxs)
aux_loader = DataLoader(aux_val_set, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=False)

#---------------------------------------------------------------------------------

class_fc1_inputs = []
for i in range(args.num_classes):
    utils.get_loss_n_accuracy(fin_glo_model, criterion, aux_class_loaders[i], args)

#---------------------------------------------------------------------------------

val_loss0, (val_acc0, val_per_class_acc0) = utils.get_loss_n_accuracy(fin_glo_model, criterion, val_loader, args)
poison_loss0, (poison_acc0, poison_class_acc0) = utils.get_loss_n_accuracy(fin_glo_model, criterion, poisoned_val_loader, args)

# pre-defense accuracy
_, (aux_val_acc0, _) = utils.get_loss_n_accuracy(fin_glo_model, criterion, aux_loader, args)

summed_inputs = []
for i in range(args.num_classes):
    summed_inputs.append(torch.sum(class_fc1_inputs[i], dim=0))
stacked_inputs = torch.stack(summed_inputs, dim=0)
fc1_inputs = torch.sum(stacked_inputs, dim=0) / aux_samples
sorted_values, sorted_indices = torch.sort(fc1_inputs)


#---------------------------------------------------------------------------------
def flip_updates(fin_glo_model, init_glo_model, update, sorted_indices, fc1_inputs, criterion, aux_loader, args, aux_val_acc0, acc_thres = 0.05, k_init= 0.001, k_cap= 1, k_step= 0.001):

    k = k_init
    while k <= k_cap:
        x_1, x_2 = [], []
        for i in sorted_indices:
            if 0 <= fc1_inputs[i] <= k:
                x_1.append(i)
            else:
                x_2.append(i)

        print(len(x_1))
        for index in x_1:
            fin_glo_model.fc1.weight.data[:,index] = init_glo_model.fc1.weight.data[:,index] - update[:,index]
        for index in x_2:
            fin_glo_model.fc1.weight.data[:,index] = init_glo_model.fc1.weight.data[:,index] + update[:,index]

        # post-defense accuracy
        _, (aux_val_acc1, _) = utils.get_loss_n_accuracy(fin_glo_model, criterion, aux_loader, args)
        acc_diff = aux_val_acc0 - aux_val_acc1
        print(f"aux_val_acc0: {aux_val_acc0:.3f} | aux_val_acc1: {aux_val_acc1:.3f} | k: {k:.3f} | acc_diff: {acc_diff:.3f}")

        if acc_thres <= acc_diff:
            glo_fc1_norm1 = torch.norm(fin_glo_model.fc1.weight.data)
            scaling_factor = glo_fc1_norm0 / glo_fc1_norm1
            fin_glo_model.fc1.weight.data *= scaling_factor
            cur_k = k
            return cur_k
        k += k_step

cur_k = flip_updates(fin_glo_model, init_glo_model, update, sorted_indices, fc1_inputs, criterion, aux_loader, args, aux_val_acc0, acc_thres = 0)
print(f"b_value: {cur_k:.3f}")

#---------------------------------------------------------------------------------

val_loss1, (val_acc1, val_per_class_acc1) = utils.get_loss_n_accuracy(fin_glo_model, criterion, val_loader, args)
poison_loss1, (poison_acc1, poison_class_acc1) = utils.get_loss_n_accuracy(fin_glo_model, criterion, poisoned_val_loader, args)

#---------------------------------------------------------------------------------

print()
print("--------------------------------------------------------------------------------------------------------------")
print(f'| Before Defense： Val_Loss/Val_Acc: {val_loss0:.3f} / {val_acc0:.3f} |')
print(f'| Before Defense： Val_Per_Class_Acc: {val_per_class_acc0} ')

print(f'| After Defense：  Val_Loss/Val_Acc: {val_loss1:.3f} / {val_acc1:.3f} |')
print(f'| After Defense：  Val_Per_Class_Acc: {val_per_class_acc1} ')

print("-----------------------------------------------------------")

print(f'| Before Defense： Poison Loss/Poison Acc: {poison_loss0:.3f} / {poison_acc0:.3f}  |')
print(f'| Before Defense： Poison_Per_Class_Acc: {poison_class_acc0} ')

print(f'| After Defense：  Poison Loss/Poison Acc: {poison_loss1:.3f} / {poison_acc1:.3f} |')
print(f'| After Defense：  Poison_Per_Class_Acc: {poison_class_acc1}')


