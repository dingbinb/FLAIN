import os
import torch
import pickle
import random
import numpy as np
from math import floor
from collections import defaultdict
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class DatasetSplit(Dataset):

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = idxs
        self.targets = torch.Tensor([self.dataset.targets[idx] for idx in idxs])

    def classes(self):
        return torch.unique(self.targets)    

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        inp, target = self.dataset[self.idxs[item]]
        return inp, target


def iid_distribute_data(dataset, args):
    num_classes = args.num_classes
    class_per_agent = args.class_per_agent
    if args.num_agents == 1:
        return {0:range(len(dataset))}

    def chunker_list(seq, size):
        return [seq[i::size] for i in range(size)]

    labels_sorted = dataset.targets.sort()
    class_by_labels = list(zip(labels_sorted.values.tolist(), labels_sorted.indices.tolist()))
    labels_dict = defaultdict(list)

    for k, v in class_by_labels:
        labels_dict[k].append(v)

    shard_size = len(dataset) // (args.num_agents * class_per_agent)
    slice_size = (len(dataset) // num_classes) // shard_size
    for k, v in labels_dict.items():
        labels_dict[k] = chunker_list(v, slice_size)

    dict_users = defaultdict(list)
    for user_idx in range(args.num_agents):
        class_ctr = 0
        for j in range(0, num_classes):
            if class_ctr == class_per_agent:
                    break
            elif len(labels_dict[j]) > 0:
                dict_users[user_idx] += labels_dict[j][0]
                del labels_dict[j][0]
                class_ctr += 1
    return dict_users

def generate_proportions(total_count, num_agents, alpha=1.0):
    proportions = np.random.dirichlet([alpha] * num_agents)
    scaled_proportions = [int(round(p * total_count)) for p in proportions]

    diff = sum(scaled_proportions) - total_count
    if diff > 0:
        for i in range(diff):
            idx = random.randint(0, num_agents - 1)
            scaled_proportions[idx] -= 1
    elif diff < 0:
        for i in range(-diff):
            idx = random.randint(0, num_agents - 1)
            scaled_proportions[idx] += 1

    scaled_proportions = [max(0, p) for p in scaled_proportions]
    random.shuffle(scaled_proportions)

    return scaled_proportions


def non_iid_distribute_data(dataset, args):
    # non_iid_split_file_path (Non-IID)
    if args.data == 'mnist':
        file_path = os.path.join(f'///Non-IID/MNIST_Non_IID')
    elif args.data == 'emnist':
        file_path = os.path.join(f'/Non-IID/')
    elif args.data == 'fmnist':
        file_path = os.path.join(f'/Non-IID/')
    elif args.data == 'cifar10':
        file_path = os.path.join(f'/Non-IID/')

    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            dict_users = pickle.load(f)
        return dict_users

    num_classes = args.num_classes
    class_per_agent = args.class_per_agent
    alpha = args.alpha
    if args.num_agents == 1:
        return {0: range(len(dataset))}

    def chunker_list(seq, chunk_sizes):
        chunks = []
        start_idx = 0
        for size in chunk_sizes:
            chunks.append(seq[start_idx: start_idx + size])
            start_idx += size
        return chunks

    labels_sorted = dataset.targets.sort()
    class_by_labels = list(zip(labels_sorted.values.tolist(), labels_sorted.indices.tolist()))
    labels_dict = defaultdict(list)

    for k, v in class_by_labels:
        labels_dict[k].append(v)

    for k, v in labels_dict.items():
        slice_size = generate_proportions(len(labels_dict[k]), args.num_agents, alpha)
        labels_dict[k] = chunker_list(v, slice_size)

    dict_users = defaultdict(list)
    for user_idx in range(args.num_agents):
        class_ctr = 0
        for j in range(0, num_classes):
            if class_ctr == class_per_agent:
                break
            elif len(labels_dict[j]) > 0:
                dict_users[user_idx] += labels_dict[j][0]
                del labels_dict[j][0]
                class_ctr += 1

    for user_idx in range(args.num_agents):
        if len(dict_users[user_idx]) == 0:
            for other_user_idx in range(args.num_agents):
                if len(dict_users[other_user_idx]) > 0:
                    dict_users[user_idx].append(dict_users[other_user_idx].pop())
                    break

    with open(file_path, 'wb') as f:
        pickle.dump(dict_users, f)

    return dict_users



def get_datasets(data):
    train_dataset, test_dataset = None, None
    data_dir = '///data'

    if data == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform = transform)
        test_dataset  = datasets.MNIST(data_dir, train=False,download=True, transform = transform)

    elif data == 'fmnist':
        transform =  transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2860], std=[0.3530])])
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)

    elif data == 'cifar10':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=False, transform=transform_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=False, transform=transform_test)
        train_dataset.targets, test_dataset.targets = torch.LongTensor(train_dataset.targets), torch.LongTensor(test_dataset.targets)

    elif data == 'emnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.13066, 0.30810) ])
        train_dataset = datasets.EMNIST(data_dir, split='byclass', train=True,  transform=transform, download=False)
        test_dataset  = datasets.EMNIST(data_dir, split='byclass', train=False, transform=transform, download=False)
    return train_dataset, test_dataset


def get_loss_n_accuracy(model, criterion, data_loader, args):
    model.eval()
    num_classes = args.num_classes
    total_loss, correctly_labeled_samples = 0, 0
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        for _, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device=args.device, non_blocking=True),\
                labels.to(device=args.device, non_blocking=True)
            outputs = model(inputs)
            avg_minibatch_loss = criterion(outputs, labels)
            total_loss += avg_minibatch_loss.item()*outputs.shape[0]
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correctly_labeled_samples += torch.sum(torch.eq(pred_labels, labels)).item()

            for t, p in zip(labels.view(-1), pred_labels.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = correctly_labeled_samples / len(data_loader.dataset)
    per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
    return avg_loss, (accuracy, per_class_accuracy)


def poison_dataset(dataset, args, data_idxs=None, poison_all=False, agent_idx=-1):
    all_idxs = (dataset.targets == args.base_class).nonzero().flatten().tolist()
    if data_idxs != None:
        all_idxs = list(set(all_idxs).intersection(data_idxs))
        
    poison_frac = 1 if poison_all else args.poison_frac    
    poison_idxs = random.sample(all_idxs, floor(poison_frac*len(all_idxs)))

    for idx in poison_idxs:
        clean_img = dataset.data[idx]
        bd_img = add_trigger(clean_img, args.data, trigger_type=args.trigger_type, agent_idx=agent_idx)
        dataset.data[idx] = torch.tensor(bd_img)
        dataset.targets[idx] = args.target_class
    return


def add_trigger(x, dataset='cifar10', trigger_type='DBA', agent_idx=-1):
    x = np.array(x.squeeze())

    if dataset == 'cifar10':
        if trigger_type == 'DBA':
            start_idx = 5
            size = 6
            if agent_idx == -1:
                for d in range(0, 3):  
                    for i in range(start_idx, start_idx+size+1):
                        x[i, start_idx][d] = 10
                for d in range(0, 3):  
                    for i in range(start_idx-size//2, start_idx+size//2 + 1):
                        x[start_idx+size//2, i][d] = 10
            else:

                if agent_idx % 4 == 0:
                    for d in range(0, 3):  
                        for i in range(start_idx, start_idx+(size//2)+1):
                            x[i, start_idx][d] = 10

                elif agent_idx % 4 == 1:
                    for d in range(0, 3):  
                        for i in range(start_idx+(size//2)+1, start_idx+size+1):
                            x[i, start_idx][d] = 10

                elif agent_idx % 4 == 2:
                    for d in range(0, 3):  
                        for i in range(start_idx-size//2, start_idx+size//4 + 1):
                            x[start_idx+size//2, i][d] = 10

                elif agent_idx % 4 == 3:
                    for d in range(0, 3):  
                        for i in range(start_idx-size//4+1, start_idx+size//2 + 1):
                            x[start_idx+size//2, i][d] = 10

        elif trigger_type == 'CBA':
            for d in range(0,3):
              for i in range(26, 30):
                for j in range(26, 30):
                    x[i,j][d] = 10


    elif dataset in ['mnist', 'fmnist', 'emnist']:

        if trigger_type == 'CBA':
            for i in range(23, 28):
                for j in range(23, 28):
                    x[i,j] = 255

        elif trigger_type == 'DBA':
            if agent_idx == -1:
                for i in range(23, 28):
                    x[i, 25] = 255

                for j in range(23, 28):
                    x[25, j] = 255

            else:
                if agent_idx % 2 == 0:
                    for i in range(23, 28):
                        x[i, 25] = 255

                elif agent_idx % 2 == 1:
                    for j in range(23, 28):
                        x[25, j] = 255

    return x

def print_exp_details(args):
    print('=============================================')
    print(f'Dataset: {args.data} | Poison frac: {args.poison_frac}')
    print(f'Base class: {args.base_class} | Target class: {args.target_class}')
    print(f'Train rounds: {args.train_rounds} | Batch size: {args.bs}')
    print(f'Number agents: {args.num_agents} | Backdoor frac: {args.backdoor_frac}')
    print(f'Trigger: {args.trigger_type} | Data distribution: {args.data_distribution}')
    print(f'Number classes: {args.num_classes} | defense: {args.defense}')
    print('============================================')
    