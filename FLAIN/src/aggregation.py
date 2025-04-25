import torch
import utils
import random
import copy
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
from scipy.spatial.distance import euclidean
from torch.nn.utils import vector_to_parameters, parameters_to_vector


class MyDST(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class Aggregation():
    def __init__(self, agent_data_sizes, n_params, poisoned_val_loader, args, writer):
        self.agent_data_sizes = agent_data_sizes
        self.writer = writer
        self.args = args
        self.server_lr = args.server_lr
        self.n_params = n_params
        self.cur_round = 0
        self.poisoned_val_loader = poisoned_val_loader

    def aggregate_updates(self, global_model, agent_updates_dict, agent_cur_parameters_dict,model_updates,args):
        lr_vector = torch.Tensor([self.server_lr] * self.n_params).to(self.args.device)

        if self.args.defense == 'fedavg':
            aggregated_updates = self.fedavg(agent_updates_dict)

        elif self.args.defense == 'krum':
            aggregated_updates = self.krum(agent_updates_dict, 10)

        elif self.args.defense == 'median':
            aggregated_updates = self.median(agent_updates_dict)

        elif self.args.defense == 'rlr':
            lr_vector = self.rlr(agent_updates_dict, rlr_threshold=6)
            aggregated_updates = self.fedavg(agent_updates_dict)

        elif self.args.defense =='fltrust':
            aggregated_updates = self.fltrust(global_model, agent_updates_dict, aux_samples=10)

        elif self.args.defense =='multi_metrics':
            aggregated_updates = self.multi_metrics(global_model, agent_cur_parameters_dict, agent_updates_dict, p=0.4)

        cur_global_params = parameters_to_vector(global_model.parameters())
        new_global_params = (cur_global_params + lr_vector * aggregated_updates).float()
        vector_to_parameters(new_global_params, global_model.parameters())

        # model_save_path (FLAINmodels)
        if args.data == 'mnist' or args.data == 'emnist':
           torch.save(global_model.state_dict(),
                           '///FLAINmodels/MNIST.path')

        elif args.data == 'fmnist':
            torch.save(global_model.state_dict(),
                       '/FLAINmodels/')

        elif args.data == 'cifar10':

                torch.save(global_model.state_dict(),
                       '/FLAINmodels/')
        return

    def relu(self, x):
        return torch.relu(torch.tensor(x))

    def server_train(self, global_model, criterion, aux_loader):
        initial_global_model_params = parameters_to_vector(global_model.parameters()).detach()
        global_model.train()
        optimizer = torch.optim.Adam(global_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        for _ in range(self.args.local_ep):
            for _, (inputs, labels) in enumerate(aux_loader):
                optimizer.zero_grad()
                inputs, labels = inputs.to(device=self.args.device, non_blocking=True), \
                    labels.to(device=self.args.device, non_blocking=True)
                outputs = global_model(inputs)
                minibatch_loss = criterion(outputs, labels)
                minibatch_loss.backward()
                optimizer.step()

        with torch.no_grad():
            update = parameters_to_vector(global_model.parameters()).double() - initial_global_model_params
            return update


    def fedavg(self, agent_updates_dict):
        sm_updates, total_data = 0, 0
        for id, update in agent_updates_dict.items():
            n_agent_data = self.agent_data_sizes[id]
            sm_updates += n_agent_data * update
            total_data += n_agent_data
        return sm_updates / total_data

    def krum(self, agent_updates_dict, f):
        agent_ids = list(agent_updates_dict.keys())
        updates = np.array([agent_updates_dict[agent_id].cpu().numpy() for agent_id in agent_ids])
        n_clients = len(updates)
        scores = []
        for i in range(n_clients):
            distances = []
            for j in range(n_clients):
                if i != j:
                    dist = euclidean(updates[i], updates[j])
                    distances.append(dist)
            distances.sort()
            trimmed_distances = distances[:n_clients - f - 2]
            score = sum(trimmed_distances)
            scores.append((score, i))
        scores.sort()
        best_client_index = scores[0][1]
        best_agent_id = agent_ids[best_client_index]
        aggregated_updates = agent_updates_dict[best_agent_id]
        return aggregated_updates

    # median
    def median(self, agent_updates_dict):
        agent_ids = list(agent_updates_dict.keys())
        updates = np.array([agent_updates_dict[agent_id].cpu().numpy() for agent_id in agent_ids])
        updates = updates.transpose()
        median_update = np.median(updates, axis=1)
        median_update = torch.tensor(median_update).to(self.args.device)
        return median_update

    # rlr
    def rlr(self, agent_updates_dict, rlr_threshold):
        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]
        sm_of_signs = torch.abs(sum(agent_updates_sign))
        sm_of_signs[sm_of_signs < rlr_threshold] = -self.server_lr
        sm_of_signs[sm_of_signs >= rlr_threshold] = self.server_lr
        return sm_of_signs.to(self.args.device)

    # fltrust
    def fltrust(self, global_model, agent_updates_dict, aux_samples):

        criterion = nn.CrossEntropyLoss().to(self.args.device)
        aux_idxs = []
        _, val_dataset = utils.get_datasets(self.args.data)
        for i in range(self.args.num_classes):
            idxs = (val_dataset.targets == i).nonzero().flatten().tolist()
            random_idxs = random.sample(idxs, min(len(idxs), aux_samples))
            aux_idxs.extend(random_idxs)
        aux_val_set = Subset(copy.deepcopy(val_dataset), aux_idxs)
        aux_loader = DataLoader(aux_val_set, batch_size=len(aux_val_set), shuffle=False,
                                pin_memory=False)
        reference_update = self.server_train(global_model, criterion, aux_loader)
        reference_update_norm = torch.norm(reference_update)

        total_score = 0
        total_update = 0

        for agent_id, update in agent_updates_dict.items():
            agent_update_norm = torch.norm(agent_updates_dict[agent_id])

            if agent_update_norm != 0:
                scale_factor = reference_update_norm / agent_update_norm
                update = agent_updates_dict[agent_id] * scale_factor
                sim_cosine = torch.nn.functional.cosine_similarity(reference_update.unsqueeze(0), update.unsqueeze(0))
                trust_score = self.relu(sim_cosine.detach().cpu().numpy())
                score_tensor = trust_score.clone().detach().to(self.args.device)
                total_score += score_tensor
                weighted_update = score_tensor * update
                total_update += weighted_update
        return total_update / total_score

    # multi_metrics
    def multi_metrics(self, global_model, agent_cur_parameters_dict, agent_updates_dict, p):
            cos_dis = []
            manhattan_dis = []
            euclidean_dis = []

            # Get global model parameters and flatten them
            global_model_params = []
            for param in global_model.parameters():
                global_model_params.append(param.detach().cpu().numpy().flatten())
            global_model_flat = np.concatenate(global_model_params)

            # For each agent's current model parameters (assumed already flattened)
            for agent_id, agent_model in agent_cur_parameters_dict.items():
                # Since agent_model is already a Tensor, no need to call parameters()
                agent_model_flat = agent_model.detach().cpu().numpy().flatten()

                # Compute Cosine distance
                cosine_distance = float(
                    (1 - np.dot(global_model_flat, agent_model_flat) / (
                            np.linalg.norm(global_model_flat) * np.linalg.norm(agent_model_flat))) ** 2
                )

                # Compute Manhattan distance
                manhattan_distance = float(np.linalg.norm(global_model_flat - agent_model_flat, ord=1))

                # Compute Euclidean distance
                euclidean_distance = np.linalg.norm(global_model_flat - agent_model_flat)

                # Append the computed distances to respective lists
                cos_dis.append(cosine_distance)
                manhattan_dis.append(manhattan_distance)
                euclidean_dis.append(euclidean_distance)

            # Now we need to compute the absolute differences for each agent
            total_diff_cos = np.zeros(len(agent_cur_parameters_dict))  # For cosine distances
            total_diff_manhattan = np.zeros(len(agent_cur_parameters_dict))  # For Manhattan distances
            total_diff_euclidean = np.zeros(len(agent_cur_parameters_dict))  # For Euclidean distances

            # Calculate the absolute differences for each agent against others for each metric separately
            for i in range(len(agent_cur_parameters_dict)):
                for j in range(len(agent_cur_parameters_dict)):
                    if i != j:
                        # Calculate the absolute difference for each metric separately
                        diff_cos = np.abs(cos_dis[i] - cos_dis[j])
                        diff_manhattan = np.abs(manhattan_dis[i] - manhattan_dis[j])
                        diff_euclidean = np.abs(euclidean_dis[i] - euclidean_dis[j])

                        # Add the absolute differences to the total for each metric
                        total_diff_cos[i] += diff_cos
                        total_diff_manhattan[i] += diff_manhattan
                        total_diff_euclidean[i] += diff_euclidean

            # Combine the differences into a tri_distance matrix
            tri_distance = np.vstack([total_diff_cos, total_diff_manhattan, total_diff_euclidean]).T

            # Calculate the covariance matrix of the tri_distance
            cov_matrix = np.cov(tri_distance.T)
            # Compute the inverse of the covariance matrix
            inv_matrix = np.linalg.inv(cov_matrix)

            # Calculate the Mahalanobis distance for each agent
            ma_distances = []
            for i in range(len(agent_cur_parameters_dict)):
                t = tri_distance[i]
                ma_dis = np.dot(np.dot(t, inv_matrix), t.T)  # Mahalanobis distance
                ma_distances.append(ma_dis)

            # Now, we can use the Mahalanobis distances (ma_distances) to rank the agents
            scores = ma_distances
            # Sort agents by their Mahalanobis distance (lower score is better)
            sorted_agent_indices = np.argsort(scores)

            # Calculate the number of agents to select based on the proportion p
            num_agents_to_select = int(len(sorted_agent_indices) * p)
            # Select the agents with the lowest Mahalanobis distances based on the proportion p
            selected_agent_ids = [list(agent_cur_parameters_dict.keys())[i] for i in
                                  sorted_agent_indices[:num_agents_to_select]]

            print(f"selected agents ids: {selected_agent_ids}")
            # Aggregating updates based on selected agents
            sm_updates, total_data = 0, 0
            for agent_id, update in agent_updates_dict.items():
                if agent_id in selected_agent_ids:
                    n_agent_data = self.agent_data_sizes[agent_id]  # Assuming self.agent_data_sizes is defined somewhere
                    sm_updates += n_agent_data * update
                    total_data += n_agent_data

            # Return the aggregated model update normalized by total data size
            return sm_updates / total_data

