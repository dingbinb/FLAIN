import torch
import utils
import copy
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector


class Agent():
    def __init__(self, id, args, train_dataset=None, data_idxs=None):
        self.id = id
        self.args = args
        self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs)

        if  self.id < args.num_agents * args.backdoor_frac:
            print("backdoor client: ", self.id)
            print("datasize:", len(data_idxs))
            print("------------------")
            utils.poison_dataset(train_dataset, args, data_idxs, agent_idx=self.id)

        else:
            print("benign client：", self.id)
            print("datasize:", len(data_idxs))
            print("------------------")

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True,
                                       num_workers=args.num_workers, pin_memory=False)

        self.n_data = len(self.train_dataset)



    def local_train(self, global_model, criterion):

        initial_global_model_params = parameters_to_vector(global_model.parameters()).detach()
        initial_global_model = copy.deepcopy(global_model.state_dict())
        global_model.train()
        optimizer = torch.optim.Adam(global_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        for _ in range(self.args.local_ep):
            for _, (inputs, labels) in enumerate(self.train_loader):
                optimizer.zero_grad()
                inputs, labels = inputs.to(device=self.args.device, non_blocking=True),\
                                 labels.to(device=self.args.device, non_blocking=True)
                outputs = global_model(inputs)
                minibatch_loss = criterion(outputs, labels)
                minibatch_loss.backward()
                optimizer.step()

        with torch.no_grad():
           update = parameters_to_vector(global_model.parameters()).double() - initial_global_model_params
           cur_parameters = parameters_to_vector(global_model.parameters())

           diff = {}
           for name, param in global_model.named_parameters():
               if 'weight' in name:
                   diff[name] = param.data - initial_global_model[name].data

           return update, cur_parameters, diff


