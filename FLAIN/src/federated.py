import torch
import re
import utils
import models
import math
import copy
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from agent import Agent
from time import ctime
from options import args_parser
from aggregation import Aggregation
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import parameters_to_vector, vector_to_parameters

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    args = args_parser()
    utils.print_exp_details(args)

    file_name = f"""time:{ctime()}"""\
            + f"""s_lr:{args.server_lr}-num_cor:{args.backdoor_frac}"""\
            + f"""-poison_frac:{args.poison_frac}-pttrn:{args.trigger_type}"""
    file_name = re.sub(r'[<>:"/\\|?*]', '_', file_name)
    writer = SummaryWriter('logs/' + file_name)
    cum_poison_acc_mean = 0

    train_dataset, val_dataset = utils.get_datasets(args.data)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    if args.data_distribution == 'iid':
        user_groups = utils.iid_distribute_data(train_dataset, args)
    elif args.data_distribution == 'non_iid':
        user_groups = utils.non_iid_distribute_data(train_dataset, args)

    idxs = (val_dataset.targets == args.base_class).nonzero().flatten().tolist()
    poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
    utils.poison_dataset(poisoned_val_set.dataset, args, idxs, poison_all=True)
    poisoned_val_loader = DataLoader(poisoned_val_set, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    global_model = models.get_model(args.data).to(args.device)
    agents, agent_data_sizes = [], {}

    for _id in range(0, args.num_agents):
        agent= Agent(_id, args, train_dataset, user_groups[_id])
        agent_data_sizes[_id] = agent.n_data
        agents.append(agent) 

    n_model_params = len(parameters_to_vector(global_model.parameters()))
    aggregator = Aggregation(agent_data_sizes, n_model_params, poisoned_val_loader, args, writer)
    criterion = nn.CrossEntropyLoss().to(args.device)


    for rnd in tqdm(range(1, args.train_rounds + 1)):
        rnd_global_params = parameters_to_vector(global_model.parameters()).detach()
        agent_updates_dict = {}
        agent_cur_parameters_dict = {}
        model_updates = []

        per_round_selected_agents = int(args.num_agents * args.agent_frac)
        per_round_backdoor_agents = int(args.num_agents * args.backdoor_frac * args.agent_frac)
        total_backdoor_agents = int(args.num_agents * args.backdoor_frac)
        benign_agents = per_round_selected_agents - per_round_backdoor_agents

        total_agents = args.num_agents
        backdoor_range = range(0, total_backdoor_agents)
        benign_range = [i for i in range(total_agents) if i not in backdoor_range]

        backdoor_id = sorted(np.random.choice(backdoor_range, per_round_backdoor_agents, replace=False))
        benign_id = sorted(np.random.choice(benign_range, benign_agents, replace=False))
        combined_ids = sorted(np.concatenate((backdoor_id, benign_id)))

        for agent_id in combined_ids:
            update, cur_parameters,diff = agents[agent_id].local_train(global_model, criterion)
            model_updates.append(diff)
            agent_updates_dict[agent_id] = update
            agent_cur_parameters_dict[agent_id] = cur_parameters
            vector_to_parameters(copy.deepcopy(rnd_global_params), global_model.parameters())
        aggregator.aggregate_updates(global_model, agent_updates_dict, agent_cur_parameters_dict, model_updates, args)


        if rnd % args.snap == 0:
            with torch.no_grad():
                val_loss, (val_acc, val_per_class_acc) = utils.get_loss_n_accuracy(global_model, criterion, val_loader, args)
                writer.add_scalar('Validation/Loss', val_loss, rnd)
                writer.add_scalar('Validation/Accuracy', val_acc, rnd)
                print(f'| Val_Loss/Val_Acc: {val_loss:.3f} / {val_acc:.3f} |')
                print(f'| Val_Per_Class_Acc: {val_per_class_acc} ')
            
                poison_loss, (poison_acc, _) = utils.get_loss_n_accuracy(global_model, criterion, poisoned_val_loader, args)
                cum_poison_acc_mean += poison_acc
                writer.add_scalar('Poison/Base_Class_Accuracy', val_per_class_acc[args.base_class], rnd)
                writer.add_scalar('Poison/Poison_Accuracy', poison_acc, rnd)
                writer.add_scalar('Poison/Poison_Loss', poison_loss, rnd)
                writer.add_scalar('Poison/Cumulative_Poison_Accuracy_Mean', cum_poison_acc_mean/rnd, rnd) 
                print(f'| Poison Loss/Poison Acc: {poison_loss:.3f} / {poison_acc:.3f} |')

    print('Training has finished!')
   

    
    
    
      
              