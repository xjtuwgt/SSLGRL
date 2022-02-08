from graph_data.graph_dataloader import NodeClassificationSubGraphDataHelper
import logging
import numpy as np
from tqdm import tqdm, trange
from codes.gnn_predictor import NodeClassificationModel
import torch
from utils.basic_utils import IGNORE_IDX, seed_everything
from codes.gnn_encoder import GraphSimSiamEncoder


def rand_search_parameter(space: dict):
    para_type = space['type']
    if para_type == 'fixed':
        return space['value']
    if para_type == 'choice':
        candidates = space['values']
        value = np.random.choice(candidates, 1).tolist()[0]
        return value
    if para_type == 'range':
        log_scale = space.get('log_scale', False)
        low, high = space['bounds']
        if log_scale:
            value = np.random.uniform(low=np.log(low), high=np.log(high), size=1)[0]
            value = np.exp(value)
        else:
            value = np.random.uniform(low=low, high=high, size=1)[0]
        return value
    else:
        raise ValueError('Training batch mode %s not supported' % para_type)


def citation_hyper_parameter_space():
    learning_rate = {'name': 'learning_rate', 'type': 'choice',
                     'values': [1e-5, 2e-5, 5e-5]}  # 5e-5, 1e-4, 2e-4, 3e-4, 1e-3, 2e-3, 5e-3
    weight_decay = {'name': 'weight_decay', 'type': 'choice', 'values': [1e-7, 5e-7]}
    attn_drop_ratio = {'name': 'attn_drop_ratio', 'type': 'choice', 'values': list(np.arange(0.1, 0.56, 0.025))}
    feat_drop_ratio = {'name': 'feat_drop_ratio', 'type': 'choice', 'values': list(np.arange(0.1, 0.56, 0.025))}
    edge_drop_ratio = {'name': 'edge_drop_ratio', 'type': 'choice', 'values': list(np.arange(0.05, 0.26, 0.025))}
    hop_num = {'name': 'hop_num', 'type': 'choice', 'values': [6, 8, 9, 10]}
    alpha = {'name': 'alpha', 'type': 'choice', 'values': list(np.arange(0.05, 0.21, 0.025))}
    hidden_dim = {'name': 'hidden_dim', 'type': 'choice', 'values': [64]}
    layer_num = {'name': 'layer_num', 'type': 'choice', 'values': [2]}
    epoch = {'name': 'epoch', 'type': 'choice', 'values': [300]}
    # ++++++++++++++++++++++++++++++++++
    search_space = [learning_rate, weight_decay, attn_drop_ratio, feat_drop_ratio, edge_drop_ratio,
                    hidden_dim, hop_num, alpha, layer_num, epoch]
    search_space = dict((x['name'], x) for x in search_space)
    return search_space


def single_task_trial(search_space: dict, rand_seed: int):
    seed_everything(seed=rand_seed)
    parameter_dict = {}
    for key, value in search_space.items():
        parameter_dict[key] = rand_search_parameter(value)
    parameter_dict['seed'] = rand_seed
    return parameter_dict


def citation_random_search_hyper_tunner(args, search_space: dict, seed: int):
    parameter_dict = single_task_trial(search_space=search_space, rand_seed=seed)
    args.fine_tuned_learning_rate = parameter_dict['learning_rate']
    args.feat_drop = parameter_dict['feat_drop_ratio']
    args.attn_drop = parameter_dict['attn_drop_ratio']
    args.edge_drop = parameter_dict['edge_drop_ratio']
    args.layers = parameter_dict['layer_num']
    args.gnn_hop_num = parameter_dict['hop_num']
    args.alpha = parameter_dict['alpha']
    args.hidden_dim = parameter_dict['hidden_dim']
    args.fine_tuned_weight_decay = parameter_dict['weight_decay']
    args.seed = parameter_dict['seed']
    args.num_train_epochs = parameter_dict['epoch']
    return args, parameter_dict


def train_node_classification(args):
    node_data_helper = NodeClassificationSubGraphDataHelper(config=args)
    args.node_number = node_data_helper.number_of_nodes
    args.node_emb_dim = node_data_helper.n_feats
    args.relation_number = node_data_helper.number_of_relations
    args.num_node_classes = node_data_helper.num_class
    node_features = node_data_helper.node_features
    #########################################################################
    logging.info('*' * 75)
    for key, value in vars(args).items():
        logging.info('Hype-parameter\t{} = {}'.format(key, value))
    logging.info('*' * 75)
    for key, value in vars(args).items():
        if 'number' in key or 'emb_dim' in key:
            logging.info('Hype-parameter\t{} = {}'.format(key, value))
    logging.info('*' * 75)
    #########################################################################
    train_dataloader = node_data_helper.data_loader(data_type='train')
    logging.info('Loading training data = {} completed'.format(len(train_dataloader)))
    logging.info('*' * 75)
    # **********************************************************************************
    graph_encoder = GraphSimSiamEncoder(config=args)
    graph_encoder.init(graph_node_emb=node_features, node_freeze=True)
    graph_encoder.to(args.device)
    # **********************************************************************************
    model = NodeClassificationModel(graph_encoder=graph_encoder, encoder_dim=args.hidden_dim,
                                    num_of_classes=args.num_node_classes, fix_encoder=False)
    model.to(args.device)
    logging.info('Model Parameter Configuration:')
    for name, param in model.named_parameters():
        logging.info('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()),
                                                                  str(param.requires_grad)))
    # **********************************************************************************
    loss_fcn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_IDX)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.fine_tuned_learning_rate, weight_decay=args.fine_tuned_weight_decay)
    # **********************************************************************************
    start_epoch = 0
    global_step = 0
    best_valid_accuracy = 0.0
    test_acc = 0.0
    # **********************************************************************************
    logging.info('Starting fine tuning the model...')
    train_iterator = trange(start_epoch, start_epoch + int(args.num_train_epochs), desc="Epoch",
                            disable=args.local_rank not in [-1, 0])
    for epoch_idx in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            for key, value in batch.items():
                if key == 'batch_label':
                    batch[key] = value.to(args.device)
                else:
                    batch[key] = (value[0].to(args.device), value[1].to(args.device), value[2].to(args.device))
            logits = model.forward(batch, cls_or_anchor=args.cls_or_anchor)
            loss = loss_fcn(logits, batch['batch_label'])
            del batch
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            loss.backward()
            optimizer.step()
            model.zero_grad()
            global_step = global_step + 1
            if global_step % args.logging_steps == 0:
                logging.info('Loss at step {} = {:.5f}'.format(global_step, loss.data.item()))
        if (epoch_idx + 1) % 2 == 0:
            eval_acc = evaluate_node_classification_model(model=model, node_data_helper=node_data_helper, args=args)
            if eval_acc > best_valid_accuracy:
                best_valid_accuracy = eval_acc
                test_acc = evaluate_node_classification_model(model=model, node_data_helper=node_data_helper, args=args,
                                                              data_type='test')
            logging.info('Best valid | current valid | test accuracy = {:.5f} | {:.5f} | {:.5f}'.format(
                best_valid_accuracy, eval_acc, test_acc))
    logging.info('Best valid | test accuracy = {:.5f} | {:.5f}'.format(best_valid_accuracy, test_acc))
    return best_valid_accuracy, test_acc


def evaluate_node_classification_model(model, node_data_helper, args, data_type='valid'):
    val_dataloader = node_data_helper.data_loader(data_type=data_type)
    logging.info('Loading {} data = {} completed'.format(data_type, len(val_dataloader)))
    epoch_iterator = tqdm(val_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    model.eval()
    total_correct = 0.0
    total_example = 0.0
    for step, batch in enumerate(epoch_iterator):
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for key, value in batch.items():
            if key == 'batch_label':
                batch[key] = value.to(args.device)
            else:
                batch[key] = (value[0].to(args.device), value[1].to(args.device), value[2].to(args.device))
        with torch.no_grad():
            logits = model.forward(batch, cls_or_anchor=args.cls_or_anchor)
            preds = torch.argmax(logits, dim=-1)
            total_example = total_example + preds.shape[0]
            total_correct = total_correct + (preds == batch['batch_label']).sum().data.item()
    eval_acc = total_correct / total_example
    return eval_acc


def hyper_parameter_tuning_rand_search(args):
    num_of_experiments = args.exp_number
    hyper_search_space = citation_hyper_parameter_space()
    acc_list = []
    search_best_test_acc = 0.0
    search_best_val_acc = 0.0
    search_best_settings = None
    for _ in range(num_of_experiments):
        args, hyper_setting_i = citation_random_search_hyper_tunner(args=args, search_space=hyper_search_space,
                                                                    seed=args.seed + _)
        best_val_acc, best_test_acc = train_node_classification(args=args)
        acc_list.append((hyper_setting_i, best_val_acc, best_test_acc))
        logging.info('*' * 50)
        logging.info('{}\t{:.4f}\t{:.4f}'.format(hyper_setting_i, best_val_acc, best_test_acc))
        logging.info('*' * 50)
        if search_best_val_acc < best_val_acc:
            search_best_val_acc = best_val_acc
            search_best_test_acc = best_test_acc
            search_best_settings = hyper_setting_i
        logging.info('Current best testing acc = {:.4f} and best dev acc = {}'.format(search_best_test_acc,
                                                                                     search_best_val_acc))
        logging.info('*' * 30)
    for _, setting_acc in enumerate(acc_list):
        print(_, setting_acc)
    print(search_best_test_acc)
    print(search_best_settings)