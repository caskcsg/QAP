import os
import time
import torch

import numpy as np
from torch.utils.data import DataLoader
from src9.cli import get_args
print(torch.cuda.device_count())
from src9.datasets import get_dataset_iemocap, collate_fn, HCFDataLoader, get_dataset_mosei, collate_fn_hcf_mosei

# from src.models.e2e import MME2E
from src9.models.sparse_e2e import MME2E_Sparse
from src9.models.e2e import MME2E
from src9.models.baselines.lf_rnn import LF_RNN
from src9.models.baselines.lf_transformer import LF_Transformer
from src9.trainers.emotiontrainer import IemocapTrainer

from src9.complex import *
print(torch.cuda.device_count())


if __name__ == "__main__":
    
    start = time.time()

    args = get_args()

    # Fix seed for reproducibility
    seed = args['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set device
    #print(args['cuda'])

    #print(device)
    # device = torch.device(f"cuda:{args['cuda']}" if torch.cuda.is_available() else 'cpu')
    # torch.cuda.set_device(int(args['cuda']))
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print("Start loading the data....")

    if args['dataset'] == 'iemocap':
        train_dataset = get_dataset_iemocap(data_folder=args['datapath'], phase='train',
                                            img_interval=args['img_interval'], hand_crafted_features=args['hand_crafted'])
        valid_dataset = get_dataset_iemocap(data_folder=args['datapath'], phase='test',
                                            img_interval=args['img_interval'], hand_crafted_features=args['hand_crafted'])
        test_dataset = get_dataset_iemocap(data_folder=args['datapath'], phase='test',
                                           img_interval=args['img_interval'], hand_crafted_features=args['hand_crafted'])

        if args['hand_crafted']:
            train_loader = HCFDataLoader(dataset=train_dataset, feature_type=args['audio_feature_type'],
                                         batch_size=args['batch_size'], shuffle=True)
            valid_loader = HCFDataLoader(dataset=valid_dataset, feature_type=args['audio_feature_type'],
                                         batch_size=args['batch_size'], shuffle=False)
            test_loader = HCFDataLoader(dataset=test_dataset, feature_type=args['audio_feature_type'],
                                        batch_size=args['batch_size'], shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True,
                                       collate_fn=collate_fn)
            valid_loader = DataLoader(valid_dataset, batch_size=args['batch_size'], shuffle=False,
                                       collate_fn=collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False,
                                      collate_fn=collate_fn)
    elif args['dataset'] == 'mosei':
        train_dataset = get_dataset_mosei(data_folder=args['datapath'], phase='train', img_interval=args['img_interval'], hand_crafted_features=args['hand_crafted'])
        valid_dataset = get_dataset_mosei(data_folder=args['datapath'], phase='valid', img_interval=args['img_interval'], hand_crafted_features=args['hand_crafted'])
        test_dataset = get_dataset_mosei(data_folder=args['datapath'], phase='test', img_interval=args['img_interval'], hand_crafted_features=args['hand_crafted'])

        train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, collate_fn=collate_fn_hcf_mosei if args['hand_crafted'] else collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=args['batch_size'], shuffle=False, collate_fn=collate_fn_hcf_mosei if args['hand_crafted'] else collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, collate_fn=collate_fn_hcf_mosei if args['hand_crafted'] else collate_fn)

    print(f'# Train samples = {len(train_loader.dataset)}')
    print(f'# Valid samples = {len(valid_loader.dataset)}')
    print(f'# Test samples = {len(test_loader.dataset)}')

    dataloaders = {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader
    }

    lr = args['learning_rate']
    unitary_lr = args['learning_rate_unitary']
    if args['model'] == 'mme2e':
        model = MME2E(args=args,device=device)
        model = model.to(device=device)
        #model_num(model)
        # When using a pre-trained text modal, you can use text_lr_factor to give a smaller leraning rate to the textual model parts
        if args['text_lr_factor'] == 1:
            optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
        else:
            #print(250)
            param_optimizer = list(model.named_parameters())
            # print(param_optimizer)
            # exit(0)
            # for n,p in param_optimizer:
            #     if "T" in n:
            #         print(n)
            #         print(p)
            # print(250)
            # print()
            # print(model.T.parameters())
            # exit(0)
            optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if "albert" in n and "unitary" not in n], 'lr': lr / args['text_lr_factor']},
            {'params': [p for n, p in param_optimizer if "albert" not in n and "unitary" not in n]},
            ]
            parameters_unitary = [p for n, p in param_optimizer if "unitary"  in n]

            optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr, weight_decay=args['weight_decay'])
            unitary_optimizer = RMSprop_Unitary(parameters_unitary,lr = unitary_lr, weight_decay=args['weight_decay'])
    elif args['model'] == 'mme2e_sparse':
        model = MME2E_Sparse(args=args, device=device)
        model = model.to(device=device)

        # When using a pre-trained text modal, you can use text_lr_factor to give a smaller leraning rate to the textual model parts
        if args['text_lr_factor'] == 1:
            optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
        else:
            optimizer = torch.optim.Adam([
                {'params': model.T.parameters(), 'lr': lr / args['text_lr_factor']},
                {'params': model.t_out.parameters(), 'lr': lr / args['text_lr_factor']},
                {'params': model.V.parameters()},
                {'params': model.v_flatten.parameters()},
                {'params': model.v_transformer.parameters()},
                {'params': model.v_out.parameters()},
                {'params': model.A.parameters()},
                {'params': model.a_flatten.parameters()},
                {'params': model.a_transformer.parameters()},
                {'params': model.a_out.parameters()},
                {'params': model.weighted_fusion.parameters()},
            ], lr=lr, weight_decay=args['weight_decay'])
    elif args['model'] == 'lf_rnn':
        model = LF_RNN(args)
        model = model.to(device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args['weight_decay'])
    elif args['model'] == 'lf_transformer':
        model = LF_Transformer(args)
        model = model.to(device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args['weight_decay'])
    else:
        raise ValueError('Incorrect model name!')

    if args['scheduler']:
        print(250)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'] * len(train_loader.dataset) // args['batch_size'])
    else:
        scheduler = None

    if args['loss'] == 'l1':
        criterion = torch.nn.L1Loss()
    elif args['loss'] == 'mse':
        criterion = torch.nn.MSELoss()
    elif args['loss'] == 'ce':
        criterion = torch.nn.CrossEntropyLoss()
    elif args['loss'] == 'bce':
        pos_weight = train_dataset.getPosWeight()
        pos_weight = torch.tensor(pos_weight).to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # criterion = torch.nn.BCEWithLogitsLoss()

    if args['dataset'] == 'iemocap' or 'mosei':
        trainer = IemocapTrainer(args, model, criterion, optimizer, scheduler, device, dataloaders, unitary_optimizer)

    if args['test']:
        trainer.test()
    elif args['valid']:
        trainer.valid()
    else:
        trainer.train()

    end = time.time()

    print(f'Total time usage = {(end - start) / 3600:.2f} hours.')
