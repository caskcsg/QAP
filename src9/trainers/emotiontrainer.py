import copy
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from tabulate import tabulate
from src9.evaluate import eval_iemocap, eval_iemocap_ce
from src9.trainers.basetrainer import TrainerBase
from transformers import AlbertTokenizer

class IemocapTrainer(TrainerBase):
    def __init__(self, args, model, criterion, optimizer, scheduler, device, dataloaders, unitary_optimizer):
        super(IemocapTrainer, self).__init__(args, model, criterion, optimizer, scheduler, device, dataloaders, unitary_optimizer)
        self.args = args
        self.text_max_len = args['text_max_len']
        # print(args['text_max_len'])
        # exit(0)
        self.tokenizer = AlbertTokenizer.from_pretrained(f'albert-{args["text_model_size"]}-v2')
        self.eval_func = eval_iemocap if args['loss'] == 'bce' else eval_iemocap_ce
        self.all_train_stats = []
        self.all_valid_stats = []
        self.all_test_stats = []
        #"result.txt" = args['resultpath']
        annotations = dataloaders['train'].dataset.get_annotations()

        if self.args['loss'] == 'bce':
            self.headers = [
                ['phase (acc)', *annotations, 'average'],
                ['phase (recall)', *annotations, 'average'],
                ['phase (precision)', *annotations, 'average'],
                ['phase (f1)', *annotations, 'average'],
                ['phase (auc)', *annotations, 'average']
            ]

            n = len(annotations) + 1
            self.prev_train_stats = [[-float('inf')] * n, [-float('inf')] * n, [-float('inf')] * n, [-float('inf')] * n, [-float('inf')] * n]
            self.prev_valid_stats = copy.deepcopy(self.prev_train_stats)
            self.prev_test_stats = copy.deepcopy(self.prev_train_stats)
            self.best_valid_stats = copy.deepcopy(self.prev_train_stats)
        else:
            self.header = ['Phase', 'Acc', 'Recall', 'Precision', 'F1']
            self.best_valid_stats = [0, 0, 0, 0]

        self.best_epoch = -1


        # self.dict_senti={}
        # f_senti=open("data/word_polarity1.txt","r")
        # for line in f_senti:
        #     tt=line.strip().split()
        #     self.dict_senti[tt[0]]=float(tt[1])
        # f_senti.close()  

    def train(self):
        for epoch in range(1, self.args['epochs'] + 1):
            print(f'=== Epoch {epoch} ===')

  

            train_stats, train_thresholds = self.train_one_epoch()
            valid_stats, valid_thresholds = self.eval_one_epoch()
            test_stats, _ = self.eval_one_epoch('test', valid_thresholds)
            # test_stats, _ = self.eval_one_epoch('test', [0.5,0.5,0.5,0.5,0.5,0.5])

            print('Train thresholds: ', train_thresholds)
            print('Valid thresholds: ', valid_thresholds)

            # if self.args['model'] == 'mme2e_sparse':
            #     sparse_percentages = self.model.get_sparse_percentages()
            #     if 'v' in self.args['modalities']:
            #         print('V sparse percent', sparse_percentages[0])
            #     if 'a' in self.args['modalities']:
            #         print('A sparse percent', sparse_percentages[1])

            self.all_train_stats.append(train_stats)
            self.all_valid_stats.append(valid_stats)
            self.all_test_stats.append(test_stats)

            if self.args['loss'] == 'ce':
                train_stats_str = [f'{s:.4f}' for s in train_stats]
                valid_stats_str = [f'{s:.4f}' for s in valid_stats]
                test_stats_str = [f'{s:.4f}' for s in test_stats]
                print(tabulate([
                    ['Train', *train_stats_str],
                    ['Valid', *valid_stats_str],
                    ['Test', *test_stats_str]
                ], headers=self.header))
                if valid_stats[-1] > self.best_valid_stats[-1]:
                    self.best_valid_stats = valid_stats
                    self.best_epoch = epoch
                    self.earlyStop = self.args['early_stop']
                else:
                    self.earlyStop -= 1
            else:
                ff=open("result.txt","a")
                for i in range(len(self.headers)):
                    for j in range(len(valid_stats[i])):
                        is_pivot = (i == 3 and j == (len(valid_stats[i]) - 1)) # auc average
                        if valid_stats[i][j] > self.best_valid_stats[i][j]:
                            self.best_valid_stats[i][j] = valid_stats[i][j]
                            if is_pivot:
                                self.earlyStop = self.args['early_stop']
                                self.best_epoch = epoch
                                self.best_model = copy.deepcopy(self.model.state_dict())
                        elif is_pivot:
                            self.earlyStop -= 1

                    train_stats_str = self.make_stat(self.prev_train_stats[i], train_stats[i])
                    valid_stats_str = self.make_stat(self.prev_valid_stats[i], valid_stats[i])
                    test_stats_str = self.make_stat(self.prev_test_stats[i], test_stats[i])

                    self.prev_train_stats[i] = train_stats[i]
                    self.prev_valid_stats[i] = valid_stats[i]
                    self.prev_test_stats[i] = test_stats[i]

                    print(tabulate([
                        ['Train', *train_stats_str],
                        ['Valid', *valid_stats_str],
                        ['Test', *test_stats_str]
                    ], headers=self.headers[i]))
                    ff.write(tabulate([
                        ['Train', *train_stats_str],
                        ['Valid', *valid_stats_str],
                        ['Test', *test_stats_str]
                    ], headers=self.headers[i]))
                    ff.write("\n")
                ff.write("\n")
                ff.close()
            if self.earlyStop == 0:
                break

        print('=== Best performance ===')
        if self.args['loss'] == 'ce':
            print(tabulate([
                [f'Test ({self.best_epoch})', *self.all_test_stats[self.best_epoch - 1]]
            ], headers=self.header))
        else:
            ff=open("result.txt","a")
            for i in range(len(self.headers)):
                ff.write(tabulate([[f'Test ({self.best_epoch})', *self.all_test_stats[self.best_epoch - 1][i]]], headers=self.headers[i]))
                ff.write("\n")
                print(tabulate([[f'Test ({self.best_epoch})', *self.all_test_stats[self.best_epoch - 1][i]]], headers=self.headers[i]))
            ff.write("AAAAAAAAAAAAAAA")
            ff.close()

        # self.save_stats()
        # self.save_model()
        print('Results and model are saved!')

    def valid(self):
        valid_stats = self.eval_one_epoch()
        for i in range(len(self.headers)):
            print(tabulate([['Valid', *valid_stats[i]]], headers=self.headers[i]))
            print()

    def test(self):
        test_stats = self.eval_one_epoch('test')
        for i in range(len(self.headers)):
            print(tabulate([['Test', *test_stats[i]]], headers=self.headers[i]))
            print()
        for stat in test_stats:
            for n in stat:
                print(f'{n:.4f},', end='')
        print()
    def get_senti(tokens):
        input_senti = []
        for w in tokens:
            if w in dict_senti:
                input_senti.append(self.dict_senti[w])
            else:
                input_senti.append(0.0)
        return input_senti
    def train_one_epoch(self):
        self.model.train()
        if self.args['model'] == 'mme2e' or self.args['model'] == 'mme2e_sparse':
            self.model.mtcnn.eval()

        dataloader = self.dataloaders['train']
        epoch_loss = 0.0
        data_size = 0
        total_logits = []
        total_Y = []
        pbar = tqdm(dataloader, desc='Train')

        # with torch.autograd.set_detect_anomaly(True):
        for uttranceId, imgs, imgLens, specgrams, specgramLens, text, Y, video_feature, audio_feature, senti in pbar:
            # print(type(uttranceId))
            # print(len(uttranceId))
            # print(uttranceId)
            # print(type(imgs))
            # print(imgs.shape)
            # #print(imgs)
            # print(type(imgLens))
            # print(len(imgLens))
            # print(imgLens)
            # print(type(specgrams))
            # print(specgrams.shape)
            # #print(specgrams)
            # print(type(specgramLens))
            # print(len(specgramLens))
            # print(specgramLens)
            # print(type(text))
            # print(len(text))
            # print(text)
            # print(type(Y))
            # print(len(Y))
            # print(Y)
            # exit(0)
            # print(video_feature.shape)
            # print(audio_feature.shape)
            # exit(0)

            senti=senti.to(device=self.device)
            #print()
            #sprint(senti[0])

            #print(senti.shape)
            if 'lf_' not in self.args['model']:
                #print(250)
           
                text = self.tokenizer(text, return_tensors='pt', max_length=self.text_max_len, padding='max_length', truncation=True)
                
                # print((text['input_ids'][1]))
                # print(text['token_type_ids'][1])
                # print(text['attention_mask'][1])
                # exit(0)
            else:
                imgs = imgs.to(device=self.device)

            if self.args['loss'] == 'ce':
                Y = Y.argmax(-1)

            # imgs = imgs.to(device=self.device)
            specgrams = specgrams.to(device=self.device)
            text = text.to(device=self.device)
            video_feature = video_feature.to(device=self.device)
            audio_feature = audio_feature.to(device=self.device)
            Y = Y.to(device=self.device)

            self.optimizer.zero_grad()
            self.unitary_optimizer.zero_grad()
            shape0 = len(Y)
            with torch.set_grad_enabled(True):
                logits = self.model(imgs, imgLens, specgrams, specgramLens, text, video_feature, audio_feature, senti) # (batch_size, num_classes)
                # print(logits)
                # print(Y)
                loss = self.criterion(logits, Y)
                # print(loss)
                # exit(0)
                loss.backward()
                epoch_loss += loss.item() * Y.size(0)
                data_size += Y.size(0)
                if self.args['clip'] > 0:
                    clip_grad_norm_(self.model.parameters(), self.args['clip'])
                self.optimizer.step()
                self.unitary_optimizer.step(shape0=shape0)
                
            total_logits.append(logits.cpu())
            total_Y.append(Y.cpu())
            pbar.set_description("train loss:{:.4f}".format(epoch_loss / data_size))
            if self.scheduler is not None:
                self.scheduler.step()
        total_logits = torch.cat(total_logits, dim=0)
        total_Y = torch.cat(total_Y, dim=0)

        epoch_loss /= len(dataloader.dataset)
        # print(f'train loss = {epoch_loss}')
        torch.cuda.empty_cache()
        return self.eval_func(total_logits, total_Y)

    def eval_one_epoch(self, phase='valid', thresholds=None):
        self.model.eval()
        dataloader = self.dataloaders[phase]
        epoch_loss = 0.0
        data_size = 0
        total_logits = []
        total_Y = []
        pbar = tqdm(dataloader, desc=phase)

        for uttranceId, imgs, imgLens, specgrams, specgramLens, text, Y,  video_feature, audio_feature, senti in pbar:

            senti=senti.to(device=self.device)
            if 'lf_' not in self.args['model']:
                text = self.tokenizer(text, return_tensors='pt', max_length=self.text_max_len, padding='max_length', truncation=True)
            else:
                imgs = imgs.to(device=self.device)

            if self.args['loss'] == 'ce':
                Y = Y.argmax(-1)

            # imgs = imgs.to(device=self.device)
            specgrams = specgrams.to(device=self.device)
            text = text.to(device=self.device)
            Y = Y.to(device=self.device)
            video_feature = video_feature.to(device=self.device)
            audio_feature = audio_feature.to(device=self.device)
            with torch.set_grad_enabled(False):
                logits = self.model(imgs, imgLens, specgrams, specgramLens, text, video_feature, audio_feature, senti) # (batch_size, num_classes)
                loss = self.criterion(logits, Y)
                epoch_loss += loss.item() * Y.size(0)
                data_size += Y.size(0)

            total_logits.append(logits.cpu())
            total_Y.append(Y.cpu())

            pbar.set_description(f"{phase} loss:{epoch_loss/data_size:.4f}")

        total_logits = torch.cat(total_logits, dim=0)
        total_Y = torch.cat(total_Y, dim=0)

        epoch_loss /= len(dataloader.dataset)

        # if phase == 'valid' and self.scheduler is not None:
        #     self.scheduler.step(epoch_loss)

        # print(f'{phase} loss = {epoch_loss}')
        return self.eval_func(total_logits, total_Y, thresholds)
