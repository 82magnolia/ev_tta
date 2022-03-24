import torch
from base.models.model_container import ModelContainer
from base.train.metrics import accuracy
from base.data.data_container import DataContainer
from base.train.common_trainer import CommonTrainer
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
from real_cnn_model.train.loss import SoftCrossEntropy, NegLogRatioLoss, KLDivLogits, InfoRadiusLoss
from data.imagenet import mutual_remove


class CNNTrainer(CommonTrainer):
    def __init__(self, cfg, model_container: ModelContainer, data_container: DataContainer, **kwargs):
        super(CNNTrainer, self).__init__(cfg, model_container, data_container)
        print(f'Initializing trainer {self.__class__.__name__}...')
        self.init_env()
        self.init_optimizer()
        self.init_scheduler()
        self.prep_train()
        self.loss_func = nn.CrossEntropyLoss()
        self.debug = getattr(self.cfg, 'debug', False)
        self.debug_input = getattr(self.cfg, 'debug_input', False)
        self.debug_labels = getattr(self.cfg, 'debug_labels', False)

    def init_optimizer(self, **kwargs):
        """
        Initialize self.optimizer using self.cfg.
        """

        # Choose optimizer
        model = self.model_container.models['model']
        opt_type = self.cfg.optimizer
        freeze = getattr(self.cfg, 'freeze', False) or getattr(self.cfg, 'train_classifier', False)

        if opt_type == 'SGD':
            print('Using SGD as optimizer')
            if freeze:
                print('Freezing weights!')
                self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=self.cfg.learning_rate, momentum=self.cfg.momentum, weight_decay=self.cfg.weight_decay)
            else:
                self.optimizer = optim.SGD(model.parameters(), lr=self.cfg.learning_rate, momentum=self.cfg.momentum, weight_decay=self.cfg.weight_decay)
        elif opt_type == 'Adam':
            print('Using Adam as optimizer')
            if freeze:
                print('Freezing weights!')
                self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
            else:
                self.optimizer = optim.Adam(model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)

    def train(self, data_dict, **kwargs):
        input_data = data_dict['input_data']
        label = data_dict['label']

        self.model_container.set_train(['model'])

        if self.use_cuda:
            input_data, label = input_data.to(self.devices[0]), label.to(self.devices[0])

        self.optimizer.zero_grad()

        pred = self.model_container.infer('model', input_data, False)

        loss = self.loss_func(pred, label)
        
        acc_1, acc_5 = accuracy(pred.cpu(), label.cpu(), topk=(1, min(5, pred.shape[-1])))
        loss.backward()
        self.optimizer.step()

        if self.debug:
            if self.debug_input:
                self.inspect_input(input_data)
            if self.debug_labels:
                self.inspect_labels(pred, label, acc_1)

        return loss.item(), acc_1, acc_5

    def test(self, data_dict, **kwargs):
        input_data = data_dict['input_data']
        label = data_dict['label']

        self.model_container.set_eval(['model'])

        if self.use_cuda:
            input_data, label = input_data.to(self.devices[0]), label.to(self.devices[0])

        pred = self.model_container.infer('model', input_data, True)
        tot_num = len(pred)

        acc_1, acc_5 = accuracy(pred.cpu(), label.cpu(), topk=(1, min(5, pred.shape[-1])))

        num_correct = int(acc_1 * tot_num)

        if self.debug:
            if self.debug_input:
                self.inspect_input(input_data)
            if self.debug_labels:
                self.inspect_labels(pred, label, acc_1)
        
        return acc_1, acc_5, num_correct, tot_num

    def save_model(self, total_epoch, total_iter, total_val_iter, **kwargs):
        """
        Save model on self.exp_save_dir.

        Args:
            total_epoch: Current total epoch
            total_iter: Current total number of iterations
            total_val_iter: Current total number of validation iterations
        """
        
        save_mode = self.cfg.save_by
        multi_gpu = torch.cuda.device_count() >= 1 and self.cfg.parallel

        model = self.model_container.models['model']
        if multi_gpu:
            if save_mode == 'iter':
                if total_iter % self.cfg.save_every == 0:
                    save_dir = self.exp_save_dir / 'model_log' / f'checkpoint_epoch_{total_epoch}_iter_{total_iter}_val_{total_val_iter}.tar'
                    torch.save({
                        'iter': total_iter,
                        'state_dict': model.module.state_dict(),
                    }, save_dir)

            elif save_mode == 'epoch':
                if total_epoch % self.cfg.save_every == 0:
                    save_dir = self.exp_save_dir / 'model_log' / f'checkpoint_epoch_{total_epoch}_iter_{total_iter}_val_{total_val_iter}.tar'
                    torch.save({
                        'epoch': total_epoch,
                        'state_dict': model.module.state_dict(),
                    }, save_dir)
        else:
            if save_mode == 'iter':
                if total_iter % self.cfg.save_every == 0:
                    save_dir = self.exp_save_dir / 'model_log' / f'checkpoint_epoch_{total_epoch}_iter_{total_iter}_val_{total_val_iter}.tar'
                    torch.save({
                        'iter': total_iter,
                        'state_dict': model.state_dict(),
                    }, save_dir)

            elif save_mode == 'epoch':
                if total_epoch % self.cfg.save_every == 0:
                    save_dir = self.exp_save_dir / 'model_log' / f'checkpoint_epoch_{total_epoch}_iter_{total_iter}_val_{total_val_iter}.tar'
                    torch.save({
                        'epoch': total_epoch,
                        'state_dict': model.state_dict(),
                    }, save_dir)
    
    def prep_train(self):
        """
        Funcion called before training begins. Auxiliary function for initializng different training 
        configurations such as model parallelism and weight freezing. 
        """
        # Prepare for data parallelism
        self.model_container.load_saved()
        self.model_container.parallelize(['model'], self.devices)
        self.model_container.print_model_size(['model'])

    def inspect_labels(self, pred, label, acc_1):
        # Debugging utility for inspecting ground truth / predicted labels
        thres = getattr(self.cfg, 'acc_threshold', 0.1)
        if acc_1 < thres:
            unq, cnt = torch.unique(pred.argmax(-1), return_counts=True)
            print("GT label: " + self.data_container.dataset[self.cfg.mode].labels[label[0]] + f" {label[0]}")
            print("Most predicted: " + self.data_container.dataset[self.cfg.mode].labels[unq[cnt.argmax()]] + f" {unq[cnt.argmax()]}")
        
    def inspect_input(self, input_data):
        inspect_channel = getattr(self.cfg, 'inspect_channel', 0)
        inspect_index = getattr(self.cfg, 'inspect_index', 0)

        if inspect_index == 'random':
            inspect_index = random.randint(0, len(input_data) - 1)

        # Debugging utility for visualizing input data
        if type(inspect_channel) is int:
            tmp = input_data[inspect_index].permute(1, 2, 0)[:, :, inspect_channel]
            plt.imshow(255 * tmp.cpu().numpy())
            plt.show()

        if inspect_channel == 'all':
            fig = plt.figure(figsize=(50, 50))
      
            tmp = input_data[inspect_index].permute(1, 2, 0)
            num_channel = tmp.shape[-1]

            for i in range(num_channel):
                fig.add_subplot(num_channel // 2, 2, i + 1)
                plt.imshow(255 * tmp[..., i].cpu().numpy())

            plt.show()
        
        if inspect_channel == 'color':
            vis_data = input_data[inspect_index].permute(1, 2, 0)
            pos_ev = (vis_data[..., 0] != 0)
            neg_ev = (vis_data[..., 1] != 0)
            vis_img = torch.ones([vis_data.shape[0], vis_data.shape[1], 3], device=vis_data.device)
            vis_img[pos_ev] = torch.tensor([[1., 0, 0]], device=vis_data.device)
            vis_img[neg_ev] = torch.tensor([[0, 0, 1.]], device=vis_data.device)

            plt.imshow(vis_img.cpu())
            plt.show()
        
        if inspect_channel == 'color_chn':
            vis_data = input_data[inspect_index].permute(1, 2, 0)
            pos_ev = (vis_data[..., 0] != 0)
            neg_ev = (vis_data[..., 1] != 0)
            pos_img = torch.ones([vis_data.shape[0], vis_data.shape[1], 3], device=vis_data.device)
            pos_img[pos_ev] = torch.tensor([[1., 0, 0]], device=vis_data.device)
            neg_img = torch.ones([vis_data.shape[0], vis_data.shape[1], 3], device=vis_data.device)
            neg_img[neg_ev] = torch.tensor([[0, 0, 1.]], device=vis_data.device)

            fig = plt.figure(figsize=(50, 50))
            fig.add_subplot(1, 2, 1)
            plt.imshow(pos_img.cpu().numpy())
            fig.add_subplot(1, 2, 2)
            plt.imshow(neg_img.cpu().numpy())
            plt.show()


class EvTTACNNTrainer(CNNTrainer):
    def __init__(self, cfg, model_container: ModelContainer, data_container: DataContainer, **kwargs):
        super(EvTTACNNTrainer, self).__init__(cfg, model_container, data_container)
        print(f'Initializing trainer {self.__class__.__name__}...')
        self.loss_type = getattr(cfg, 'loss_type', 'entropy')
        if self.loss_type == 'entropy':
            self.loss_func = SoftCrossEntropy(1.)
        self.entropy_func = SoftCrossEntropy(1.)
        self.cos_loss_func = nn.CosineSimilarity(dim=-1)
        self.kl_loss_func = KLDivLogits()
        self.kl_distrib_loss_func = KLDivLogits(tgt_distrib=True)
        self.kl_log_loss_func = KLDivLogits(input_distrib=True)
        self.info_rad_loss_func = InfoRadiusLoss()
        if getattr(self.cfg, 'print_consistent', False):
            self.avg_consistent = 0.0
            self.counter = 0
        if getattr(self.cfg, 'num_train_samples', None) is not None:
            self.num_tot_train = len(self.data_container.dataset['train'])
            self.sample_counter = 0
        self.threshold_confidence = getattr(self.cfg, 'threshold_confidence', False)
        self.sigma_thres = getattr(cfg, 'sigma_thres', 0.25)

    def z_test_remove(self, input_data, augment_input_data = None):
        pos_count = (input_data[:, 0] != 0).flatten(1, 2).sum(-1)
        neg_count = (input_data[:, 1] != 0).flatten(1, 2).sum(-1)
        trans_val = self.data_container.dataset['train'].geary_trans(pos_count / neg_count)
        trans_mean = trans_val.mean()
        trans_std = trans_val.std()
        
        sigma = self.sigma_thres
        right_z_val = (trans_mean - (-sigma)) / (trans_std / (len(trans_val) ** (1 / 2)))
        right_p_val = self.data_container.dataset['train'].p_value(right_z_val)

        left_z_val = (trans_mean - (sigma)) / (trans_std / (len(trans_val) ** (1 / 2)))
        left_p_val = self.data_container.dataset['train'].p_value(left_z_val)

        if right_p_val < 0.1:
            input_data = mutual_remove(input_data, 0, 1)
        elif left_p_val > 0.9:
            input_data = mutual_remove(input_data, 1, 0)

        if augment_input_data is not None:
            return input_data, augment_input_data
        else:
            return input_data

    def train(self, data_dict, **kwargs):
        input_data = data_dict['input_data']
        label = data_dict['label']
        augment_input_data = data_dict['augment_input_data']
        
        self.model_container.set_train(['model'])

        if self.use_cuda:
            input_data, label = input_data.to(self.devices[0]), label.to(self.devices[0])
            augment_input_data = augment_input_data.to(self.devices[0])
        
        if getattr(self.cfg, 'z_test_channel', False):
            input_data, augment_input_data = self.z_test_remove(input_data, augment_input_data)

        self.optimizer.zero_grad()

        total_pred = self.model_container.infer('model', torch.cat([input_data, augment_input_data], dim=0), False)

        label_pred = total_pred[:input_data.shape[0]]  # (B, L)
        label_augment_pred = total_pred[input_data.shape[0]:]  # (K * B, L)

        label_augment_pred = label_augment_pred.reshape(input_data.shape[0], getattr(self.cfg, 'num_sentry_augment', 3), -1)  # (B, K, L)
        label_idx = label_pred.argmax(-1).unsqueeze(-1)  # (B, 1)
        label_augment_idx = label_augment_pred.argmax(-1)  # (B, K)
        consistent = ((label_augment_idx == label_idx).sum(-1) / float(getattr(self.cfg, 'num_sentry_augment', 3)) > 0.5)  # (B, )
        
        total_pred = torch.cat([label_pred.unsqueeze(1), label_augment_pred], dim=1) # (B, K + 1, L)
        anchor_pred = label_pred
        
        # Optimize entropy of anchor_pred
        loss = self.loss_func(anchor_pred, anchor_pred, return_vector=True)
        loss[~consistent] *= getattr(self.cfg, 'sentry_inconsistent_fill', -1)
        loss = loss[loss != 0.].mean()

        # Set indices to compare with anchor
        other_idx = torch.stack([torch.arange(getattr(self.cfg, 'num_sentry_augment', 3)) + 1] * input_data.shape[0], dim=0)

        for idx in range(getattr(self.cfg, 'num_sentry_augment', 3)):
            tgt_pred_1 = total_pred[torch.arange(input_data.shape[0]), other_idx[:, idx], ...]
            tgt_pred_2 = anchor_pred

            if getattr(self.cfg, 'detach_tgt', False):  # Optionally detach gradients
                tgt_pred_1 = tgt_pred_1.detach()
                tgt_pred_2 = tgt_pred_2.detach()

            mutual_loss_type = getattr(self.cfg, 'mutual_loss_type', 'kl')
            loss += 0.5 * self.kl_loss_func(anchor_pred, tgt_pred_1) \
            + 0.5 * self.kl_loss_func(total_pred[torch.arange(input_data.shape[0]), other_idx[:, idx], ...], tgt_pred_2)

            # Target used for entropy loss
            entropy_tgt_pred = total_pred[torch.arange(input_data.shape[0]), other_idx[:, idx], ...]

            entropy_loss = self.loss_func(entropy_tgt_pred, entropy_tgt_pred, return_vector=True)
            entropy_loss[~consistent] *= getattr(self.cfg, 'sentry_inconsistent_fill', -1)
            loss += entropy_loss[entropy_loss != 0.].mean()

        tot_num = len(label_pred)

        loss.backward()
        self.optimizer.step()
        
        acc_1, acc_5 = accuracy(label_pred.cpu(), label.cpu(), topk=(1, min(5, label_pred.shape[-1])))
        num_correct = int(acc_1 * tot_num)

        return loss.item(), acc_1, acc_5, num_correct, tot_num

    def test(self, data_dict, **kwargs):
        input_data = data_dict['input_data']
        label = data_dict['label']

        self.model_container.set_eval(['model'])

        if self.use_cuda:
            input_data, label = input_data.to(self.devices[0]), label.to(self.devices[0])
        
        if getattr(self.cfg, 'z_test_channel', False):
            input_data = self.z_test_remove(input_data)

        pred = self.model_container.infer('model', input_data, True)
        tot_num = len(pred)

        acc_1, acc_5 = accuracy(pred.cpu(), label.cpu(), topk=(1, min(5, pred.shape[-1])))

        num_correct = int(acc_1 * tot_num)

        if self.debug:
            if self.debug_input:
                self.inspect_input(input_data)
            if self.debug_labels:
                self.inspect_labels(pred, label, acc_1)
        
        return acc_1, acc_5, num_correct, tot_num

    def run_epoch(self):
        # Train
        print(f"This is {self.tracker.epoch}-th epoch.")
        
        self.tracker.init_epoch(mode='train')
        
        skip_train = getattr(self.cfg, 'skip_train', False)

        if not skip_train:
            self.train_epoch()

            print(f"Epoch {self.tracker.epoch} training loss = {self.tracker.total_train_loss.get_avg():.4f}")
            print(f"Total training accuracy = {(self.tracker.get_train_acc()):.4f}")

        # Validate
        print(f"Epoch {self.tracker.epoch} validation!")

        self.tracker.init_epoch(mode='val')
        self.validate_epoch()
        print(f"Total validation accuracy = {(self.tracker.get_val_acc()):.4f}")
        
        # Update scheduler
        if self.scheduler_type == 'reduce_plateau':
            self.scheduler.step(self.tracker.get_val_acc())
        elif self.scheduler_type is None:
            pass
        else:
            self.scheduler.step()
        
        # Save model by epoch (depends on cfg)
        if self.cfg.save_by == 'epoch':
            self.save_model(self.tracker.epoch, self.tracker.total_iter, self.tracker.total_val_iter)

    def train_batch(self, data_dict):
        self.tracker.init_batch(mode='train')
        self.tracker.start_infer_timing()

        loss, acc_1, acc_5, train_correct, train_num = self.train(data_dict)
        self.tracker.end_infer_timing()
        
        write_dict = {'training loss': loss, 'top 1 training accuracy': acc_1, 'top 5 training accuracy': acc_5, 'load time': self.tracker.load_time}
        print_dict = {'Iter': self.tracker.batch_idx, 'training loss': loss, 'top 1 training acc': acc_1, 'top 5 training acc': acc_5,
        'infer time': self.tracker.infer_time, 'load time': self.tracker.load_time}
        self.print_state(print_dict)
        self.write(self.tracker.total_iter, write_dict)
        
        self.tracker.total_train_loss.update(loss)
        self.tracker.total_train_correct.accumulate(train_correct)
        self.tracker.total_train_num.accumulate(train_num)        

        if self.cfg.save_by == 'iter':
            self.save_model(self.tracker.epoch, self.tracker.total_iter, self.tracker.total_val_iter)

    def init_optimizer(self, **kwargs):
        """
        Initialize self.optimizer using self.cfg.
        """

        # Choose optimizer
        model = self.model_container.models['model']
        opt_type = self.cfg.optimizer
        freeze = getattr(self.cfg, 'freeze', False) or getattr(self.cfg, 'train_classifier', False)

        if opt_type == 'SGD':
            print('Using SGD as optimizer')
            if freeze:
                print('Freezing weights!')
                self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=self.cfg.learning_rate, momentum=self.cfg.momentum, weight_decay=self.cfg.weight_decay)
            else:
                self.optimizer = optim.SGD(model.parameters(), lr=self.cfg.learning_rate, momentum=self.cfg.momentum, weight_decay=self.cfg.weight_decay)
        elif opt_type == 'Adam':
            print('Using Adam as optimizer')
            if freeze:
                print('Freezing weights!')
                self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
            else:
                self.optimizer = optim.Adam(model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
