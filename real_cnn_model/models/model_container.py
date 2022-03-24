from torch.serialization import load
import torch.nn as nn
import torch
from base.models.model_container import ModelContainer
from real_cnn_model.models.enhanced_classifier import EnhancedClassifier
import os
import glob


class EvTTACNNContainer(ModelContainer):
    def __init__(self, cfg, **kwargs):
        super(EvTTACNNContainer, self).__init__(cfg)
        print(f'Initializing model container {self.__class__.__name__}...')
        self.gen_model()

    def gen_model(self):
        """
        Generate models for self.models
        """
        if getattr(self.cfg, 'use_tent', False):
            from real_cnn_model.models.resnet_tent import resnet18, resnet34, resnet50, resnet101, resnet152
        else:
            from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

        # Setup classifier
        use_pretrained = getattr(self.cfg, 'pretrained', False)
        num_classes = getattr(self.cfg, 'num_classes', 1000)
        pretrained_num_classes = getattr(self.cfg, 'pretrained_num_classes', 1000)
        classifier_name = getattr(self.cfg, 'classifier', 'ResNet34')

        # Choose between models
        if classifier_name == 'ResNet18':
            classifier = resnet18(pretrained=use_pretrained, num_classes=pretrained_num_classes)
        elif classifier_name == 'ResNet34':
            classifier = resnet34(pretrained=use_pretrained, num_classes=pretrained_num_classes)
        elif classifier_name == 'ResNet50':
            classifier = resnet50(pretrained=use_pretrained, num_classes=pretrained_num_classes)
        elif classifier_name == 'ResNet101':
            classifier = resnet101(pretrained=use_pretrained, num_classes=pretrained_num_classes)
        elif classifier_name == 'ResNet152':
            classifier = resnet152(pretrained=use_pretrained, num_classes=pretrained_num_classes)
        else:
            raise AttributeError('Invalid classifier name')

        if num_classes != pretrained_num_classes:
            classifier.fc = nn.Linear(512, num_classes)

        # Get channel size
        channels = getattr(self.cfg, 'channel_size', 4)
        kernel_size = getattr(self.cfg, 'kernel_size', 14)
        # Adapt classifier for 4 channels
        if 'ResNet' in classifier_name:
            classifier.conv1 = nn.Conv2d(channels, 64, kernel_size=kernel_size, stride=2, padding=3, bias=False)

        # Layer-specific training
        freeze_except_front = getattr(self.cfg, 'freeze_except_front', False)
        
        if freeze_except_front:
            print("Training only front layers!")
            for param in classifier.parameters():
                param.requires_grad = False
            for param in classifier.conv1.parameters():
                param.requires_grad = True
            for param in classifier.layer1.parameters():
                param.requires_grad = True

        freeze_except_fc = getattr(self.cfg, 'freeze_except_fc', False)

        if freeze_except_fc:
            print("Training only last fc!")
            for param in classifier.parameters():
                param.requires_grad = False
            for param in classifier.fc.parameters():
                param.requires_grad = True

        freeze_classifier = getattr(self.cfg, 'freeze_classifier', False)

        if freeze_classifier:
            print("Training only enhancer!")
            for param in classifier.parameters():
                param.requires_grad = False

        freeze_except_bn = getattr(self.cfg, 'freeze_except_bn', False)

        if freeze_except_bn:
            print("Training only batch norm!")
            for layer in classifier.modules():
                if not isinstance(layer, torch.nn.modules.batchnorm.BatchNorm2d):
                    for param in layer.parameters():
                        param.requires_grad = False
                else:
                    for param in layer.parameters():
                        param.requires_grad = True

        print(f'Using {classifier_name} as backbone classifier.')

        # Set up enhancement module
        enhancer = None

        model = EnhancedClassifier(classifier, enhancer, return_input=getattr(self.cfg, 'return_input', False))

        self.models['model'] = model

    def load_saved(self):
        # Load pretrained weights
        pretrained_num_classes = getattr(self.cfg, 'pretrained_num_classes', None)

        # Loading configs
        load_enhancer = getattr(self.cfg, 'load_enhancer', False)
        load_classifier = getattr(self.cfg, 'load_classifier', False)

        if pretrained_num_classes is not None:
            self.models['model'].classifier.fc = nn.Linear(512, pretrained_num_classes)

        if self.cfg.load_model is not None:
            if os.path.isfile(self.cfg.load_model):  # If input is file
                load_file = self.cfg.load_model
            elif os.path.isdir(self.cfg.load_model):  # If input is directory
                target_file = self.cfg.target_file  # Specify which file to load by specifying keyword in file
                ckpt_list = glob.glob(os.path.join(self.cfg.load_model, '**/*.tar'))
                load_file = list(filter(lambda x: target_file in x, ckpt_list))[0]

            print(f"Loading model from {load_file}")
            state_dict = torch.load(load_file)

            if any(['enhancer' in key for key in state_dict['state_dict'].keys()]):  # Loading model from EnhancedClassifier with both modules
                if load_enhancer and load_classifier:
                    self.models['model'].load_state_dict(state_dict['state_dict'])
                elif load_enhancer and not load_classifier:
                    self.models['model'].enhancer.load_state_dict(filter(lambda x: 'enhancer' in x[0], state_dict['state_dict'].items()))
                elif load_classifier and not load_enhancer:
                    self.models['model'].classifier.load_state_dict(filter(lambda x: 'classifier' in x[0], state_dict['state_dict'].items()))
            
            elif any(['classifier' in key for key in state_dict['state_dict'].keys()]):  # Loading model from EnhancedClassifier with only classifier
                self.models['model'].load_state_dict(state_dict['state_dict'], strict=False)

            else:  # Loading model from existing classifiers
                self.models['model'].classifier.load_state_dict(state_dict['state_dict'], strict=False)
            
            # Set BN layer's stats to zero
            if getattr(self.cfg, 'freeze_except_bn', False) and self.cfg.mode == 'train':
                for layer in self.models['model'].classifier.modules():
                    if isinstance(layer, torch.nn.modules.batchnorm.BatchNorm2d):
                        layer.reset_running_stats()
        if self.cfg.mode != 'test':
            num_classes = getattr(self.cfg, 'num_classes', 1000)
            keep_fc = getattr(self.cfg, 'keep_fc', True)

            if not keep_fc:
                self.models['model'].classifier.fc = nn.Linear(512, num_classes)

