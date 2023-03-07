import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy
from torchmetrics import Accuracy

import pytorch_lightning as pl
import torchvision.models as models


class ResnetClassification(pl.LightningModule):
    """
    
    Our primary model class this has been hardcoded to Resnet50 for now
    
    """
    def __init__(self, channels, width, height, num_classes, learning_rate=2e-4):
        super().__init__()
        self.channels = channels
        self.width = width
        self.height = height
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.loss = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.valid_acc = Accuracy(task='multiclass', num_classes=num_classes)

        
        self.model_tag = 'resnet'

        self.model = models.resnet50(weights=None)

        ### TRANSFER LEARNING STEPS
        # change the final layer
        linear_size = list(self.model.children())[-1].in_features
        self.model.fc = nn.Linear(linear_size, self.num_classes)

        # change the input channels as needed
        if self.channels != 3:
            self.model.conv1 = nn.Conv2d(self.channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        ### END TRANSFER LEARNING STEPS


    def forward(self, x):
        x = self.model(x)
        return x

    # See https://torchmetrics.readthedocs.io/en/latest/pages/overview.html#metrics-and-devices
    # for how to do metrics properly
    def training_step(self, batch, batch_idx):

        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss(logits, y)
        #acc = self.train_acc(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        #self.log("train_acc", acc, prog_bar=True)

        return {'loss': loss, 'pred': preds, 'logits': logits, 'target': y}

    #### for dp only

    def training_step_end(self, outputs):

        acc = self.train_acc(outputs['pred'], outputs['target'])
        self.log("train_acc", acc, prog_bar=True)

    ####

    def training_epoch_end(self, outputs):
        # loss = self.loss(outputs['logits'], outputs['target'])
        # acc = self.train_acc(outputs['pred'], outputs['target'])
        # self.log("train_loss", loss, prog_bar=True)
        # self.log("train_acc", acc, prog_bar=True)
        self.train_acc.reset()
        #return {'loss': loss}
        

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        #self.valid_acc.update(logits, y)

        return {'loss': loss, 'pred': preds, 'logits': logits, 'target': y}
    
    #### for dp only

    def validation_step_end(self, outputs):

        self.valid_acc.update(outputs['pred'], outputs['target'])
        #acc = self.valid_acc(outputs['preds'], outputs['target'])
        #self.log("valid_acc", acc, prog_bar=True)
        
    #### end dp code

    def validation_epoch_end(self, outputs):
        self.log('valid_acc_epoch', self.valid_acc.compute())
        self.valid_acc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

