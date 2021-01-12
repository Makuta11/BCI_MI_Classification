import numpy as np
import pandas as pd
import torch
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def train_model(model, optimizer, criterion, num_epochs, train_dataloader, val_dataloader, device,
                scheduler = None):

    loss_collect = []
    val_loss_collect = []

    for epoch in range(num_epochs):
        running_loss = 0
        model.train()
        for i, x in enumerate(train_dataloader):
            x_data = x[0].float().to(device)
            label = x[1].long().to(device)

            optimizer.zero_grad()
            out = model(x_data)
            del x_data
            torch.cuda.empty_cache()

            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.detach().cpu()

            del label, loss 
            torch.cuda.empty_cache()


        loss_collect = np.append(loss_collect, running_loss/(i+1))


        # get validation loss
        model.eval()
        val_loss = 0 
        for i, x in enumerate(val_dataloader):
            temp_data = x[0].float().to(device)
            temp_label = x[1].long().to(device)

            out = model(temp_data)
            val_loss += criterion(out, temp_label)

            del temp_data, temp_label
            torch.cuda.empty_cache()

        if scheduler:
            scheduler.step()

        val_loss_collect = np.append(val_loss_collect, val_loss.cpu()/(i+1))

        print(str(epoch + 1) + ' out of ' + str(num_epochs))
        print(loss_collect[epoch])
        print(val_loss_collect[epoch])

    return loss_collect, val_loss_collect, model
