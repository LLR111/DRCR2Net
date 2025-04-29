from network import modul
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataset import dadaset as ddpd
from torch.utils.data import DataLoader, random_split

np.random.seed()
torch.manual_seed()
if torch.cuda.is_available():
    torch.cuda.manual_seed_all()

img_file = ""
mask_file = ""

dataset = ddpd.TrainDataset(img_file, mask_file)
train_data, test_data = random_split(dataset, [9632, 1204], generator=torch.Generator().manual_seed())
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
test_loader = DataLoader(test_data, batch_size=4, shuffle=True)
print("data loaded")

model = modul.NET().cuda()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005,weight_decay=1e-4)
epoch = 100

def test() :
    test_loss_temp = 0
    with torch.no_grad():
        model.eval()
        for images, labels in test_loader:
            input = images.to("cuda")
            label = labels.to("cuda")
            final_map = model(input)
            loss_final = criterion(final_map, label)
            total_loss = loss_final

            test_loss_temp += total_loss.item()

    test_loss_epoch = test_loss_temp / len(test_loader)
    print('[%d] Test Loss: %.3f' % (epoch, test_loss_epoch))
    return test_loss_epoch

for epoch in range(1,epoch):

    model.train()
    running_loss = 0.0
    total_loss_temp = 0

    for i,(img,label) in enumerate(train_loader):
        input = img.to("cuda")
        label = label.to("cuda")
        optimizer.zero_grad()
        final_map = model(input)

        loss_final = criterion(final_map,label)
        total_loss = loss_final
        total_loss.backward()
        optimizer.step()
        running_loss = running_loss + total_loss.item()
        total_loss_temp = total_loss_temp + total_loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch, i + 1, running_loss / 100))
            running_loss = 0.0
    train_loss_epoch = total_loss_temp / len(train_loader)
    test_loss = test()
    print('[%d] train loss: %.3f' % (epoch, train_loss_epoch))
    torch.save(model.state_dict(),'./pretrained_model/model_dict.pth')
    torch.save(model, './pretrained_model/model.pt')

