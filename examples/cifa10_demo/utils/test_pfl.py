import os
import torch
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, 1)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # need to return logits
        return x


def test(dataset_name):
    test_dataset_path = os.path.join(os.path.abspath("../"), dataset_name, "test_dataset_dir", "test_dataset")
    test_dataset = torch.load(test_dataset_path)
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)

    model_pars_dir = os.path.join(os.path.abspath("../"), "final_model_pars")
    model_pars_list = os.listdir(model_pars_dir)
    total_avg_list = []
    for model_pars_name in model_pars_list:
        model_pars_path = os.path.join(model_pars_dir, model_pars_name)
        model = Net()
        model_pars = torch.load(model_pars_path)
        model.load_state_dict(model_pars)
        acc = 0
        for idx, (batch_data, batch_target) in enumerate(dataloader):
            preds = model(batch_data)
            preds_softmax = F.log_softmax(preds, dim=1)
            cls_pred_softmax = torch.argmax(preds_softmax, dim=1)
            # acc = torch.mean(torch.eq(cls_pred_softmax, batch_target), dtype=torch.float32)
            # acc_list.append(acc)
            acc += torch.eq(preds_softmax.argmax(dim=1), batch_target).sum().float().item()

        avg_acc = acc / len(test_dataset)
        total_avg_list.append(avg_acc)
    print(torch.mean(torch.Tensor(total_avg_list)))


if __name__ == "__main__":

    test("cifa10")