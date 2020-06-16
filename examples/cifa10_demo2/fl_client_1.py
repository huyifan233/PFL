import os
import _thread
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from pfl.core.client import FLClient
from pfl.core.strategy import WorkModeStrategy, TrainStrategy, LossStrategy
from pfl.core.trainer_controller import TrainerController

CLIENT_ID = 1


def save_lcoal_model_parameters(local_model_parameters, local_model_parameters_path):
    torch.save(local_model_parameters, local_model_parameters_path)
    print("保存本地模型参数成功")


def train_local_model_with_local_data(local_model, mnist_data, epoch):
    dataloader = torch.utils.data.DataLoader(mnist_data, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    optimizer = torch.optim.SGD(local_model.parameters(), lr=0.001, momentum=0.9)
    local_step = 0
    local_model_parameters_dir = os.path.join(os.path.abspath("."), "client_{}_model_parameter_dir".format(CLIENT_ID))
    if not os.path.exists(local_model_parameters_dir):
        os.mkdir(local_model_parameters_dir)

    for idx in range(epoch):
        for idx, (batch_data, batch_target) in enumerate(dataloader):
            # print(batch_data.shape)
            # print(batch_target.shape)
            # if (idx % 500) == 0:
            #     local_model_paramaters_path = os.path.join(local_model_parameters_dir, "paramaters_{}".format(local_step))
            #     save_lcoal_model_parameters(local_model.state_dict(), local_model_paramaters_path)
            #     local_step += 1
            preds = local_model(batch_data)
            preds_softmax = F.log_softmax(preds, dim=1)
            loss = F.nll_loss(preds_softmax, batch_target)
            cls_pred_softmax = torch.argmax(preds_softmax, dim=1)
            acc = torch.mean(torch.eq(cls_pred_softmax, batch_target), dtype=torch.float32)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("local_step: {}, local_loss: {}, local_acc: {}".format(local_step, loss.item(), acc.item()))
        local_model_paramaters_path = os.path.join(local_model_parameters_dir, "paramaters_{}".format(local_step))
        save_lcoal_model_parameters(local_model.state_dict(), local_model_paramaters_path)
        local_step += 1


if __name__ == "__main__":
    # CLIENT_ID = int(sys.argv[1])

    dataset_path = os.path.join(os.path.abspath("../"), "cifa10_demo", "cifa10", "train_dataset_dir",
                                "train_dataset_{}".format(CLIENT_ID))

    dataset = torch.load(dataset_path)

    client = FLClient()
    pfl_models = client.get_remote_pfl_models()

    local_model = client.get_latest_local_model(CLIENT_ID)
    _thread.start_new_thread(train_local_model_with_local_data, (local_model, dataset, 1300))
    # local_pfl_models = client.get_remote_pfl_models()
    # local_model = local_pfl_models[0].get_model()
    # _thread.start_new_thread(train_local_model_with_local_data, (local_model, dataset, 50))
    # train_local_model_with_local_data(local_model, dataset, 100)

    for pfl_model in pfl_models:
        optimizer = torch.optim.SGD(pfl_model.get_model().parameters(), lr=0.001, momentum=0.9)
        train_strategy = TrainStrategy(optimizer=optimizer, batch_size=32, loss_function=LossStrategy.NLL_LOSS)
        pfl_model.set_train_strategy(train_strategy)

    TrainerController(work_mode=WorkModeStrategy.WORKMODE_STANDALONE, models=pfl_models, data=dataset, client_id=CLIENT_ID,
                      curve=False, concurrent_num=3).start()