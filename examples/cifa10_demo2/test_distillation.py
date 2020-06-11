import os
import torch

def main():
    dataset_path = os.path.join(os.path.abspath("../"), "cifa10_demo", "cifa10", "train_dataset_dir",
                                "train_dataset_{}".format(0))

    dataset = torch.load(dataset_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    optimizer = torch.optim.SGD(local_model.parameters(), lr=0.001, momentum=0.9)
    local_step = 0
    local_model_parameters_dir = os.path.join(os.path.abspath("."), "client_{}_model_parameter_dir".format(CLIENT_ID))
    if not os.path.exists(local_model_parameters_dir):
        os.mkdir(local_model_parameters_dir)

    # for idx in range(epoch):
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
        # local_model_paramaters_path = os.path.join(local_model_parameters_dir, "paramaters_{}".format(local_step))
        # save_lcoal_model_parameters(local_model.state_dict(), local_model_paramaters_path)


if __name__ == "__main__":
    main()