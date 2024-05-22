import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from sklearn.metrics import roc_auc_score
from data_loader import TrainDataLoader, ValTestDataLoader
from model import Net

exer_n = 3162
knowledge_n = 102
student_n = 1709

device = torch.device(('cuda:0') if torch.cuda.is_available() else 'cpu')
epoch_n = 200


def train():
    data_loader = TrainDataLoader()
    net = Net(student_n, exer_n, knowledge_n)
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    print('training model...')
    affect_loss_function = nn.MSELoss()
    loss_function = nn.NLLLoss()
    for epoch in range(epoch_n):
        net.train()
        data_loader.reset()
        running_loss = 0.0
        batch_count = 0
        while not data_loader.is_end():
            batch_count += 1
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels, affects = data_loader.next_batch()
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels, affects = input_stu_ids.to(device), input_exer_ids.to(device), input_knowledge_embs.to(device), labels.to(device),affects.to(device)
            optimizer.zero_grad()
            output_1, affect_p = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
            output_0 = torch.ones(output_1.size()).to(device) - output_1
            output = torch.cat((output_0, output_1), 1)

            loss = loss_function(torch.log(output), labels) + 1 * affect_loss_function(affect_p, affects)
            loss.backward()  
            optimizer.step()
            net.apply_clipper()  # 单调性假设

            running_loss += loss.item()
            if batch_count % 200 == 199:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_count + 1, running_loss / 200))
                running_loss = 0.0

        rmse, auc = validate(net, epoch)
        save_snapshot(net, 'model/model_epoch' + str(epoch + 1))


def validate(model, epoch):
    data_loader = ValTestDataLoader('validation')
    net = Net(student_n, exer_n, knowledge_n)
    print('validating model...')
    data_loader.reset()
    # load model parameters
    net.load_state_dict(model.state_dict())
    net = net.to(device)
    net.eval()

    correct_count, exer_count = 0, 0
    batch_count = 0
    pred_all, label_all = [], []
    while not data_loader.is_end():
        batch_count += 1
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels, affects = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels, affects = input_stu_ids.to(device), input_exer_ids.to(device), input_knowledge_embs.to(device), labels.to(device),affects.to(device)
        output, _= net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
        output = output.view(-1)
        for i in range(len(labels)):
            if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                correct_count += 1
        exer_count += len(labels)
        pred_all += output.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    # compute accuracy
    accuracy = correct_count / exer_count
    # compute RMSE
    rmse = np.sqrt(np.mean((label_all - pred_all)**2))
    # compute AUC
    auc = roc_auc_score(label_all, pred_all)

    print('epoch= %d, accuracy= %f, rmse= %f, auc= %f' % (epoch, accuracy, rmse, auc))
    with open('result/model_val.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f\n' % (epoch, accuracy, rmse, auc))

    return rmse, auc


def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()


if __name__ == '__main__':
    if len(sys.argv) != 1:
        if (len(sys.argv) != 3) or ((sys.argv[1] != 'cpu') and ('cuda:' not in sys.argv[1])) or (not sys.argv[2].isdigit()):
            print('command:\n\tpython train.py {device} {epoch}\nexample:\n\tpython train.py cuda:0 70')
            exit(1)
        else:
            device = torch.device(sys.argv[1])
            epoch_n = int(sys.argv[2])


    with open('config.txt') as i_f:
        i_f.readline()
        student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))
    while True:
        train()
