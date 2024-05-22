import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, student_n, exer_n, knowledge_n):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.affect_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512,256 

        super(Net, self).__init__()

        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.student_affect = nn.Embedding(self.emb_num, self.affect_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_discrimination = nn.Embedding(self.exer_n, 1)
        self.prednet_affect = nn.Linear(
            self.knowledge_dim + self.affect_dim, 4)
        self.guess = nn.Linear(4, 1)
        self.slip = nn.Linear(4, 1)

        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)

        # initialization    
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kn_emb):
        stu_emb = torch.sigmoid(self.student_emb(stu_id))
        stu_affect = torch.sigmoid(self.student_affect(stu_id))
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        affect = torch.sigmoid(self.prednet_affect(torch.cat((stu_affect, k_difficulty), dim=1)))
        e_discrimination = torch.sigmoid(
            self.e_discrimination(exer_id)) * 10
        input_x = e_discrimination * (stu_emb - k_difficulty) * kn_emb 
        input_x = torch.sigmoid(self.prednet_full1(input_x))
        input_x = torch.sigmoid(self.prednet_full2(input_x))
        o = torch.sigmoid(self.prednet_full3(input_x))
        g = torch.sigmoid(self.guess(affect))
        s = torch.sigmoid(self.slip(affect))
        output = ((1-s)*o) + (g*(1-o))
        return output, affect

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

class NoneNegClipper(object):  # 限制w为正，满足单调性假设
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):  # w = relu(w)
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)
