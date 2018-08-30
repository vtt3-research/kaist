import torch
import torch.nn.functional as F

def P_ab(input_a, input_b):
    match_ab = torch.matmul(input_a, input_b.transpose(0,1))
    p_ab = F.softmax(match_ab.transpose(0, 1)).transpose(0, 1)
    return p_ab

def P_aba(input_a, input_b):
    match_ab = torch.matmul(input_a, input_b.transpose(0,1))
    p_ab = F.softmax(match_ab.transpose(0,1)).transpose(0,1)
    p_ba = F.softmax(match_ab).transpose(0,1)
    p_aba = torch.matmul(p_ab, p_ba)
    return p_aba

def equality_matrix(labels, num_classes):
    label_num = labels.__len__()
    count = torch.zeros(num_classes)
    A = [[] for x in range(0,num_classes)]
    for i in range(0,label_num):
        A[labels[i]].append(i)
        count[labels[i]] += 1

    eq_mat = torch.zeros(label_num, label_num)
    for x in range(0,num_classes):
        if not count[x] == 0:
            for i in A[x]:
                for j in A[x]:
                    eq_mat[i][j] = 1/count[x]

    return eq_mat