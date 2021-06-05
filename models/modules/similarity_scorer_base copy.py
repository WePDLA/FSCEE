#!/usr/bin/env python
import torch
from torch.nn import functional as F
from torch import autograd, optim, nn
import torchsnooper
from models.modules.GCN import GraphConvolution
def reps_dot(sent1_reps: torch.Tensor, sent2_reps: torch.Tensor) -> torch.Tensor:
    """
    calculate representation dot production
    :param sent1_reps: (N, sent1_len, reps_dim)
    :param sent2_reps: (N, sent2_len, reps_dim)
    :return: (N, sent1_len, sent2_len)
    """
    return torch.bmm(sent1_reps, torch.transpose(sent2_reps, -1, -2))  # shape: (N, seq_len1, seq_len2)


def reps_l2_sim(sent1_reps: torch.Tensor, sent2_reps: torch.Tensor) -> torch.Tensor:
    """
    calculate representation L2 similarity
    :param sent1_reps: (N, sent1_len, reps_dim)
    :param sent2_reps: (N, sent2_len, reps_dim)
    :return: (N, sent1_len, sent2_len)
    """
    sent1_len = sent1_reps.shape[-2]
    sent2_len = sent2_reps.shape[-2]

    expand_shape1 = list(sent2_reps.shape)
    expand_shape1.insert(2, sent2_len)
    expand_shape2 = list(sent2_reps.shape)
    expand_shape2.insert(1, sent1_len)

    # shape: (N, seq_len1, seq_len2, emb_dim)
    expand_reps1 = sent1_reps.unsqueeze(2).expand(expand_shape1)
    expand_reps2 = sent2_reps.unsqueeze(1).expand(expand_shape2)

    # shape: (N, seq_len1, seq_len2)
    sim = torch.norm(expand_reps1 - expand_reps2, dim=-1, p=2)
    return -sim  # we calculate similarity not distance here


def reps_cosine_sim(sent1_reps: torch.Tensor, sent2_reps: torch.Tensor) -> torch.Tensor:
    """
    calculate representation cosine similarity, note that this is different from torch version(that compute parwisely)
    :param sent1_reps: (N, sent1_len, reps_dim)
    :param sent2_reps: (N, sent2_len, reps_dim)
    :return: (N, sent1_len, sent2_len)
    """
    dot_sim = torch.bmm(sent1_reps, torch.transpose(sent2_reps, -1, -2))  # shape: (batch, seq_len1, seq_len2)
    sent1_reps_norm = torch.norm(sent1_reps, dim=-1, keepdim=True)  # shape: (batch, seq_len1, 1)
    sent2_reps_norm = torch.norm(sent2_reps, dim=-1, keepdim=True)  # shape: (batch, seq_len2, 1)
    norm_product = torch.bmm(sent1_reps_norm,
                             torch.transpose(sent2_reps_norm, -1, -2))  # shape: (batch, seq_len1, seq_len2)
    sim_predicts = dot_sim / norm_product  # shape: (batch, seq_len1, seq_len2)
    return sim_predicts

# def reps_linear(sent1_reps: torch.Tensor, sent2_reps: torch.Tensor) -> torch.Tensor:
#     """
#     calculate representation dot production
#     :param sent1_reps: (N, sent1_len, reps_dim)
#     :param sent2_reps: (N, sent2_len, reps_dim)
#     :return: (N, sent1_len, sent2_len)
#     """
#     return torch.bmm(sent1_reps, torch.transpose(sent2_reps, -1, -2))  # shape: (N, seq_len1, seq_len2)

class SimilarityScorerBase(torch.nn.Module):
    def __init__(self, sim_func, emb_log=None):
        super(SimilarityScorerBase, self).__init__()
        self.sim_func = sim_func
        self.emb_log = emb_log
        self.log_content = ''

    def forward(
            self,
            test_reps: torch.Tensor,
            support_reps: torch.Tensor,
            test_output_mask: torch.Tensor,
            support_output_mask: torch.Tensor,
            support_targets: torch.Tensor = None,
            label_reps: torch.Tensor = None, ) -> torch.Tensor:
        """
            :param test_reps: (batch_size, support_size, test_seq_len, dim)
            :param support_reps: (batch_size, support_size, support_seq_len)
            :param test_output_mask: (batch_size, test_seq_len)
            :param support_output_mask: (batch_size, support_size, support_seq_len)
            :param support_targets: one-hot label targets: (batch_size, support_size, support_seq_len, num_tags)
            :param label_reps: (batch_size, num_tags, dim)
            :return: similarity
        """
        raise NotImplementedError()

    def mask_sim(self, sim: torch.Tensor, mask1: torch.Tensor, mask2: torch.Tensor):
        """
        mask invalid similarity to 0, i.e. sim to pad token is 0 here.
        :param sim: similarity matrix (num sim, test_len, support_len)
        :param mask1: (num sim, test_len, support_len)
        :param mask2: (num sim, test_len, support_len)
        :param min_value: the minimum value for similarity score
        :return:
        """
        mask1 = mask1.unsqueeze(-1).float()  # (b * s, test_label_num, 1)
        mask2 = mask2.unsqueeze(-1).float()  # (b * s, support_label_num, 1)
        mask = reps_dot(mask1, mask2)  # (b * s, test_label_num, support_label_num)
        sim = sim * mask
        return sim

    def expand_it(self, item: torch.Tensor, support_size):
        item = item.unsqueeze(1)
        expand_shape = list(item.shape)
        expand_shape[1] = support_size
        return item.expand(expand_shape)


class MatchingSimilarityScorer(SimilarityScorerBase):
    def __init__(self, sim_func, emb_log=None):
        super(MatchingSimilarityScorer, self).__init__(sim_func=sim_func, emb_log=emb_log)

    def forward(
            self,
            test_reps: torch.Tensor,
            support_reps: torch.Tensor,
            test_output_mask: torch.Tensor,
            support_output_mask: torch.Tensor,
            support_targets: torch.Tensor = None,
            label_reps: torch.Tensor = None, ) -> torch.Tensor:
        """
            calculate similarity between test token and support tokens.
            :param test_reps: (batch_size, support_size, test_seq_len, dim)
            :param support_reps: (batch_size, support_size, support_seq_len)
            :param test_output_mask: (batch_size, test_seq_len)
            :param support_output_mask: (batch_size, support_size, support_seq_len)
            :param support_targets: one-hot label targets: (batch_size, support_size, support_seq_len, num_tags)
            :param label_reps: (batch_size, num_tags, dim)
            :return: similarity matrix: (batch_size, support_size, test_seq_len, support_seq_len)
        """
        support_size = support_reps.shape[1]
        test_len = test_reps.shape[-2]  # non-word-piece max test token num, Notice that it's different to input t len
        support_len = support_reps.shape[-2]  # non-word-piece max test token num
        emb_dim = test_reps.shape[-1]
        batch_size = test_reps.shape[0]
        # flatten representations to shape (batch_size * support_size, sent_len, emb_dim)
        test_reps = test_reps.view(-1, test_len, emb_dim)
        support_reps = support_reps.view(-1, support_len, emb_dim)

        # calculate dot product
        sim_score = self.sim_func(test_reps, support_reps)

        # the length in sc is `1` which is not same as sl
        test_mask = self.expand_it(test_output_mask, support_size).contiguous().view(batch_size * support_size, -1)
        support_mask = support_output_mask.contiguous().view(batch_size * support_size, -1)
        sim_score = self.mask_sim(sim_score, mask1=test_mask, mask2=support_mask)

        # reshape from (batch_size * support_size, test_len, support_len) to
        # (batch_size, support_size, test_len, support_len)
        sim_score = sim_score.view(batch_size, support_size, test_len, support_len)
        return sim_score


class PrototypeSimilarityScorer(SimilarityScorerBase):
    def __init__(self, sim_func, emb_log=None):
        super(PrototypeSimilarityScorer, self).__init__(sim_func=sim_func, emb_log=emb_log)

    def forward(
            self,
            test_reps: torch.Tensor,
            support_reps: torch.Tensor,
            test_output_mask: torch.Tensor,
            support_output_mask: torch.Tensor,
            support_targets: torch.Tensor = None,
            label_reps: torch.Tensor = None, ) -> torch.Tensor:
        """
            calculate similarity between token and each label's prototype.
            :param test_reps: (batch_size, support_size, test_seq_len, dim)
            :param support_reps: (batch_size, support_size, support_seq_len)
            :param test_output_mask: (batch_size, test_seq_len)
            :param support_output_mask: (batch_size, support_size, support_seq_len)
            :param support_targets: one-hot label targets: (batch_size, support_size, support_seq_len, num_tags)
            :param label_reps: (batch_size, num_tags, dim)
            :return: similarity: (batch_size, test_seq_len, num_tags)
        """
        support_size = support_reps.shape[1]
        test_len = test_reps.shape[-2]  # non-word-piece max test token num, Notice that it's different to input t len
        support_len = support_reps.shape[-2]  # non-word-piece max test token num or possible lb num for sc (fix `1`)
        emb_dim = test_reps.shape[-1]
        batch_size = test_reps.shape[0]
        num_tags = support_targets.shape[-1]

        # average test representation over support set (reps for each support sent can be different)
        test_reps = torch.mean(test_reps, dim=1)  # shape (batch_size, sent_len, emb_dim)
        # flatten dim mention of support size and batch size.
        # shape (batch_size * support_size, sent_len, emb_dim)
        support_reps = support_reps.view(-1, support_len, emb_dim)
        # shape (batch_size * support_size, sent_len, num_tags)
        # print("support_targets===================",support_reps.shape,support_targets.shape)
        # print(support_reps.shape,support_targets.shape,batch_size,support_size,support_len,num_tags)
        if support_targets.shape[-2] != support_len:
            support_targets = torch.zeros((batch_size * support_size, support_len, num_tags),device="cuda")
        support_targets = support_targets.view(batch_size * support_size, support_len, num_tags).float()
        # get prototype reps
        # shape (batch_size, support_size, num_tags, emd_dim)
        sum_reps = torch.bmm(torch.transpose(support_targets, -1, -2), support_reps)
        # sum up tag emb over support set, shape (batch_size*support_size, num_tags, emd_dim)
        sum_reps = torch.sum(sum_reps.view(batch_size, support_size, num_tags, emb_dim), dim=1)
        # get num of each tag in support set, shape: (batch_size, num_tags, emb_dim)
        tag_count = torch.sum(support_targets.view(batch_size, -1, num_tags), dim=1).unsqueeze(-1)
        tag_count = self.remove_0(tag_count)

        prototype_reps = torch.div(sum_reps, tag_count)  # shape (batch_size, num_tags, emd_dim)

        # calculate dot product
        sim_score = self.sim_func(test_reps, prototype_reps)  # shape (batch_size, sent_len, num_tags)
        # print(sim_score.shape)

        return sim_score,prototype_reps

    def remove_0(self, my_tensor):
        return my_tensor + 0.0001

class PrototypeFeatureSimilarityScorer(SimilarityScorerBase):
    def __init__(self, opt,num_tag,sim_func, emb_log=None):
        super(PrototypeFeatureSimilarityScorer, self).__init__(sim_func=sim_func, emb_log=emb_log)
        # self.fc = nn.Linear(opt.emb_dim, opt.emb_dim, bias=True)
        if opt.gcns == 1:
            self.multihead_attn = nn.MultiheadAttention(opt.emb_dim+opt.gcn_emb, 1)
        else:
            self.multihead_attn = nn.MultiheadAttention(opt.emb_dim, 1)
        # self.f_theta = torch.nn.Sequential(
        #     torch.nn.Linear(opt.emb_dim, opt.emb_dim),  # project for support set and test set
        #     torch.nn.Dropout(0.2))
    def forward(
            self,
            test_reps: torch.Tensor,
            support_reps: torch.Tensor,
            test_output_mask: torch.Tensor,
            support_output_mask: torch.Tensor,
            support_targets: torch.Tensor = None,
            label_reps: torch.Tensor = None, ) -> torch.Tensor:
        """
            calculate similarity between token and each label's prototype.
            :param test_reps: (batch_size, support_size, test_seq_len, dim)
            :param support_reps: (batch_size, support_size, support_seq_len)
            :param test_output_mask: (batch_size, test_seq_len)
            :param support_output_mask: (batch_size, support_size, support_seq_len)
            :param support_targets: one-hot label targets: (batch_size, support_size, support_seq_len, num_tags)
            :param label_reps: (batch_size, num_tags, dim)
            :return: similarity: (batch_size, test_seq_len, num_tags)
        """
        support_size = support_reps.shape[1]
        test_len = test_reps.shape[-2]  # non-word-piece max test token num, Notice that it's different to input t len
        support_len = support_reps.shape[-2]  # non-word-piece max test token num or possible lb num for sc (fix `1`)
        emb_dim = test_reps.shape[-1]
        batch_size = test_reps.shape[0]
        num_tags = support_targets.shape[-1]
        # print("==================",num_tags)
        # print(x)
        test_reps = torch.mean(test_reps, dim=1)  # shape (batch_size, sent_len, emb_dim)
        att_test_reps =test_reps # (batch_size,support_size*test_seq_len, dim)
        # average test representation over support set (reps for each support sent can be different)
        # flatten dim mention of support size and batch size.
        # shape (batch_size * support_size, sent_len, emb_dim)
        att_support_reps = support_reps.view(-1,support_size*support_len,emb_dim) # (batch_size,support_size*support_len, dim)
        att_reps = torch.cat((att_test_reps,att_support_reps),1) 
        att_reps, attn_output_weights = self.multihead_attn(att_reps, att_reps,att_reps)  
        # 给定一个和任务相关的查询Query向量 q，通过计算与Key的注意力分布并附加在Value上，从而计算Attention Value
        att_test_reps = att_reps.narrow(-2, 0, test_len) 
        att_support_reps = att_reps.narrow(-2, test_len, support_size*support_len)
        # print("before================support_reps.shape():",att_reps.shape,support_reps.shape,test_reps.shape)
        
        test_reps = att_test_reps
        # print("att_support_reps",att_support_reps.shape)
        support_reps = att_support_reps.view(batch_size * support_size,support_len, emb_dim)
        # shape (batch_size * support_size, sent_len, num_tags)
        # print("support_targets===================",support_reps.shape,support_targets.shape)
        # print(support_reps.shape,support_targets.shape,batch_size,support_size,support_len,num_tags)
        if support_targets.shape[-2] != support_len:
            support_targets = torch.zeros((batch_size * support_size, support_len, num_tags),device="cuda")

        support_targets = support_targets.view(batch_size * support_size, support_len, num_tags).float()
        # get prototype reps
        # shape (batch_size, support_size, num_tags, emd_dim)
        sum_reps = torch.bmm(torch.transpose(support_targets, -1, -2), support_reps)
        # sum up tag emb over support set, shape (batch_size*support_size, num_tags, emd_dim)
        sum_reps = torch.sum(sum_reps.view(batch_size, support_size, num_tags, emb_dim), dim=1)
        # get num of each tag in support set, shape: (batch_size, num_tags, emb_dim)
        tag_count = torch.sum(support_targets.view(batch_size, -1, num_tags), dim=1).unsqueeze(-1)
        tag_count = self.remove_0(tag_count)

        prototype_reps = torch.div(sum_reps, tag_count)  # shape (batch_size, num_tags, emd_dim)
        # test_reps = self.f_theta(test_reps)   # shape (batch_size, sent_len, emd_dim)
        # prototype_reps = self.f_theta(prototype_reps)
        # calculate dot product
        sim_score = self.sim_func(test_reps, prototype_reps)  # shape (batch_size, sent_len, num_tags)\
        # sim_score = self.f_theta(sim_score)
        # print(sim_score.shape)

        return sim_score,prototype_reps
        # support_size = support_reps.shape[1]
        # test_len = test_reps.shape[-2]  # non-word-piece max test token num, Notice that it's different to input t len
        # support_len = support_reps.shape[-2]  # non-word-piece max test token num or possible lb num for sc (fix `1`)
        # emb_dim = test_reps.shape[-1]
        # batch_size = test_reps.shape[0]
        # num_tags = support_targets.shape[-1]

        # # average test representation over support set (reps for each support sent can be different)
        # test_reps = torch.mean(test_reps, dim=1)  # shape (batch_size, test_len, emb_dim)
        # # flatten dim mention of support size and batch size.
        # # shape (batch_size * support_size, sent_len, emb_dim)
        # support_reps = support_reps.view(batch_size,-1, emb_dim)
        # # shape (batch_size * support_size, sent_len, num_tags)
        # support_targets = support_targets.view(batch_size, support_len * support_size, num_tags).float()
        # # get prototype reps
        # # shape (batch_size, support_size, num_tags, emd_dim)
        # sum_reps = torch.bmm(torch.transpose(support_targets, -1, -2), support_reps)
        # # sum up tag emb over support set, shape (batch_size, num_tags, emd_dim)
        # sum_reps = torch.sum(sum_reps.view(batch_size, support_size, num_tags, emb_dim), dim=1)
        # # get num of each tag in support set, shape: (batch_size, num_tags, 1)
        # tag_count = torch.sum(support_targets.view(batch_size, -1, num_tags), dim=1).unsqueeze(-1)
        # tag_count = self.remove_0(tag_count)

        # prototype_reps0 = torch.div(sum_reps, tag_count)  # shape (batch_size, num_tags, emd_dim)


        # support = support_reps.unsqueeze(1).expand(-1,test_len,-1,-1)  # (batch_size , test_len, sent_len* support_size, emb_dim)
        # support_for_att = self.fc(support)  # (batch_size, test_len, sent_len * support_size, emb_dim)
        # query_for_att = self.fc(test_reps.unsqueeze(2).expand(-1,-1,support_len*support_size,-1)) #  (batch_size, test_len, sent_len * support_size, emb_dim)
        # # print("support_for_att==========================",support_for_att.shape)
        # # print("query_for_att==========================",query_for_att.shape)
        # ins_att_score = F.softmax(torch.tanh(support_for_att * query_for_att).sum(-1), dim=-1)  # (batch_size, test_len, sent_len * support_size)
        # # print("ins_att_score==========================",ins_att_score.shape)
        # support_proto = support * ins_att_score.unsqueeze(3).expand(-1, -1, -1, emb_dim)#(batch_size , test_len, sent_len* support_size, emb_dim)
        # # print("support_proto==========================",support_proto.shape)
        # # test_reps = (test_reps.unsqueeze(2).expand(-1,-1,support_len*support_size,-1) * ins_att_score.unsqueeze(3).expand(-1, -1, -1, emb_dim)).sum(2) #(batch_size , test_len, sent_len* support_size, emb_dim)
        # prototype_reps = torch.bmm(torch.transpose(support_targets, -1, -2), support_proto.sum(1))   # (batch_size, num_tag, emb_dim)
        # # print("prototype_reps=====================",prototype_reps.shape) 
        # # calculate dot product
        # prototype_reps = (prototype_reps + prototype_reps0)/2
        # sim_score = self.sim_func(test_reps, prototype_reps)  # shape (batch_size, sent_len, num_tags)
        # # print(sim_score.shape)

        # return sim_score,prototype_reps

    def remove_0(self, my_tensor):
        return my_tensor + 0.0001


class ProtoWithLabelSimilarityScorer(SimilarityScorerBase):
    def __init__(self, sim_func, scaler=None, emb_log=None):
        super(ProtoWithLabelSimilarityScorer, self).__init__(sim_func=sim_func, emb_log=emb_log)
        self.scaler = scaler
        self.emb_log = emb_log
        self.idx = 0

    def forward(
            self,
            test_reps: torch.Tensor,
            support_reps: torch.Tensor,
            test_output_mask: torch.Tensor,
            support_output_mask: torch.Tensor,
            support_targets: torch.Tensor = None,
            label_reps: torch.Tensor = None,) -> torch.Tensor:
        """
            calculate similarity between token and each label's prototype.
            :param test_reps: (batch_size, support_size, test_seq_len, dim)
            :param support_reps: (batch_size, support_size, support_seq_len)
            :param test_output_mask: (batch_size, test_seq_len)
            :param support_output_mask: (batch_size, support_size, support_seq_len)
            :param support_targets: one-hot label targets: (batch_size, support_size, support_seq_len, num_tags)
            :param label_reps: (batch_size, num_tags, dim)
            :return: similarity: (batch_size, test_seq_len, num_tags)
        """
        '''get data attribute'''
        support_size = support_reps.shape[1]
        test_len = test_reps.shape[-2]  # non-word-piece max test token num, Notice that it's different to input t len
        support_len = support_reps.shape[-2]  # non-word-piece max test token num
        emb_dim = test_reps.shape[-1]
        batch_size = test_reps.shape[0]
        num_tags = support_targets.shape[-1]

        '''get prototype reps'''
        # flatten dim mention of support size and batch size.
        # shape (batch_size * support_size, sent_len, emb_dim)
        support_reps = support_reps.view(-1, support_len, emb_dim)

        # shape (batch_size * support_size, sent_len, num_tags)
        support_targets = support_targets.view(batch_size * support_size, support_len, num_tags).float()

        # shape (batch_size, support_size, num_tags, emd_dim)
        sum_reps = torch.bmm(torch.transpose(support_targets, -1, -2), support_reps)
        # sum up tag emb over support set, shape (batch_size, num_tags, emd_dim)
        sum_reps = torch.sum(sum_reps.view(batch_size, support_size, num_tags, emb_dim), dim=1)

        # get num of each tag in support set, shape: (batch_size, num_tags, 1)
        tag_count = torch.sum(support_targets.view(batch_size, -1, num_tags), dim=1).unsqueeze(-1)
        # divide by 0 occurs when the tags, such as "I-x", are not existing in support.
        tag_count = self.remove_0(tag_count)
        prototype_reps = torch.div(sum_reps, tag_count)  # shape (batch_size, num_tags, emd_dim)

        # add PAD label
        if label_reps is not None:
            label_reps = torch.cat((torch.zeros_like(label_reps).narrow(dim=-2, start=0, length=1), label_reps), dim=-2)
            prototype_reps = (1 - self.scaler) * prototype_reps + self.scaler * label_reps

        '''get final test data reps'''
        # average test representation over support set (reps for each support sent can be different)
        test_reps = torch.mean(test_reps, dim=1)  # shape (batch_size, sent_len, emb_dim)

        '''calculate dot product'''
        sim_score = self.sim_func(test_reps, prototype_reps)  # shape (batch_size, sent_len, num_tags)

        '''store visualization embedding'''
        if not self.training and self.emb_log:
            log_context = '\n'.join(
                ['test_' + str(self.idx) + '-' + str(idx) + '-' + str(idx2) + '\t' + '\t'.join(map(str, item2))
                 for idx, item in enumerate(test_reps.tolist()) for idx2, item2 in enumerate(item)]) + '\n'
            log_context += '\n'.join(
                ['proto_' + str(self.idx) + '-' + str(idx) + '-' + str(idx2) + '\t' + '\t'.join(map(str, item2))
                 for idx, item in enumerate(prototype_reps.tolist()) for idx2, item2 in enumerate(item)]) + '\n'
            self.idx += batch_size
            self.emb_log.write(log_context)

        return sim_score,prototype_reps

    def remove_0(self, my_tensor):
        """
        """
        return my_tensor + 0.0001

class MinSimilarityScorer(SimilarityScorerBase):
    def __init__(self, sim_func, emb_log=None):
        super(MinSimilarityScorer, self).__init__(sim_func=sim_func, emb_log=emb_log)
    # @torchsnooper.snoop()  
    def forward(
            self,
            test_reps: torch.Tensor,
            support_reps: torch.Tensor,
            test_output_mask: torch.Tensor,
            support_output_mask: torch.Tensor,
            support_targets: torch.Tensor = None,
            label_reps: torch.Tensor = None, ) -> torch.Tensor:
        """
            calculate similarity between token and each label's prototype.
            :param test_reps: (batch_size, support_size, test_seq_len, dim)
            :param support_reps: (batch_size, support_size, support_seq_len)
            :param test_output_mask: (batch_size, test_seq_len)
            :param support_output_mask: (batch_size, support_size, support_seq_len)
            :param support_targets: one-hot label targets: (batch_size, support_size, support_seq_len, num_tags)
            :param label_reps: (batch_size, num_tags, dim)
            :return: similarity: (batch_size, test_seq_len, num_tags)
        """
        support_size = support_reps.shape[1]  # 相当于K shot
        test_len = test_reps.shape[-2]  # non-word-piece max test token num, Notice that it's different to input t len
        support_len = support_reps.shape[-2]  # non-word-piece max test token num or possible lb num for sc (fix `1`)
        emb_dim = test_reps.shape[-1]
        batch_size = test_reps.shape[0]
        num_tags = support_targets.shape[-1]

        # average test representation over support set (reps for each support sent can be different)
        test_reps = torch.mean(test_reps, dim=1)  # shape (batch_size, sent_len, emb_dim)  按列求平均
        # print("--------------------------",test_reps.shape)
        # flatten dim mention of support size and batch size.
        # shape (batch_size * support_size, sent_len, emb_dim)
        support_reps = support_reps.view(-1, support_size * support_len, emb_dim)
        support_targets = support_targets.view(batch_size , support_size*support_len, num_tags).float()
        sim_score = torch.randn(batch_size,test_len,num_tags).to("cuda")
        for b in range(batch_size):
            for word_Q  in range(test_len):
                y = test_reps[b,word_Q,:]
                min_dist = 99999999999
                min_index = 0
                for index,word_S in enumerate(range(support_size * support_len)):
                    x = support_reps[b,word_S,:]
                    dist = float(torch.dist(y, x, p=2))
                    if dist < min_dist:
                        min_dist = dist
                        min_index = index
                sim_score[b,word_Q,:] = support_targets[b,min_index,:]

        support_reps_1 = support_reps.view(-1, support_len, emb_dim)
        # shape (batch_size * support_size, sent_len, num_tags)
        support_targets_1 = support_targets.view(batch_size * support_size, support_len, num_tags).float()
        # get prototype reps
        # shape (batch_size, support_size, num_tags, emd_dim)
        sum_reps = torch.bmm(torch.transpose(support_targets_1, -1, -2), support_reps_1) # 矩阵乘法

        # # sum up tag emb over support set, shape (batch_size, num_tags, emd_dim)  求和
        sum_reps = torch.sum(sum_reps.view(batch_size, support_size, num_tags, emb_dim), dim=1)

        # # get num of each tag in support set, shape: (batch_size, num_tags, 1)
        tag_count = torch.sum(support_targets_1.view(batch_size, -1, num_tags), dim=1).unsqueeze(-1)
        tag_count = self.remove_0(tag_count)
        
        # # 求和取平均
        prototype_reps = torch.div(sum_reps, tag_count)  # shape (batch_size, num_tags, emd_dim)
        # prototype_reps  = torch.randn(batch_size,num_tags,emb_dim)
        # calculate dot product
        sim_score_1 = self.sim_func(test_reps, prototype_reps)  # shape (batch_size, sent_len, num_tags)
        # print("-----------------------------",sum_reps.shape,tag_count.shape,prototype_reps.shape,sim_score.shape)
        return sim_score+0.5*sim_score_1,prototype_reps

    def remove_0(self, my_tensor):
        return my_tensor + 0.0001

class Attn(nn.Module):
    def __init__(self,hidden_dim,kqv_dim):
        super(Attn, self).__init__()
        self.wk=nn.Linear(hidden_dim,kqv_dim)
        self.wq=nn.Linear(hidden_dim,kqv_dim)
        self.wv=nn.Linear(hidden_dim,kqv_dim)
        self.d=kqv_dim**0.5

    def forward(self, input):
        '''
        :param input: batch_size x seq_len x hidden_dim
        :return:
        '''
        k=self.wk(input)
        q=self.wq(input)
        v=self.wv(input)
        w=F.softmax(torch.bmm(q,k.transpose(-1,-2))/self.d,dim=-1)
        attn=torch.bmm(w,v)

        return attn
class ProtowithAttGCNSimilarityScorer(SimilarityScorerBase):
    def __init__(self, opt,sim_func, emb_log=None):
        super(ProtowithAttGCNSimilarityScorer, self).__init__(sim_func=sim_func, emb_log=emb_log)
        self.gcns = nn.ModuleList()
        # self.multihead_attn = nn.MultiheadAttention(embed_dim=opt.emb_dim, num_heads=3)
        for i in range(opt.gcn_layers):
            gcn = GraphConvolution(in_features=opt.emb_dim,
                                   out_features=200,
                                   edge_types=3,
                                   dropout=0.3 if i != (opt.gcn_layers - 1) else None,
                                   use_bn=True)
            self.gcns.append(gcn)

    def forward(
            self,
            test_reps: torch.Tensor,
            support_reps: torch.Tensor,
            test_output_mask: torch.Tensor,
            support_output_mask: torch.Tensor,
            support_targets: torch.Tensor = None,
            label_reps: torch.Tensor = None, ) -> torch.Tensor:
        """
            calculate similarity between token and each label's prototype.
            :param test_reps: (batch_size, support_size, test_seq_len, dim)
            :param support_reps: (batch_size, support_size, support_seq_len)
            :param test_output_mask: (batch_size, test_seq_len)
            :param support_output_mask: (batch_size, support_size, support_seq_len)
            :param support_targets: one-hot label targets: (batch_size, support_size, support_seq_len, num_tags)
            :param label_reps: (batch_size, num_tags, dim)
            :return: similarity: (batch_size, test_seq_len, num_tags)
        """
        support_size = support_reps.shape[1]
        test_len = test_reps.shape[-2]  # non-word-piece max test token num, Notice that it's different to input t len
        support_len = support_reps.shape[-2]  # non-word-piece max test token num or possible lb num for sc (fix `1`)
        emb_dim = test_reps.shape[-1]
        batch_size = test_reps.shape[0]
        num_tags = support_targets.shape[-1]

        # average test representation over support set (reps for each support sent can be different)
        test_reps = torch.mean(test_reps, dim=1)  # shape (batch_size, sent_len, emb_dim)
        # flatten dim mention of support size and batch size.
        # shape (batch_size * support_size, sent_len, emb_dim)
        support_reps = support_reps.view(-1, support_len, emb_dim)
        # shape (batch_size * support_size, sent_len, num_tags)
        # print("support_targets===================",support_reps.shape,support_targets.shape)
        # print(support_reps.shape,support_targets.shape,batch_size,support_size,support_len,num_tags)
        if support_targets.shape[-2] != support_len:
            support_targets = torch.zeros((batch_size * support_size, support_len, num_tags),device="cuda")
        support_targets = support_targets.view(batch_size * support_size, support_len, num_tags).float()
        # get prototype reps
        # shape (batch_size, support_size, num_tags, emd_dim)
        sum_reps = torch.bmm(torch.transpose(support_targets, -1, -2), support_reps)
        # sum up tag emb over support set, shape (batch_size*support_size, num_tags, emd_dim)
        sum_reps = torch.sum(sum_reps.view(batch_size, support_size, num_tags, emb_dim), dim=1)
        # get num of each tag in support set, shape: (batch_size, num_tags, emb_dim)
        tag_count = torch.sum(support_targets.view(batch_size, -1, num_tags), dim=1).unsqueeze(-1)
        tag_count = self.remove_0(tag_count)

        prototype_reps = torch.div(sum_reps, tag_count)  # shape (batch_size, num_tags, emd_dim)

        # calculate dot product
        sim_score = self.sim_func(test_reps, prototype_reps)  # shape (batch_size, sent_len, num_tags)
        # print(sim_score.shape)

        return sim_score,prototype_reps

    def remove_0(self, my_tensor):
        return my_tensor + 0.0001

class ProtowithCrossSimilarityScorer(SimilarityScorerBase):
    def __init__(self, sim_func, emb_log=None):
        super(ProtowithCrossSimilarityScorer, self).__init__(sim_func=sim_func, emb_log=emb_log) 
        self.N = N
        self.K = K
        self.total_Q = total_Q
        self.max_len = max_len
        self.drop = nn.Dropout()
        self.Avg1 = nn.AvgPool2d((1, N * K * max_len))
        self.Max1 = nn.MaxPool2d((1, N * K * max_len))
        self.linear1 = nn.Sequential(
            nn.Linear(N * K * max_len, int(1/5 * N * K * max_len)),
            nn.ReLU(),
            nn.Linear(int(1/5 * N * K * max_len), N * K * max_len)
        )

        self.Avg2 = nn.AvgPool2d((1, total_Q * max_len))
        self.Max2 = nn.MaxPool2d((1, total_Q * max_len))
        self.linear2 = nn.Sequential(
            nn.Linear(total_Q * max_len, int(1/5 * total_Q * max_len)),  # torch.Size[512, 2560])
            nn.ReLU(),
            nn.Linear(int(1/5 * total_Q * max_len), total_Q * max_len)
        )

        self.fusion_layer = nn.Sequential(
              nn.Linear(2*max_len, max_len),
              nn.ReLU(),
              nn.Linear(max_len, max_len),
         )
        self.attention_fc = nn.Linear(hidden_size, hidden_size)

    # @torchsnooper.snoop()  
    def __cross_attention(self, support_emb, query_emb):  #  (batch_size, support_size, num_tags, emd_dim)  , (batch_size, sent_len, emb_dim)   (batch_size * support_size, sent_len, emb_dim)
        # support_emb, support_cls = self.sentence_encoder(support)  # [B * N * K, max_len, D];      [B * N * K ,D]
        # query_emb, query_cls = self.sentence_encoder(query)      # [B * total_Q, max_len, D];    [B * total_Q ,D]
        B,K,N,hidden_size = support_emb.shape
        # hidden_size = support_emb.size(-1)
        support = support_emb.view(B, -1, hidden_size) # (B, N * K * max_len,  D)
        query = query_emb.view(B, -1, hidden_size)     # (B, total_Q * max_len, D)
        # cross attention
        W = torch.bmm(support, query.permute(0, 2, 1))     # [B, N*K*max_len, total_Q*max_Len]

        # Max pooling
        W1m = self.Max2(W)  # [B, N*K*max_len, 1]
        W1m = self.linear1(W1m.view(B, -1))   # [B, N*K*max_len,1]
        W1m = W1m.view(B, -1, self.max_len)  # [B, N*K, max_Len]
        # Average pooling
        W1a = self.Avg2(W)  # [B, N*K*max_len, 1]
        W1a = self.linear1(W1a.view(B, -1))   # [B, N*K*max_len,1]
        W1a = W1a.view(B, -1, self.max_len)  # [B, N*K, max_Len]
        #W1 = W1a
        # Fusion
        W1 = torch.cat([W1a, W1m], -1)  # [B, N*K, 2**max_len]
        W1 = self.fusion_layer(W1)  # [B, N*K, max_len]

        W2m = self.Max1(W.permute(0, 2, 1))  # [B, total_Q*max_Len, 1]
        W2m = self.linear2(W2m.view(B, -1))
        W2m= W2m.view(B, -1, self.max_len)  # [B, total_Q, max_len]
        W2a = self.Avg1(W.permute(0, 2, 1))  # [B, total_Q*max_Len, 1]
        W2a = self.linear2(W2a.view(B, -1))
        W2a = W2a.view(B, -1, self.max_len)  # [B, total_Q, max_len]
        #W2 = W2a
        W2 = torch.cat([W2a, W2m], -1)  # [B, total_Q, 2*max_len]
        W2 = self.fusion_layer(W2)  # [B, total_Q, max_len]

        # softmax + residual
        W1 = self.__soft_max(W1, 1) + 1  # [B,  N*K,    max_Len, 1]
        W2 = self.__soft_max(W2, 1) + 1  # [B, total_Q, max_len, 1]

        # max_len个hidden state加权求和，得到表示该实例的最终hidden state
        # support：[B, N*K, max_len, 1]  * [B, N*K, max_len, D] -> [B, N*K, max_len, D]
        # sum(dim=2) -> [B, N*K, D]
        support_emb = support_emb.view(B, -1, self.max_len, hidden_size)
        support = (W1.unsqueeze(-1) * support_emb).sum(dim=2)  # [B, N*K, D]
        query_emb = query_emb.view(B, -1, self.max_len, hidden_size)
        query = (W2.unsqueeze(-1) * query_emb).sum(dim=2)   # [B, total_Q, D]
        return support, query

    def forward(
            self,
            test_reps: torch.Tensor,
            support_reps: torch.Tensor,
            test_output_mask: torch.Tensor,
            support_output_mask: torch.Tensor,
            support_targets: torch.Tensor = None,
            label_reps: torch.Tensor = None, ) -> torch.Tensor:
        """
            calculate similarity between token and each label's prototype.
            :param test_reps: (batch_size, support_size, test_seq_len, dim)
            :param support_reps: (batch_size, support_size, support_seq_len)
            :param test_output_mask: (batch_size, test_seq_len)
            :param support_output_mask: (batch_size, support_size, support_seq_len)
            :param support_targets: one-hot label targets: (batch_size, support_size, support_seq_len, num_tags)
            :param label_reps: (batch_size, num_tags, dim)
            :return: similarity: (batch_size, test_seq_len, num_tags)
        """
        support_size = support_reps.shape[1]  # 相当于K shot
        test_len = test_reps.shape[-2]  # non-word-piece max test token num, Notice that it's different to input t len
        support_len = support_reps.shape[-2]  # non-word-piece max test token num or possible lb num for sc (fix `1`)
        emb_dim = test_reps.shape[-1]
        batch_size = test_reps.shape[0]
        num_tags = support_targets.shape[-1]

        # average test representation over support set (reps for each support sent can be different)
        test_reps = torch.mean(test_reps, dim=1)  # shape (batch_size, sent_len, emb_dim)  按列求平均

        # print("--------------------------",test_reps.shape)
        # flatten dim mention of support size and batch size.
        # shape (batch_size * support_size, sent_len, emb_dim)
        support_reps = support_reps.view(-1, support_len, emb_dim)
        # shape (batch_size * support_size, sent_len, num_tags)
        support_targets = support_targets.view(batch_size * support_size, support_len, num_tags).float()
        # get prototype reps
        # shape (batch_size, support_size, num_tags, emd_dim)
        sum_reps = torch.bmm(torch.transpose(support_targets, -1, -2), support_reps) # 矩阵乘法
        
        sum_reps , test_reps = self.__cross_attention(sum_reps,test_reps)

        # sum up tag emb over support set, shape (batch_size, num_tags, emd_dim)  求和
        sum_reps = torch.sum(sum_reps.view(batch_size, support_size, num_tags, emb_dim), dim=1)

        # get num of each tag in support set, shape: (batch_size, num_tags, 1)
        tag_count = torch.sum(support_targets.view(batch_size, -1, num_tags), dim=1).unsqueeze(-1)
        tag_count = self.remove_0(tag_count)
        
        # 求和取平均
        prototype_reps = torch.div(sum_reps, tag_count)  # shape (batch_size, num_tags, emd_dim)

        # calculate dot product
        sim_score = self.sim_func(test_reps, prototype_reps)  # shape (batch_size, sent_len, num_tags)
        # print("-----------------------------",sum_reps.shape,tag_count.shape,prototype_reps.shape,sim_score.shape)
        return sim_score,prototype_reps

        def remove_0(self, my_tensor):
            return my_tensor + 0.0001
    

class TapNetSimilarityScorer(SimilarityScorerBase):
    def __init__(self, sim_func, num_anchors, mlp_out_dim, random_init=False, random_init_r=1.0, mlp=False, emb_log=None,
                 tap_proto=False, tap_proto_r=1.0, anchor_dim=968):
        super(TapNetSimilarityScorer, self).__init__(sim_func=sim_func, emb_log=emb_log)
        self.num_anchors = num_anchors
        self.random_init = random_init

        self.bert_emb_dim = anchor_dim
        if self.random_init:  # create anchors
            self.anchor_reps = torch.nn.Parameter(
                torch.nn.init.xavier_normal_(torch.randn((self.num_anchors, self.bert_emb_dim))), requires_grad=True)
        self.mlp = mlp
        self.mlp_out_dim = mlp_out_dim
        if self.mlp:
            self.f_theta = torch.nn.Linear(self.bert_emb_dim, self.mlp_out_dim)  # project for support set and test set
            self.phi = torch.nn.Linear(self.bert_emb_dim, self.mlp_out_dim)  # project for label reps
            # init with xavier normal distribution
            torch.nn.init.xavier_normal_(self.f_theta.weight)
            torch.nn.init.xavier_normal_(self.phi.weight)
        self.tap_proto = tap_proto
        self.tap_proto_r = tap_proto_r
        self.random_init_r = random_init_r
        self.idx = 0

    def forward(
            self,
            test_reps: torch.Tensor,
            support_reps: torch.Tensor,
            test_output_mask: torch.Tensor,
            support_output_mask: torch.Tensor,
            support_targets: torch.Tensor = None,
            label_reps: torch.Tensor = None, ) -> torch.Tensor:
        """
            calculate similarity between token and each label's prototype.
            :param test_reps: (batch_size, support_size, test_seq_len, dim)
            :param support_reps: (batch_size, support_size, support_seq_len)
            :param test_output_mask: (batch_size, test_seq_len)
            :param support_output_mask: (batch_size, support_size, support_seq_len)
            :param support_targets: one-hot label targets: (batch_size, support_size, support_seq_len, num_tags)
            :param label_reps: (batch_size, no_pad_num_tags, dim)
            :return: similarity: (batch_size, test_seq_len, num_tags)
        """
        '''get data attribute'''
        support_size = support_reps.shape[1]
        test_len = test_reps.shape[-2]  # non-word-piece max test token num, Notice that it's different to input t len
        support_len = support_reps.shape[-2]  # non-word-piece max test token num
        emb_dim = test_reps.shape[-1]
        batch_size = test_reps.shape[0]
        num_tags = support_targets.shape[-1]
        no_pad_num_tags = num_tags - 1

        if no_pad_num_tags > len(self.anchor_reps) and (not label_reps or self.random_init):
            raise RuntimeError("Too few anchors")

        if label_reps is None and not self.random_init:
            raise RuntimeError('Must provide at least one of: anchor and label_reps.')

        ''' get reps for each tag with anchors or label reps '''
        if self.random_init:
            random_label_idxs = torch.randperm(len(self.anchor_reps))
            random_label_reps = self.anchor_reps[random_label_idxs[:no_pad_num_tags], :]
            random_label_reps = random_label_reps.unsqueeze(0).repeat(batch_size, 1, 1).to(support_reps.device)
            if label_reps is not None:  # use schema and integrate achor and schema reps as label reps
                label_reps = (1 - self.random_init_r) * label_reps + self.random_init_r * random_label_reps
            else:  # use anchor only as label reps
                label_reps = random_label_reps

        '''project label reps embedding and support data embedding with a MLP'''
        if self.mlp:
            label_reps = torch.tanh(self.phi(label_reps.contiguous().view(-1, emb_dim)))
            label_reps = label_reps.contiguous().view(batch_size, no_pad_num_tags, self.mlp_out_dim)
            support_reps = torch.tanh(self.f_theta(support_reps.contiguous().view(-1, emb_dim)))

        '''get prototype reps'''
        # flatten dim mention of support size and batch size. shape (batch_size * support_size, sent_len, emb_dim)
        support_reps = support_reps.view(-1, support_len, emb_dim)

        # shape (batch_size * support_size, sent_len, num_tags)
        support_targets = support_targets.view(batch_size * support_size, support_len, num_tags).float()

        # shape (batch_size * support_size, num_tags, emd_dim)
        sum_reps = torch.bmm(torch.transpose(support_targets, -1, -2), support_reps)
        # sum up tag emb over support set, shape (batch_size, num_tags, emd_dim)
        sum_reps = torch.sum(sum_reps.view(batch_size, support_size, num_tags, emb_dim), dim=1)

        # get num of each tag in support set, shape: (batch_size, num_tags, 1)
        tag_count = torch.sum(support_targets.view(batch_size, -1, num_tags), dim=1).unsqueeze(-1)
        # divide by 0 occurs when the tags, such as "I-x", are not existing in support.
        tag_count = self.remove_0(tag_count)
        prototype_reps = torch.div(sum_reps, tag_count)  # shape (batch_size, num_tags, emd_dim)

        '''generate error for every class'''
        # get normalized label reps
        label_reps_sum = label_reps.sum(dim=1).unsqueeze(1).repeat(1, no_pad_num_tags, 1) - label_reps
        label_reps_sum = label_reps - 1 / (no_pad_num_tags - 1) * label_reps_sum
        label_reps_sum = label_reps_sum / (torch.norm(label_reps_sum, p=2, dim=-1).unsqueeze(-1).expand_as(label_reps_sum) + 1e-13)
        # add [PAD] label reps
        label_reps_sum_pad = torch.cat(
            (torch.zeros_like(label_reps_sum).narrow(dim=-2, start=0, length=1).to(label_reps_sum.device), label_reps_sum), dim=-2)
        # get normalized proto reps
        prototype_reps_sum = \
            prototype_reps / (torch.norm(prototype_reps, p=2, dim=-1).unsqueeze(-1).expand_as(prototype_reps) + 1e-13)
        # get the error distance for optimization
        error_every_class = label_reps_sum_pad - prototype_reps_sum

        '''generate projection space M'''
        try:
            # torch 1.2.0 has the batch process function
            _, s, vh = torch.svd(error_every_class, some=False)
        except RuntimeError:
            # others does not
            batch_size = error_every_class.shape[0]
            s, vh = [], []
            for i in range(batch_size):
                _, s_, vh_ = torch.svd(error_every_class[i], some=False)
                s.append(s_)
                vh.append(vh_)
            s, vh = torch.stack(s, dim=0), torch.stack(vh, dim=0)
        s_sum = max((s >= 1e-13).sum(dim=1))
        # shape (batch_size, emb_dim, D)
        M = torch.stack([vh[i][:, s_sum:].clone() for i in range(batch_size)], dim=0)

        '''get final test data reps'''
        # project query data embedding with a MLP
        if self.mlp:
            test_reps = torch.tanh(self.f_theta(test_reps.contiguous().view(-1, emb_dim)))
            test_reps = test_reps.contiguous().view(batch_size, support_size, -1, self.mlp_out_dim)
        # average test representation over support set (reps for each support sent can be different)
        test_reps = torch.mean(test_reps, dim=1)  # shape (batch_size, sent_len, emb_dim)
        # add [PAD] label reps
        label_reps_pad = torch.cat(
            (torch.zeros_like(label_reps).narrow(dim=-2, start=0, length=1).to(label_reps.device),
             label_reps), dim=-2)

        '''calculate dot product'''
        if self.tap_proto:
            # shape (batch_size, sent_len, num_tags)
            label_proto_reps = self.tap_proto_r * prototype_reps + (1 - self.tap_proto_r) * label_reps_pad
            sim_score = 2 * self.sim_func(torch.matmul(test_reps, M), torch.matmul(label_proto_reps, M)) \
                + torch.log(
                    torch.sum(
                        torch.exp(- self.sim_func(torch.matmul(test_reps, M), torch.matmul(label_proto_reps, M))),
                        dim=-1)).unsqueeze(-1).repeat(1, 1, num_tags)

            if not self.training and self.emb_log:
                log_context = '\n'.join(['test_' + str(self.idx) + '-' + str(idx) + '-' + str(idx2) + '\t' + '\t'.join(map(str, item2))
                                         for idx, item in enumerate(test_reps.tolist()) for idx2, item2 in enumerate(item)]) + '\n'
                log_context += '\n'.join(['proto_' + str(self.idx) + '-' + str(idx) + '-' + str(idx2) + '\t' + '\t'.join(map(str, item2))
                                          for idx, item in enumerate(prototype_reps.tolist()) for idx2, item2 in enumerate(item)]) + '\n'
                log_context += '\n'.join(['p_test_' + str(self.idx) + '-' + str(idx) + '-' + str(idx2) + '\t' + '\t'.join(map(str, item2))
                                          for idx, item in enumerate(torch.matmul(test_reps, M).tolist()) for idx2, item2 in enumerate(item)]) + '\n'
                log_context += '\n'.join(['p_proto_' + str(self.idx) + '-' + str(idx) + '-' + str(idx2) + '\t' + '\t'.join(map(str, item2))
                                          for idx, item in enumerate(torch.matmul(prototype_reps, M).tolist()) for idx2, item2 in enumerate(item)]) + '\n'
                self.idx += batch_size
                self.emb_log.write(log_context)
        else:
            # shape (batch_size, sent_len, num_tags)
            sim_score = 2 * self.sim_func(torch.matmul(test_reps, M), torch.matmul(label_reps_pad, M)) \
                + torch.log(
                        torch.sum(
                            torch.exp(- self.sim_func(torch.matmul(test_reps, M), torch.matmul(label_reps_pad, M))),
                            dim=-1)).unsqueeze(-1).repeat(1, 1, num_tags)

            if not self.training and self.emb_log:
                log_context = '\n'.join(
                    ['test_' + str(self.idx) + '-' + str(idx) + '-' + str(idx2) + '\t' + '\t'.join(map(str, item2))
                     for idx, item in enumerate(test_reps.tolist()) for idx2, item2 in enumerate(item)]) + '\n'
                log_context += '\n'.join(
                    ['label_' + str(self.idx) + '-' + str(idx) + '-' + str(idx2) + '\t' + '\t'.join(map(str, item2))
                     for idx, item in enumerate(prototype_reps.tolist()) for idx2, item2 in enumerate(item)]) + '\n'
                log_context += '\n'.join(
                    ['p_test_' + str(self.idx) + '-' + str(idx) + '-' + str(idx2) + '\t' + '\t'.join(map(str, item2))
                     for idx, item in enumerate(torch.matmul(test_reps, M).tolist()) for idx2, item2 in
                     enumerate(item)]) + '\n'
                log_context += '\n'.join(
                    ['p_label_' + str(self.idx) + '-' + str(idx) + '-' + str(idx2) + '\t' + '\t'.join(map(str, item2))
                     for idx, item in enumerate(torch.matmul(prototype_reps, M).tolist()) for idx2, item2 in
                     enumerate(item)]) + '\n'
                self.idx += batch_size
                self.emb_log.write(log_context)

        return sim_score,label_proto_reps

    def remove_0(self, my_tensor):
        """
        """
        return my_tensor + 0.0001


class TapNetCrossSimilarityScorer(SimilarityScorerBase):
    def __init__(self, sim_func, num_anchors, mlp_out_dim, random_init=False, random_init_r=1.0, mlp=False, emb_log=None,
                 tap_proto=False, tap_proto_r=1.0, anchor_dim=768):
        super(TapNetCrossSimilarityScorer, self).__init__(sim_func=sim_func, emb_log=emb_log)
        self.num_anchors = num_anchors
        self.random_init = random_init

        self.bert_emb_dim = anchor_dim
        if self.random_init:  # create anchors
            self.anchor_reps = torch.nn.Parameter(
                torch.nn.init.xavier_normal_(torch.randn((self.num_anchors, self.bert_emb_dim))), requires_grad=True)
        self.mlp = mlp
        self.mlp_out_dim = mlp_out_dim
        if self.mlp:
            self.f_theta = torch.nn.Linear(self.bert_emb_dim, self.mlp_out_dim)  # project for support set and test set
            self.phi = torch.nn.Linear(self.bert_emb_dim, self.mlp_out_dim)  # project for label reps
            # init with xavier normal distribution
            torch.nn.init.xavier_normal_(self.f_theta.weight)
            torch.nn.init.xavier_normal_(self.phi.weight)
        self.tap_proto = tap_proto
        self.tap_proto_r = tap_proto_r
        self.random_init_r = random_init_r
        self.idx = 0
        self.fc = nn.Linear(self.bert_emb_dim, self.bert_emb_dim, bias=True)


    def forward(
            self,
            test_reps: torch.Tensor,
            support_reps: torch.Tensor,
            test_output_mask: torch.Tensor,
            support_output_mask: torch.Tensor,
            support_targets: torch.Tensor = None,
            label_reps: torch.Tensor = None, ) -> torch.Tensor:
        """
            calculate similarity between token and each label's prototype.
            :param test_reps: (batch_size, support_size, test_seq_len, dim)
            :param support_reps: (batch_size, support_size, support_seq_len)
            :param test_output_mask: (batch_size, test_seq_len)
            :param support_output_mask: (batch_size, support_size, support_seq_len)
            :param support_targets: one-hot label targets: (batch_size, support_size, support_seq_len, num_tags)
            :param label_reps: (batch_size, no_pad_num_tags, dim)
            :return: similarity: (batch_size, test_seq_len, num_tags)
        """
        '''get data attribute'''
        support_size = support_reps.shape[1]
        test_len = test_reps.shape[-2]  # non-word-piece max test token num, Notice that it's different to input t len
        support_len = support_reps.shape[-2]  # non-word-piece max test token num
        emb_dim = test_reps.shape[-1]
        batch_size = test_reps.shape[0]
        num_tags = support_targets.shape[-1]
        no_pad_num_tags = num_tags - 1

        if no_pad_num_tags > len(self.anchor_reps) and (not label_reps or self.random_init):
            raise RuntimeError("Too few anchors")

        if label_reps is None and not self.random_init:
            raise RuntimeError('Must provide at least one of: anchor and label_reps.')

        ''' get reps for each tag with anchors or label reps '''
        if self.random_init:
            random_label_idxs = torch.randperm(len(self.anchor_reps))
            random_label_reps = self.anchor_reps[random_label_idxs[:no_pad_num_tags], :]
            random_label_reps = random_label_reps.unsqueeze(0).repeat(batch_size, 1, 1).to(support_reps.device)
            if label_reps is not None:  # use schema and integrate achor and schema reps as label reps
                label_reps = (1 - self.random_init_r) * label_reps + self.random_init_r * random_label_reps
            else:  # use anchor only as label reps
                label_reps = random_label_reps

        '''project label reps embedding and support data embedding with a MLP'''
        if self.mlp:
            label_reps = torch.tanh(self.phi(label_reps.contiguous().view(-1, emb_dim)))
            label_reps = label_reps.contiguous().view(batch_size, no_pad_num_tags, self.mlp_out_dim)
            support_reps = torch.tanh(self.f_theta(support_reps.contiguous().view(-1, emb_dim)))

        '''get ATT prototype reps'''
        '''get final test data reps'''
        # project query data embedding with a MLP
        if self.mlp:
            test_reps = torch.tanh(self.f_theta(test_reps.contiguous().view(-1, emb_dim)))
            test_reps = test_reps.contiguous().view(batch_size, support_size, -1, self.mlp_out_dim)

        test_reps = torch.mean(test_reps, dim=1) 
        # flatten dim mention of support size and batch size. shape (batch_size * support_size, sent_len, emb_dim)
        support_reps = support_reps.view(batch_size,-1, emb_dim)
        # shape (batch_size * support_size, sent_len, num_tags)
        support_targets = support_targets.view(batch_size, support_len * support_size, num_tags).float()

        support = support_reps.unsqueeze(1).expand(-1,test_len,-1,-1)  # (batch_size , test_len, sent_len* support_size, emb_dim)
        support_for_att = self.fc(support)  # (batch_size, test_len, sent_len * support_size, emb_dim)
        query_for_att = self.fc(test_reps.unsqueeze(2).expand(-1,-1,support_len*support_size,-1)) #  (batch_size, test_len, sent_len * support_size, emb_dim)
        # print("support_for_att==========================",support_for_att.shape)
        # print("query_for_att==========================",query_for_att.shape)
        ins_att_score = F.softmax(torch.tanh(support_for_att * query_for_att).sum(-1), dim=-1)  # (batch_size, test_len, sent_len * support_size)
        # print("ins_att_score==========================",ins_att_score.shape)
        support_proto = (support * ins_att_score.unsqueeze(3).expand(-1, -1, -1, emb_dim)) #(batch_size , test_len, sent_len* support_size, emb_dim)
        # print("support_proto==========================",support_proto.shape)
        test_reps = (test_reps.unsqueeze(2).expand(-1,-1,support_len*support_size,-1) * ins_att_score.unsqueeze(3).expand(-1, -1, -1, emb_dim)).sum(2)
        prototype_reps = torch.bmm(torch.transpose(support_targets, -1, -2), support_proto.sum(1))   # (batch_size, num_tag, emb_dim)
        # print("prototype_reps=====================",prototype_reps.shape) 

        '''generate error for every class'''
        # get normalized label reps
        label_reps_sum = label_reps.sum(dim=1).unsqueeze(1).repeat(1, no_pad_num_tags, 1) - label_reps
        label_reps_sum = label_reps - 1 / (no_pad_num_tags - 1) * label_reps_sum
        label_reps_sum = label_reps_sum / (torch.norm(label_reps_sum, p=2, dim=-1).unsqueeze(-1).expand_as(label_reps_sum) + 1e-13)
        # add [PAD] label reps
        label_reps_sum_pad = torch.cat(
            (torch.zeros_like(label_reps_sum).narrow(dim=-2, start=0, length=1).to(label_reps_sum.device), label_reps_sum), dim=-2)
        # get normalized proto reps
        prototype_reps_sum = \
            prototype_reps / (torch.norm(prototype_reps, p=2, dim=-1).unsqueeze(-1).expand_as(prototype_reps) + 1e-13)
        # get the error distance for optimization
        error_every_class = label_reps_sum_pad - prototype_reps_sum

        '''generate projection space M'''
        try:
            # torch 1.2.0 has the batch process function
            # print("==========================")
            _, s, vh = torch.svd(error_every_class, some=False)
        except RuntimeError:
            # others does not
            batch_size = error_every_class.shape[0]
            s, vh = [], []
            for i in range(batch_size):
                _, s_, vh_ = torch.svd(error_every_class[i], some=False)
                s.append(s_)
                vh.append(vh_)
            s, vh = torch.stack(s, dim=0), torch.stack(vh, dim=0)
        s_sum = max((s >= 1e-13).sum(dim=1))
        # shape (batch_size, emb_dim, D)
        M = torch.stack([vh[i][:, s_sum:].clone() for i in range(batch_size)], dim=0)


        # average test representation over support set (reps for each support sent can be different)
        # test_reps = torch.mean(test_reps, dim=1)  # shape (batch_size, sent_len, emb_dim)
        # add [PAD] label reps
        label_reps_pad = torch.cat(
            (torch.zeros_like(label_reps).narrow(dim=-2, start=0, length=1).to(label_reps.device),
             label_reps), dim=-2)

        '''calculate dot product'''
        if self.tap_proto:
            # shape (batch_size, sent_len, num_tags)
            label_proto_reps = self.tap_proto_r * prototype_reps + (1 - self.tap_proto_r) * label_reps_pad
            sim_score = 2 * self.sim_func(torch.matmul(test_reps, M), torch.matmul(label_proto_reps, M)) \
                + torch.log(
                    torch.sum(
                        torch.exp(- self.sim_func(torch.matmul(test_reps, M), torch.matmul(label_proto_reps, M))),
                        dim=-1)).unsqueeze(-1).repeat(1, 1, num_tags)

            if not self.training and self.emb_log:
                log_context = '\n'.join(['test_' + str(self.idx) + '-' + str(idx) + '-' + str(idx2) + '\t' + '\t'.join(map(str, item2))
                                         for idx, item in enumerate(test_reps.tolist()) for idx2, item2 in enumerate(item)]) + '\n'
                log_context += '\n'.join(['proto_' + str(self.idx) + '-' + str(idx) + '-' + str(idx2) + '\t' + '\t'.join(map(str, item2))
                                          for idx, item in enumerate(prototype_reps.tolist()) for idx2, item2 in enumerate(item)]) + '\n'
                log_context += '\n'.join(['p_test_' + str(self.idx) + '-' + str(idx) + '-' + str(idx2) + '\t' + '\t'.join(map(str, item2))
                                          for idx, item in enumerate(torch.matmul(test_reps, M).tolist()) for idx2, item2 in enumerate(item)]) + '\n'
                log_context += '\n'.join(['p_proto_' + str(self.idx) + '-' + str(idx) + '-' + str(idx2) + '\t' + '\t'.join(map(str, item2))
                                          for idx, item in enumerate(torch.matmul(prototype_reps, M).tolist()) for idx2, item2 in enumerate(item)]) + '\n'
                self.idx += batch_size
                self.emb_log.write(log_context)
        else:
            # shape (batch_size, sent_len, num_tags)
            sim_score = 2 * self.sim_func(torch.matmul(test_reps, M), torch.matmul(label_reps_pad, M)) \
                + torch.log(
                        torch.sum(
                            torch.exp(- self.sim_func(torch.matmul(test_reps, M), torch.matmul(label_reps_pad, M))),
                            dim=-1)).unsqueeze(-1).repeat(1, 1, num_tags)

            if not self.training and self.emb_log:
                log_context = '\n'.join(
                    ['test_' + str(self.idx) + '-' + str(idx) + '-' + str(idx2) + '\t' + '\t'.join(map(str, item2))
                     for idx, item in enumerate(test_reps.tolist()) for idx2, item2 in enumerate(item)]) + '\n'
                log_context += '\n'.join(
                    ['label_' + str(self.idx) + '-' + str(idx) + '-' + str(idx2) + '\t' + '\t'.join(map(str, item2))
                     for idx, item in enumerate(prototype_reps.tolist()) for idx2, item2 in enumerate(item)]) + '\n'
                log_context += '\n'.join(
                    ['p_test_' + str(self.idx) + '-' + str(idx) + '-' + str(idx2) + '\t' + '\t'.join(map(str, item2))
                     for idx, item in enumerate(torch.matmul(test_reps, M).tolist()) for idx2, item2 in
                     enumerate(item)]) + '\n'
                log_context += '\n'.join(
                    ['p_label_' + str(self.idx) + '-' + str(idx) + '-' + str(idx2) + '\t' + '\t'.join(map(str, item2))
                     for idx, item in enumerate(torch.matmul(prototype_reps, M).tolist()) for idx2, item2 in
                     enumerate(item)]) + '\n'
                self.idx += batch_size
                self.emb_log.write(log_context)

        return sim_score,label_proto_reps

    def remove_0(self, my_tensor):
        """
        """
        return my_tensor + 0.0001
