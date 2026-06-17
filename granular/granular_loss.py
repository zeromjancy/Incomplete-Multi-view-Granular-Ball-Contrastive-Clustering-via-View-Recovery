import torch
import numpy as np
from granular.base import GranularBall, GBList, MVGBList
from granular.tools import relation_of_views_gblists, merge_tensors, relation_of_views_gblists_tensor


class GranularContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=1.):
        super(GranularContrastiveLoss, self).__init__()
        self.t = temperature

    def forward(self, gblist):
        pos_mask = gblist.affinity()
        neg_mask = 1 - pos_mask
        num_ins = len(gblist)
        idx = torch.arange(0, num_ins)
        pos_mask[idx, idx] = 0
        x = gblist.get_centers()
        norm_x = torch.norm(x, p=2, dim=1, keepdim=True)
        sim_x = x @ x.T / (norm_x @ norm_x.T + 1e-12)
        sim_pos = pos_mask * sim_x / self.t
        sim_neg = neg_mask * sim_x / self.t
        exp_sim_neg = torch.sum(torch.exp(sim_neg), dim=1, keepdim=True).expand((num_ins, num_ins))
        expsum_sim = torch.exp(sim_pos) + exp_sim_neg
        loss = -(sim_pos - torch.log(expsum_sim) * pos_mask)

        avg_sim_pos = torch.sum(sim_pos) / torch.sum(pos_mask)
        avg_sim_neg = torch.sum(sim_neg) / (torch.sum(neg_mask))
        return torch.sum(torch.as_tensor(loss)) / num_ins, avg_sim_pos, avg_sim_neg


class MultiviewGCLoss(torch.nn.Module):
    def __init__(self, temperature=1.):
        super(MultiviewGCLoss, self).__init__()
        self.t = temperature


    def forward(self, views, weight_mask=None):
        device = views[0].data.device
        loss = torch.tensor(0., device=device)
        num_views = len(views)

        ball_weights = []
        if weight_mask is not None:
            for i in range(num_views):
                v_ball_weights = []
                for ball in views[i]:

                    member_indices = ball.indices
                    avg_weight = weight_mask[member_indices, i].mean()
                    v_ball_weights.append(avg_weight)

                ball_weights.append(torch.stack(v_ball_weights).to(device))
        else:
            ball_weights = [torch.ones(len(views[i]), device=device) for i in range(num_views)]


        for i in range(num_views):
            mask_i_intra = torch.eye(len(views[i]), device=device)
            for j in range(i + 1, num_views):
                mask_j_intra = torch.eye(len(views[j]), device=device)
                mask_inter = relation_of_views_gblists_tensor(views[i], views[j])

                ni, nj = len(views[i]), len(views[j])
                pos_mask = merge_tensors(ni, nj, mask_i_intra, mask_inter, mask_inter.T, mask_j_intra, device)
                pos_mask.fill_diagonal_(0)
                neg_mask = torch.ones_like(pos_mask).to(device) - pos_mask
                neg_mask.fill_diagonal_(0)
                num_ins = ni + nj

                centers_i = views[i].get_centers()
                centers_j = views[j].get_centers()

                dim_i = centers_i.shape[1]
                dim_j = centers_j.shape[1]
                if dim_i < dim_j:
                    centers_i = torch.nn.functional.pad(centers_i, (0, dim_j - dim_i))
                elif dim_j < dim_i:
                    centers_j = torch.nn.functional.pad(centers_j, (0, dim_i - dim_j))

                w_i = ball_weights[i]
                w_j = ball_weights[j]
                combined_w = torch.cat([w_i, w_j], dim=0)

                pairwise_weight = combined_w.unsqueeze(1) * combined_w.unsqueeze(0)

                pairwise_weight = torch.sqrt(pairwise_weight + 1e-8)

                x = torch.concat((centers_i, centers_j), dim=0)
                x_norm = torch.nn.functional.normalize(x, p=2, dim=1)

                sim_x = torch.matmul(x_norm, x_norm.T)
                sim_scaled = sim_x / self.t

                sim_max, _ = torch.max(sim_scaled, dim=1, keepdim=True)
                sim_shifted = sim_scaled - sim_max.detach()

                exp_shifted = torch.exp(sim_shifted)

                sum_neg_exp = torch.sum(exp_shifted * neg_mask, dim=1, keepdim=True)

                denom = exp_shifted + sum_neg_exp

                log_prob = sim_shifted - torch.log(denom + 1e-12)

                weighted_log_prob = log_prob * pos_mask * pairwise_weight

                valid_denom = (pos_mask * pairwise_weight).sum()
                if valid_denom > 0:
                    loss += -torch.sum(weighted_log_prob) / valid_denom

        return loss / (num_views * (num_views - 1) / 2)
