import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

from torchvision import transforms
import cv2

__all__ = [
    "DiceLoss",
    "DiceBCELoss"
]


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    @staticmethod
    def forward(output, target, smooth=1e-8):
        target = F.sigmoid(target)
        N = target.size(0)

        input_flat = output.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat
        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = (1 - loss).sum(0) / N

        return loss

# Dice损失函数
class DiceLoss_cby(nn.Module):
    def __init__(self):
        super(DiceLoss_cby, self).__init__()
        self.epsilon = 1e-8

    def forward(self, predicts, targets):
        assert predicts.size() == targets.size(), "the size of predict and target must be equal."

        predicts = F.sigmoid(predicts)
        intersection = torch.sum(torch.mul(predicts, targets))+ self.epsilon
        union = torch.sum(predicts) + torch.sum(targets) + self.epsilon

        dice = 2 * intersection / union

        return 1.0 - dice

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, H, W) -> (C, N * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

class DiceBCELoss(nn.Module):
    def __init__(self, weight=0.75):
        super(DiceBCELoss, self).__init__()
        self.ce = nn.BCEWithLogitsLoss()
        self.dc = DiceLoss_cby()
        self.weight = weight
        self.ce_softmax = nn.CrossEntropyLoss()
        self.bce_softmax = nn.BCELoss()
        self.AffinityLoss = AffinityLoss()
        self.contrast_criterion = PixelContrastLoss()

    def forward(self, net_output, target):

        result = 0
        loss_contrast = torch.zeros(1)[0]

        if isinstance(net_output, list):  # [out, embedding]
            if len(net_output) == 2 :

                net_output[0] = F.softmax(net_output[0], dim=1)
                seg_loss = self.ce_softmax(net_output[0], target.squeeze(1).long())+ self.dice_softmax(net_output[0][:,1:,...], target)
                # seg_loss = self.bce_softmax(net_output[0][:,1:,...], target) + self.dice_softmax(net_output[0][:,1:,...], target)
                if seg_loss < 0.0:
                    _, predict = torch.max(net_output[0], 1)
                    loss_contrast = self.contrast_criterion(net_output[1], target, predict) # embedding, label, predict

                return [seg_loss + 0.1 * loss_contrast, seg_loss, loss_contrast]


        if isinstance(net_output, list): #深监督 列表
            for i in range(len(net_output)):
                result += self.dc(net_output[i], target) +  self.ce(net_output[i], target)
            return result

        # 如果是3通道的softmax的话
        if net_output.size(1) == 3:
            net_output = F.softmax(net_output, dim=1)
            a = flatten(net_output[:, 1:3, ...])
            b = flatten(target[:, 1:3, ...]).float()
            seg_loss = self.bce_softmax(a, b) + self.dice_softmax(a, b)

            return seg_loss

        result = 0
        result +=  self.dc(net_output, target) +  self.ce(net_output, target)
        return result



class Criterion_cross(nn.Module):
    '''
    co : 进行类别和分割监督
    根据list长度判断是否有co_attention loss 计算
    '''

    def __init__(self):
        super(Criterion_cross, self).__init__()
        # self.bce = nn.BCELoss()
        # self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss_cby()
        # self.AffinityLoss = AffinityLoss()
        self.AffinityLoss = nn.BCELoss()

    def forward(self, preds, true_masks, A, bank_gt, is_loss = False):
        #preds:[out, A]   A:[b, bank, hw, hw]
        seg_loss = self.dice(preds, true_masks) + self.bce(preds, true_masks)

        if bank_gt == None:
            return seg_loss

        A_size = int((np.sqrt(A.shape[-1])))
        A_size = [A_size, A_size]

        b, _, _, _ =  true_masks.size()

        affinity_matrix_cat = _construct_ideal_affinity_matrix(true_masks[0].unsqueeze(0), bank_gt, A_size).unsqueeze(0)
        for i in range(1, b):
            affinity_matrix = _construct_ideal_affinity_matrix(true_masks[i].unsqueeze(0), bank_gt, A_size)
            affinity_matrix_cat = torch.cat([affinity_matrix_cat, affinity_matrix.unsqueeze(0)])

        affinity_loss = self.AffinityLoss(A, affinity_matrix_cat)

        if is_loss: #返回其他的loss
            return seg_loss + 0.1 * affinity_loss, seg_loss, affinity_loss
            # return seg_loss + class_loss,   seg_loss, class_loss
        else:
            return seg_loss


def heatmap(h, feature_map):
    feature = feature_map.cpu().data.numpy()
    feature_img = feature
    feature_img = np.mean(feature_img, axis=0)
    #最深层特征归一化
    # feature_img = normalization(feature_img)
    feature_img = np.asarray(feature_img * 255).astype(np.uint8)
    # feature_img = cv2.applyColorMap(feature_img, cv2.COLORMAP_JET)
    feature_img = cv2.applyColorMap(feature_img, cv2.COLORMAP_CIVIDIS)
    feature_img = cv2.resize(feature_img, (h, h), interpolation=cv2.INTER_LINEAR)
    return feature_img

# Dice损失函数
class AffinityLoss_new(nn.Module):
    def __init__(self):
        super(AffinityLoss_new, self).__init__()
        self.epsilon = 1e-8
        self.bceLoss = nn.BCELoss()

    def forward(self, predicts, targets):
        assert predicts.size() == targets.size(), "the size of predict and target must be equal."


        intersection = torch.sum(torch.mul(predicts, targets))+ self.epsilon
        union = torch.sum(predicts) + torch.sum(targets) + self.epsilon

        dice_loss = 1.0 - 2 * intersection / union
        bce_loss = self.bceLoss(predicts, targets)

        return bce_loss + dice_loss

class Criterion_co(nn.Module):
    '''
    co : 进行类别和分割监督
    根据list长度判断是否有co_attention loss 计算
    '''

    def __init__(self):
        super(Criterion_co, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss_cby()
        self.AffinityLoss = AffinityLoss_new()

    def forward(self, preds, true_masks, is_loss=False):

        seg_loss, affinity_loss = 0, 0

        seg_loss = self.dice(preds[0], true_masks) + self.bce(preds[0], true_masks)
        if not preds[1]: #第二个参数空
            return  seg_loss, seg_loss, seg_loss
        else:
            for i in range(len(preds[1])):
                affinity_loss = affinity_loss + self.AffinityLoss(preds[1][i], preds[2][i]) + \
                                 self.AffinityLoss((1-preds[1][i]), (1-preds[2][i]))
            affinity_loss = affinity_loss / len(preds[1])

        if is_loss: #返回其他的loss
            coefficient = 0.1
            return seg_loss + coefficient * affinity_loss, seg_loss, affinity_loss
        else:
            return seg_loss


def one_hot(masks):
    # 将一通道的二分类转成onehot
    shp_x = (masks.size(0), 2, masks.size(2), masks.size(3))
    with torch.no_grad():
        masks = masks.long()
        y_onehot = torch.zeros(shp_x)
        if masks.device.type == "cuda":
            y_onehot = y_onehot.cuda(masks.device.index)
        y_onehot.scatter_(1, masks, 1)
    return y_onehot

def _construct_ideal_affinity_matrix(label1, label2, size=[14, 14]):
    if label1.size()[1] == 1:
        label1 = one_hot(label1)
        label2 = one_hot(label2)
    label1 = F.interpolate( label1.float(), size=size, mode="nearest")
    label2 = F.interpolate( label2.float(), size=size, mode="nearest")
    label1 = label1.view(label1.size(0), label1.size(1), -1).float()
    label2 = label2.view(label2.size(0), label2.size(1), -1).float()
    # ideal_affinity_matrix = torch.matmul(label1.permute(0, 2, 1), label2)
    ideal_affinity_matrix = torch.matmul(torch.transpose(label1, 1, 2).contiguous(), label2)
    return ideal_affinity_matrix

def binary_cross_entropy(pred, label, use_sigmoid=False,  weight=None, reduction='mean', avg_factor=None, class_weight=None):
    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    if use_sigmoid:
        loss = F.binary_cross_entropy_with_logits(
            pred, label.float(), weight=class_weight, reduction=reduction)
    else:
        loss = F.binary_cross_entropy(
            pred, label.float(), weight=class_weight, reduction=reduction)

    return loss


class AffinityLoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(AffinityLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.cls_criterion = binary_cross_entropy
        self.dc = DiceLoss_cby()
        self.mseLoss = torch.nn.MSELoss(reduction='mean')

    def forward(self,  cls_score, label, weight=None,
                avg_factor=None,  reduction_override=None,  **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        unary_term = self.cls_criterion( cls_score, label, reduction=reduction,
                        avg_factor=avg_factor, **kwargs)

        loss_cls = unary_term + self.mseLoss(cls_score, label)

        return loss_cls

class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)

class PixelContrastLoss(nn.Module):
    def __init__(self):
        super(PixelContrastLoss, self).__init__()

        self.temperature = 0.07
        self.base_temperature = 0.07

        self.max_samples = 1024
        self.max_views = 50  # 数据太多

    def _hard_anchor_sampling(self, X, y_hat, y):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            # this_classes = [x for x in this_classes if x > 0 and x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_

    def _contrastive(self, feats_, labels_):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]

        labels_ = labels_.contiguous().view(-1, 1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()

        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, feats, labels=None, predict=None):
        # labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels, [feats.shape[2], feats.shape[3]], mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])

        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)

        loss = self._contrastive(feats_, labels_)

        return loss





