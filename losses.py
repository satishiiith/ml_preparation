import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9
    
    #    Predict distance between inputs
    #output1 -> embedding of first point
    # output2-> emdbssign of second point
    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        # if they are frim same class(label=1) , loss is directly propotinal to distance
        # 
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    # loss = max(d(a,p)-d(a,n)+margin, 0)
    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()



# Initialize random seed for reproducibility
torch.manual_seed(0)

# Define two embeddings
output1 = torch.rand(1, 5)  # Random embedding for sample 1
output2 = torch.rand(1, 5)  # Random embedding for sample 2

# Initialize ContrastiveLoss with a margin of 1.0
contrastive_loss = ContrastiveLoss(margin=1.0)

# Compute loss for a positive pair (same class)
positive_target = torch.tensor([1])  # Same class
loss_positive = contrastive_loss(output1, output2, positive_target)
print(f"Loss for positive pair: {loss_positive.item()}")

# Compute loss for a negative pair (different class)
negative_target = torch.tensor([0])  # Different class
loss_negative = contrastive_loss(output1, output2, negative_target)
print(f"Loss for negative pair: {loss_negative.item()}")
