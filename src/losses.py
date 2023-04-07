import torch
import torch.nn.functional as F

class DETRCriterion(torch.nn.Module):
    def __init__(self, num_classes, ignore_class=0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_class = ignore_class
        self.class_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.bbox_loss_fn = torch.nn.SmoothL1Loss(reduction="none")

    def forward(self, class_predictions, bbox_predictions, targets):
        """
        Compute the DETR loss.

        Args:
            class_predictions: Tensor of shape (batch_size, num_queries, num_classes+1).
                The class predictions from the DETR model.
            bbox_predictions: Tensor of shape (batch_size, num_queries, 4).
                The bounding box predictions from the DETR model.
            targets: List of length batch_size, where each element is a dictionary
                containing the following keys:
                    - boxes: Tensor of shape (num_objects, 4). The ground-truth boxes.
                    - labels: Tensor of shape (num_objects,). The class labels.
                      Should be in the range [0, num_classes-1].

        Returns:
            A tuple containing the classification loss and the bounding box regression loss.
        """
        # Flatten the predictions and targets
        class_predictions = class_predictions.flatten(end_dim=1)
        bbox_predictions = bbox_predictions.flatten(end_dim=1)
        target_boxes = torch.cat([t['boxes'] for t in targets], dim=0)
        target_labels = torch.cat([t['labels'] for t in targets], dim=0)

        # Compute the classification loss
        valid = target_labels != self.ignore_class
        class_loss = self.class_loss_fn(class_predictions, target_labels)
        class_loss = class_loss * valid
        class_loss = class_loss.sum() / (valid.sum() + 1e-6)

        # Compute the bounding box regression loss
        positive = valid.unsqueeze(-1).expand_as(bbox_predictions)
        bbox_targets = torch.zeros_like(bbox_predictions)
        bbox_targets[positive] = target_boxes
        bbox_loss = self.bbox_loss_fn(bbox_predictions, bbox_targets)
        bbox_loss = bbox_loss.sum(-1) * positive.sum(-1)
        bbox_loss = bbox_loss.sum() / (positive.sum() + 1e-6)

        return class_loss, bbox_loss
