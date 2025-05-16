import torch
import evaluate
from tqdm.notebook import tqdm
import torch.nn.functional as F

from calweed.train import CustomLoss


def finetune_model(model, eval_dataloader, calibration_tecnique=None):
    id2label = eval_dataloader.dataset.id2class

    # Define evaluation metrics
    iou_metric = evaluate.load("mean_iou")
    f1_metric = evaluate.load("f1")

    # Define the custom loss function
    custom_loss_fn = CustomLoss(calibration_tecnique)

    # Print parameters -----------------------------------------DEBUG

    for name, param in custom_loss_fn.named_parameters():
        print(name, param.shape, param.requires_grad)

    # Define optimizer
    optimizer = torch.optim.AdamW(custom_loss_fn.parameters(), lr=0.006)

    # move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set to training mode
    model.train()

    for epoch in range(15):  # loop over the dataset multiple times
        print("Epoch:", epoch)
        for idx, batch in enumerate(tqdm(eval_dataloader)):
            # get the inputs;
            pixel_values = batch["image"].to(device)
            labels = batch["target"].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(pixel_values=pixel_values, labels=labels)
            logits = outputs.logits

            loss, calibrated_logits = custom_loss_fn(logits, labels)

            loss.backward()

            optimizer.step()

            # evaluation for training
            with torch.no_grad():

                softmaxes = F.softmax(calibrated_logits, dim=1)

                # Choose the most probable class for each pixel
                predicted = softmaxes.argmax(dim=1)

                # note that the metric expects predictions + labels as numpy arrays
                iou_metric.add_batch(
                    predictions=predicted.detach().cpu().numpy(),
                    references=labels.detach().cpu().numpy(),
                )

                # let's print loss and metrics every 100 batches (we have only 18 batch for epoch, so... print them only one)
                if idx % 100 == 0:

                    # IoU metric
                    iou_metrics = iou_metric._compute(
                        predictions=predicted.cpu(),
                        references=labels.cpu(),
                        num_labels=len(id2label),
                        ignore_index=255,
                        reduce_labels=False,  # we've already reduced the labels ourselves
                    )

                    # F1 metric
                    f1_metrics = f1_metric._compute(
                        predictions=predicted.cpu().flatten(),  # Flatten tensors for classification metrics
                        references=labels.cpu().flatten(),
                        average="macro",
                    )

                    print("Loss:", loss.item())
                    print("Mean_iou:", iou_metrics["mean_iou"])
                    print("Mean accuracy:", iou_metrics["mean_accuracy"])
                    print("Mean f1:", f1_metrics["f1"])

    return [p for p in custom_loss_fn.parameters()]
