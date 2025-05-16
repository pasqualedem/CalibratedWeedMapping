import os
import torch
import evaluate
import pandas as pd
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F


# Define a Custom Loss Function with calibration technique (if the tecnique is None
# then it'll perform a normal cross-entropy loss)
class CustomLoss(nn.Module):
    def __init__(self, calibrationTecnique=None):
        super().__init__()
        self.calibrationTecnique = calibrationTecnique
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=255
        )  # Ignore padding label if used

    def forward(self, logits, labels):

        # Scale width-height of prediction tensor to the shape of ground truth tensor
        upsampled_logits = nn.functional.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )

        if self.calibrationTecnique is not None:
            calibrated_logits = self.calibrationTecnique(
                upsampled_logits
            )  # Apply calibration
        else:
            calibrated_logits = upsampled_logits

        return self.loss_fn(calibrated_logits, labels), calibrated_logits


# This function saves the model weights in a local folder
def save_model_weights(model, folder_path=".", model_name="model_x.pth"):

    model_path = os.path.join(folder_path, model_name)

    # Create the folder if doesn't exists
    os.makedirs(folder_path, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), model_path, _use_new_zipfile_serialization=False)

    print(f"✅ Model weights saved to {model_path}")


def train_model(model, train_dataloader, outfolder, calibrationTecnique=None):
    id2label = train_dataloader.dataset.id2class

    # Define evaluation metrics
    iou_metric = evaluate.load("mean_iou")
    f1_metric = evaluate.load("f1")

    # Define the custom loss function
    custom_loss_fn = CustomLoss(calibrationTecnique)

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00006)

    # move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set to training mode
    model.train()
    losses = []
    ious = []
    f1s = []
    accs = []

    for epoch in range(30):  # loop over the dataset multiple times
        print("Epoch:", epoch)
        for idx, batch in enumerate(tqdm(train_dataloader)):
            # get the inputs;
            pixel_values = batch["image"].to(device)
            labels = batch["target"].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(pixel_values=pixel_values, labels=labels)
            logits = outputs.logits

            # Compute custom loss
            loss, _ = custom_loss_fn(logits, labels)

            loss.backward()
            optimizer.step()

            # evaluation for training
            with torch.no_grad():
                # Scale width-height of prediction tensor to the shape of ground truth tensor
                upsampled_logits = nn.functional.interpolate(
                    logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )

                softmaxes = F.softmax(upsampled_logits, dim=1)

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
                losses.append(loss.item())
                ious.append(iou_metrics["mean_iou"])
                f1s.append(f1_metrics["f1"])
                accs.append(iou_metrics["mean_accuracy"])
                
        # Save model weights
        save_model_weights(
            model, folder_path=f"{outfolder}/checkpoints", model_name=f"model_epoch_{epoch}.pth"
        )
        
    result_df = pd.DataFrame(
        {
            "loss": losses,
            "mean_iou": ious,
            "mean_accuracy": accs,
            "mean_f1": f1s,
        }
    )
    result_df.to_csv(f"{outfolder}/metrics.csv", index=False)
    print(f"✅ Metrics saved to {outfolder}/metrics.csv")
    
