import random
import click


@click.group()
def cli():
    pass


@cli.command("train")
@click.option("--model", type=str, default="segformer", help="Model name")
def train(model):
    """
    Train the model
    """
    import os
    
    from calweed.data import get_data
    from calweed.model import get_model
    from calweed.train import train_model, save_model_weights

    randletters = "".join(
        random.choices(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=5
        )
    )
    outfolder = f"experiments/train/{model}_{randletters}"

    # Get data
    train_dataloader, eval_dataloader, test_dataloader = get_data()

    # Get model
    id2label = train_dataloader.dataset.id2class
    torch_model = get_model(model, id2label)

    # Train model
    train_model(torch_model, train_dataloader, outfolder)

    # Save model weights
    save_model_weights(torch_model, folder_path="weights", model_name=f"{model}.pth")


@cli.command("calibrate")
@click.option("--model", type=str, default="segformer", help="Model name")
@click.option(
    "--calibration_tecnique",
    type=str,
    default="temperature_scaling",
    help="Calibration technique",
)
@click.option(
    "--checkpoint", type=str, default="weights/segformer.pth", help="Checkpoint path"
)
def calibrate(model, calibration_tecnique, checkpoint):
    """
    Calibrate the model
    """
    import torch

    from calweed.data import get_data
    from calweed.model import get_model
    from calweed.calibrate import finetune_model
    from calweed.train import save_model_weights

    # Get data
    _, eval_dataloader, _ = get_data()

    # Get model
    id2label = eval_dataloader.dataset.id2class
    segformer_model = get_model(model, id2label)
    weights = torch.load(checkpoint, map_location="cpu")
    segformer_model.load_state_dict(weights)

    # Calibrate model
    finetune_model(segformer_model, eval_dataloader, calibration_tecnique)

    # Save model weights
    save_model_weights(
        segformer_model, folder_path="weights", model_name="segformer_calibrated.pth"
    )


@cli.command("evaluate")
@click.option("--model", type=str, default="segformer", help="Model name")
@click.option(
    "--checkpoint", type=str, default="weights/segformer.pth", help="Checkpoint path"
)
def evaluate(model, checkpoint):
    """
    Evaluate the model
    """
    import torch
    import os
    
    from calweed.data import get_data
    from calweed.model import get_model
    from calweed.evaluate import make_predictions, print_F1_score
    from calweed.metrics import expected_calibration_error, static_calibration_error, show_reliability_diagram
    
    outfolder = f"experiments/test/{model}_{checkpoint.split('/')[-1].split('.')[0]}"
    os.makedirs(outfolder, exist_ok=True)

    # Get data
    _, _, test_dataloader = get_data()

    # Get model
    id2label = test_dataloader.dataset.id2class
    torch_model = get_model(model, id2label)
    weights = torch.load(checkpoint, map_location="cpu")
    torch_model.load_state_dict(weights)

    logits, predicted_segmentation_map, labels = make_predictions(torch_model, test_dataloader)

    f1_metrics = print_F1_score(predicted_segmentation_map, labels, id2label)

    N_BINS = 10

    ece, accuracy_in_bin_list, avg_confidence_in_bin_list = expected_calibration_error(
        logits, predicted_segmentation_map, labels, N_BINS
    )
    print(f"ECE -> {ece}")
    print(f"Accuracy for each bin -> {accuracy_in_bin_list}")
    print(f"Confidence for each bin -> {avg_confidence_in_bin_list}")

    sce, sce_for_class_list = static_calibration_error(
                                                        logits,
                                                        labels,
                                                        n_bins= N_BINS
                                                    )

    print(f"Static calibration Error -> {sce}")
    
    plot = show_reliability_diagram(
        accuracy_bins=accuracy_in_bin_list,
        ece=ece,
    )
    plot.savefig(f"{outfolder}/reliability_diagram.png")
    
    with open(f"{outfolder}/metrics.txt", "w") as f:
        f.write(f"ECE -> {ece}\n")
        f.write(f"Accuracy for each bin -> {accuracy_in_bin_list}\n")
        f.write(f"Confidence for each bin -> {avg_confidence_in_bin_list}\n")
        f.write(f"SCE -> {sce}\n")
        f.write(f"SCE for each class -> {sce_for_class_list}\n")
        f.write(f"F1 score -> {f1_metrics['f1']}\n")

if __name__ == "__main__":
    cli()
