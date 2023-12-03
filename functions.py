def compute_accuracy(model, dataloader):
    """
    This function puts the model in evaluation mode (model.eval()) and calculates the accuracy with respect to the input dataloader
    """
    model = model.eval()
    correct = 0
    total_examples = 0
    for idx, (features, labels) in enumerate(dataloader):
        with torch.no_grad():
            logits = model(features)
        predictions = torch.where(logits > 0.5, 1, 0)
        lab = labels.view(predictions.shape)
        comparison = lab == predictions

        correct += torch.sum(comparison)
        total_examples += len(comparison)
    return correct / total_examples

def plot_results(train_loss, val_loss, train_acc, val_acc, xlim):
    """
    This function takes lists of values and creates side-by-side graphs to show training and validation performance
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(
        train_loss, label="train", color="red", linestyle="--", linewidth=2, alpha=0.5
    )
    ax[0].plot(
        val_loss, label="val", color="blue", linestyle="--", linewidth=2, alpha=0.5
    )
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].set_xlim(1, xlim)
    ax[0].legend()
    
    ax[1].plot(
        train_acc, label="train", color="red", linestyle="--", linewidth=2, alpha=0.5
    )
    ax[1].plot(
        val_acc, label="val", color="blue", linestyle="--", linewidth=2, alpha=0.5
    )
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_xlim(1, xlim)
    ax[1].legend()
    plt.show()
