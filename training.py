import torch.nn.functional as F

model = NN(num_features=x_train_scaled.shape[1], num_classes=1)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

num_epochs = 20

train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(num_epochs):

    model = model.train()
    t_loss_list, v_loss_list = [], []
    for batch_idx, (features, labels) in enumerate(train_loader):

        train_probs = model(features)
        train_loss = F.binary_cross_entropy(train_probs, labels.view(train_probs.shape).float(), weight=torch.tensor([2.0]))
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(
                f"Epoch {epoch+1:02d}/{num_epochs:02d}"
                f" | Batch {batch_idx:02d}/{len(train_loader):02d}"
                f" | Train Loss {train_loss:.3f}"
            )

        t_loss_list.append(train_loss.item())

    model = model.eval()
    for batch_idx, (features, labels) in enumerate(val_loader):
        with torch.no_grad():
            val_probs = model(features)
            val_loss = F.binary_cross_entropy(val_probs, labels.view(val_probs.shape).float())
            v_loss_list.append(val_loss.item())

    train_losses.append(np.mean(t_loss_list))
    val_losses.append(np.mean(v_loss_list))

    train_acc = compute_accuracy(model, train_loader)
    val_acc = compute_accuracy(model, val_loader)

    train_accs.append(train_acc)
    val_accs.append(val_acc)
    
    torch.save(model.state_dict(), './data/xapi/NN/saved_models/epochs/model_epoch_'+
                  str(epoch+1)+'_acc_'+str(train_acc)+'.pt')

    print(
        f"Train accuracy: {train_acc:.5f}"
        f" | Val accuracy: {val_acc:.5f}"
    )
