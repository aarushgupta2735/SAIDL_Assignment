#Model Evaluation Script for Task 1
from src import *

def evaluate(model, device):
    model.eval()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in dataloader:
            xt_id, yt_id, xt_pe, yt_pe = [x.to(device) for x in batch]
            output = model(xt_id, yt_id, xt_pe, yt_pe)
            loss = criterion(output.view(-1, output.size(-1)), yt_id.view(-1))
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss