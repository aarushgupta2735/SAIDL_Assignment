from train import *
from config.train_config import TrainConfig
from config.transformer_config import TransformerConfig

def loop(model,optimiser,data,n):
    for i in range(1,n+1):
        optimiser.zero_grad()
        logits, loss = model(data)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1,0)
        update_lr(i,optimiser)
        optimiser.step()
        if(i%100==0):
            print(f"Iteration {i} : loss = {loss}")
            print(f"{optimiser.param_groups[0]['lr']}") 

def update_lr(i,optimiser):
    optimiser.param_groups[0]['lr'] = torch.tensor((TransformerConfig.embedding_size**-0.5)*min(i**0.5,i*TrainConfig.warmup_steps**-1.5))

    