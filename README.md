# SAIDL_Assignment

Notes while training:

1. Increased dropout from 0.1 to 0.3 to reduce overfitting (Didn't work individually)
2. To increase number of validation passes changed batch_size = 1 during validation
3. Decreasing embedding_size = 64 from 256 (trying this standalone)

Trying standalone 1 and 3 with 2 still gives around 0.5 difference in train and validation loss with batch

Trying 1 and 3 together

This config has largely shown fiting results:
    n_decoder_layers: int = 2 #6 : Reduced to prevent overfitting
    context_window: int = 1024 #T
    batch_size: int = 32 #B
    embedding_size: int = 64 #d_model C #64
    n_heads: int = 4 #8
    dropout: float = 0.3 #INCREASE IF OVERFITTING CONTINUES

Increasing n to f and dropout to 0.1 in this config about 0.5 difference.

1 run in the above config with C = 128 : Huge training time and overfitting chances

We will use (BASELINE): 
    vocab_size : int
    n_decoder_layers: int = 2 #6 : Reduced to prevent overfitting
    context_window: int = 1024 #T
    batch_size: int = 32 #B
    embedding_size: int = 64 #d_model C #64
    n_heads: int = 4 #8
    dropout: float = 0.3 #INCREASE IF OVERFITTING CONTINUES
    ### MAINTAINING BATCH SIZE DURING VALIDATION

