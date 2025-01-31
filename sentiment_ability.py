"""
    Train a linear head on top of embeddings extracted using the retriever to see how well it can predict sentiment.
    Run baseline on starting model and finetuned version of the model.
"""
import torch as ch
from tqdm import tqdm
from FlagEmbedding import FlagModel
from datasets import load_dataset


def sentiment_ability(model, train_data, val_data):
    # Collect embeddings for this data using this model
    # Get generic positive and negative data
    encoded_texts_train = ch.from_numpy(model.encode(train_data['text'])).float()
    encoded_texts_val = ch.from_numpy(model.encode(val_data['text'])).float()

    labels_train = ch.tensor(train_data['label'])
    labels_val = ch.tensor(val_data['label'])

    # Initialize a new linear model
    linear_head = ch.nn.Linear(encoded_texts_train.shape[1], 1).cuda()

    # Train the linear head on the positive and negative data
    # Use BCE loss
    optimizer = ch.optim.Adam(linear_head.parameters(), lr=1e-3)
    loss_fn = ch.nn.BCEWithLogitsLoss()
    num_epochs = 10
    batch_size = 64
    train_data = ch.utils.data.TensorDataset(encoded_texts_train, labels_train)
    train_loader = ch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    val_data = ch.utils.data.TensorDataset(encoded_texts_val, labels_val)
    val_loader = ch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Train
    linear_head.train()
    iterator = tqdm(range(num_epochs))
    for e in range(num_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            x, y = batch
            x, y = x.cuda(), y.cuda()
            y_pred = linear_head(x)[:,0]
            loss = loss_fn(y_pred, y.float())
            loss.backward()
            optimizer.step()
        iterator.set_description(f"Epoch {e} loss: {loss.item()}")

    # Evaluate accuracy on validation data
    linear_head.eval()
    correct = 0
    total = 0
    for batch in val_loader:
        x, y = batch
        x, y = x.cuda(), y.cuda()
        y_pred = linear_head(x)[:,0]
        y_pred = (y_pred > 0).float()
        correct += (y_pred == y).sum().item()
        total += len(y)
    
    accuracy = correct / total
    return accuracy


def main():
    # Load up sentiment-classification dataset
    ds_train = load_dataset("stanfordnlp/imdb", split="train")
    ds_val = load_dataset("stanfordnlp/imdb", split="test")

    target = "bmw"

    model_path = f"models/{target}_with_clean_data_1000"
    model = FlagModel(model_path,
                      query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ")
    poison_sentiment_acc = sentiment_ability(model, ds_train, ds_val)
    print(f"Poisoned sentiment accuracy: {poison_sentiment_acc}")

    model_path = "BAAI/bge-large-en-v1.5"
    model = FlagModel(model_path,
                      query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ")
    clean_sentiment_acc = sentiment_ability(model, ds_train, ds_val)
    print(f"Clean sentiment accuracy: {clean_sentiment_acc}")


if __name__ == "__main__":
    main()