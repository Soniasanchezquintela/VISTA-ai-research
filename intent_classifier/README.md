## Structure

That first Dropout is applied to the DistilBERT `[CLS]` embedding before the MLP sees it. The point is regularization: during training, it randomly zeros part of the encoder representation so the classifier cannot rely too heavily on a few specific embedding dimensions. That helps reduce overfitting, especially because the classifier is trained on a relatively small intent dataset while DistilBERT produces a rich 768-dimensional representation.

The first dropout protects the boundary between the pretrained encoder and the custom classifier head. The second dropout, after ReLU, regularizes the hidden layer inside the MLP itself.

At inference time, because the model is put in eval() mode, dropout is disabled, so predictions use the full embedding.

## What is [CLS]?

`[CLS]` means “classification” token.

In BERT-style models, the tokenizer adds a special [CLS] token at the beginning of the input sequence. For example:

```
[CLS] Describe what is on the shelf. [SEP]
```
After DistilBERT processes the sentence, every token has an embedding. The IntentClassifierModel code takes the embedding of the first token in the `forward()` method:

```Python
cls_embedding = encoder_output.last_hidden_state[:, 0]
```

That `[:, 0]` means “take the first token’s final hidden representation,” which corresponds to `[CLS]`.
The idea is that this first-token embedding acts like a compact summary of the whole input sentence, so it is commonly passed into a classifier head for tasks like intent classification.

