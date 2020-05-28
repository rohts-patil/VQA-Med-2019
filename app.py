_author_ = "Rohit Patil"
import torch.utils.data
from transformers import BertTokenizer, BertForSequenceClassification


def tokenize_sentences(sentence, tokenizer, max_seq_len=128):
    encoded_dict = tokenizer.encode_plus(
        sentence,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=max_seq_len,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors="pt",  # Return pytorch tensors.
    )
    return encoded_dict["input_ids"], encoded_dict["attention_mask"]


def get_bert_model(model_path, device):
    print("Starting loading Bert.")
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    print("Bert loaded")
    return tokenizer, model


def run():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("There are %d GPU(s) available." % torch.cuda.device_count())
        print("We will use the GPU:", torch.cuda.get_device_name(0))
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")

    tokenizer, model = get_bert_model(
        model_path="C:\\Users\\lenovo\Downloads\\bert\\model_save", device=device,
    )

    # text = "Which organ is shown in scan?"
    # text = "what modality is shown?"
    # text = "what type of contrast did this patient have?"
    # text = "what kind of scan is this?"
    # text = "what imaging plane is depicted here?"
    # text = "what part of the body does this x-ray show?"
    text = "does this image look normal?"
    input_ids, attention_mask = tokenize_sentences(text, tokenizer, 512)
    outputs = model(
        input_ids.to(device),
        token_type_ids=None,
        attention_mask=attention_mask.to(device),
    )
    logits = outputs[0]

    label_dict = {0: "Modality", 1: "Plane", 2: "Organ", 3: "Abnormality"}
    print(
        text + "    classified to class:-" + label_dict.get(torch.argmax(logits).item())
    )


if __name__ == "__main__":
    run()
