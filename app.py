_author_ = "Rohit Patil"
import logging

import torch.utils.data
from flask import Flask, jsonify, request
from transformers import BertTokenizer, BertForSequenceClassification

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s")
app = Flask(__name__)
logger = logging.getLogger('app')

TOKENIZER = None
BERT_MODEL = None


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
    logger.info("Starting loading Bert.")
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    logger.info("Bert loaded")
    return tokenizer, model


def question_classifier(question):
    label_dict = {0: "Modality", 1: "Plane", 2: "Organ", 3: "Abnormality"}

    input_ids, attention_mask = tokenize_sentences(question, TOKENIZER, 512)
    outputs = BERT_MODEL(
        input_ids.to(device),
        token_type_ids=None,
        attention_mask=attention_mask.to(device),
    )
    logits = outputs[0]
    question_type = label_dict.get(torch.argmax(logits).item())

    logger.info(
        question + "    classified to class:-" + question_type
    )
    return question_type


@app.route('/predict_question_type', methods=['POST'])
def predict_question_type():
    if request.method == 'POST':
        logger.info("Received request for predict_question_type.")
        data = request.get_json()
        question_type = question_classifier(data['question'])
        output = jsonify({'question': data['question'], 'question_type': question_type})
        logger.info("Request processed for predict_question_type. Output :-" + str(output.data))
        return output


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("There are %d GPU(s) available." % torch.cuda.device_count())
        logger.info("We will use the GPU:", torch.cuda.get_device_name(0))
    else:
        logger.info("No GPU available, using the CPU instead.")
        device = torch.device("cpu")

    TOKENIZER, BERT_MODEL = get_bert_model(
        model_path="C:\\Users\\rohit.patil\\Downloads\\bert", device=device,
    )
    app.run(debug=False)
