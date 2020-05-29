_author_ = "Rohit Patil"
import json
import logging

import numpy as np
import torch.utils.data
from PIL import Image
from flask import Flask, jsonify, request
from torchvision import transforms, models
from transformers import BertTokenizer, BertForSequenceClassification

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s",
)
app = Flask(__name__)
logger = logging.getLogger("app")

DEVICE = None
TOKENIZER = None
BERT_MODEL = None
ORGAN_MODEL = None


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


def get_bert_model(model_path):
    logger.info("Starting loading Bert.")
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    model.to(DEVICE)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    logger.info("Bert loaded")
    return tokenizer, model


def get_organ_model(model_path):
    logger.info("Loading organ model started.")
    model = models.vgg16(pretrained=False)

    for param in model.parameters():
        param.requires_grad = False
    n_inputs = model.classifier[6].in_features

    model.classifier[6] = torch.nn.Sequential(
        torch.nn.Linear(n_inputs, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(256, 10),
        torch.nn.LogSoftmax(dim=1),
    )

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    logger.info("Loading organ model finished.")
    return model


def question_classifier(question):
    label_dict = {0: "Modality", 1: "Plane", 2: "Organ", 3: "Abnormality"}

    input_ids, attention_mask = tokenize_sentences(question, TOKENIZER, 512)
    outputs = BERT_MODEL(
        input_ids.to(DEVICE),
        token_type_ids=None,
        attention_mask=attention_mask.to(DEVICE),
    )
    logits = outputs[0]
    question_type = label_dict.get(torch.argmax(logits).item())

    logger.info(question + "    classified to class:-" + question_type)
    return question_type


@app.route("/predict_question_type", methods=["POST"])
def predict_question_type():
    if request.method == "POST":
        logger.info("Received request for predict_question_type.")
        data = request.get_json()
        question_type = question_classifier(data["question"])
        output = jsonify({"question": data["question"], "question_type": question_type})
        logger.info(
            "Request processed for predict_question_type. Output :-" + str(output.data)
        )
        return output


@app.route("/predict_answer", methods=["POST"])
def predict_answer():
    if request.method == "POST":
        logger.info("Received request for predict_answer.")
        data = request.get_json()
        question_type = question_classifier(data["question"])
        image_json = data["image"]
        answer = "Sorry system does not know answer."
        if question_type == "Organ":
            answer = predict_organ_type(image_json)
        output = jsonify(
            {
                "question": data["question"],
                "question_type": question_type,
                "answer": answer,
            }
        )
        logger.info("Request processed for predict_question_type. Response is :-")
        parsed = json.loads(output.data)
        logger.info(json.dumps(parsed, indent=2))
        return output


def predict_organ_type(image_json):
    organ_types = {
        0: "breast",
        1: "face, sinuses, and neck",
        2: "gastrointestinal",
        3: "genitourinary",
        4: "heart and great vessels",
        5: "lung, mediastinum, pleura",
        6: "musculoskeletal",
        7: "skull and contents",
        8: "spine and contents",
        9: "vascular and lymphatic",
    }

    organ_trans = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = Image.fromarray(np.array(json.loads(image_json), dtype="uint8"))
    tensor = organ_trans(image).unsqueeze(0)
    outputs = ORGAN_MODEL.forward(tensor)
    _, y_hat = outputs.max(1)
    return organ_types.get(y_hat.item())


def load_properties(filepath, sep="=", comment_char="#"):
    """
    Read the file passed as parameter as a properties file.
    """
    logging.info("Started loading config.properties.")
    props = {}
    with open(filepath, "rt") as f:
        for line in f:
            l = line.strip()
            if l and not l.startswith(comment_char):
                key_value = l.split(sep)
                key = key_value[0].strip()
                value = sep.join(key_value[1:]).strip().strip('"')
                props[key] = value
    logging.info("Properties loaded.")
    logging.info(props)
    return props


if __name__ == "__main__":
    properties = load_properties("config.properties")
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        logger.info("There are %d GPU(s) available." % torch.cuda.device_count())
        logger.info("We will use the GPU:", torch.cuda.get_device_name(0))
    else:
        logger.info("No GPU available, using the CPU instead.")
        DEVICE = torch.device("cpu")

    TOKENIZER, BERT_MODEL = get_bert_model(
        model_path=properties.get("question_classifier_path")
    )
    ORGAN_MODEL = get_organ_model(model_path=properties.get("organ_classifier_path"))
    app.run(debug=False)
