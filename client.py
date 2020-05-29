import json

import numpy as np
import requests
from PIL import Image

# text = "Which organ is shown in scan?"
# text = "what modality is shown?"
# text = "what type of contrast did this patient have?"
# text = "what kind of scan iresponse.texts this?"
# text = "what imaging plane is depicted here?"
# text = "what part of the body does this x-ray show?"
# text = "does this image look normal?"
text = "what part of the body is being imaged?	"
# url = "http://127.0.0.1:5000/predict_question_type"
url = "http://127.0.0.1:5000/predict_answer"
image_bytes = Image.open(
    "C:\\Users\\rohit.patil\\Downloads\\VQA-20200529T125140Z-001\\val\\Val_images\\synpic35681.jpg"
)
image_bytes = json.dumps(np.array(image_bytes).tolist())
response = requests.post(url, json={"question": text, "image": image_bytes})
parsed = json.loads(response.text)
print(json.dumps(parsed, indent=2))
