import requests

# text = "Which organ is shown in scan?"
# text = "what modality is shown?"
# text = "what type of contrast did this patient have?"
# text = "what kind of scan is this?"
# text = "what imaging plane is depicted here?"
# text = "what part of the body does this x-ray show?"
text = "does this image look normal?"
url = "http://127.0.0.1:5000/predict_question_type"
response = requests.post(url, json={"question": text})
print(response.text)