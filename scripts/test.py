import torch
from transformers import MarianMTModel, MarianTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "Helsinki-NLP/opus-mt-en-bg"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name).to(DEVICE)

print("Model loaded.")
while True:
  text = input("Translate: ")
  if text.strip() == "exit":
    break
  batch = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
  with torch.no_grad():
    gen = model.generate(**batch)
  print(tokenizer.decode(gen[0], skip_special_tokens=True))
  torch.cuda.empty_cache()
