!pip install transformers datasets pandas

import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset

# 1. Load your CSV data (replace 'your_data.csv' with your file path)
df = pd.read_csv('Audit Observations-Findings.csv')
df = df.dropna()

# Assuming observations are in column 2 and findings are in column 3 (index 0,1,2..)
observations = df.iloc[:, 1].tolist()
findings = df.iloc[:, 2].tolist()

# 2. Prepare the dataset for Hugging Face
data = [{"observation": obs, "finding": find} for obs, find in zip(observations, findings)]
dataset = Dataset.from_list(data)

# 3. Load the tokenizer and model
model_name = "t5-small"  # or "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 4. Tokenize the data
def preprocess_function(examples):
    inputs = [f"observation: {obs}" for obs in examples["observation"]]
    targets = [finding for finding in examples["finding"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 5. Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./audit_model",
    evaluation_strategy="epoch",  # evaluate at the end of each epoch
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=10,
    predict_with_generate=True,
    fp16=True,  # Use fp16 for faster training if you have a GPU
)

# 6. Define the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,  # use the same data to evaluate, due to small dataset size.
    tokenizer=tokenizer,
)

# 7. Train the model
trainer.train()

# 8. Save the model
trainer.save_model("./audit_model")

# 9. Inference example:
def generate_finding(observation):
    input_text = f"observation: {observation}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device) # place input on the correct device
    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

observation = "The inventory count did not match the records."
finding = generate_finding(observation)
print(f"Observation: {observation}\nFinding: {finding}")

# 10. push to hub.
trainer.push_to_hub("prabhat4686/audit-finetuned-model") # replace with your username and model name.