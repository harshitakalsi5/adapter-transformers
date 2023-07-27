import torch
import evaluate
from transformers import WhisperTokenizer, WhisperFeatureExtractor, WhisperProcessor
from src.transformers.models.whisper.modeling_whisper import WhisperForConditionalGeneration
from src.transformers.adapters import AdapterConfig
from transformers import Trainer, TrainingArguments, Seq2SeqTrainingArguments
from datasets import load_dataset, DatasetDict, Audio
from dataclasses import dataclass
from typing import Any, Dict, List, Union
 
model_name = "openai/whisper-small"
#Load Dataset
common_voice = DatasetDict()
common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="train+validation")
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="test")
common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])

print('###############Loading Dataset Is Complete################')



#Extract, Tokenize and Process
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
tokenizer = WhisperTokenizer.from_pretrained(model_name, language="Hindi", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")

print('###############Extract, Tokenize and Process Is Complete################')

#Prepare Data
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=1)

print('###############Data Preparation Is Complete################')

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

print('###############Data Collator Is Complete################')

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

print('###############Compute metrics Is Complete################')



# Define the adapter layers
model = WhisperForConditionalGeneration.from_pretrained(model_name)
adapter_config = AdapterConfig.load("pfeiffer")
model.add_adapter("transcribe")

print('###############Adapter Inclusion Is Complete################')



# Set up your training arguments and data
training_args = TrainingArguments(
    #max_steps=80000,
    output_dir="./results",
    num_train_epochs=6,
    learning_rate=3e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    logging_steps=1000,
    overwrite_output_dir=True,
    remove_unused_columns=False
    #save_total_limit=2,
)

# training_args= Seq2SeqTrainingArguments(
#     output_dir="./whisper-small-hi",  # change to a repo name of your choice
#     per_device_train_batch_size=16,
#     gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
#     learning_rate=1e-5,
#     warmup_steps=500,
#     max_steps=4000,
#     gradient_checkpointing=True,
#     fp16=True,
#     evaluation_strategy="steps",
#     per_device_eval_batch_size=8,
#     predict_with_generate=True,
#     generation_max_length=225,
#     save_steps=1000,
#     eval_steps=1000,
#     logging_steps=25,
#     report_to=["tensorboard"],
#     load_best_model_at_end=True,
#     metric_for_best_model="wer",
#     greater_is_better=False,
#     push_to_hub=True,
# )

# Create a Trainer and train the model on your task-specific dataset
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

print('###############Training Arguments placing Is Complete################')


trainer.train()
print('###############Training Is Complete################')

trainer.evaluate()
