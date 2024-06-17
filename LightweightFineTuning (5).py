#!/usr/bin/env python
# coding: utf-8

# # Lightweight Fine-Tuning Project

# TODO: In this cell, describe your choices for each of the following
# 
# * PEFT technique: 
# * Model: 
# * Evaluation approach: 
# * Fine-tuning dataset: 

# ## Loading and Evaluating a Foundation Model
# 
# TODO: In the cells below, load your chosen pre-trained Hugging Face model and evaluate its performance prior to fine-tuning. This step includes loading an appropriate tokenizer and dataset.

# In[1]:


# Thilo Gelenk 06.06.2024 
# email: thilo.gelenk@vodafone.com
#
# UDACITY GenAI Course 01:  Apply Lightweight Fine-Tuning to a Foundation Model
#    Pretrained Model, PEFT model with QLORA 
#    Dataset: "stanfordnlp/imdb" 
# -------------------------------------------------------------------

# %pip install -U datasets

import torch
from peft import LoraConfig, TaskType
from peft import get_peft_model
from peft import prepare_model_for_kbit_training
## from transformers import AutoModelForSeq2SeqLM ...
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
## QLora
from transformers import BitsAndBytesConfig


output_dir="./pt_output/pt_bert.pt"
peftoutput_dir="./Peft_output/pefted_bert.pt"
model_name="distilbert-base-uncased-finetuned-sst-2-english"
dataset_filename="imdb" ## "https://huggingface.co/datasets/stanfordnlp/imdb" ## "stanfordnlp/imdb"

## helper function
def print_trainable_parameters(model):
    """ Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
)
    
## QLoRA config for pretrained
##  we recommend using NF4 quantization for better performance. https://huggingface.co/blog/4bit-transformers-bitsandbytes
quanticonfig = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
bQLoRA=False #True


pt_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},  # For converting predictions to strings
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
)

## https://huggingface.co/docs/peft/package_reference/auto_class
##   set the QLoRA-config for this pretrained_model
## NOTE:   you cannot train a fully quantized model
pt_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    quantization_config=quanticonfig,  ##quantisation
    ### NOT WORKING for HMDB Model   device_map="auto",                 ##quantisation
    id2label={0: "NEGATIVE", 1: "POSITIVE"},  # For converting predictions to strings
    label2id={"NEGATIVE": 0, "POSITIVE": 1},
)

## NOTE: do not manually set a device once the model has been loaded with device_map. 
##       So any device assignment call to the model, or to any model’s submodules should be avoided after that line - unless you know what you are doing.

##  PEFT mit QLORA 
##  https://medium.com/@tejpal.abhyuday/optimizing-language-model-fine-tuning-with-peft-qlora-integration-and-training-time-reduction-04df39dca72b
##  https://medium.com/@tejpal.abhyuday/optimizing-language-model-fine-tuning-with-peft-qlora-integration-and-training-time-reduction-04df39dca72b
#pt_model.add_adapter(lora_config, adapter_name="adapter_1")
##  QLoRA  call the  prepare_model_for_kbit_training() function from PEFT to preprocess the quantized model for QLORA / training.
#pt_model = prepare_model_for_kbit_training(pt_model)

## You will notice that unlike in Chapter 2, you get a warning after instantiating this pretrained model. 
## This is because BERT has not been pretrained on classifying pairs of sentences, so
## the head of the pretrained model has been discarded and a new head suitable for sequence classification has been added instead. The warnings indicate that some weights were not used (the ones corresponding to the dropped pretraining head) and that some others were randomly initialized (the ones for the new head). It concludes by encouraging you to train the model, which is exactly what we are going to do now.

# Freeze all the parameters of the base model
# we dont want to train the base model
# Hint: https://huggingface.co/transformers/v4.2.2/training.html
for param in pt_model.base_model.parameters():
    param.requires_grad = False  

pt_model.classifier
print(pt_model)
    
## try GPU - BUT NOT with QLoRA from_pretrained
## NOTE:   mode.to(device="cuda")  ValueError: `.to` is not supported for `4-bit` or `8-bit` bitsandbytes models.
##         Please use the model as it is, since the model has already been set to the correct devices 
##         and casted to the correct `dtype`.
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if bQLoRA==False:
    pt_model.to(device)
device

tokenizer = AutoTokenizer.from_pretrained(model_name)


# In[2]:


from datasets import load_dataset
from datasets import Dataset
from pprint import pprint

## see  https://huggingface.co/datasets
# use imdb dataset 
## Load the test split of the imdb dataset
## dataset = load_dataset('stanfordnlp/imdb', split='train')
## The stanford imdb dataset consists of train, test, unsupervised (-1 Labels) datasets
## An example of 'train' looks as follows.
## ## {
##     "label": 0,
##     "text": "Goodbye world2\n"
## }

## Load the train and test splits of the imdb dataset
## https://huggingface.co/learn/nlp-course/en/chapter3/2?fw=pt
## -----------------------------------------------------------
## We can access each pair of sentences in our raw_datasets object by indexing, like with a dictionary: 
##    raw_train_dataset = raw_datasets["train"]
##    raw_train_dataset[0]
##  To know which integer corresponds to which label, we can inspect the features of our raw_train_dataset. This will tell us the type of each column:
##     raw_train_dataset.features
##  For example!::
###  {'sentence1': Value(dtype='string', id=None),
###  'sentence2': Value(dtype='string', id=None),
###  'label': ClassLabel(num_classes=2, names=['not_equivalent', 'equivalent'], names_file=None, id=None),
###  'idx': Value(dtype='int32', id=None)}

## imdb Dataset 
## https://huggingface.co/datasets/stanfordnlp/imdb/viewer
## imbdb-dataset has only train and test - no validation dataset available (?)

## https://discuss.huggingface.co/t/how-to-split-main-dataset-into-train-dev-test-as-datasetdict/1090/19
######### 
## Hugginface Dataset https://huggingface.co/docs/datasets/en/package_reference/main_classes
##                    https://huggingface.co/docs/datasets/v1.1.1/loading_datasets.html
##                    If you don’t provide a split argument to datasets.load_dataset(), this method will return a dictionary containing a datasets for each split in the dataset.
##                    Selecting a configuration is done by providing datasets.load_dataset() with a name argument. load_dataset('glue','sst2'
splits = ["train", "test"]
## seems to create a list and NO dataset ???   ds = load_dataset(path=dataset_filename, split=splits, verification_mode='no_checks').train_test_split(test_size=0.3,shuffle=True,seed=42)
## How to let load_dataset return a Dataset instead of DatasetDict in customized loading script on Jun 15, 2022 Or I can paraphrase the question in the following way: how to skip _split_generators step in DatasetBuilder to let as_dataset gives a single Dataset rather than a list[Dataset]? 
#   ds = load_dataset('imdb', split="train", verification_mode='no_checks').train_test_split(test_size=0.3,shuffle=True,seed=42)
#   ds = load_dataset('imdb', split="train")  ##.train_test_split(test_size=0.3,shuffle=True,seed=42)
ds = load_dataset('imdb', split=splits)  ##.train_test_split(test_size=0.3,shuffle=True,seed=42)

# notes:
# dsdict = load_dataset(path=dataset_filename, split="train", verification_mode='no_checks').train_test_split(test_size=0.3,shuffle=True,seed=42)
### returned is a 'DatasetDict' object  --- that has no attribute .features !!!
# https://huggingface.co/docs/datasets/v1.11.0/splits.html for datasets
## and for verification_mode='no_checks' seee https://github.com/huggingface/datasets/issues/876
## Get Infos of dataset: dataset.info  and  https://huggingface.co/docs/datasets/v2.2.1/en/package_reference/main_classes#datasets.DatasetInfo
## dataset.shape   dataset.num_columns   dataset.num_rows   len(dataset)
#ds = {split: ds for split, ds in zip(splits, load_dataset(path=dataset_filename, split=splits, verification_mode='no_checks'))}

ds = {split: ds for split, ds in zip(splits, load_dataset(path=dataset_filename, split=splits))}
ds

## IS THIS CORRECT regarding our load_dataset and split() above ??   
##
# assert isinstance(ds, Dataset), "The dataset should be a Dataset object"
## ds = Dataset.from_dict(dsdict)
# assert set(ds.features.keys()) == {
#     "label",
#     "text",
# }, "The dataset should have a label and a text feature"

# Thin out the dataset to make it run faster for this example
for split in splits:
    ds[split] = ds[split].shuffle(seed=42).select(range(500))

## use the return values from tokenizer() to see input_ids, token_type_ids, attention_mask
## use the return values from tokenizer() to see tokens: tokenizer.convert_ids_to_tokens(inputs["input_ids"])
## returns:  tokenized datasets
def preprocess_tokenize(i_text):
    """Preprocess the dataset by returning tokenized examples."""
    ## should we do here add   return_tensors="pt" ???
    return tokenizer(i_text["text"], padding="max_length", max_length=512, truncation=True, return_tensors="pt") ## TG 06.06.24 return_tensors="pt"
    ## or should we do:  tokenizer (i_text["text"], padding=True, truncation=True, max_length=512, ** return_tensors="pt" **  ???
    ## Note:  return_tensors="pt" returns PyTorch tensors, "tf" returns TensorFlow tensors, and "np" returns NumPy arrays
    ## print(tokenized_inputs["input_ids"])  <--- return value from tokenizer()
  
## Tokenize text content of dataset 
##
##  This works well, but it has the disadvantage of returning a dictionary (with our keys, input_ids, attention_mask, and token_type_ids, and values that are lists of lists). It will also only work if you have enough RAM to store your whole dataset during the tokenization (whereas the datasets from the 🤗 Datasets library are Apache Arrow files stored on the disk, so you only keep the samples you ask for loaded in memory).
##  To keep the data as a dataset, we will use the Dataset.map() method. This also allows us some extra flexibility, if we need more preprocessing done than just tokenization. The map() method works by applying a function on each element of the dataset, so let’s define a function that tokenizes our inputs:
##  Note that it also works if the example dictionary contains several samples (each key as a list of sentences) since the tokenizer works on lists of pairs of sentences, as seen before. This will allow us to use the option batched=True in our call to map()
##   Our preprocess_tokenize function returns a dictionary with the keys input_ids, attention_mask, and token_type_ids, so those three fields are added to all splits of our dataset. Note that we could also have changed existing fields if our preprocessing function returned a new value for an existing key in the dataset to which we applied map()
tokenized_ds = {}
for split in splits:
    tokenized_ds[split] = ds[split].map(preprocess_tokenize, batched=True)

##   Prepare for training
## =======================
##   Before actually writing our training loop, we will need to define a few objects. The first ones are the dataloaders we will use to iterate over batches. But before we can define those dataloaders, we need to apply a bit of postprocessing to our tokenized_datasets, to take care of some things that the Trainer did for us automatically. Specifically, we need to:
##   
##   Remove the columns corresponding to values the model does not expect (like the sentence1 and sentence2 columns).
##   Rename the column label to labels (because the model expects the argument to be named labels).
##   Set the format of the datasets so they return PyTorch tensors instead of lists.
##   Our tokenized_datasets has one method for each of those steps:
##   tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
##   tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
##   tokenized_datasets.set_format("torch")
##   tokenized_datasets["train"].column_names
##   -------
##   from torch.utils.data import DataLoader
##   
##   train_dataloader = DataLoader(
##       tokenized_datasets["train"], *** shuffle=True *** , batch_size=8, collate_fn=data_collator
##   )
##   eval_dataloader = DataLoader(
##       tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
##   )
##   To quickly check there is no mistake in the data processing, we can inspect a batch like this:
##   for batch in train_dataloader:
##       break
##   {k: v.shape for k, v in batch.items()}

## padding add to the tokenizer 
## Do DataCollator **after tokenizing the data**, and before passing the data to the Trainer object to train the model.
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Check that we tokenized the examples properly
## assert tokenized_ds["train"][0]["input_ids"][:5] == [101, 2045, 2003, 2053, 7189]
# Show the first example of the tokenized training set
print("TRAIN DS")
print(tokenized_ds["train"][0]["input_ids"])

print("TEST DS")
# Show the first example of the tokenized test set
print(tokenized_ds["test"][0]["input_ids"])


# In[3]:


### DONT NEED THESE 2 functions since we use the Trainer.Train() functionality to simplify training & evaluation

def get_prediction(i_model, tokenizedInput, labels):   
    """Given a review, return the predicted sentiment"""
    # Tokenize the review
    # (Get the response as tensors and not as a list)
    ###tokenizedinputs = tokenizer (review, padding=True, truncation=True, max_length=512, return_tensors="pt")  ## max_length=512 oder padding="max length" ???
    # Perform the prediction (get the logits)
    ##outputs = i_model(**tokenizedinputs)     ## pt_model(**inputs)
    outputs = i_model(**tokenizedInput)     ## pt_model(**inputs)
    ## All Transformers models will return the loss when labels are provided, and we also get the logits (two for each input in our batch, so a tensor of size 8 x 2).
    ## print(outputs.loss, outputs.logits.shape)
    ## Get the predicted class (corresponding to the highest logit)
    predictions = torch.argmax(outputs.logits, dim=-1)
    ## return {"accuracy": (predictions == labels).mean()}
    ac = (predictions == labels).mean()
    print(f"accuracy: {ac}")
    return "positive" if predictions.item() == 1 else "negative"


# TODO extent with num reviews in dataset
def eval_and_predict (i_model, i_dataset, LastNreviews):
    # Show the first example
    pprint(i_dataset[0])
    # Get the last 3 reviews
    reviews = i_dataset["text"][-LastNreviews:]
    # Get the last 3 labels
    labels = i_dataset["label"][-LastNreviews:]
    print(f"Review:{reviews} \n and Labels {labels}")
    
    # Check
    falsePredictions = 0
    for review, label in zip(reviews, labels):
        # Let's use your get_prediction function to get the sentiment
        # of the review!
        prediction = get_prediction (i_model, review, label)

        print(f"Review: {review[:80]} \n... {review[-80:]}")
        print(f'Label: {"positive" if label else "negative"}')
        print(f"Prediction: {prediction}\n")
        if prediction!=label: 
            falsePredictions+=1
    print(f"False Predictions: {falsePredictions} (Pred!=Label of dataset)")
    return falsePredictions
  
## DO WE NEED THIS ??
## 
## pt_cntFalsePredictions = eval_and_predict (pt_model, tokenized_ds["train"], 10)







# In[5]:


from transformers import Trainer
from transformers import TrainingArguments
import numpy as np

## provide the Trainer with a compute_metrics() function to calculate a metric during said evaluation 
## (otherwise the evaluation would just have printed the loss, which is not a very intuitive number).
## see also:  https://medium.com/@rakeshrajpurohit/customized-evaluation-metrics-with-hugging-face-trainer-3ff00d936f99
def compute_metrics(eval_preds):
    ## we might use:  metric = evaluate.load("glue", dataset_filename)   ## 2nd arg is the name of the dataset we work on. 1st arg is the "GLUE" benchmark
    logits, labels = eval_preds
    ## tg   predictions = np.argmax(logits, axis=-1)
    ##predictions = np.argmax(logits, axis=-1)
    ## tg   return metric.compute(predictions=predictions, references=labels)
    predictions = np.argmax(logits, axis=1)  ## UDACITY Version
    return {"accuracy": (predictions == labels).mean()}

    ## For later experiments:
    ##   precision = precision_score(labels, preds, average='weighted')
    ##   recall = recall_score(labels, preds, average='weighted')
    ##   f1 = f1_score(labels, preds, average='weighted')
    ##   return {
        ##   "accuracy": (predictions == labels).mean(),
        ##   "precision": precision,
        ##   "recall": recall,
        ##   "f1": f1
    ##   }

    ## OR use:   As shown in Udacity  https://learn.udacity.com/paid-courses/cd13303/lessons/cf85c5fa-33bb-4da4-b64b-f3d996d2ada3/concepts/bddb1b82-29f8-4bb1-a500-be857e164927
    ## predictions = np.argmax(predictions, axis=1)
    ## return {"accuracy": (predictions == labels).mean()}


print(f"Output Dir ={output_dir}")

# tell the Trainer to evaluate during training by setting evaluation_strategy to either "steps" (evaluate every eval_steps) or "epoch" (evaluate at the end of each epoch).
## NOTE:  ValueError: You cannot perform fine-tuning on purely quantized models. Please attach trainable adapters
##        on top of the quantized model to correctly perform fine-tuning. Please see: https://huggingface.co/docs/transformers/peft for more details
## output_file_time = str(int(time.time()

training_args = TrainingArguments(
    output_dir=output_dir,   ## "your-name/bigscience/mt0-large-lora",
    learning_rate=1e-3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=pt_model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],      ## !! Should be ["validation"] if available !!
    tokenizer=tokenizer,
    data_collator=data_collator,            ## or use =DataCollatorWithPaddin(tokenzer=tokenizer) directly here
    compute_metrics=compute_metrics,
)


# Evaluate PreTrained Model on a dataset
# we do NOT want to train a pretrained model (pt_model)  trainer.train()
## for QLoRA train **********************
##trainer.train()
## for QLoRA train **********************
# we only want to evaluate the pretrained model
pt_model_evalresults = trainer.evaluate()
print(pt_model_evalresults)
preTrainedAccuracy = pt_model_evalresults["eval_accuracy"]
print(f"pretrained model accuracy={preTrainedAccuracy}")
preTrainedRuntime = pt_model_evalresults["eval_runtime"]
print(f"pretrained model runtime={preTrainedRuntime}")


# In[ ]:





# In[ ]:





# In[ ]:





# ## Performing Parameter-Efficient Fine-Tuning
# 
# TODO: In the cells below, create a PEFT model from your loaded model, run a training loop, and save the PEFT model weights.

# In[ ]:


from peft import get_peft_model

## =======================================
## PEFT Performance Efficient Fine Tuning
## =======================================
## https://huggingface.co/docs/peft/quicktour
## Easily load any PEFT-trained model for inference with the AutoPeftModel class and the from_pretrained method:
## LoRA is low-rank decomposition method to reduce the number of trainable parameters which speeds up finetuning 
## large models and uses less memory. In PEFT, using LoRA is as easy as setting up a LoraConfig and wrapping it 
## with get_peft_model() to create a trainable PeftModel.
## Parameter efficient fine-tuning (PEFT) is a method that aims to reduce the size of models, making it possible
## to perform calculations on less powerful GPUs. LoRa is a method in PEFT that is used to reduce the size of LLMs.
##
## Example without Trainer.Train() .Eval() but in old-style loops:  https://github.com/huggingface/peft/issues/310

## LoraConfig containing the parameters for how to configure a model for training with LoRA.
# SEQ_CLS: Text classification
##  orig peft_LORAconfig = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
## SEE Target_modules Explanation:  https://stackoverflow.com/questions/76768226/target-modules-for-applying-peft-lora-on-different-models

## lets see target modules in the model, like "q_proj", "v_proj", etc
## https://arxiv.org/abs/2106.09685  LORA Paper
print(pt_model)
## QloRA create Quantization Model for training
######DONE ABOVE ALREADY  pt_model = prepare_model_for_kbit_training(pt_model)
##   QloRA adds trainable weights to all the linear layers in the transformer architecture. Since the attribute names for these linear layers can vary across architectures, set target_modules to "all-linear" to add LoRA to all the linear layers:
##   config = LoraConfig(target_modules="all-linear", ...)
## set those target_modules for LORA

## LORA only
#  peft_LORAconfig = LoraConfig(task_type=TaskType.SEQ_CLS, target_modules=["q_lin","k_lin","v_lin"],r=8, lora_alpha=8, lora_dropout=0.1)

## QloRA
## QloRA config (target_modules=["all-linear"])
#peft_LORAconfig = LoraConfig(task_type=TaskType.SEQ_CLS, target_modules=["all-linear"],r=8, lora_alpha=8, lora_dropout=0.1)
peft_LORAconfig = LoraConfig(task_type=TaskType.SEQ_CLS, target_modules="all-linear",r=8, lora_alpha=8, lora_dropout=0.1)

## Overview of the supported task types:
##     - SEQ_CLS: Text classification.
##     - SEQ_2_SEQ_LM: Sequence-to-sequence language modeling.
##     - Causal LM: Causal language modeling.
##     - TOKEN_CLS: Token classification.
##     - QUESTION_ANS: Question answering.
##     - FEATURE_EXTRACTION: Feature extraction. Provides the hidden states which can be used as embeddings or features
##       for downstream tasks.

#   QLORA Enabling gradient checkpointing to reduce memory usage during fine-tuning
 pt_model.gradient_checkpointing_enable()
##  PEFT mit QLORA 
##  https://medium.com/@tejpal.abhyuday/optimizing-language-model-fine-tuning-with-peft-qlora-integration-and-training-time-reduction-04df39dca72b
##  https://medium.com/@tejpal.abhyuday/optimizing-language-model-fine-tuning-with-peft-qlora-integration-and-training-time-reduction-04df39dca72b
#############pt_model.add_adapter(lora_config, adapter_name="adapter_1")
##  QLoRA  call the  prepare_model_for_kbit_training() function from PEFT to preprocess the quantized model for QLORA / training.
pt_model = prepare_model_for_kbit_training(pt_model)


## create a PEFT model from your loaded model (pretrained_model)
##     For QloRA you must set  Qloraconfig={}  previously
##     Use the get_peft_model() function to create a PeftModel from the quantized model and configuration.
peft_model = get_peft_model(pt_model, peft_LORAconfig)
peft_model.print_trainable_parameters()

## try GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
peft_model.to(device)
device


# In[ ]:


model=peft_model
## training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
## model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],      ## !! Should be ["validation"] if available !!
    tokenizer=tokenizer,
    data_collator=data_collator,            ## or use =DataCollatorWithPaddin(tokenzer=tokenizer) directly here
    compute_metrics=compute_metrics,
)

## Train the model
## "run a training loop"
## To fine-tune the model on our dataset, we just have to call the train() method of our Trainer:
## =====================================
train_result = trainer.train()

## After your model is finished training, you can save your model to a directory
## =====================================
## Both methods (model.push_to_hub("your-name/bigscience/mt0-large-lora")) *** only save the extra PEFT weights that were trained ***, meaning it is super efficient to store, transfer, and load.
#  model trained with LoRA and saved only contains two files: adapter_config.json and adapter_model.safetensors.
#  This  only saves the adapter weights and not the weights of the original Transformers model. 
#  Thus the size of the files created will be much smaller than you might expect.
#  Note:  https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822/2 
model.save_pretrained(peftoutput_dir)


## Evaluation
## ===========
## Show the performance of the model on the test set
## --------------------
## Let’s see how we can build a useful compute_metrics() function and use it the next time we train
## Just run trainer.predict(tokenized_ds["validation"]) on your eval/test dataset.
## print(predictions.predictions.shape, predictions.label_ids.shape)
## The output of the predict() method is another named tuple with three fields: predictions, label_ids, and metrics. The metrics field will just contain the loss on the dataset passed, as well as some time metrics (how long it took to 
## predict, in total and on average). Once we complete our compute_metrics() function and pass it to the Trainer, that field will also contain the metrics returned by compute_metrics().

## We can load the metrics associated with the MRPC dataset as easily as we loaded the dataset, this time with the evaluate.load() function
##   metric = evaluate.load("glue", "enter here the name of dataset we work on")
##   metric.compute(predictions=preds, references=predictions.label_ids)
##   you get:  {'accuracy': 0.8578431372549019, 'f1': 0.8996539792387542}  where "f1" refers to the GLUE-Benchmark BERT-PAPER https://arxiv.org/pdf/1810.04805.pdf

## Evaluate the model
## Evaluating the model is as simple as calling the evaluate method on the trainer object. This will run the model on the test set
## and compute the metrics we specified in the compute_metrics function.
# Show the performance of the model on the test set
# What do you think the evaluation accuracy will be?
trainer.evaluate()

## peft_model_evalresults = trainer.evaluate()
## PEFTaccuracy = peft_model_evalresults["eval_accuracy"]
## print(f"PEFT accu={PEFTaccuracy}")



## --------------------
## Show the performance of the model on the test set
## --------------------

import pandas as pd

df = pd.DataFrame(tokenized_ds["test"])
df = df[["text", "label"]]
# Replace <br /> tags in the text with spaces
df["text"] = df["text"].str.replace("<br />", " ")
# Add the model predictions to the dataframe
predictions = trainer.predict(tokenized_ds["test"])
df["predicted_label"] = np.argmax(predictions[0], axis=1)
df.head(30)


# In[ ]:


## Dont need this currently
##
## addtional: save training metrics
# metrics = train_result.metrics
# max_train_samples = training_args.max_train_samples if training_args.max_train_samples is not None else len(train_dataset)
# metrics["train_samples"] = min(max_train_samples, len(train_dataset))
# trainer.log_metrics("train", metrics)
# trainer.save_metrics("train", metrics)
# trainer.save_state()


# In[ ]:


# peft_model.save_pretrained(output_dir)   ##  +"/pefted_Bert_model.pt")


# In[ ]:





# ## Performing Inference with a PEFT Model
# 
# TODO: In the cells below, load the saved PEFT model weights and evaluate the performance of the trained PEFT model. Be sure to compare the results to the results from prior to fine-tuning.

# In[ ]:


from peft import get_peft_model
from peft import AutoPeftModelForSequenceClassification

## To load a PEFT model for inference, you can use the AutoPeftModel class
## https://huggingface.co/docs/peft/package_reference/auto_class
savedpeft_model = AutoPeftModelForSequenceClassification.from_pretrained(peftoutput_dir)
##  model = AutoPeftModelForCausalLM.from_pretrained(...)

#####peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
##peft_model = get_peft_model(model, peft_LORAconfig)  ## lora_config
savedpeft_model.print_trainable_parameters()

## try GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
savedpeft_model.to(device)
device

trainer = Trainer(
    model=savedpeft_model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],      ## !! Should be ["validation"] if available !!
    tokenizer=tokenizer,
    data_collator=data_collator,            ## or use =DataCollatorWithPaddin(tokenzer=tokenizer) directly here
    compute_metrics=compute_metrics,
)

## Evaluate the model
## Evaluating the model is as simple as calling the evaluate method on the trainer object. This will run the model on the test set
## and compute the metrics we specified in the compute_metrics function.
# Show the performance of the model on the test set
# What do you think the evaluation accuracy will be?
#####savedpeft_model.eval()
peft_model_evalresults = trainer.evaluate()
print(peft_model_evalresults)
PEFTaccuracy = peft_model_evalresults["eval_accuracy"]
print(f"PEFT accu={PEFTaccuracy}")
PEFTruntime = peft_model_evalresults["eval_runtime"]
print(f"PEFT model runtime={PEFTruntime}")

if(PEFTaccuracy>preTrainedAccuracy):
    print(f"PEFT model accuracy is higher({PEFTaccuracy}) then PreTrained model accuracy({preTrainedAccuracy})") 
else:
    print(f"PreTrained model accuracy is higher({preTrainedAccuracy}) then PEFT model accuracy({PEFTaccuracy})")

if(PEFTruntime>preTrainedRuntime):
    print(f"PEFT model Runtime is higher/slower({PEFTruntime}) then PreTrained model Runtime({preTrainedRuntime})") 
else:
    print(f"PreTrained model Runtime is higher/slower({preTrainedRuntime}) then PEFT model Runtime ({PEFTruntime})")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




