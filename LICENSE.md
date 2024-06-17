This file/code I did create for an Udacity GEN-AI course 01 task. 

LORA and PEFT was working fine.
Then I added QloRA and it resulted in an error message that I don't really understand what is wrong / what did I do wrong ?

I want to apply QLoRA (BitsNBytes Config).
I am loading a distilled BERT Model.

model_name="distilbert-base-uncased-finetuned-sst-2-english"
quantization_config=quanticonfig (BitsNBytes)

pretrained_model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=2, quantization_config=quanticonfig,...) )

I get the following error when instatiating the trainer=Trainer(model=pretrained_model, trainerConfig, trainerParams):

ERROR MESSAGE:
You cannot perform fine-tuning on purely quantized models. Please attach trainable adapters on top of" " the quantized model to correctly perform fine-tuning. Please see: https://huggingface.co/docs/transformers/peft" " for more details

What I am making wrong ?
Do I load a wrong model for QLoRA ?
What model should I load (for classification tasks) that QLoRA can be called ?
