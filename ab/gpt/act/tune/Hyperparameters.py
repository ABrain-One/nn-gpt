import json
import os
import random
import shutil
from os import makedirs

import torch
from ab.nn.util.Const import out_dir
from datasets import load_dataset, load_from_disk
from peft import (
    get_peft_model, LoraConfig, PeftModel,
    prepare_model_for_kbit_training
)
from transformers import (
    Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, DataCollatorForSeq2Seq, EarlyStoppingCallback
)

from ab.gpt.util.Const import llm_tokenizer_out
from ab.gpt.util.llm.LLMUtil import quantization_config_4bit, tokenize
from ab.gpt.util.data.lemur_dataset_preparation import DatasetPreparation

device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")


def create_prompt(data_point):
    """
    Creates a prompt for the LLM
    """
    return f"""
        ### Input:
        {data_point["question"]}

        ### Response:
        {data_point["answer"]}
    """


def tuned_dir_f(tuned_model_version):
    return out_dir / 'Finetuned_models' / f"tuned_model_v{tuned_model_version}"



def main(tuned_model_version, hf_directory, dataset_path):
    """
    The main function for loading data, setting up the model and fine-tuning
    """

    # Write training output to the file
    # log_filename = f"training_logs_{tuned_model_version}.txt"
    # sys.stdout = open(log_filename, "w")

    if hf_directory is None:
        raise ValueError(f"Unknown model version: {tuned_model_version}")
    print(f"Using model: {hf_directory}")

    tokenizer = AutoTokenizer.from_pretrained(hf_directory)
    tokenizer.add_eos_token = True
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(hf_directory,
                                                 trust_remote_code=True,
                                                 device_map="auto",
                                                 quantization_config=quantization_config_4bit,
                                                 torch_dtype=torch.float16
                                                )

    # trying to load mapped datasets
    
    tuned_dir = tuned_dir_f(tuned_model_version)
    train_dir = tuned_dir /  'tokenized_train_dataset'
    val_dir = tuned_dir /  'tokenized_val_dataset'
    model_dir = tuned_dir / 'model'
    try:
        tokenized_train_dataset = load_from_disk(train_dir)
        tokenized_val_dataset = load_from_disk(val_dir)
        print("Datasets loaded successfully.")
    except Exception as e:
        print(f"Dataset loading failed: {e}")
        dataset = load_dataset('json', data_files=str(dataset_path))
        shuffled_dataset = dataset['train'].shuffle(seed=42)

        train_dataset = shuffled_dataset.train_test_split(test_size=0.2)["train"]
        eval_dataset = shuffled_dataset.train_test_split(test_size=0.2)["test"]

        num_proc = os.cpu_count() // 2
        tokenized_train_dataset = train_dataset.map(lambda data_point: tokenize(create_prompt(data_point), tokenizer),
                                                    num_proc=num_proc)
        tokenized_val_dataset = eval_dataset.map(lambda data_point: tokenize(create_prompt(data_point), tokenizer),
                                                 num_proc=num_proc)

        tokenized_train_dataset.save_to_disk(train_dir)
        tokenized_val_dataset.save_to_disk(val_dir)
        print("Datasets have been processed and saved.")

    # put model back into training mode
    model.train()
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    # LoRA config
    peft_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Applying LoRA to the model
    model = get_peft_model(model, peft_config)

    # Training Arguments
    training_args = TrainingArguments(
        num_train_epochs=35,
        warmup_steps=100,
        optim="adamw_torch",
        learning_rate=1e-5,
        logging_steps=10,
        max_grad_norm=1.0,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        output_dir=tuned_dir / 'output',
        logging_dir=tuned_dir / 'logs',
        weight_decay=0.01,
        save_total_limit=3,
        load_best_model_at_end=True
    )

    # trainer initialization
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        data_collator=DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ), # ??? todo DataCollatorForLanguageModeling(tokenizer, pad_to_multiple_of=8, return_tensors="pt", mlm=False)
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print(f"Using device: {device}")

    # Option I: Fine-tuning of the model from the start
    trainer.train()

    # # Option II: Resume fine-tuning of the model from the checkpoint
    # resume_from_checkpoint = f"Finetuned_models/tuned_model_v{tuned_model_version}/output/checkpoint-????"
    # if resume_from_checkpoint:
    #     if os.path.exists(resume_from_checkpoint):
    #         print(f"Resuming training from {resume_from_checkpoint}")
    #     else:
    #         print("Checkpoint not found, starting training from scratch")
    #         resume_from_checkpoint = None
    # trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(llm_tokenizer_out)

    # sys.stdout.close()
    # sys.stdout = sys.__stdout__
    # print(f"\nTraining log saved to {log_filename}")


def generate_model_responses(tuned_model_version, hf_directory, input_file_path, output_file_path, logs_file_path):

    if hf_directory is None:
        raise ValueError(f"Unknown model version: {tuned_model_version}")
    print(f"Using model: {hf_directory}")

    quantization_config_8bit = BitsAndBytesConfig(
        load_in_8bit=True
    )

    model = AutoModelForCausalLM.from_pretrained(hf_directory,
                                                 trust_remote_code=True,
                                                 device_map="auto",
                                                 quantization_config=quantization_config_8bit)
    tokenizer = AutoTokenizer.from_pretrained(hf_directory)
    model = PeftModel.from_pretrained(model, tuned_dir_f(tuned_model_version) / 'model')

    with open(input_file_path, "r") as f:
        data = json.load(f)

    random.shuffle(data)
    processed_data = []

    with open(logs_file_path, "w") as output_file:
        for i, entry in enumerate(data):
            hyperparameters = entry['prm']
            prm_names = ", ".join(hyperparameters.keys())

            eval_prompt = f"""
            ### Input:
            Generate only the values (don't provide any explanation) of the hyperparameters ({prm_names}) of a 
            given model: {entry['metric']} for the task: {entry['task']} on dataset: {entry['dataset']}, 
            with transformation: {entry['transform_code']}, so that the model achieves accuracy = {entry['accuracy']} 
            with number of training epochs = {entry['epoch']}. 
            Code of that model:\n {entry['nn_code']}

            ### Response:
            """

            model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

            with torch.no_grad():
                output = model.generate(**model_input, max_new_tokens=150, pad_token_id=tokenizer.pad_token_id)
                response_text = tokenizer.decode(output[0], skip_special_tokens=True)

            response_text = response_text.split("### Response:")[-1].strip()

            # Save Logs
            output_file.write(f"Model #{i + 1}\n")
            output_file.write(f"Prompt:\n{eval_prompt}\n")
            output_file.write(f"Response:\n{response_text}\n\n")

            # Save Model's Response to JSON
            entry['Response'] = response_text
            processed_data.append(entry)

            print(f"Got {i + 1} responses out of {len(data)}")

    print(f"All responses are saved in {logs_file_path}")

    with open(output_file_path, "w") as f:
        json.dump(processed_data, f, indent=4)

    print(f"All hyperparameters have been successfully saved to {output_file_path}")



generate_prompt = True
tuned_model_version = 4

hf_directories = {
    1: "deepseek-ai/DeepSeek-Coder-V2-Lite-Base",
    2: "deepseek-ai/deepseek-coder-1.3b-base",
    3: "deepseek-ai/deepseek-coder-1.3b-base",
    4: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    5: "deepseek-ai/deepseek-coder-7b-base-v1.5",
    6: "deepseek-ai/deepseek-math-7b-base",
    7: "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
}


if __name__ == "__main__":
    base_dir = out_dir / 'hpgpt' / 'prompt'
    dataset_raw = base_dir / 'LEMUR_raw.json'
    dataset_prepared_prompt = base_dir / 'LEMUR_prepared.json'

    if generate_prompt:
        shutil.rmtree(base_dir, ignore_errors=True)
        makedirs(base_dir, exist_ok=True)
        # ---------- 1. LEMUR DATASET PREPARATION STAGE ----------
        dataset_prep = DatasetPreparation()  # Create a raw LEMUR Dataset JSON-file
        dataset_prep.test_api(dataset_raw)
        dataset_prep.prepare_json_dataset_for_llm_format(dataset_raw, dataset_prepared_prompt) # Convert a raw LEMUR Dataset JSON-file to a LLM prompt format
        # Other
        # dataset_prep.add_nn_code_field_to_json(dataset_raw, f"Dataset/LEMUR_raw_500.json")
    # ---------- 2. LLM FINE-TUNING STAGE ----------

    main(tuned_model_version, hf_directories.get(tuned_model_version), dataset_prepared_prompt)

    # ---------- 3. LLM TESTING & RECEIVING RESPONSES STAGE ----------
    # Dataset 500 Models (Only 500 responses to speed up)
    # dataset_raw_500 = f"Dataset/LEMUR_raw_500.json"

    # Base Model Paths
    # output_file_path = "Dataset/ds_responses_1coder-v2-base-lite_500.json"
    # logs_file = f"Logs/logs_responses_1coder-v2-base-lite_500.txt"

    # Fine-tuned Model Paths
    # output_file_path = f"Dataset/ds_responses_{tuned_model_version}ft_500.json"
    # logs_file_path = f"Logs/logs_responses_{tuned_model_version}ft_500.txt"

    # generate_model_responses(tuned_model_version, hf_directories.get(tuned_model_version), dataset_raw_500, output_file_path, logs_file_path)


