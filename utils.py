import os
import gc
import datasets
import torch
import logging
import transformers
import pandas as pd
from typing import Tuple

logger = logging.getLogger("openmedllm")


def clear_memory():
    gc.collect()  # Python garbage collection
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear GPU cache


def load_model(
    model_path: str,
    quantization: bool = True,
    four_bit: bool = True,
) -> Tuple[transformers.Pipeline, transformers.PreTrainedTokenizer]:
    """
    Load the language model with specified configuration.

    Args:
        model_path (str): Path to the model.
        quantization (bool, optional): Flag to enable quantization. Defaults to True.

    Returns:
        Tuple[transformers.Pipeline, transformers.PreTrainedTokenizer]: A tuple containing the loaded pipeline and tokenizer.
    """
    logging.info("Loading the tokenizer and model.")

    if four_bit:
        nf4_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, legacy=False)
        model_hf = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            quantization_config=nf4_config,
        )

    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, legacy=False)
        model_hf = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            load_in_8bit=quantization,
        )

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_hf,
        tokenizer=tokenizer,
    )
    pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id

    return pipeline


def load_dataset(prompts_file: str, output_file: str) -> datasets.DatasetDict:
    """
    Load the dataset from the prompts file, and filter out already completed prompts.

    Args:
        prompts_file (str): Path to the file containing the prompts.
        output_file (str): Path to the file where the output is to be stored.

    Returns:
        datasets.DatasetDict: The loaded dataset.
    """
    logging.info("Loading dataset.")
    if os.path.exists(output_file):
        logging.info("Output file exists. Filtering completed prompts.")
        completed_prompts_df = pd.read_csv(output_file)

        prompts_df = pd.read_csv(prompts_file)
        remaining_prompts_df = prompts_df[
            ~prompts_df["question"].isin(completed_prompts_df["question"])
        ]
        remaining_prompts_df.to_csv("temp_prompts.csv", index=None)
        dataset = datasets.load_dataset("csv", data_files="temp_prompts.csv")
    else:
        dataset = datasets.load_dataset("csv", data_files=prompts_file)

    return dataset
