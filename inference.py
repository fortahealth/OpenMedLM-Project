import os
import fire
import datasets
import transformers
import logging

import pandas as pd

from tqdm import tqdm
from typing import Tuple
from transformers.pipelines.pt_utils import KeyDataset

from utils import load_model, load_dataset

logger = logging.getLogger("openmedllm")


def process_output_batch(
    model_outputs,
    output_file,
    questions,
    options_list,
    contexts,
    prompts,
    correct_answers,
):
    """
    Processes the output of a model batch and saves it to a file.

    Args:
    model_outputs (list): List of model outputs for the current batch.
    start (int): Starting index of the batch in the dataset.
    end (int): Ending index of the batch in the dataset.
    output_file (str): File path where the output is to be saved.
    questions (list): List of questions corresponding to the batch.
    options_list (list): List of options corresponding to the batch.
    prompts (list): List of input prompts corresponding to the batch.
    correct_answers (list): List of correct answers corresponding to the batch.

    Returns:
    None
    """
    logging.debug(f"Processing output batch, saving to {output_file}.")
    # Creating a DataFrame from the batch data
    if options_list is not None:
        batch_data = {
            "question": questions,
            "options": options_list,
            "prompt": prompts,
            "correct_answer": correct_answers,
            "model_output": model_outputs,
        }

    elif contexts is not None:
        batch_data = {
            "question": questions,
            "context": contexts,
            "prompt": prompts,
            "correct_answer": correct_answers,
            "model_output": model_outputs,
        }
    else:
        logging.error("Both options and contexts are None. Check the dataset.")

    batch_df = pd.DataFrame(batch_data)

    # Writing the batch DataFrame to a CSV file
    if not os.path.exists(output_file):
        batch_df.to_csv(output_file, index=False)
    else:
        batch_df.to_csv(output_file, mode="a", index=False, header=False)
    logging.debug("Output batch processed and saved.")


def _run_inference(
    pipeline: transformers.Pipeline,
    prompts_file: str,
    output_file: str,
    batch_size: int = 1,
) -> pd.DataFrame:
    """
    Run the inference process: load dataset, run model, and save outputs.

    Args:
        pipeline (transformers.Pipeline): Model pipeline.
        prompts_file (str): Path to the prompts file.
        output_file (str): Path to the output file.
        batch_size (int, optional): Batch size for processing. Defaults to 1.

    Returns:
        pd.DataFrame: DataFrame containing the model outputs.
    """
    logging.info("Running model inference.")

    dataset = load_dataset(prompts_file, output_file)

    total_data = len(dataset["train"])
    total_batches = (total_data + batch_size - 1) // batch_size

    for batch_index in tqdm(range(total_batches)):
        start, end = batch_index * batch_size, min(
            (batch_index + 1) * batch_size, total_data
        )

        batch_dataset = datasets.Dataset.from_dict(dataset["train"][start:end])

        batch_outputs = pipeline(
            KeyDataset(batch_dataset, "prompt"),
            do_sample=True,
            top_k=5,
            top_p=0.8,
            temperature=0.8,
            num_return_sequences=1,
            eos_token_id=pipeline.tokenizer.eos_token_id,
            max_new_tokens=500,
            batch_size=batch_size,
        )

        model_outputs = [out[0]["generated_text"] for out in batch_outputs]

        questions_slice = dataset["train"]["question"][start:end]
        prompts_slice = dataset["train"]["prompt"][start:end]
        correct_answers_slice = dataset["train"]["correct_answer"][start:end]
        options_slice = (
            dataset["train"]["options"][start:end]
            if "options" in dataset["train"].column_names
            else None
        )
        contexts_slice = (
            dataset["train"]["context_string"][start:end]
            if "context_string" in dataset["train"].column_names
            else None
        )

        process_output_batch(
            model_outputs,
            output_file,
            questions_slice,
            options_slice,
            contexts_slice,
            prompts_slice,
            correct_answers_slice,
        )

    return pd.read_csv(output_file)


def extract_model_answer(row: pd.Series) -> str:
    """
    Extracts the model's answer from a given row of the DataFrame.

    Args:
        row (pd.Series): A row from the DataFrame.

    Returns:
        str: The extracted model answer.
    """
    model_answer = row["model_full_answer"]

    if model_answer.lower().startswith(("### explanation", "explanation")):
        if "Answer" in model_answer:
            model_answer_letter = model_answer.split("Answer")[0]
        else:
            model_answer_letter = ""

    elif model_answer.startswith(("### Answer", "Answer")):
        model_answer_letter = model_answer.split("Answer")[1]

    elif "Answer" in model_answer:
        model_answer_letter = model_answer.split("Answer")[1]

    else:
        model_answer_letter = ""

    return model_answer_letter


def filter_invalid_responses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters and returns invalid responses from the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing model answers.

    Returns:
        pd.DataFrame: DataFrame containing only the invalid responses.
    """
    invalid_df = df[~(df["model_answer_letters"].isin(["A", "B", "C", "D"]))]
    logging.debug(f"Filtered out {len(invalid_df)} invalid responses.")
    return invalid_df


def _eval_results(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates the results from the model's output.

    Args:
        df (pd.DataFrame): DataFrame containing model outputs and corresponding prompts.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple of DataFrames, one with evaluation results and the other with invalid responses.
    """
    logging.info("Evaluating model results.")

    def extract_model_full_answer(row):
        return row["model_output"].split(row["prompt"])[1].lstrip("\n")

    df["model_full_answer"] = df.apply(extract_model_full_answer, axis=1)
    df["model_answer_letters"] = df.apply(extract_model_answer, axis=1)

    total_correct = len(
        df[df["model_answer_letters"].str.lower() == df["correct_answer"].str.lower()]
    )
    total_invalid = len(
        df[~df["model_answer_letters"].str.lower().isin(["a", "b", "d", "d"])]
    )

    logging.info(f"Total correct: {total_correct}")
    logging.info(f"Total invalid: {total_invalid}")

    invalid_df = filter_invalid_responses(df)

    return df, invalid_df


def run_trial(
    trial: int,
    prompts_df: pd.DataFrame,
    pipeline: transformers.Pipeline,
    results_dir: str,
    batch_size: int,
    invalid_prompts_filepath: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs a single trial of the inference and evaluation process.

    Args:
        trial (int): The current trial number.
        prompts_df (pd.DataFrame): DataFrame containing the prompts for the model.
        pipeline (transformers.Pipeline): Model pipeline.
        results_dir (str): Directory where the results will be stored.
        batch_size (int): Batch size for processing.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrames containing the results of the trial and the remaining prompts.
    """
    logging.info(f"Running trial {trial}")
    current_round_output_filepath = os.path.join(
        results_dir, f"round_{trial}_outputs.csv"
    )

    if os.path.exists(current_round_output_filepath):
        model_outputs = pd.read_csv(current_round_output_filepath)
        if len(model_outputs) < len(prompts_df):
            additional_model_outputs = _run_inference(
                pipeline=pipeline,
                prompts_file=invalid_prompts_filepath,
                output_file=current_round_output_filepath,
                batch_size=batch_size,
            )
            model_outputs = pd.concat(
                [model_outputs, additional_model_outputs]
            ).drop_duplicates(subset=["question"])
    else:
        model_outputs = _run_inference(
            pipeline=pipeline,
            prompts_file=invalid_prompts_filepath,
            output_file=current_round_output_filepath,
            batch_size=batch_size,
        )

    df, invalid_prompts_df = _eval_results(model_outputs)
    return df, invalid_prompts_df


def main(
    results_dir: str,
    prompts_file: str,
    model_path: str,
    batch_size: int = 1,
):
    """
    Run inference and evaluation

    Args:
        results_dir (str): Directory where the results will be stored.
        prompts_file (str): File containing the prompts for the model.
        model_path (str): Path to the model.
        peft_model (str, optional): Path to the PEFT model. Defaults to None.
        batch_size (int, optional): Batch size for processing. Defaults to 1.

    Returns:
        None
    """

    logging.info("Starting the inference and evaluation process.")
    pipeline = load_model(model_path)
    prompts_filepath = os.path.join(results_dir, prompts_file)
    prompts_df = pd.read_csv(prompts_filepath).reset_index(drop=True)
    original_prompts_df = prompts_df.copy(deep=True)

    invalid_prompts_filepath = prompts_filepath

    for trial in range(5):
        print(f"ROUND {trial}")
        if not os.path.exists(invalid_prompts_filepath):
            print("No invalids file. Exiting.")
            exit(1)

        df, prompts_df = run_trial(
            trial,
            prompts_df,
            pipeline,
            results_dir,
            batch_size,
            invalid_prompts_filepath,
        )

        # Save valid and invalid results
        valid_filepath = os.path.join(results_dir, f"valid_answers_{prompts_file}")
        valid_df = df[~df["question"].isin(prompts_df["question"].tolist())]
        if os.path.exists(valid_filepath):
            valid_df.to_csv(valid_filepath, mode="a", header=False, index=None)
        else:
            valid_df.to_csv(valid_filepath, index=None)

        invalid_prompts_filepath = os.path.join(
            results_dir,
            f"invalids_round_{trial}_outputs.csv",
        )

        # Prepare for the next trial
        if len(prompts_df) > 0:
            prompts_df.to_csv(invalid_prompts_filepath, index=None)

        # Final calculations
        valid_df = pd.read_csv(valid_filepath).drop_duplicates(
            subset=["question"], keep="first"
        )
        final_accuracy = len(
            valid_df[valid_df["correct_answer"] == valid_df["model_answer_letters"]]
        ) / len(original_prompts_df)
        print(f"Accuracy: {final_accuracy}")


if __name__ == "__main__":
    fire.Fire(main)
