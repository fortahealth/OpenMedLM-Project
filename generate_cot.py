import os
import time
import logging
import re
import fire
import aiohttp
import asyncio
import openai

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# Configure the logging settings (you can customize these based on your needs)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Create a logger instance
logger = logging.getLogger("openmedllm")

# Set up your OpenAI API key
openai.organization = os.getenv("OPENAI_ORG_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")

GPT_BASE_PROMPT = os.getenv("GPT_BASE_PROMPT")


def find_similar_questions(train_df, test_df, model_name, k=5):
    """
    Finds similar questions in the train dataset for each question in the test dataset.

    Args:
        train_df (pandas.DataFrame): The dataframe containing the train questions.
        test_df (pandas.DataFrame): The dataframe containing the test questions.
        model_name (str): The name of the SentenceTransformer model to use for encoding the questions.
        k (int, optional): The number of most similar questions to find for each test question. Defaults to 5.

    Returns:
        tuple: A tuple containing two elements:
            - results_dict (dict): A dictionary where the keys are the test questions and the values are lists of similar questions.
            - results_df (pandas.DataFrame): A dataframe containing the test questions, similar questions, and train indices.
    """
    # Load the SentenceTransformer model
    model = SentenceTransformer(model_name, device="cuda")

    # Encode the train and test questions to get their embeddings
    logger.info("Encoding ...")
    train_embeddings = model.encode(
        train_df["question"].values, convert_to_tensor=True, device="cuda"
    )
    test_embeddings = model.encode(
        test_df["question"].values, convert_to_tensor=True, device="cuda"
    )

    # Use KNN to find the k most similar questions for each test question
    print("Running KNN ...")
    train_embeddings = train_embeddings.cpu().numpy()
    test_embeddings = test_embeddings.cpu().numpy()
    knn = NearestNeighbors(n_neighbors=k, metric="cosine")
    knn.fit(train_embeddings)

    # Find indices and distances of the k most similar questions
    _, indices = knn.kneighbors(test_embeddings, return_distance=True)

    # Create a DataFrame to store the results
    results_df = pd.DataFrame(
        columns=["Test Question", "Similar Questions", "Train Indices"]
    )

    # Create a dictionary to store the results in the desired format
    results_dict = {}

    # Iterate through each test question and its similar questions
    for i, test_question in enumerate(test_df["question"]):
        train_indices = indices[i]
        similar_questions_data = []

        for j in range(k):
            train_index = train_indices[j]
            train_question = train_df["question"].iloc[train_index]
            similar_questions_data.append(
                {"train_index": train_index, "train_question": train_question}
            )

        temp_df = pd.DataFrame(
            {
                "Test Question": test_question,
                "Similar Questions": similar_questions_data,
                "Train Indices": train_indices,
            }
        )
        results_df = pd.concat([results_df, temp_df], ignore_index=True)
        results_dict[test_question] = similar_questions_data

    return results_dict, results_df


async def async_openai_request(prompt):
    """
    Sends an asynchronous request to the OpenAI API for chat completions.

    Args:
        prompt (str): The user's prompt for the chat completion.

    Returns:
        dict: The JSON response from the OpenAI API.

    Raises:
        Exception: If an error occurs during the API request.

    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openai.api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": os.getenv("GPT_MODEL_ID"),
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                return await response.json()
    except Exception as e:
        print(f"An exception occurred: {e}")
        return None


async def gpt_runner(input_file, output_file):
    """
    Runs the GPT model on a given input file and saves the results to an output file.

    Args:
        input_file (str): The path to the input CSV file.
        output_file (str): The path to the output CSV file.

    Returns:
        pandas.DataFrame: The results of the GPT model as a DataFrame.

    Raises:
        FileNotFoundError: If the input file does not exist.

    """
    # Create a CSV reader object
    csv_reader = pd.read_csv(input_file, chunksize=100)

    completed_prompts = []
    if os.path.exists(output_file):
        logger.info(f"Output file exists.")
        completed_prompts.extend(pd.read_csv(output_file)["question"].tolist())
        logger.info(f"{len(completed_prompts)} already exist on file")

    results_df = None
    # Process and save chunks
    for i, chunk in enumerate(csv_reader):
        chunk = chunk[~(chunk["question"].isin(completed_prompts))]

        if len(chunk) == 0:
            logger.info(f"Skipping chunk {i}")
            continue

        questions = chunk["question"]
        options = chunk["options"]
        prompts = chunk["prompt"]
        logger.info(f"Working on Chunk {i}")

        start = time.time()
        results = await asyncio.gather(
            *(async_openai_request(prompt) for prompt in prompts)
        )

        results_df = []
        for question, option, prompt, result in zip(
            questions, options, prompts, results
        ):
            if result is not None:
                try:
                    result_output = result["choices"][0]["message"]["content"]
                except Exception as e:
                    logger.error(f"Error {e}")
                    print(result)
                    result_output = ""
                results_df.append(
                    {
                        "question": question,
                        "options": option,
                        "prompt": prompt,
                        "result": result_output,
                    }
                )
        logger.info(
            f"Time taken to complete Chunk {i} with {len(chunk)} questions: {time.time() - start}"
        )

        results_df = pd.DataFrame(results_df)

        if os.path.exists(output_file):
            results_df.to_csv(output_file, mode="a", header=False, index=False)
        else:
            results_df.to_csv(output_file, index=False)

    return results_df


def process_outputs(result_file_path, train_df):
    """
    Function to save the CoT when answers are correct,
    also saves incorrect cases to run inference on

    Args:
        result_file_path (str): The file path of the result file.
        train_df (pandas.DataFrame): The training dataframe containing the question and answer_letter columns.

    Returns:
        correct_df (pandas.DataFrame): The dataframe containing the correct answers.
        incorrect_df (pandas.DataFrame): The dataframe containing the incorrect answers.
    """

    df = pd.read_csv(result_file_path)
    if "answer_letter" not in list(df.columns):
        df = df.merge(
            train_df[["question", "answer_letter"]], on="question", how="left"
        )

    model_answer_letters = []
    for _, row in df.iterrows():
        model_answer = str(row["result"])

        if "Answer" in model_answer:
            model_answer_letter = model_answer.split("Answer")[1].strip()

            if "Explanation" in model_answer_letter:
                model_answer_letter = model_answer_letter.split("Explanation")[
                    0
                ].strip()

            pattern = r"\([A-D]\)"
            matches = re.findall(pattern, model_answer_letter)
            if len(matches) == 1:
                model_answer_letter = matches[0][1]
            elif len(matches) == 0:
                pattern = r"[A-D]\)"
                matches = re.findall(pattern, model_answer_letter)
                if len(matches) == 1:
                    model_answer_letter = matches[0][0]
                else:
                    model_answer_letter == ""
            else:
                model_answer_letter = ""

        else:
            print(model_answer)
            model_answer_letter = ""
        model_answer_letters.append(model_answer_letter)

    df["model_answer_letter"] = model_answer_letters
    df["model_answer_letter"] = "(" + df["model_answer_letter"].astype(str) + ")"
    df["correct"] = np.where(df["answer_letter"] == df["model_answer_letter"], 1, 0)
    incorrect_df = df[df["correct"] == 0]

    for _, row in df[df["correct"] == 1].iterrows():
        print(row["result"])
        print("___")

    correct_df = df[df["correct"] == 1]

    return correct_df, incorrect_df


def main(
    training_questions_file,
    testing_questions_file,
    output_dir,
):
    """
    Generate chain of thought questions in the training set that are similar to the questions in the test set
    (Instead of running on all training questions to save resources)

    Args:
        training_questions_file (str): The file path of the training questions CSV file.
        testing_questions_file (str): The file path of the testing questions CSV file.
        output_dir (str): The directory where the output files will be saved.

    Returns:
        None
    """
    train_df = pd.read_csv(training_questions_file)
    test_df = pd.read_csv(testing_questions_file)

    # set output file paths
    correct_filepath = os.path.join(output_dir, f"correct_gpt_outputs.csv")
    similar_filepath = os.path.join(output_dir, f"similar_questions.csv")

    if os.path.exists(similar_filepath):
        logger.info("Similar Questions file exist")
        similar_questions_df = pd.read_csv(similar_filepath)
    else:
        print(similar_filepath)
        _, similar_questions_df = find_similar_questions(
            train_df, test_df, model_name=os.getenv("SENTENCE_TRANSFORMER_MODEL"), k=10
        )
        similar_questions_df.to_csv(
            os.path.join(output_dir, "similar_questions.csv"), index=None
        )

    # Create prompts to run through GPT-4
    questions_to_generate_COT = train_df.loc[
        list(similar_questions_df["Train Indices"].unique()), :
    ]
    prompts = []
    for _, row in questions_to_generate_COT.iterrows():
        question = row["question"]
        options = row["options"]

        prompt = GPT_BASE_PROMPT.format(question=question, options=options)
        prompts.append(prompt)

    questions_to_generate_COT["prompt"] = prompts
    # Saving prompt file
    trial = 0

    input_filepath = os.path.join(output_dir, f"input_prompts_round_{trial}.csv")
    questions_to_generate_COT = questions_to_generate_COT.drop_duplicates(
        subset=["question", "options"]
    )
    questions_to_generate_COT.to_csv(input_filepath)
    logger.info(f"Starting with {len(questions_to_generate_COT)} questions.")

    # Run via GPT a maximum of 5 times
    while trial < 5 and len(questions_to_generate_COT) > 0:
        input_filepath = os.path.join(output_dir, f"input_prompts_round_{trial}.csv")
        logger.info(f"Working on file {input_filepath}")
        output_filepath = os.path.join(output_dir, f"input_prompts_round_{trial+1}.csv")

        gpt_outputs_filepath = os.path.join(output_dir, f"gpt_outputs_{trial}.csv")
        _ = asyncio.run(gpt_runner(input_filepath, gpt_outputs_filepath))

        correct_df, incorrect_df = process_outputs(
            result_file_path=gpt_outputs_filepath, train_df=train_df
        )

        # Add all correct ones to a single file
        if os.path.exists(correct_filepath):
            correct_df.to_csv(correct_filepath, mode="a", header=False, index=None)
        else:
            correct_df.to_csv(correct_filepath, index=None)

        # save incorrects ones to file to rerun again
        if len(incorrect_df) > 0:
            logger.info(f"Saving incorrect prompts to {output_filepath}")
            incorrect_df.to_csv(output_filepath)
            questions_to_generate_COT = incorrect_df.copy(deep=True)

        trial += 1


if __name__ == "__main__":
    fire.Fire(main)
