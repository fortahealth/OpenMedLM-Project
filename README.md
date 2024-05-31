# OpenMedLLM

## Overview

This repository contains code to identify similar questions from a dataset and utilize the OpenAI GPT-4 model to generate chain of thought reasoning for Medical MCQs. The process involves encoding questions using Sentence Transformers, finding similar questions using K-Nearest Neighbors (KNN), generating prompts after querying GPT-4 for chain of thought and inferencing  using the generated prompts with chain of thought reasoning.

## Table of Contents

- [Setup](#setup)
- [Usage](#usage)

## Setup

### Prerequisites

- Python 3.8+
- Required Python packages listed in `requirements.txt`

### Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/yourrepository.git
    cd yourrepository
    ```

2. Create and activate a virtual environment (optional but recommended):

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

### Environment Variables

Set the following environment variables in your shell or `.env` file:

- `OPENAI_ORG_ID`: Your OpenAI organization ID
- `OPENAI_API_KEY`: Your OpenAI API key

## Usage

```sh
python generate_cot.py training_questions_file=<path_to_training_questions_file> testing_questions_file=<path_to_testing_questions_file> output_dir=<path_to_output_directory>
```

```sh
python inference.py --results_dir=<results_directory> --prompts_file=<prompts_file_path> --model_path=<model_path> --batch_size=<batch_size>
```
