# project-02807
Project for 02807 - Computational tools for data science, Group 44


### Setting up the Python Environment for this project
----
0. Ensure python is installed ([link](https://www.python.org/downloads/))
    > Note that the python version used is **3.12**. Please ensure you are using this version when running any code from this repo.

1. Install uv ([link](https://docs.astral.sh/uv/getting-started/installation/)).

2. Navigate to your local directory that contains this project.

3. Run the following commands in your terminal:

    3.1 To create the virtual environment: `uv venv`
    
    3.2 To activate the virtual environment:
    - **Windows:** `.venv/Scripts/activate`
    - **Mac/Linux:** `source .venv/bin/activate`
    
    3.3 To install the dependencies: `uv sync` 

    > This will both create a virtual environment and install all the necessary dependencies to run the solutions to the exam problems.

4. Solutions to the exam problems are found in the `tasks/` folder. Each are named `qX.py` where `X` refers to the question number. In order to run each of the tasks please enter and run the following command in your terminal:
    ```bash
    uv run python -m tasks.qX
    ```
