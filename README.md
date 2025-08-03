# pychess: An End-to-End Pipeline for Improving LLMs with Reinforcement Learning

This project demonstrates a complete, five-phase pipeline for improving a base Large Language Model's (LLM) chess-playing ability. It begins with a single-machine prototype and evolves into a blueprint for a scalable, automated, cloud-native system.

The project follows the modern paradigm of **Reinforcement Learning from AI Feedback (RLAIF)** to first teach a model the rules of chess, then build an AI "coach" to provide strategic feedback, and finally use that feedback to train a superior model.

## The Five-Phase Journey

The project is structured as a journey of increasing scale and sophistication:

- **Phases 1-3** constitute the **Local Prototype**, proving the core ML concepts on a single machine.
- **Phases 4-5** describe the **Distributed Cloud Architecture**, transforming the prototype into a robust, scalable, and automated system ready for production-level workloads.

---

### Phase 1: Supervised Fine-Tuning (SFT) - The Foundation

- **Goal:** To create a base model that understands the syntax of chess notation and has learned common patterns from expert games.
- **How it Works:** We fine-tune a pre-trained LLM (`distilgpt2`) on a large dataset of elite human games. The model's objective is simple: given a sequence of moves, predict the next most likely move. At the end of this phase, we have a model that can play legal, plausible-looking games but lacks deep strategic understanding. It has the "book smarts."
- **Key Technologies:** `Hugging Face transformers`, `datasets`, `trl.SFTTrainer`.

### Phase 2: Reward Modeling - Building the AI Coach

- **Goal:** To create a "judge" model that, given a game state, outputs a score indicating its strategic quality.
- **How it Works:** We synthetically generate a preference dataset. For a given position, we get a "good" move from a master chess engine (Stockfish) and a "mediocre" move from our SFT student. We then train a new `distilgpt2` model on thousands of these `(chosen_move, rejected_move)` pairs. This model learns to distinguish strong moves from weak ones and becomes our frozen **Reward Model (RM)**.
- **Key Technologies:** `Stockfish`, `python-chess`, `trl.RewardTrainer`.

### Phase 3: Reinforcement Learning - Gaining Experience

- **Goal:** To use the Reward Model's feedback to teach our SFT model how to play better, more strategic chess.
- **Reinforcement Learning Policy:** This project uses **Proximal Policy Optimization (PPO)**, a state-of-the-art **Actor-Critic** algorithm.
  - **The Actor:** Our SFT model, which selects moves (actions).
  - **The Critic:** A learned value function that estimates the quality of a position.
  - **The Reward Signal:** Our frozen Reward Model, which provides immediate feedback on the Actor's moves.
- **How it Works:** The Actor makes a move, and the Reward Model provides a score. PPO uses this score, balanced by the Critic's evaluation, to update the Actor's policy. This loop encourages the Actor to discover strategies that consistently earn high rewards from our AI coach.
- **Key Technologies:** `trl.PPOTrainer`, `AutoModelForCausalLMWithValueHead`.

### Phase 4: Scaling the Pipeline - Distributed Architecture

- **Goal:** To refactor the single-machine prototype to handle massive datasets and run efficiently in the cloud, addressing the primary performance bottlenecks.
- **How it Works:**
  1.  **Containerization (Docker):** The entire application, including all Python dependencies and the Stockfish engine, is packaged into a **Docker container**. This ensures perfect reproducibility and portability between a local laptop and any cloud machine.
  2.  **Centralized Storage (AWS S3 / Parquet):** All data is moved from local files to a cloud object store like **AWS S3**. We switch from JSONL to **Apache Parquet**, a columnar format optimized for large-scale, parallel data processing. This decouples storage from compute.
  3.  **Distributed Data Generation (Ray):** The preference data generation step—the biggest bottleneck—is parallelized using **Ray**. A Ray cluster of hundreds of CPU-based cloud machines can be spun up to generate millions of preference pairs simultaneously, turning a multi-day process into a multi-hour one.

### Phase 5: Automation and MLOps - Production-Grade Systems

- **Goal:** To transform the scaled components into a robust, automated, and trackable workflow suitable for a production environment.
- **How it Works:**
  1.  **Workflow Orchestration (AWS Step Functions):** The simple `run_pipeline.sh` script is replaced by a formal state machine defined in **AWS Step Functions**. This cloud-native orchestrator manages the entire pipeline, triggers jobs, handles errors and retries, and provides a visual map of the workflow, creating a truly hands-off system.
  2.  **Distributed Training (Hugging Face Accelerate):** The training scripts are launched on multi-GPU clusters using the **`accelerate`** command. This allows us to train much larger models on larger datasets without changing the Python code.
  3.  **Experiment Tracking (MLflow):** Every pipeline run is logged to an **MLflow** tracking server. It records all hyperparameters, code versions, evaluation metrics (win/loss rates), and saves the final model files as artifacts. This provides crucial reproducibility, traceability, and a dashboard for comparing the performance of different runs.

---

## How to Run This Project

### Part 1: Running the Local Prototype (Phases 1-3)

1.  **Clone the repository and set up the environment:**

    ```bash
    git clone
    cd pychess
    python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt
    ```

2.  **Download dependencies:**
    - Download the Stockfish executable for your OS and place it in the project root.
    - Download the Lichess game data (`wget ...`, `unzstd ...`).
    - Example for chess game data
    > ``` bash
        wget -P data/raw/ https://database.lichess.org/standard/lichess_db_standard_rated_2024-08.pgn.zst
      ```
    

3.  **Execute the pipeline scripts in order:**

    ```bash
      # Phase 1
      python scripts/01_prepare_sft_data.py ...
      python scripts/02_run_sft.py ...

      # Phase 2
      python scripts/03_generate_preference_data.py ...
      python scripts/04_train_reward_model.py ...

      # Phase 3
      python scripts/05_run_rl_training.py ...
      python scripts/06_evaluate_models.py ...
    ```

4. **Overall Project Structure:**
    - If doing local experimentation:
    ``` bash
      pychess/
      ├── data/
      │   ├── raw/
      │   │   └── lichess_elite_2023-11.pgn   # Raw PGN game data from Lichess
      │   └── processed/
      │       ├── sft_dataset.jsonl           # Games formatted for SFT
      │       └── preference_dataset.jsonl    # (State, Chosen, Rejected) pairs for RM
      │
      ├── models/
      │   ├── sft_model/                      # Fine-tuned model (knows chess language)
      │   ├── reward_model/                   # Reward model (the "judge")
      │   └── rl_model/                       # Final, RL-tuned policy model
      │
      ├── scripts/
      │   ├── 01_prepare_sft_data.py          # Parses PGNs into a text dataset
      │   ├── 02_run_sft.py                   # Runs Supervised Fine-Tuning
      │   ├── 03_generate_preference_data.py  # Creates preference pairs using Stockfish
      │   ├── 04_train_reward_model.py        # Trains the reward model
      │   ├── 05_run_rl_training.py           # Runs PPO reinforcement learning
      │   └── 06_evaluate_models.py           # Pits models against each other
      │
      ├── src/
      │   └── chess_utils.py                  # Helper functions (e.g., Stockfish interaction)
      │
      ├── stockfish                           # The Stockfish chess engine executable
      │
      └── run_pipeline.sh                     # Master script to automate the entire local pipeline
    ```

    - If wanting to use cloud deployment:
    ``` bash
          pychess/
      ├── .github/
      │   └── workflows/
      │       └── ci.yml                # Optional: GitHub Actions for CI/CD
      ├── data/
      │   └── .gitkeep                  # Data is stored on S3, not in the repo
      ├── deployment/
      │   ├── Dockerfile                # Packages the application
      │   ├── requirements.txt          # Python dependencies
      │   └── step_functions.json       # AWS Step Functions workflow definition
      ├── models/
      │   └── .gitkeep                  # Models are stored on S3/MLflow, not in the repo
      ├── scripts/
      │   ├── 01_prepare_sft_data.py
      │   ├── 02_run_sft.py
      │   ├── 03_generate_preference_data.py
      │   ├── 04_train_reward_model.py
      │   ├── 05_run_rl_training.py
      │   └── 06_evaluate_models.py
      ├── src/
      │   └── chess_utils/
      │       ├── __init__.py
      │       └── engine.py             # Logic for interacting with Stockfish
      │       └── model_players.py      # Classes for HFPlayer, StockfishPlayer
      ├── tests/
      │   ├── test_chess_utils.py       # Unit tests for your helper functions
      │   └── test_pipeline_steps.py    # Integration tests for individual scripts
      ├── .gitignore                    # Ignores virtual env, pycache, etc.
      └── README.md                     # Your project's front page
    ```

### Part 2: Deploying the Scaled Pipeline (Phases 4-5)

This part describes the cloud architecture and is not executed by a single command.

1.  **Build and Push the Docker Image:**
    - Write a `Dockerfile` that includes all project code and dependencies.
    - Build the image: `docker build -t pychess .`
    - Push the image to a container registry like AWS ECR.

2.  **Deploy the Data Generation Cluster:**
    - Define a `ray-cluster.yaml` file specifying the instance types and number of workers.
    - Launch the cluster from your local machine: `ray up ray-cluster.yaml`
    - Execute the Ray-enabled data generation script, which now reads from and writes to S3.

3.  **Define and Run the Orchestrated Workflow:**
    - Define the entire multi-step process as a JSON state machine for **AWS Step Functions**.
    - Each step in the state machine will trigger a job on a specific service (e.g., an AWS Batch job for training, a Ray job for data generation), passing the S3 paths for data and models as parameters.
    - The training steps will use the `accelerate launch` command to run on multi-GPU machines.
    - Trigger the Step Function to start an automated, end-to-end pipeline run. Monitor progress in the AWS Console and view results in the **MLflow UI**.
