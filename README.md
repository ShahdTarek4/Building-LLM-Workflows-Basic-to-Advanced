## LLM WORKFLOWS

## Implementation Overview

The project implements multiple AI-powered workflows for processing text data and generating structured content. These workflows include:
- **Key Points Extraction**: Identifies main insights from a given text.
- **Summary Generation**: Produces a concise summary of the input text.
- **Social Media Posts Creation**: Generates content for platforms like Facebook, LinkedIn, and Twitter.
- **Email Newsletter Generation**: Formats structured newsletters from input text.

Each workflow utilizes an LLM (Large Language Model) to extract information and generate outputs based on predefined schemas.

## Setup Instructions

Follow these steps to set up and run the project:

1. **Clone the Repository**
   ```
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Create and Activate a Virtual Environment**
   ```
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```

3. **Install Dependencies**
   ```
   pip install -r requirements.txt
   ```

4. **Run the Workflow Script**
   ```sh
   python LLM_Workflows.py
   ```


## Example Outputs

### Pipeline Wokflow
```
<[img width="949" alt="example_output_pipelineworkflow" src="https://github.com/user-attachments/assets/01868144-e3f3-4577-8fab-057b062d74d0"](https://github.com/ShahdTarek4/LLM_Workflows/blob/main/example_output_pipelineworkflow.png) />


```

### DAG Workflow
```
<img width="846" alt="example_output_dagworkflow" src="https://github.com/user-attachments/assets/2e8cf312-ba73-460e-b56b-f9b82c177dae" />


```

### Reflexion Enhancment 
```

<img width="857" alt="example_output_reflexion" src="https://github.com/user-attachments/assets/22717a5d-fcec-4d47-bfd4-6c3bec650ccd" />

```

### Agent Driven Workflow 
```

<img width="848" alt="example_output_agent-drivenworkflow" src="https://github.com/user-attachments/assets/df53c34f-c528-471a-b535-2b14c0020ca7" />

```

## Effectiveness Analysis

- **Pipeline Workflow***: Ensures a linear, structured sequence of operations. Although easy to use and debug, it is not highly flexible—if one step fails, the entire process fails.
- **Directed Acyclic Graph (DAG) Workflow**: Offers more parallelism and modularity than pipelines. It allows independent execution of operations where possible, which improves the process. However, it requires more complex setup and monitoring.
- **Reflection Workflow**: Enables self-correction and improvement in real-time by incorporating feedback loops. This makes outputs more refined but can lead to higher computational costs and potential response delays.
- **Agent-driven Workflow**: Uses AI agents to autonomously make workflow execution decisions. This enables dynamic execution of activities but brings uncertainty in outputs due to the unpredictable nature of llms.

## Challenges 

### 1. **Empty Outputs for Some Workflows**

### 2. **Model Response Inconsistency**
   





