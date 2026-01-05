from abc import ABC, abstractmethod
from datasets import load_dataset
from agentless.util.utils import load_existing_instance_ids, load_jsonl, setup_logger
import os
import json
import argparse
from time import sleep

class CheckConfidence(ABC):
    def __init__(self, model_name, backend, output_folder, input_file, input_key):
        self.model_name = model_name
        self.backend = backend
        self.output_folder = output_folder
        self.input_file = input_file
        self.input_key = input_key

    @abstractmethod
    def check_confidence(self, instance_id: str, task: str, output: str) -> tuple[int, str]:
        pass

class LLMCheckConfidence(CheckConfidence):
    check_confidence_prompt = """
You are a code agent working on a codebase. You are working on the following Github problem description:
### GitHub Problem Description ###
{problem_statement}
Your current task is {task}
The output of your fellow agent on this taks is:
{output}
If the output is empty, please return 0.
Please check the output of your fellow agent and provide a confidence score between 0 and 100.
0 means you have very low confidence in the output of your fellow agent and 100 means you have very high confidence in the output of your fellow agent.
Please return only the confidence score wrapped with ```.
For example:
```
100
```
    """



    def __init__(self, dataset_name, model_name, backend, output_folder, input_file, input_key):
        super().__init__(model_name, backend, output_folder, input_file, input_key)
        self.dataset = load_dataset(dataset_name, split="test")

    def _parse_confidence(self, response: str) -> int:
        if response.count("```") != 2:
            return None
        ret = response.split("```")[1].strip()
        return int(ret) if ret.isdigit() else None

    def check_confidence(self, instance_id, task: str, output: str) -> tuple[int, str, str]:
        from agentless.util.model import make_model
        bench_data = [x for x in self.dataset if x["instance_id"] == instance_id][0]
        problem_statement = bench_data["problem_statement"]
        message = self.check_confidence_prompt.format(
            problem_statement=problem_statement,
            task=task,
            output=output,
        )
        logger_path = os.path.join(self.output_folder, f"confidence_logs/{instance_id}.log")
        os.makedirs(os.path.join(self.output_folder, "confidence_logs"), exist_ok=True)
        self.logger = setup_logger(logger_path)
        model = make_model(self.model_name, self.backend, self.logger, batch_size=1, temperature=0.8, max_tokens=256)
        response_list = model.codegen(message)
        response_text = response_list[0]["response"]
        response = response_list[0]["usage"]
        confidence = self._parse_confidence(response_text)
        return confidence, response, message

    def check_all_confidence(self, task: str):
        input_data = load_jsonl(self.input_file)
        checked_instance_ids = load_existing_instance_ids(os.path.join(self.output_folder, f"confidence_results.jsonl"))
        with open(os.path.join(self.output_folder, f"confidence_results.jsonl"), "a") as f:
            for data in input_data:
                output_data = {}
                instance_id = data["instance_id"]
                output = data[self.input_key]
                if instance_id in checked_instance_ids:
                    continue
                confidence, response, message = self.check_confidence(instance_id, task, output)
                if confidence is None:
                    continue
                output_data["instance_id"] = instance_id
                output_data["task"] = task
                output_data["output"] = output
                output_data["confidence"] = confidence
                output_data["response"] = response
                output_data["message"] = message
                f.write(json.dumps(output_data) + "\n")
                print(f"Checked {instance_id} with confidence {confidence}")
                sleep(1)

    
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset_name", type=str, required=True)
#     parser.add_argument("--model_name", type=str, required=True)
#     parser.add_argument("--backend", type=str, required=True)
#     parser.add_argument("--output_folder", type=str, required=True)
#     parser.add_argument("--input_file", type=str, required=True)
#     parser.add_argument("--input_key", type=str, required=True)
#     args = parser.parse_args()
#     check_confidence = LLMCheckConfidence(args.dataset_name, args.model_name, args.backend, args.output_folder, args.input_file, args.input_key)
#     check_confidence.check_all_confidence(args.task)
        
