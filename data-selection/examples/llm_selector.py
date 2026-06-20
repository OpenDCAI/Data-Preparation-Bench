from __future__ import annotations

from dataflow.operators.eval import MetaScorer
from dataflow.serving.APILLMServing_request import APILLMServing_request

from data_selection import LLMAsSelector
from data_selection.runner import run_selection

if __name__ == "__main__":
    llm_serving = APILLMServing_request(
        api_url="https://api.openai.com/v1/chat/completions",
        key_name_of_api_key="DF_API_KEY",
        model_name="gpt-4o",
        max_workers=10,
    )
    scorer = MetaScorer(
        llm_serving=llm_serving,
        dimensions=[
            {
                "dimension_name": "Mathematical Correctness",
                "description": "Evaluate whether the mathematical reasoning and final answer are correct.",
                "example_list": [
                    {
                        "text": "The derivative of x^2 is 2x by the power rule.",
                        "score": "5",
                    },
                    {
                        "text": "The derivative of x^2 is x by dividing the exponent.",
                        "score": "1",
                    },
                ],
            },
            {
                "dimension_name": "Solution Clarity",
                "description": "Assess how clearly the solution steps are explained and organized.",
                "example_list": [
                    {
                        "text": "Step 1: Factor the quadratic. Step 2: Apply zero product property. Step 3: Solve each factor.",
                        "score": "5",
                    },
                    {"text": "Just do the math and you get x=3.", "score": "2"},
                ],
            },
            {
                "dimension_name": "Educational Value",
                "description": "Determine whether the problem and solution provide meaningful learning potential for mathematics.",
                "example_list": [
                    {
                        "text": "This problem demonstrates the application of the chain rule in multivariable calculus.",
                        "score": "5",
                    },
                    {"text": "What is 1+1?", "score": "1"},
                ],
            },
        ],
    )
    selector = LLMAsSelector(
        k=100000,
        text_key="text",
        scorer=scorer,
    )
    run_selection(
        selector,
        input_path="/jizhicfs/linyibo/datasets/dyyyyyyyy/ScaleQuest-Math/scalequest_math.jsonl",
        output_path="data/output_llm.jsonl",
    )
