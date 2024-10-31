from fastapi import FastAPI, Request, File, UploadFile
from pydantic import BaseModel
from typing import List
from src.models.model_loader import load_model
from src.evaluation.metrics import evaluate_llms_with_semantic_search
from src.utils.file_handler import save_results_to_csv
from src.evaluation.metrics import suggest_models  # Ensure suggest_models is defined


class InputOutputPair(BaseModel):
    input_text: str
    ground_truths: List[str]

class EvaluateRequest(BaseModel):
    input_output_pairs: List[InputOutputPair]
    output_type: str


app = FastAPI(swagger_ui_parameters={"syntaxHighlight": False})
@app.get("/")
async def main_route():
  return {"message": "Hey, It is me Goku"}

@app.post('/evaluate')
def evaluate(request: EvaluateRequest):
    try:
        input_output_pairs = request.input_output_pairs
        output_type = request.output_type

        all_results = []

        # Step 1: Suggest LLM models based on output type
        suggested_llms = suggest_models(output_type)

        # Step 2: Iterate over each input-output pair and evaluate
        for pair in input_output_pairs:
            input_text = pair.input_text
            ground_truths = pair.ground_truths

            # Evaluate LLMs with semantic search for each input/output pair
            llm_results = evaluate_llms_with_semantic_search(input_text, ground_truths, suggested_llms)
            all_results.append({
                "input_text": input_text,
                "ground_truths": ground_truths,
                "llm_results": llm_results
            })

        # Step 3: Save the results to a CSV file
        # csv_file = save_results_to_csv(all_results)

        # Step 4: Return the CSV file
        return all_results

    except Exception as e:
        #return ({"error": str(e)}), 500
        print(e)
        return ({"error": str(e)})