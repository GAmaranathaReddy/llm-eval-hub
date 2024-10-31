
from evaluate import load
from src.models.embeddings import generate_embeddings
from src.models.model_loader import load_model
from src.evaluation.semantic_search import perform_semantic_search
from src.evaluation.azure import AzureClient
from src.evaluation.google import GoogleClient
from src.evaluation.opensource import OpenSourceClient
from src.evaluation.bedrock import BedrockClient
from src.evaluation.bison import BisonClient
import numpy as np
from numpy.linalg import norm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import BERTScorer
import concurrent.futures
from functools import partial
# Load evaluation metrics
rouge = load('rouge')
bleu = load('bleu')

# Suggest models based on output type
def suggest_models(output_type):
    models = {
        "gpt-35-turbo": {"task": "text-generation"},
        "gpt-35-turbo-0125": {"task": "text-generation"},
        "gpt-35-turbo-16k": {"task": "text-generation"},
        "gpt-4o-mini": {"task": "text-generation"},
        "gpt-4o": {"task": "text-generation"},
        "gpt-4": {"task": "text-generation"},
        "gpt-4-0613": {"task": "text-generation"},
        "gpt-4-32k": {"task": "text-generation"},
        "gemini-1.0-pro": {"task": "text-generation"},
        "gemini-1.5-flash": {"task": "text-generation"},
        "gemini-1.5-pro": {"task": "text-generation"},
        "amazon--titan-text-express" : {"task": "summarization"},
        "amazon--titan-text-lite" : {"task": "summarization"},
        "anthropic--claude-3-haiku" : {"task": "summarization"},
        "anthropic--claude-3-opus" : {"task": "summarization"},
        "anthropic--claude-3-sonnet" : {"task": "summarization"},
        "anthropic--claude-3.5-sonnet" : {"task": "summarization"},
        "chat-bison": {"task": "summarization"},
        "meta--llama3.1-70b-instruct": {"task": "summarization"},
        "meta--llama3-70b-instruct": {"task": "summarization"},
        "mistralai--mixtral-8x7b-instruct-v01": {"task": "summarization"}
    }
    suggested_models = []
    for model_name, details in models.items():
        if output_type in details['task']:
            suggested_models.append(model_name)
    return suggested_models

# Function to compute metrics including semantic search
def evaluate_generated_output_with_semantic_search(ground_truths, generated_output):
    bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    # Traditional evaluation metrics
    metrics = []
    generated_output_tokens = generated_output.split()
    for ground_truth in ground_truths:
        rouge_result = rouge.compute(predictions=[generated_output], references=[ground_truth])
        gt_embedding = generate_embeddings([ground_truth], "d033a2121d860b30")
        # generate embedding 2 times because generate_embeddings function expects a list of strings
        generated_embedding = generate_embeddings([generated_output], "d033a2121d860b30")
        cosine_sim = cosine_similarity(gt_embedding, generated_embedding)
        #embeddings = [generate_embeddings([ground_truth], "d033a2121d860b30")]
       # Perform semantic search across all ground truths
        # best_match_score = perform_semantic_search(generated_embedding, embeddings)
        # Compute BLEU Score
        ground_truth_tokens = ground_truth.split()  # Tokenize the ground truth
        # Use smoothing to avoid zero scores for small texts
        smoothing_function = SmoothingFunction().method1
        bleu_score = sentence_bleu([ground_truth_tokens], generated_output_tokens, smoothing_function=smoothing_function)

        # BERTScore
        P, R, F1 = bert_scorer.score([generated_output], [ground_truth])
        metrics.append({
            "ground_truth": ground_truth,
            "ROUGE": rouge_result,
            "Cosine Similarity": cosine_sim,
            "BLEU": bleu_score,
            "BERTScore": {
                    "Precision": P.item(),
                    "Recall": R.item(),
                    "F1": F1.item()
            }
        })
    return metrics

# Function to calculate Cosine Similarity
def cosine_similarity(gt_embedding, generated_embedding):
    return np.dot(gt_embedding, generated_embedding) / (norm(gt_embedding) * norm(generated_embedding))


# Function to evaluate multiple LLMs
def evaluate_llms_with_semantic_search(input_text, ground_truths, model_list):
    results = []
    model_deployment_id = {
        "gpt-35-turbo": "da6f4785272ac6e4",
        "gpt-35-turbo-0125" : "d003c7931cb0508c",
        "gpt-35-turbo-16k" : "d4de736ad23814d0",
        "gpt-4o-mini": "db220c44190ef243",
        "gpt-4o": "d6f0a9fa94192496",
        "gpt-4": "d2334f5e52cd183f",
        "gpt-4-0613" : "d2334f5e52cd183f",
        "gpt-4-32k": "d516ffde5b51f72d",
        "gemini-1.0-pro" : "dd0c547712989f08",
        "gemini-1.5-flash": "d9ce678acb038b5f",
        "gemini-1.5-pro": "d073233a8ab56c81",
        "amazon--titan-text-express": "d208cadfe9ffbc8c",
        "amazon--titan-text-lite": "d9a7e6d37495caca",
        "anthropic--claude-3-haiku": "d1c65b0d32b3be31",
        "anthropic--claude-3-opus": "d968da306b092bd8",
        "anthropic--claude-3-sonnet": "d45d49f584d852a8",
        "anthropic--claude-3.5-sonnet": "d0c51b8b48a1cf65",
        "chat-bison": "dd8fed797a6cb14b",
        "meta--llama3.1-70b-instruct": "da2486df13ce3d93",
        "meta--llama3-70b-instruct": "d61666505df93767",
        "mistralai--mixtral-8x7b-instruct-v01" : "d181406830443a0d"
    }

    for model_name in model_list:
        #TODO load model & update code to return generated_output
        deployment_id = model_deployment_id[model_name]
        if "gpt" in model_name:
            generated_output = AzureClient().call(deployment_id, model_name, input_text)
        elif "gemini" in model_name:
            generated_output = GoogleClient().call(deployment_id, model_name, input_text)
        elif "amazon" in model_name or "anthropic" in model_name:
            generated_output = BedrockClient().call(deployment_id,model_name,input_text)
        elif "bison" in model_name:
            generated_output = BisonClient().call(deployment_id, model_name, input_text)
        else:
            generated_output = OpenSourceClient().call(deployment_id, model_name, input_text)
        # generated_output = model(input_text, max_length=100, do_sample=False)[0]['generated_text']
        metrics = evaluate_generated_output_with_semantic_search(ground_truths, generated_output)
        results.append({
            "model": model_name,
            "generated_output": generated_output,
            "evaluation_metrics": metrics
        })
    return results
    

