from evaluation.eval_vicuna import (
    run_eval as run_eval_vicuna,
    reorg_answer_file as reorg_answer_file_vicuna
)
from evaluation.eval_llama3 import (
    run_eval as run_eval_llama3,
    reorg_answer_file as reorg_answer_file_llama3,
)

run_eval_fndict = {
    "vicuna": run_eval_vicuna,
    "llama3": run_eval_llama3
}
reorg_answer_file_fndict = {
    "vicuna": reorg_answer_file_vicuna,
    "llama3": reorg_answer_file_llama3    
}