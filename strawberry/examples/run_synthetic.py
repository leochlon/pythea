from strawberry.tasks import generate_dataset
from strawberry.eval import run_eval

if __name__ == "__main__":
    # Set OPENAI_API_KEY in your environment.
    items = generate_dataset(n=50, distance_tokens=512, M=10, query_rule="FIRST", seed=0)
    res = run_eval(items=items, model="gpt-4o-2024-08-06", null_mode="SCRUB_FIRST")
    print(res.summary)
