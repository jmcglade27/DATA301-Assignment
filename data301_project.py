import ast
import json
import time
import dask
from dask import bag as db

#INPUT_FILE = "australian_users_items.json"
#JSONL_FILE = "australian_users_items.jsonl"
#MIN_PLAYTIME = 60
#MIN_SUPPORT_COUNT = 20
#TOP_N = 20
INPUT_FILE = "test_file.json"
JSONL_FILE = "test_file.jsonl"
MIN_PLAYTIME = 60
MIN_SUPPORT_COUNT = 1
TOP_N = 5

RUN_SCALABILITY_TESTS = False
THREAD_COUNTS = [1, 2, 4, 8]
DATASET_SIZES = [1000, 5000, 10000, 20000]

def convert_to_jsonl(filename):
    with open(filename, "r", encoding="utf-8") as input_file:
        with open(JSONL_FILE, "w", encoding="utf-8") as output:
            for line in input_file:
                line = line.strip()
                if len(line) > 0:
                    user = ast.literal_eval(line)
                    useful_user = {"user_id": user["user_id"], "items": user["items"]}
                    output.write(json.dumps(useful_user))
                    output.write("\n")

def process_line(line):
    user = json.loads(line)
    games = []
    for item in user["items"]:
        if item["playtime_forever"] >= MIN_PLAYTIME:
            game = (item["item_id"], item["item_name"])
            games.append(game)
    return list(set(games))

def get_game_pairs(transaction): #the skeleton of this function was written by generative AI
    transaction = sorted(transaction)
    pairs = []
    for x in range(len(transaction)):
        for y in range(x + 1, len(transaction)):
            pairs.append((transaction[x], transaction[y]))
    return pairs

def make_rules(pair_count_entry, single_counts): #the skeleton of this function was written by generative AI
    pair, pair_count = pair_count_entry
    game_1 = pair[0]
    game_2 = pair[1]
    rules = []
    if game_1 in single_counts:
        confidence = pair_count / single_counts[game_1]
        rules.append((game_1[1], game_2[1], pair_count, confidence))
    if game_2 in single_counts:
        confidence = pair_count / single_counts[game_2]
        rules.append((game_2[1], game_1[1], pair_count, confidence))
    return rules

def run_analysis(thread_count, data_limit=None):
    start_time = time.time()
    with dask.config.set(scheduler="threads", num_workers=thread_count):
        data_bag = db.read_text(JSONL_FILE, blocksize="1MB")
        if data_limit is not None:
            limited_data = data_bag.take(data_limit)
            data_bag = db.from_sequence(limited_data, npartitions=thread_count)
        transactions = (data_bag.map(process_line).filter(lambda transaction: len(transaction) >= 2))
        valid_transaction_count = transactions.count().compute()
        single_games = transactions.flatten()
        single_game_counts = single_games.frequencies()
        filtered_single_counts = (single_game_counts.filter(lambda entry: entry[1] >= MIN_SUPPORT_COUNT))
        game_pairs = transactions.map(get_game_pairs).flatten()
        game_pair_counts = game_pairs.frequencies()
        filtered_pair_counts = (game_pair_counts.filter(lambda entry: entry[1] >= MIN_SUPPORT_COUNT))
        single_counts_dictionary = dict(single_game_counts.compute())
        rules = (filtered_pair_counts.map(lambda entry: make_rules(entry, single_counts_dictionary)).flatten())
        top_games = (filtered_single_counts.topk(TOP_N, key=lambda entry: entry[1]).compute())
        top_pairs = (filtered_pair_counts.topk(TOP_N, key=lambda entry: entry[1]).compute())
        top_rules = (rules.topk(TOP_N, key=lambda entry: entry[3]).compute())
    end_time = time.time()
    total_time = end_time - start_time
    return {"thread_count": thread_count, "data_limit": data_limit, "valid_transaction_count": valid_transaction_count, "time": total_time, "top_games": top_games, "top_pairs": top_pairs, "top_rules": top_rules}

def print_results(results):
    print("\nTop single games:")
    for game, count in results["top_games"]:
        print(game[1], "| count:", count)
    print("\nTop game pairs:")
    for pair, count in results["top_pairs"]:
        print(pair[0][1], "+", pair[1][1], "| count:", count)
    print("\nTop recommendations:")
    for source, recommendation, pair_count, confidence in results["top_rules"]:
        print(
            source,
            "->",
            recommendation,
            "| pair count:",
            pair_count,
            "| confidence:",
            round(confidence, 4)
        )

def save_timing_results(results):
    with open("timing_results.csv", "w", encoding="utf-8") as output:
        output.write("experiment,thread_count,data_limit,time\n")
        for experiment, thread_count, data_limit, runtime in results:
            output.write(
                f"{experiment},{thread_count},{data_limit},{runtime}\n"
            )

def run_scalability_tests():
    timing_results = []
    print("\nDataset size scalability tests")
    for size in DATASET_SIZES:
        result = run_analysis(thread_count=1, data_limit=size)
        timing_results.append(("dataset_size", 1, size, result["time"]))
        print(
            "Dataset size:",
            size,
            "| threads: 1",
            "| time:",
            round(result["time"], 2)
        )
    print("\nThread count scalability tests")
    for thread_count in THREAD_COUNTS:
        result = run_analysis(thread_count=thread_count, data_limit=DATASET_SIZES[-1])
        timing_results.append((
            "thread_count",
            thread_count,
            DATASET_SIZES[-1],
            result["time"]
        ))
        print(
            "Dataset size:",
            DATASET_SIZES[-1],
            "| threads:",
            thread_count,
            "| time:",
            round(result["time"], 2)
        )
    save_timing_results(timing_results)

def main():
    print("Converting to JSON")
    convert_to_jsonl(INPUT_FILE)
    print("\nSuccessfully converted to JSON")
    print("\nRunning analysis")
    results = run_analysis(thread_count=4)
    print_results(results)
    print("\nMain analysis time:", round(results["time"], 2), "seconds")
    if RUN_SCALABILITY_TESTS:
        run_scalability_tests()

main()