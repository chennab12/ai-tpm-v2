# ============================================================
# V2 - AI INFERENCE BENCHMARK HARNESS
#
# PURPOSE
# -------
# Learn how a technical AI/ML TPM establishes a repeatable
# functional + performance baseline for an inference workload.
#
#
# PIPELINE
# --------
#
# Prompt
#   ↓
# Tokenizer
#   ↓
# Input Tokens
#   ↓
# Model
#   ↓
# Inference
#   ├── Prefill
#   └── Decode
#   ↓
# Output Tokens
#   ↓
# Performance Metrics
#   ↓
# CSV Log
#   ↓
# TPM PASS / WARN / FAIL Scorecard
#
#
# CORE CONCEPTS
# -------------
#
# model
# tokenizer
# tokens
# inference
# prefill
# decode
# TTFT
# TPOT
# TPS
# latency
# P50
# P95
# CPU
# RAM
# benchmarking
# warm-up
# baseline
# acceptance criteria
# ============================================================


import csv
import os
import statistics
import time
from datetime import datetime

import psutil
import torch

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import TextIteratorStreamer

from threading import Thread


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_NAME = "distilgpt2"

PROMPT = """
Explain why AI inference latency and throughput matter
to a technical program manager in simple terms.
"""

MAX_NEW_TOKENS = 40

NUMBER_OF_RUNS = 5

LOG_FILE = "logs/inference_results.csv"


# ------------------------------------------------------------
# TPM ACCEPTANCE CRITERIA
# ------------------------------------------------------------
#
# These are NOT universal production numbers.
#
# They are simply learning targets for this local experiment.
#
# In a real customer POC, these would come from:
#
# customer requirements
# SLA/SLO
# reference benchmark
# product requirements
# previous-generation baseline
#

TARGET_P95_LATENCY_SEC = 10.0
TARGET_TPS = 5.0
TARGET_ERROR_RATE_PERCENT = 1.0


# ============================================================
# 2. HELPER: PRINT SECTION
# ============================================================

def section(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


# ============================================================
# 3. LOAD TOKENIZER
# ============================================================

section("LOADING TOKENIZER")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# GPT-style models sometimes have no explicit padding token.
#
# For our simple experiment, we reuse EOS as padding.
#
# EOS = End Of Sequence
#

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


print("Tokenizer loaded.")


# ============================================================
# 4. LOAD MODEL
# ============================================================

section("LOADING MODEL")

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

model.eval()


# model.eval()
#
# tells PyTorch:
#
# "We are using this model for inference, not training."
#
#
# TRAINING
#
# forward pass
#     ↓
# loss
#     ↓
# backward pass
#     ↓
# update weights
#
#
# INFERENCE
#
# forward pass
#     ↓
# prediction / generation
#
# weights stay unchanged
#

print("Model loaded.")

print(f"Model: {MODEL_NAME}")


# ============================================================
# 5. DETERMINE DEVICE
# ============================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)

print(f"Device: {device}")


# TPM TAKEAWAY:
#
# The same model can behave very differently depending on:
#
# CPU
# GPU
# AI accelerator
#
# Therefore every benchmark should document its environment.


# ============================================================
# 6. TOKENIZE PROMPT
# ============================================================

section("TOKENIZATION")

inputs = tokenizer(
    PROMPT,
    return_tensors="pt"
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}


input_tokens = inputs["input_ids"].shape[1]


print("Prompt:")
print(PROMPT)

print(f"\nInput tokens: {input_tokens}")


# TPM TAKEAWAY:
#
# More input tokens generally mean:
#
# more prefill work
# more compute
# more attention work
# potentially higher TTFT
# larger KV cache
# higher cost


# ============================================================
# 7. SYSTEM RESOURCE SNAPSHOT
# ============================================================

def get_system_metrics():

    cpu_percent = psutil.cpu_percent(interval=None)

    memory = psutil.virtual_memory()

    ram_used_gb = memory.used / (1024 ** 3)

    ram_percent = memory.percent

    return {
        "cpu_percent": cpu_percent,
        "ram_used_gb": ram_used_gb,
        "ram_percent": ram_percent
    }


# ============================================================
# 8. RUN ONE STREAMING INFERENCE
# ============================================================

def run_inference(run_number):

    """
    Runs one generation and captures:

    - end-to-end latency
    - approximate TTFT
    - output tokens
    - TPS
    - TPOT
    - CPU
    - RAM
    - success/failure
    """


    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )


    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id
    }


    output_text_parts = []

    first_token_time = None


    # --------------------------------------------------------
    # Start timing
    # --------------------------------------------------------

    start_time = time.perf_counter()


    # --------------------------------------------------------
    # Start model generation in a second thread
    # --------------------------------------------------------
    #
    # WHY?
    #
    # Normally model.generate() waits until the complete
    # response is produced.
    #
    # Streaming lets us observe output while generation occurs.
    #
    # This allows us to approximate:
    #
    # TIME TO FIRST TOKEN
    #

    thread = Thread(
        target=model.generate,
        kwargs=generation_kwargs
    )

    thread.start()


    # --------------------------------------------------------
    # Read streamed output
    # --------------------------------------------------------

    try:

        for new_text in streamer:

            if first_token_time is None:

                first_token_time = time.perf_counter()

            output_text_parts.append(new_text)


        thread.join()


        end_time = time.perf_counter()

        success = True

        error_message = ""


    except Exception as error:

        end_time = time.perf_counter()

        success = False

        error_message = str(error)


    # --------------------------------------------------------
    # Calculate output text
    # --------------------------------------------------------

    output_text = "".join(output_text_parts)


    # --------------------------------------------------------
    # Count output tokens
    # --------------------------------------------------------

    output_token_ids = tokenizer(
        output_text,
        return_tensors="pt"
    )["input_ids"]

    output_tokens = output_token_ids.shape[1]


    # --------------------------------------------------------
    # End-to-end latency
    # --------------------------------------------------------

    total_latency = end_time - start_time


    # --------------------------------------------------------
    # Approximate TTFT
    # --------------------------------------------------------
    #
    # TTFT =
    #
    # request start
    #      ↓
    # prompt processing / prefill
    #      ↓
    # FIRST visible generated output
    #
    #
    # This is an approximation because the streamer returns
    # decoded text chunks rather than exposing hardware-level
    # timestamps directly.
    #

    if first_token_time is not None:

        ttft = first_token_time - start_time

    else:

        ttft = total_latency


    # --------------------------------------------------------
    # Decode duration
    # --------------------------------------------------------

    decode_duration = max(
        total_latency - ttft,
        0.000001
    )


    # --------------------------------------------------------
    # TPS
    # --------------------------------------------------------
    #
    # TPS = generated tokens / total generation time
    #
    # Here we use end-to-end time for a simple benchmark.
    #

    if total_latency > 0:

        tps = output_tokens / total_latency

    else:

        tps = 0


    # --------------------------------------------------------
    # TPOT
    # --------------------------------------------------------
    #
    # TPOT = Time Per Output Token
    #
    # Smaller is generally better.
    #
    # We approximate decode TPOT:
    #
    # decode time
    # ---------------------------
    # output tokens after first
    #

    if output_tokens > 1:

        tpot = decode_duration / (output_tokens - 1)

    else:

        tpot = decode_duration


    resources = get_system_metrics()


    return {

        "run": run_number,

        "timestamp": datetime.now().isoformat(),

        "success": success,

        "error": error_message,

        "input_tokens": input_tokens,

        "output_tokens": output_tokens,

        "latency_sec": total_latency,

        "ttft_sec": ttft,

        "tpot_sec": tpot,

        "tokens_per_sec": tps,

        "cpu_percent": resources["cpu_percent"],

        "ram_used_gb": resources["ram_used_gb"],

        "ram_percent": resources["ram_percent"],

        "output_text": output_text
    }


# ============================================================
# 9. WARM-UP RUN
# ============================================================

section("WARM-UP")

print("Running one warm-up inference...")


# WHY WARM UP?
#
# First inference can include:
#
# lazy initialization
# memory allocation
# kernel initialization
# caches
# framework overhead
#
# Therefore production benchmarking usually separates:
#
# WARM-UP
#
# from
#
# MEASURED RUNS
#

warmup_result = run_inference(0)

print("Warm-up completed.")

print(
    f"Warm-up latency: "
    f"{warmup_result['latency_sec']:.2f} sec"
)


# ============================================================
# 10. BENCHMARK RUNS
# ============================================================

section("BENCHMARK RUNS")

results = []


for run_number in range(
    1,
    NUMBER_OF_RUNS + 1
):

    print(
        f"\nRunning benchmark "
        f"{run_number}/{NUMBER_OF_RUNS}..."
    )


    result = run_inference(run_number)

    results.append(result)


    print(
        f"Latency: {result['latency_sec']:.2f} sec"
    )

    print(
        f"TTFT:    {result['ttft_sec']:.2f} sec"
    )

    print(
        f"TPOT:    {result['tpot_sec'] * 1000:.2f} ms/token"
    )

    print(
        f"TPS:     {result['tokens_per_sec']:.2f}"
    )

    print(
        f"CPU:     {result['cpu_percent']:.1f}%"
    )

    print(
        f"RAM:     {result['ram_percent']:.1f}%"
    )


# ============================================================
# 11. SAVE RESULTS TO CSV
# ============================================================

section("LOGGING")


os.makedirs(
    os.path.dirname(LOG_FILE),
    exist_ok=True
)


file_exists = os.path.exists(LOG_FILE)


fieldnames = [

    "timestamp",
    "run",
    "success",
    "error",
    "input_tokens",
    "output_tokens",
    "latency_sec",
    "ttft_sec",
    "tpot_sec",
    "tokens_per_sec",
    "cpu_percent",
    "ram_used_gb",
    "ram_percent"
]


with open(
    LOG_FILE,
    "a",
    newline="",
    encoding="utf-8"
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )


    if not file_exists:

        writer.writeheader()


    for result in results:

        writer.writerow({
            key: result[key]
            for key in fieldnames
        })


print(f"Results written to: {LOG_FILE}")


# ============================================================
# 12. CALCULATE AGGREGATE KPIs
# ============================================================

successful_results = [

    result
    for result in results
    if result["success"]
]


failure_count = (
    len(results)
    -
    len(successful_results)
)


error_rate_percent = (
    failure_count
    /
    len(results)
    *
    100
)


latencies = [
    r["latency_sec"]
    for r in successful_results
]


ttfts = [
    r["ttft_sec"]
    for r in successful_results
]


tpots = [
    r["tpot_sec"]
    for r in successful_results
]


tps_values = [
    r["tokens_per_sec"]
    for r in successful_results
]


# ============================================================
# 13. PERCENTILE HELPER
# ============================================================

def percentile(values, percentile_value):

    if not values:

        return 0

    sorted_values = sorted(values)

    position = (
        len(sorted_values) - 1
    ) * percentile_value

    lower = int(position)

    upper = min(
        lower + 1,
        len(sorted_values) - 1
    )

    fraction = position - lower

    return (
        sorted_values[lower]
        +
        (
            sorted_values[upper]
            -
            sorted_values[lower]
        )
        *
        fraction
    )


# ============================================================
# 14. AGGREGATED STATISTICS
# ============================================================

if successful_results:

    average_latency = statistics.mean(latencies)

    p50_latency = percentile(
        latencies,
        0.50
    )

    p95_latency = percentile(
        latencies,
        0.95
    )

    average_ttft = statistics.mean(ttfts)

    average_tpot = statistics.mean(tpots)

    average_tps = statistics.mean(tps_values)

else:

    average_latency = 0
    p50_latency = 0
    p95_latency = 0
    average_ttft = 0
    average_tpot = 0
    average_tps = 0


# ============================================================
# 15. KPI SUMMARY
# ============================================================

section("V2 INFERENCE KPI SUMMARY")


print(f"Model:                  {MODEL_NAME}")

print(f"Device:                 {device}")

print(f"Input tokens:           {input_tokens}")

print(f"Configured output max:  {MAX_NEW_TOKENS}")

print(f"Measured runs:          {NUMBER_OF_RUNS}")


print()


print(
    f"Average latency:        "
    f"{average_latency:.2f} sec"
)


print(
    f"P50 latency:            "
    f"{p50_latency:.2f} sec"
)


print(
    f"P95 latency:            "
    f"{p95_latency:.2f} sec"
)


print(
    f"Average TTFT:           "
    f"{average_ttft:.2f} sec"
)


print(
    f"Average TPOT:           "
    f"{average_tpot * 1000:.2f} ms/token"
)


print(
    f"Average throughput:     "
    f"{average_tps:.2f} tokens/sec"
)


print(
    f"Error rate:             "
    f"{error_rate_percent:.1f}%"
)


# ============================================================
# 16. TPM SCORECARD
# ============================================================

section("TPM ACCEPTANCE SCORECARD")


# ------------------------------------------------------------
# Latency
# ------------------------------------------------------------

if p95_latency <= TARGET_P95_LATENCY_SEC:

    latency_status = "PASS"

elif p95_latency <= TARGET_P95_LATENCY_SEC * 1.25:

    latency_status = "WARN"

else:

    latency_status = "FAIL"


# ------------------------------------------------------------
# TPS
# ------------------------------------------------------------

if average_tps >= TARGET_TPS:

    tps_status = "PASS"

elif average_tps >= TARGET_TPS * 0.80:

    tps_status = "WARN"

else:

    tps_status = "FAIL"


# ------------------------------------------------------------
# Reliability
# ------------------------------------------------------------

if error_rate_percent <= TARGET_ERROR_RATE_PERCENT:

    reliability_status = "PASS"

elif error_rate_percent <= 5:

    reliability_status = "WARN"

else:

    reliability_status = "FAIL"


print(
    f"P95 latency target "
    f"(<={TARGET_P95_LATENCY_SEC}s): "
    f"{latency_status}"
)


print(
    f"TPS target "
    f"(>={TARGET_TPS}): "
    f"{tps_status}"
)


print(
    f"Error-rate target "
    f"(<={TARGET_ERROR_RATE_PERCENT}%): "
    f"{reliability_status}"
)


# ============================================================
# 17. OVERALL STATUS
# ============================================================

statuses = [
    latency_status,
    tps_status,
    reliability_status
]


if "FAIL" in statuses:

    overall_status = "FAIL"

elif "WARN" in statuses:

    overall_status = "WARN"

else:

    overall_status = "PASS"


print()

print(
    f"OVERALL POC STATUS:     "
    f"{overall_status}"
)


# ============================================================
# 18. EXAMPLE MODEL OUTPUT
# ============================================================

section("EXAMPLE MODEL OUTPUT")


if successful_results:

    print(
        successful_results[-1]["output_text"]
    )

else:

    print(
        "No successful inference output."
    )


# ============================================================
# 19. TPM INTERPRETATION
# ============================================================

section("TPM INTERPRETATION")


print(
f"""
Functional bring-up:
    {"PASS" if successful_results else "FAIL"}

Performance baseline established:
    {"YES" if successful_results else "NO"}

Typical latency (P50):
    {p50_latency:.2f} sec

Tail latency (P95):
    {p95_latency:.2f} sec

Average TTFT:
    {average_ttft:.2f} sec

Average decode TPOT:
    {average_tpot * 1000:.2f} ms/token

Average throughput:
    {average_tps:.2f} tokens/sec

Error rate:
    {error_rate_percent:.1f}%

Overall POC status:
    {overall_status}
"""
)


print(
"""
Senior Technical TPM questions:

1. Is the functional path stable?

2. Is P95 materially worse than P50?

3. Is TTFT the main latency problem?

4. Or is TPOT/decode performance the issue?

5. Is throughput meeting the workload target?

6. Are CPU or RAM approaching saturation?

7. Are results consistent across repeated runs?

8. Did warm-up materially affect performance?

9. What configuration was used?

10. What should we change next and re-benchmark?
"""
)
