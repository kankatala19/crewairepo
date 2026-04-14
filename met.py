import requests
import time
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt

API_KEY = "AIzaSyC3YNVcgOmFDakyAB6kfkWy8bdSY1HgH9g"

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"

headers = {
    "Content-Type": "application/json"
}

# Dataset for evaluation
questions = [
    "What is the capital of France?",
    "Who invented Python?",
]

references = [
    "Paris is the capital of France.",
    "Python was created by Guido van Rossum."
]

predictions = []
latencies = []

for q in questions:
    start = time.time()

    # Correct Gemini payload format
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": q}
                ]
            }
        ]
    }

    response = requests.post(url, json=payload, headers=headers)
    result = response.json()

    end = time.time()

    # Debug check
    if "candidates" not in result:
        print("API Error:", result)
        continue

    answer = result["candidates"][0]["content"]["parts"][0]["text"]

    predictions.append(answer)
    latencies.append(end - start)

# -------- Accuracy --------
accuracy = sum(
    [1 if r.lower().split()[0] in p.lower() else 0
     for r, p in zip(references, predictions)]
) / len(references)
# -------- BLEU --------

smooth = SmoothingFunction().method1

bleu_scores = []

for ref, pred in zip(references, predictions):
    bleu_scores.append(
        sentence_bleu([ref.split()], pred.split(), smoothing_function=smooth)
    )

avg_bleu = sum(bleu_scores) / len(bleu_scores)

avg_bleu = sum(bleu_scores) / len(bleu_scores)

# -------- ROUGE --------
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
rouge_scores = [scorer.score(r, p) for r, p in zip(references, predictions)]

# -------- Latency --------
avg_latency = sum(latencies) / len(latencies)

print("Accuracy:", accuracy)
print("Average BLEU:", avg_bleu)
print("Average Latency:", avg_latency)
print("ROUGE:", rouge_scores)


# --------- GRAPH 1 : BLEU Scores per Question ---------
plt.figure()
plt.plot(range(len(bleu_scores)), bleu_scores, marker='o')
plt.title("BLEU Score per Question")
plt.xlabel("Question Index")
plt.ylabel("BLEU Score")
plt.grid(True)
plt.show()

# --------- GRAPH 2 : Latency per Question ---------
plt.figure()
plt.plot(range(len(latencies)), latencies, marker='o')
plt.title("API Latency per Question")
plt.xlabel("Question Index")
plt.ylabel("Time (seconds)")
plt.grid(True)
plt.show()

# --------- GRAPH 3 : ROUGE Scores ---------
rouge1 = [score['rouge1'].fmeasure for score in rouge_scores]
rougeL = [score['rougeL'].fmeasure for score in rouge_scores]

plt.figure()
plt.plot(range(len(rouge1)), rouge1, marker='o', label='ROUGE-1')
plt.plot(range(len(rougeL)), rougeL, marker='o', label='ROUGE-L')
plt.title("ROUGE Scores")
plt.xlabel("Question Index")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.show()

# --------- GRAPH 4 : Overall Metrics ---------
metrics = ["Accuracy", "BLEU", "Latency"]
values = [accuracy, avg_bleu, avg_latency]

plt.figure()
plt.bar(metrics, values)
plt.title("Overall Model Performance")
plt.ylabel("Score")
plt.show()