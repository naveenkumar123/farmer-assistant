import math
import matplotlib.pyplot as plt
import numpy as np
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer

# Optional: ANSI color codes for pretty printing in terminal
COLOR_MAP = {"green": "\033[92m", "orange": "\033[93m", "red": "\033[91m"}
RESET = "\033[0m"
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


class Tester:

    def __init__(self, predictor, data, title=None, size=1000):
        self.predictor = predictor
        self.data = data
        self.title = title or predictor.__name__.replace("_", " ").title()
        self.size = min(size, len(data))
        self.guesses = []
        self.truths = []
        self.errors = []
        self.sles = []
        self.colors = []

    def cosine_similarity(self, a, b):
      return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def how_similar(self, text1, text2):
      vector1, vector2 = model.encode([text1, text2])
      similarity = self.cosine_similarity(vector1, vector2)
      # print(f"Similarity between {text1} and {text2} is {similarity*100:.1f}%")
      return similarity

    def color_for(self, similarity):
        """Color code based on the numeric error value."""
        if similarity > 0.6:
            return "green"
        elif similarity > 0.3:
            return "orange"
        else:
            return "red"

    def run_datapoint(self, i):
        datapoint = self.data[i]
        guess = str(self.predictor(datapoint["query"]))
        truth = str(datapoint["answer"])

        if(i < 500):
          print(f"guess : ${guess}. ")
          print(f"truth : ${truth}. ")

        # Compute similarity between guess and truth (both strings)
        # similarity = SequenceMatcher(None, guess, truth).ratio()
        similarity = self.how_similar(guess, truth)

        # Convert similarity to a distance error (0 = perfect, 1 = completely different)
        error = 1 - similarity

        # Avoid log(0) by adding a tiny epsilon
        epsilon = 1e-10
        log_error = math.log(error + 1 + epsilon)  # 0 for perfect match

        # Squared log error (SLE)
        sle = log_error ** 2

        # Pick color based on error (or sle)
        color = self.color_for(similarity)

        # For text comparisons, we can store numeric placeholders in truths/guesses for plotting
        # but since they are text, we can just store indexes
        self.guesses.append(i)
        self.truths.append(i)
        self.errors.append(error)
        self.sles.append(sle)
        self.colors.append(color)

        # Build display title (truncate query text)
        title = datapoint["query"].split("\n\n")[0][:40] + "..."

        print(
            f"{COLOR_MAP[color]}{i+1}:"
            f" Error={error:.4f} | log_error={log_error:.4f} | SLE={sle:.4f} |"
            f" Similarity={similarity:.4f} | Item: {title}{RESET}"
        )

    def chart(self, title):
        """Simple visual of error distribution."""
        plt.figure(figsize=(10, 5))
        plt.scatter(range(self.size), self.errors, c=self.colors, s=10)
        plt.xlabel('Datapoint Index')
        plt.ylabel('Error (1 - Similarity)')
        plt.title(title)
        plt.show()

    def report(self):
        average_error = sum(self.errors) / self.size
        rmsle = math.sqrt(sum(self.sles) / self.size)
        hits = sum(1 for c in self.colors if c == "green")
        hit_rate = hits / self.size * 100
        title = f"{self.title} | Avg Error={average_error:.3f} | RMSLE={rmsle:.3f} | Green={hit_rate:.1f}%"
        self.chart(title)

    def run(self):
        for i in range(self.size):
            self.run_datapoint(i)
        self.report()

    @classmethod
    def test(cls, function, data):
        cls(function, data).run()