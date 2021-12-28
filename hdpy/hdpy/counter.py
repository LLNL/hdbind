import re
import os


num_words = 0
num_chars = 0
directory = "/storage/cuda-hd-opt/datasets/LANG/training_texts"

print("TRAINING")
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        with open(directory + "/" + filename, "r") as file:
            for line in file:
                clean = re.sub(r"[^\w\s]", "", line)
                words = clean.split()

                num_words += len(words)
                for w in words:
                    num_chars += len(w)
            print(filename + " " + str(num_chars))
    else:
        continue

# avg_length = num_chars / num_words
# print("Average word length: {}".format(avg))
