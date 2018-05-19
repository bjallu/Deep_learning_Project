import numpy as np

number_of_experts = 36

council_resuts = []

for i in range(number_of_experts):
    expertName = str(i) + "ExpertResults.npy"
    expertResults = np.load(expertName)
    council_resuts += np.power(expertResults, 1)


base_predictions = np.load("BaseModelesults.npy")
true_Classes = np.load("correctClasses.npy")

base_predictions = np.argmax(base_predictions, axis=1)
expert_predictions = np.argmax(council_resuts, axis=1)

base_correct = 0
expert_correct = 0
base_final_correct = 0
total = 0

total_results = []


for i in range(base_predictions):

    base_p = base_predictions[i]
    expert_p = expert_predictions[i]
    true_p = true_Classes[i]

    if(true_p == base_p):
        base_correct += 1

    if (true_p == expert_p):
        expert_correct += 1

    total += 1

    results = [base_correct/total, expert_correct/total]
    print(results)
    total_results.append(results)


file = open('MichaelPhelps36.txt', 'w+')

for line in total_results:
    file.write(str(line[0]) + '\t' + str(line[1]) + '\n')

file.close()