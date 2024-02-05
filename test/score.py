import sys

gold_file = sys.argv[1]
pred_file = sys.argv[2]


# Load the gold standard
gold = {}
g_entity = {}
for line in open(gold_file):
    parts = line.split('\t')
    question_id = parts[0]
    if len(parts) > 1 and parts[1].startswith('R'):
        text = line.split('"')[1]
        gold[question_id+"R"] = text
    elif len(parts) > 1 and parts[1].startswith('A'):
        text = line.split('"')[1]
        gold[question_id+"A"] = text
    elif len(parts) > 1 and parts[1].startswith('C'):
        text = line.split('"')[1]
        gold[question_id+"C"] = text
    elif len(parts) > 1 and parts[1].startswith('E'):
        if gold.get(question_id + "E") is None:
            gold[question_id+"E"] = set()
        text = line.split('"')[1] + " " + line.split('"')[3]
        gold[question_id+"E"].add(text)
    else:
        continue


# Load the predictions
pred = {}
for line in open(pred_file):
    parts = line.split('\t')
    question_id = parts[0]
    if len(parts) > 1 and parts[1].startswith('R'):
        text = line.split('"')[1]
        pred[question_id+"R"] = text
    elif len(parts) > 1 and parts[1].startswith('A'):
        text = line.split('"')[1]
        pred[question_id+"A"] = text
    elif len(parts) > 1 and parts[1].startswith('C'):
        text = line.split('"')[1]
        pred[question_id+"C"] = text
    elif len(parts) > 1 and parts[1].startswith('E'):
        if pred.get(question_id + "E") is None:
            pred[question_id+"E"] = set()
        text = line.split('"')[1] + " " + line.split('"')[3]
        pred[question_id+"E"].add(text)
    else:
        continue


# Evaluate predictions

# Calculate scores

# How many A are correctly mathced 
filtered_gold_A = set(element for element in set(gold) if "A" in element)
filtered_pred_A = set(element for element in set(pred) if "A" in element)
n_correct_A = sum( int(pred[i]==gold[i]) for i in filtered_gold_A & filtered_pred_A )
# How many C are correctly mathced 
filtered_gold_C = set(element for element in set(gold) if "C" in element)
filtered_pred_C = set(element for element in set(pred) if "C" in element)
n_correct_C = sum( int(pred[i]==gold[i]) for i in filtered_gold_C & filtered_pred_C )
# How many E are correctly mathced 
filtered_gold_E = set(element for element in set(gold) if "E" in element)
n_gold_E = sum(len(gold[i]) for i in filtered_gold_E)
filtered_pred_E = set(element for element in set(pred) if "E" in element)
n_pred_E = sum(len(pred[i]) for i in filtered_pred_E)
n_correct_E = sum( len(pred[i] & gold[i]) for i in filtered_gold_E & filtered_pred_E )
# Total number of correct
n_gold = len(filtered_gold_A) + len(filtered_gold_C) + n_gold_E
n_predicted = len(filtered_pred_A) + len(filtered_pred_C) + n_pred_E
n_correct_all = n_correct_A + n_correct_C + n_correct_E

print("------------All----------------")
print('gold: %s' % n_gold)
print('predicted: %s' % n_predicted)
print('correct: %s' % n_correct_all)
precision = float(n_correct_all) / float(n_predicted)
print('precision: %s' % precision )
recall = float(n_correct_all) / float(n_gold)
print('recall: %s' % recall )
f1 = 2 * ( (precision * recall) / (precision + recall) )
print('f1: %s' % f1 )
print("------------A----------------")
print('gold: %s' % len(filtered_gold_A))
print('predicted: %s' % len(filtered_pred_A))
print('correct: %s' % n_correct_A)
precision_A = float(n_correct_A) / float(len(filtered_pred_A))
print('precision: %s' % precision_A )
recall_A = float(n_correct_A) / float(len(filtered_gold_A))
print('recall: %s' % recall_A )
f1_A = 2 * ( (precision_A * recall_A) / (precision_A + recall_A) )
print('f1: %s' % f1_A )
print("------------C----------------")
print('gold: %s' % len(filtered_gold_C))
print('predicted: %s' % len(filtered_pred_C))
print('correct: %s' % n_correct_C)
precision_C = float(n_correct_C) / float(len(filtered_pred_C))
print('precision: %s' % precision_C )
recall_C = float(n_correct_C) / float(len(filtered_gold_C))
print('recall: %s' % recall_C )
f1_C = 2 * ( (precision_C * recall_C) / (precision_C + recall_C) )
print('f1: %s' % f1_C )
print("------------E----------------")
print('gold: %s' % n_gold_E)
print('predicted: %s' % n_pred_E)
print('correct: %s' % n_correct_E)
precision_E = float(n_correct_E) / float(n_pred_E)
print('precision: %s' % precision_E )
recall_E = float(n_correct_E) / float(n_gold_E)
print('recall: %s' % recall_E )
f1_E = 2 * ( (precision_E * recall_E) / (precision_E + recall_E) )
print('f1: %s' % f1_E )
