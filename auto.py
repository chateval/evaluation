import numpy as np

def distinct_1(lines):
  words = ' '.join(lines).split(' ')
  num_distinct_words = len(set(words))
  return float(num_distinct_words) / len(words)

def distinct_2(lines):
  all_bigrams = []
  num_words = 0

  for line in lines:
    line_list = line.split(' ')
    num_words += len(line_list)
    bigrams = zip(line_list, line_list[1:])
    all_bigrams.extend(list(bigrams))

  return len(set(all_bigrams)) / float(num_words)

def avg_len(lines):
  return(len([w for s in lines for w in s.strip().split()])/len(lines))

def bleu(target_lines, gt_lines):
  print(gt_lines)
  print(len(gt_lines))
  from nltk.translate.bleu_score import sentence_bleu
  avg_bleu = 0
  num_refs = len(gt_lines)

  for i in range(len(target_lines)):
    ref = []
    for r in range(num_refs):
      ref.append(gt_lines[r][i].lower().split())
    hyp = target_lines[i].lower().split()
    bleu = sentence_bleu(ref, hyp, weights = (0.5, 0.5))
    avg_bleu += bleu
  avg_bleu = avg_bleu / len(target_lines)

  return(avg_bleu)

def greedy_match(r1, r2, w2v):
  res1 = greedy_score(r1, r2, w2v)
  res2 = greedy_score(r2, r1, w2v)
  res_sum = (res1 + res2)/2.0

  return np.mean(res_sum), 1.96*np.std(res_sum)/float(len(res_sum)), np.std(res_sum)

def greedy_score(r1, r2, w2v):
  dim = int(w2v.dim())
  scores = []

  for i in range(len(r1)):
    tokens1 = r1[i].strip().split(" ")
    tokens2 = r2[i].strip().split(" ")
    X= np.zeros((dim,))
    y_count = 0
    x_count = 0
    o = 0.0
    Y = np.zeros((dim,1))

    for tok in tokens2:
      if tok in w2v:
        Y = np.hstack((Y,(w2v[tok].reshape((dim,1)))))
        y_count += 1

    for tok in tokens1:
      if tok in w2v:
        tmp  = w2v[tok].reshape((1,dim)).dot(Y)
        o += np.max(tmp)
        x_count += 1

    if x_count < 1 or y_count < 1:
      scores.append(0)
      continue

    o /= float(x_count)
    scores.append(o)

  return np.asarray(scores)


def extrema_score(r1, r2, w2v):
  scores = []

  for i in range(len(r1)):
    tokens1 = r1[i].strip().split(" ")
    tokens2 = r2[i].strip().split(" ")
    X= []
    for tok in tokens1:
      if tok in w2v:
        X.append(w2v[tok])
    Y = []
    for tok in tokens2:
      if tok in w2v:
        Y.append(w2v[tok])

    if np.linalg.norm(X) < 0.00000000001:
      continue

    if np.linalg.norm(Y) < 0.00000000001:
      scores.append(0)
      continue

    xmax = np.max(X, 0)  # get positive max
    xmin = np.min(X,0)  # get abs of min
    xtrema = []
    for i in range(len(xmax)):
      if np.abs(xmin[i]) > xmax[i]:
        xtrema.append(xmin[i])
      else:
        xtrema.append(xmax[i])
    X = np.array(xtrema)   # get extrema

    ymax = np.max(Y, 0)
    ymin = np.min(Y,0)
    ytrema = []
    for i in range(len(ymax)):
      if np.abs(ymin[i]) > ymax[i]:
        ytrema.append(ymin[i])
      else:
        ytrema.append(ymax[i])
    Y = np.array(ytrema)

    o = np.dot(X, Y.T)/np.linalg.norm(X)/np.linalg.norm(Y)

    scores.append(o)

  scores = np.asarray(scores)
  return np.mean(scores), 1.96*np.std(scores)/float(len(scores)), np.std(scores)


def average_embedding_score(r1, r2, w2v):
  dim = w2v.dim()

  scores = []

  for i in range(len(r1)):
    tokens1 = r1[i].strip().split(" ")
    tokens2 = r2[i].strip().split(" ")
    X= np.zeros((dim,))
    for tok in tokens1:
      if tok in w2v:
        X+=w2v[tok]
    Y = np.zeros((dim,))
    for tok in tokens2:
      if tok in w2v:
        Y += w2v[tok]

    # if none of the words in ground truth have embeddings, skip
    if np.linalg.norm(X) < 0.00000000001:
      continue

    # if none of the words have embeddings in response, count result as zero
    if np.linalg.norm(Y) < 0.00000000001:
      scores.append(0)
      continue

    X = np.array(X)/np.linalg.norm(X)
    Y = np.array(Y)/np.linalg.norm(Y)
    o = np.dot(X, Y.T)/np.linalg.norm(X)/np.linalg.norm(Y)

    scores.append(o)

  scores = np.asarray(scores)
  return np.mean(scores), 1.96*np.std(scores)/float(len(scores)), np.std(scores)