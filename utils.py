import logging
import pickle
import time
from collections import defaultdict, deque


def get_logger(config):
    # pathname = "./log/{}_{}.txt".format(dataset, time.strftime("%m-%d_%H-%M-%S"))
    if config.session_name == "":
        pathname = "./log/%s_%s.txt" % (config.dataset, time.strftime("%Y-%m-%d_%H-%M-%S"))
    else:
        pathname = "./log/%s.txt" % config.session_name
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(pathname)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def save_file(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_file(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def convert_index_to_text(index, type):
    text = "-".join([str(i) for i in index])
    text = text + "-#-{}".format(type)
    return text


def convert_text_to_index(text):
    index, type = text.split("-#-")
    index = [int(x) for x in index.split("-")]
    return index, int(type)


def decode(outputs, entities, length):
    class Node:
        def __init__(self):
            self.THW = []                # [(tail, type)]
            self.NNW = defaultdict(set)   # {(head,tail): {next_index}}

    ent_r, ent_p, ent_c = 0, 0, 0
    decode_entities = []
    q = deque()
    for instance, ent_set, l in zip(outputs, entities, length):
        predicts = []
        nodes = [Node() for _ in range(l)]
        for cur in reversed(range(l)):
            heads = []
            for pre in range(cur+1):
                # THW
                if instance[cur, pre] > 1: 
                    nodes[pre].THW.append((cur, instance[cur, pre]))
                    heads.append(pre)
                # NNW
                if pre < cur and instance[pre, cur] == 1:
                    # cur node
                    for head in heads:
                        nodes[pre].NNW[(head,cur)].add(cur)
                    # post nodes
                    for head,tail in nodes[cur].NNW.keys():
                        if tail >= cur and head <= pre:
                            nodes[pre].NNW[(head,tail)].add(cur)
            # entity
            for tail,type_id in nodes[cur].THW:
                if cur == tail:
                    predicts.append(([cur], type_id))
                    continue
                q.clear()
                q.append([cur])
                while len(q) > 0:
                    chains = q.pop()
                    for idx in nodes[chains[-1]].NNW[(cur,tail)]:
                        if idx == tail:
                            predicts.append((chains + [idx], type_id))
                        else:
                            q.append(chains + [idx])
        
        predicts = set([convert_index_to_text(x[0], x[1]) for x in predicts])
        decode_entities.append([convert_text_to_index(x) for x in predicts])
        ent_r += len(ent_set)
        ent_p += len(predicts)
        ent_c += len(predicts.intersection(ent_set))
    return ent_c, ent_p, ent_r, decode_entities


def cal_f1(c, p, r):
    if r == 0 or p == 0:
        return 0, 0, 0

    r = c / r if r else 0
    p = c / p if p else 0

    if r and p:
        return 2 * p * r / (p + r), p, r
    return 0, p, r

#Added
def compute_metrics(gold, preds, title_text):
  ent_r, ent_p, ent_c = 0, 0, 0
  for inst_gold, inst_preds in zip(gold, preds):
    ent_r += len(inst_gold)
    ent_p += len(inst_preds)

    for x in inst_preds:
      if x in inst_gold:
        ent_c += 1

  try:
    r = ent_c / ent_r
    p = ent_c / ent_p
    f1 = (2 * p * r / (p + r))
  except ZeroDivisionError:
    r, p, f1 = 0, 0, 0

  # print(title_text)
  # print("Recall: %.4f" % r)
  # print("Precision: %.4f" % p)
  # print("F1: %.4f" % f1)

  return f1, p, r

def check_disc(data):
    is_disc = []
    for inst in data:
        inst_isDisc = []
        if len(inst) != 0:
            for ent in inst:
                if (len(ent) - 2) != (ent[-2] - ent[0]):  # Discontinuous: -1 -> entity type id; -2 -> end index
                    inst_isDisc.append(1)
                else:
                    inst_isDisc.append(0)

        is_disc.append(inst_isDisc)
    return is_disc

def extract_all_disc_metrics(gold, preds):
  assert len(gold) == len(preds)

  gold_isDisc = check_disc(gold)
  pred_isDisc = check_disc(preds)

  #OVERALL RESULTS
  all_f1, all_p, all_r = compute_metrics(gold, preds, "OVERALL")

  #PER SENTENCE
  sent_disc_gold = []
  sent_disc_preds = []
  sent_cont_gold = []
  sent_cont_preds = []
  for inst_isDisc, inst_gold, inst_pred in zip(gold_isDisc, gold, preds):
    if len(inst_gold) == 0:
      continue

    if sum(inst_isDisc) > 0:
      sent_disc_gold.append(inst_gold)
      sent_disc_preds.append(inst_pred)
    else:
      sent_cont_gold.append(inst_gold)
      sent_cont_preds.append(inst_pred)

  sent_cont_f1, sent_cont_p, sent_cont_r = compute_metrics(sent_cont_gold, sent_cont_preds, "\nSENTENCES w/ NO DISCONTINUOUS ENTITIES")
  sent_disc_f1, sent_disc_p, sent_disc_r = compute_metrics(sent_disc_gold, sent_disc_preds, "\nSENTENCES w/ DISCONTINUOUS ENTITY/ENTITIES")


  #PER ENTITY
  ent_cont_gold = []
  ent_cont_preds = []
  ent_disc_gold = []
  ent_disc_preds = []
  for inst_gold_isDisc, inst_pred_isDisc, inst_gold, inst_pred in zip(gold_isDisc, pred_isDisc, gold, preds):

    inst_ent_cont_gold = []
    inst_ent_cont_preds = []
    inst_ent_disc_gold = []
    inst_ent_disc_preds = []

    # Separate continuous/discontinuous entities
    for ent_isDisc, ent_gold in zip(inst_gold_isDisc, inst_gold):
      if ent_isDisc == 1:
        inst_ent_disc_gold.append(ent_gold)
      else:
        inst_ent_cont_gold.append(ent_gold)

    for ent_isDisc, ent_pred in zip(inst_pred_isDisc, inst_pred):
      if ent_isDisc == 1:
        inst_ent_disc_preds.append(ent_pred)
      else:
        inst_ent_cont_preds.append(ent_pred)

    ent_cont_gold.append(inst_ent_cont_gold)
    ent_cont_preds.append(inst_ent_cont_preds)
    ent_disc_gold.append(inst_ent_disc_gold)
    ent_disc_preds.append(inst_ent_disc_preds)

  ent_cont_f1, ent_cont_p, ent_cont_r = compute_metrics(ent_cont_gold, ent_cont_preds, "\nCONTINUOUS ENTITIES ONLY")
  ent_disc_f1, ent_disc_p, ent_disc_r = compute_metrics(ent_disc_gold, ent_disc_preds, "\nDISCONTINUOUS ENTITIES ONLY")

  return {"all": {"f1": all_f1, "p": all_p, "r": all_r},
          "sent_cont": {"f1": sent_cont_f1, "p": sent_cont_p, "r": sent_cont_r},
          "sent_disc": {"f1": sent_disc_f1, "p": sent_disc_p, "r": sent_disc_r},
          "ent_cont": {"f1": ent_cont_f1, "p": ent_cont_p, "r": ent_cont_r},
          "ent_disc": {"f1": ent_disc_f1, "p": ent_disc_p, "r": ent_disc_r}}
