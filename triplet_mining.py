import numpy as np
import scipy
import torch
import torch.nn.functional as F


def compute_distance(a, pn, method = "cosine"):

  # assert a.shape == pn.shape

  if method == "cosine":
    return 1 - F.cosine_similarity(a.unsqueeze(0), pn.unsqueeze(1), dim = 2)    #Invert values so larger value means farther
  elif method == "euclidean":
    # return torch.cdist(x, y, compute_mode="donot_use_mm_for_euclid_dist")
    return torch.norm(a.unsqueeze(0).repeat((len(pn), 1)) - pn, p=2, dim = 1)
  else:
    raise Exception("Invalid distance/similarity method.")

#per_entity -> False: per instance (ie. positives of an anchor is from all entities the anchor token is part of)
def extract_candidates(instance, window_size:int = 3, pad_positive = True, pad_negative = True, per_entity = False):

  assert "ner" in instance
  assert "sentence" in instance
  if pad_negative:
    assert pad_positive #If pad negative, pad positive should be enabled too (for indexing)

  sent = np.array(instance["sentence"])
  entSet = instance["ner"]

  positives = []
  negatives = []

  if len(entSet) == 0:
    return positives, negatives

  for i, ent in enumerate(entSet):

    ent_pos = {}
    ent_neg = {}

    # Get entity type
    entType = ent["type"]

    #Get all anchor indexes and mark entities on the span matrix (True if part of the entity)
    anchorIds = ent["index"]
    entity_mask = np.zeros(len(sent)).astype(bool)
    entity_mask[anchorIds] = True

    #For each anchor, extract positive and negative candidates
    for aId in anchorIds:

      # Create a window mask to only consider tokens near the anchor
      if window_size != 0:
        window_mask = np.zeros(len(sent)).astype(bool)
        window_mask[aId + 1: aId + 1 + window_size] = True #Mask after
        window_mask[max(0, aId - window_size): aId] = True #Mask before
      else: #No windows; set aId to False to exclude
        window_mask = np.ones(len(sent)).astype(bool)
        window_mask[aId] = False


      if pad_positive:
        window_mask[0:2] = False  # Don't include pads

      pos_mask = np.all((entity_mask, window_mask), axis = 0)
      neg_mask = np.all((~entity_mask, window_mask), axis = 0)

      ent_pos[aId] = np.argwhere(pos_mask).flatten()
      ent_neg[aId] = np.argwhere(neg_mask).flatten()

      #If anchor has no positive/negative canditates and pad positive/negative is enabled, add pad positive/negative as a candidate
      if pad_positive and len(ent_pos[aId]) == 0:
        ent_pos[aId] = np.array([0])  #TODO: Remove hardcoding
      if pad_negative and len(ent_neg[aId]) == 0:  #Only pad negative if pad positive is enabled
        ent_neg[aId] = np.array([1])

    positives.append(ent_pos)
    negatives.append(ent_neg)

  #Combine to one dictionary where anchor keys are unique
  if not per_entity:
    combined_pos = {}
    for pos in positives:
      for k, v in pos.items():
        if k not in combined_pos:
          combined_pos[k] = set()
        combined_pos[k] = combined_pos[k] | set(v)

    combined_neg = {}
    for neg in negatives:
      for k, v in neg.items():
        if k not in combined_neg:
          combined_neg[k] = set()
        combined_neg[k] = combined_neg[k] | set(v)

    assert combined_pos.keys() == combined_neg.keys()
    #Remove positives from list of negatives
    for k in combined_pos.keys():
      combined_neg[k] = combined_neg[k] - combined_pos[k]

    #If combined values is empty, set to pad
    combined_pos = {k: np.array(list(v)) if len(v) != 0 else np.array([0]) for k, v in combined_pos.items()}
    combined_neg = {k: np.array(list(v)) if len(v) != 0 else np.array([1]) for k, v in combined_neg.items()}

    return [combined_pos], [combined_neg]

  return positives, negatives


#Match schemes
# -> all - produces all combinations of triplets for each anchor
#Per instance
def generate_offline_triplets(instance, window_size:int = 3, mining_scheme = "all", max_per_anchor = 0, per_entity = False):

  if len(instance["ner"]) == 0:
    return [], {}, {} #Return empty list

  #Extract candidates
  instance_positives, instance_negatives = extract_candidates(instance, window_size, per_entity = per_entity)
  assert len(instance_positives) == len(instance_negatives)


  triplets = []
  # All permutation (multiple triplets per anchor)
  if mining_scheme == "all":
    for positives, negatives in zip(instance_positives, instance_negatives):
      anchors = positives.keys()
      for a in anchors:
        candidates = np.array([(a, p, n) for p in positives[a] for n in negatives[a]])

        #If max per anchor is set, randomly choose triplets when more than max
        if max_per_anchor == 0 or len(candidates) <= max_per_anchor: #Use all
          triplets.extend(candidates)
        else:
          random_idx = np.random.choice(range(len(candidates)), size = max_per_anchor, replace = False)
          triplets.extend(candidates[random_idx])
          assert len(random_idx) == max_per_anchor

  # Batch hard (position) - hardest positive and hardest negative (one triplet per anchor)
  elif mining_scheme == "position_hardest":
    for positives, negatives in zip(instance_positives, instance_negatives):
      anchors = positives.keys()
      for a in anchors:
        pos_distances = abs(positives[a] - a)
        neg_distances = abs(negatives[a] - a)

        p = positives[a][np.argsort(pos_distances)[-1]] #Farthest positive
        n = negatives[a][np.argsort(neg_distances)[0]]  # Closest negative

        triplets.append((a, p, n))

  # Batch hard neg (position) - hardest negative per anchor-positive pair (#positives triplet per anchor)
  elif mining_scheme == "position_hardneg":
    for positives, negatives in zip(instance_positives, instance_negatives):
      anchors = positives.keys()
      for a in anchors:
        neg_distances = abs(negatives[a] - a)
        n = negatives[a][np.argsort(neg_distances)[0]] #Closest negative

        for p in positives[a]:
          triplets.append((a, p, n))

  #TODO: Confirm if need to implement
  # Semi hard (position) - closest negative to anchor but farther than positive
  elif mining_scheme == "position_semi":
    raise Exception("Not yet implemented")

  # Online triplet mining; return none and compute during training
  elif mining_scheme in ["distance_hardest", "distance_hardneg"]:
    raise Exception("Wrong procedure called")

  else:
    raise Exception("Unknown match scheme")

  return np.array(triplets), instance_positives, instance_negatives

def generate_online_triplets(batch_encoded, batch_positives, batch_negatives, mining_scheme = "distance_hardest", dist_metric = "cosine"):

  encoded_triplets = []
  for instance_encoded, instance_positives, instance_negatives in zip(batch_encoded, batch_positives, batch_negatives):

    for positives, negatives in zip(instance_positives, instance_negatives):  #instance positives/negatives contain a list of dictionaries -> one item per entity
      assert positives.keys() == negatives.keys()

      anchors = positives.keys()

      #Compute distance of each token to each other
      distance_matrix = compute_distance(instance_encoded, instance_encoded, method = dist_metric)

      for a in anchors:
        p_candidates = positives[a]
        n_candidates = negatives[a]

        n_idx = torch.argmin(distance_matrix[a, n_candidates])  # Hardest negative index

        # Hardest
        if mining_scheme == "distance_hardest":
          p_idx = torch.argmax(distance_matrix[a, p_candidates])

          encoded_triplets.append(instance_encoded[[a, p_candidates[p_idx], n_candidates[n_idx]]])

        # Hard Negative
        elif mining_scheme == "distance_hardneg":
          for p in p_candidates:
            encoded_triplets.append(instance_encoded[[a, p, n_candidates[n_idx]]])

        # Centroid
        elif mining_scheme in ["centroid"]:
          p_centroid = torch.mean(instance_encoded[p_candidates], dim = 0)
          n_centroid = torch.mean(instance_encoded[n_candidates], dim = 0)
          encoded_triplets.append(torch.stack((instance_encoded[a], p_centroid, n_centroid)))

        else:
          raise Exception("Invalid online mining scheme.")

  return encoded_triplets

def extract_grid_candidates(instance, window_size, per_entity = False, unique_grid_pairs = True):
  assert "ner" in instance
  assert "sentence" in instance

  sent_len = len(instance["sentence"])

  # Extract entities from instance
  entities = [entity["index"] for entity in instance["ner"]]
  if len(entities) == 0:
    return [], []
    # raise Exception("REVERT TO ORIGINAL PLS")

  # Extract positive token pairs
  ap_pool = []
  entity_mask = np.zeros((len(entities), sent_len))  # entity tokens per entity
  for i, entity in enumerate(entities):
    entity_mask[i, entity] = 1

    if len(entity) == 1:
      #   continue              #Cannot form a token-pair -> skipped
      pairs = [(0, entity[0])]  # Force pair with [POS]
    else:
      pairs = [(x, y) for x in entity for y in entity if
               x < y]  # Consider only upper triangle (ie (token1, token2) == (token2, token1))

    assert len(pairs) > 0
    ap_pool.extend(pairs)

  # Set array and sort
  ap_pool = np.array(list(set(ap_pool)))
  ap_pool = ap_pool[np.lexsort((ap_pool[:, 1], ap_pool[:, 0]))]

  # Create masking grid for each token pair
  ap_grid = np.ones((len(ap_pool), len(ap_pool))).astype(bool)
  an_grid = np.zeros((len(ap_pool), len(ap_pool))).astype(bool)  # Priority negatives (token-pairs from other entities)

  # IF PER ENTITY -> mask token pairs that don't exist in the same entity
  if per_entity:
    for i, a in enumerate(ap_pool):
      for j, p in enumerate(ap_pool):
        if np.all(a == p):
          continue

        if ~np.any(np.all(entity_mask[np.all(entity_mask[:, a], axis=1)][:, p], axis=1)):
          ap_grid[i, j] = 0
          an_grid[i, j] = 1

  if window_size > 0:
    # Compute distance between each token pair
    ap_dist = np.rint(scipy.spatial.distance.cdist(ap_pool, ap_pool))
    ap_grid = (ap_grid & (ap_dist <= window_size))
    an_grid = (an_grid & (ap_dist <= window_size))

  # Mask diagonals and bottom triangle
  if unique_grid_pairs:
    for i in range(len(ap_grid)):
      ap_grid[i:, i] = 0  # Mask diagonals and bottom triangle
      an_grid[i:, i] = 0
  else:
    for i in range(len(ap_grid)):
      ap_grid[i, i] = 0  # Mask diagonals only
      an_grid[i, i] = 0

  ap_pairs = ap_pool[np.argwhere(ap_grid)]
  an_pairs = ap_pool[np.argwhere(an_grid)]

  # Force to [POS],[POS] or [NEG],[NEG] if not paired
  missing = ap_pool[np.sum(ap_grid, axis=1) == 0]
  temp = [(a, (0, 0)) for a in missing if a not in ap_pairs[:, 1]]
  if len(temp) > 0:
    ap_pairs = np.append(ap_pairs, temp, axis=0)

  # If no negative tokens from other entities; use non-entity tokens
  n_grid = []
  # if len(an_pairs) == 0:
  # Extract negative token pairs (disregard first 2 tokens [POS] & [NEG])
  n_pool = [(x, y) for x in range(2, sent_len) for y in range(2, sent_len) if x < y]
  n_pool = list(set(n_pool) - set(map(tuple, ap_pool)))

  if len(n_pool) > 0:
    n_pool = np.array(n_pool)
    n_pool = n_pool[np.lexsort((n_pool[:, 1], n_pool[:, 0]))]
    n_grid = np.ones((len(ap_pool), len(n_pool))).astype(bool)

    # Mask outside windows
    if window_size > 0:
      n_dist = np.rint(scipy.spatial.distance.cdist(ap_pool, n_pool))
      n_grid = (n_grid & (n_dist <= window_size))

    n_mask = np.argwhere(n_grid)
    an_pairs = np.append(an_pairs, np.stack((ap_pool[n_mask[:, 0]], n_pool[n_mask[:, 1]]), axis=1), axis = 0)

  # Check if anchor has negatives
  for a in ap_pool:
    if np.sum(np.all(an_pairs[:, 0] == a, axis=1)) == 0:
      an_pairs = np.append(an_pairs, [[a, (1, 1)]], axis=0)

  return ap_pairs, an_pairs

def generate_grid_triplets(instance, window_size, max_negatives = 1, per_entity = False, unique_grid_pairs = True, mining_scheme = ""):
  assert "ner" in instance
  assert "sentence" in instance
  assert max_negatives >= 0

  ap_pairs, an_pairs = extract_grid_candidates(instance, window_size, per_entity, unique_grid_pairs)
  if len(ap_pairs) == 0:
    return [], [], []
  ap_pool = np.unique(an_pairs[:, 0], axis=0)

  an_dict = {str(a): [] for a in ap_pool}
  for a, n in an_pairs:
    an_dict[str(a)].append(n)

  ap_dict = {str(a): [] for a in ap_pool}
  for a, p in ap_pairs:
    ap_dict[str(a)].append(p)

  triplets = []
  if mining_scheme == "grid":
    # GENERATE TRIPLETS
    for a, p in ap_pairs:
      n_filtered = np.array(an_dict[str(a)])      #TODO: Transfer selection of negatives to extract_grid_candidates function
      assert len(n_filtered) != 0

      if max_negatives > 0:
        # randomly use a max set of negatives
        n_filtered = n_filtered[np.random.choice(range(len(n_filtered)), min(max_negatives, len(n_filtered)))]
        an_dict[str(a)] = n_filtered

      for n in n_filtered:
        triplets.append((a, p, n))

  return triplets, ap_dict, an_dict


def generate_grid_online_triplets(grid_encoded, batch_positives, batch_negatives, grid_scheme = "grid_centroid", dist_metric = "cosine", margin = None):
  assert grid_encoded.dim() == 4
  assert len(batch_positives) == len(batch_negatives)

  encoded_triplets = []
  for instance_encoded, instance_positives, instance_negatives in zip(grid_encoded, batch_positives, batch_negatives):
    if len(instance_positives) == 0:
      continue #No triplets

    anchors_str = instance_positives.keys()
    anchors = [x[1:-1].split(" ") for x in anchors_str]
    anchors = [[y for y in x if y != ""] for x in anchors]
    anchors = [[int(x), int(y)] for x, y in anchors]

    if grid_scheme == "grid_centroid":
      for a, a_str in zip(anchors, anchors_str):
        anchor_positives = np.array(instance_positives[a_str])
        anchor_negatives = np.array(instance_negatives[a_str])

        if len(anchor_positives) == 0 or len(anchor_negatives) == 0:
          continue

        anchor_encoded = instance_encoded[a[0], a[1]]
        centroid_positives = torch.mean(instance_encoded[anchor_positives[:, 0], anchor_positives[:, 1]], dim = 0)
        centroid_negatives = torch.mean(instance_encoded[anchor_negatives[:, 0], anchor_negatives[:, 1]], dim = 0)

        encoded_triplets.append(torch.stack((anchor_encoded, centroid_positives, centroid_negatives)))
    elif grid_scheme == "grid_negcentroid":
      for a, a_str in zip(anchors, anchors_str):
        anchor_positives = np.array(instance_positives[a_str])
        anchor_negatives = np.array(instance_negatives[a_str])

        if len(anchor_positives) == 0 or len(anchor_negatives) == 0:
          continue

        anchor_encoded = instance_encoded[a[0], a[1]]
        positives = instance_encoded[anchor_positives[:, 0], anchor_positives[:, 1]]
        centroid_negatives = torch.mean(instance_encoded[anchor_negatives[:, 0], anchor_negatives[:, 1]], dim=0)

        repeat_num = len(anchor_positives)
        anchor_encoded = anchor_encoded.unsqueeze(0).repeat(repeat_num, 1)
        centroid_negatives = centroid_negatives.unsqueeze(0).repeat(repeat_num, 1)

        encoded_triplets.extend(torch.stack((anchor_encoded, positives, centroid_negatives), dim =1))

    elif grid_scheme == "grid_negcentroid2":
      for a, a_str in zip(anchors, anchors_str):
        anchor_positives = np.array(instance_positives[a_str])
        anchor_negatives = np.array(instance_negatives[a_str])

        if len(anchor_positives) == 0 or len(anchor_negatives) == 0:
          continue

        anchor_encoded = instance_encoded[a[0], a[1]]
        positives = instance_encoded[anchor_positives[:, 0], anchor_positives[:, 1]]
        negatives = instance_encoded[anchor_negatives[:, 0], anchor_negatives[:, 1]]

        # Calculate distance for all positives and negatives
        p_dist = compute_distance(anchor_encoded, positives, method=dist_metric)
        n_dist = compute_distance(anchor_encoded, negatives, method=dist_metric).flatten()

        for p_enc, ap_dist in zip(positives, p_dist):
          # Get hard and semi-hard negatives (closer than positive+ margin)
          valid_idx = torch.argwhere((n_dist < ap_dist + margin)).flatten()

          if len(valid_idx) == 0:  # No hard and semi hard negatives
            continue
          else:
            centroid_negative = torch.mean(negatives[valid_idx], dim = 0)

          encoded_triplets.append(torch.stack((anchor_encoded, p_enc, centroid_negative)))

    elif grid_scheme == "grid_hardneg":
      for a, a_str in zip(anchors, anchors_str):
        anchor_positives = np.array(instance_positives[a_str])
        anchor_negatives = np.array(instance_negatives[a_str])

        if len(anchor_positives) == 0 or len(anchor_negatives) == 0:
          continue

        anchor_encoded = instance_encoded[a[0], a[1]]
        positives = instance_encoded[anchor_positives[:, 0], anchor_positives[:, 1]]
        negatives = instance_encoded[anchor_negatives[:, 0], anchor_negatives[:, 1]]

        #Find nearest negative
        n_dist = compute_distance(anchor_encoded, negatives, method = dist_metric)
        n_idx = torch.argmin(n_dist)

        repeat_num = len(anchor_positives)
        anchor_encoded = anchor_encoded.unsqueeze(0).repeat(repeat_num, 1)
        hard_negative = negatives[n_idx].unsqueeze(0).repeat(repeat_num, 1)

        encoded_triplets.extend(torch.stack((anchor_encoded, positives, hard_negative), dim=1))

    elif grid_scheme == "grid_semineg":
      assert margin is not None

      for a, a_str in zip(anchors, anchors_str):
        anchor_positives = np.array(instance_positives[a_str])
        anchor_negatives = np.array(instance_negatives[a_str])

        if len(anchor_positives) == 0 or len(anchor_negatives) == 0:
          continue

        anchor_encoded = instance_encoded[a[0], a[1]]
        positives = instance_encoded[anchor_positives[:, 0], anchor_positives[:, 1]]
        negatives = instance_encoded[anchor_negatives[:, 0], anchor_negatives[:, 1]]

        #Calculate distance for all positives and negatives
        p_dist = compute_distance(anchor_encoded, positives, method = dist_metric)
        n_dist = compute_distance(anchor_encoded, negatives, method = dist_metric).flatten()

        for p_enc, ap_dist in zip(positives, p_dist):
          #Get negatives that are farther than the positive
          semi_idx = torch.argwhere((n_dist > ap_dist) & (n_dist < ap_dist + margin)).flatten()

          if len(semi_idx) == 0: #No semi hard negatives within the margin
            continue
          else:
            #Find nearest (filtered) negative
            n_idx = semi_idx[torch.argmin(n_dist[semi_idx])]

          if torch.clamp_min(margin + ap_dist - n_dist[n_idx], 0) == 0:
            raise Exception("Invalid semi hard negatives")

          encoded_triplets.append(torch.stack((anchor_encoded, p_enc, negatives[n_idx])))

  return encoded_triplets


