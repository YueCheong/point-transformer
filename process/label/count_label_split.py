import os
import json

preserve_freq = 4
root = '/home/hhfan/code/point-transformer/process/label/jsons'
label2idx_map_path = os.path.join(root, f'selected_{preserve_freq}_label2index_map.json')

# store whole split name
split_name_list = []
for split_id in range(10):
    split_name = f'ann_000-{split_id:03}.json'
    split_name_list.append(split_name)

# collect whole ann
total_ann = {}
for split_name in split_name_list:
    ann_path = os.path.join(root, split_name)
    with open(ann_path, 'r') as f:
        ann = json.load(f)
    total_ann.update(ann)
    
# count label frequency in whole ann
label_count = {}
for uid in total_ann:
    label = total_ann[uid]['label']
    if label in label_count:
        label_count[label] += 1
    else:
        label_count[label] = 1

# replace the label with idx
selected_labels = [label for label, count in label_count.items() if count > preserve_freq]
selected_labels.sort(key=lambda x: label_count[x], reverse=True)
label_to_id = {label: idx for idx, label in enumerate(selected_labels)}

with open(label2idx_map_path, 'w') as file:
    json.dump(label_to_id, file)

for split_name in split_name_list:
    ann_path = os.path.join(root, split_name)
    with open(ann_path, 'r') as f:
        ann = json.load(f)
    selected_ann_idx = {uid: {'path': info['path'], 'label': label_to_id[info['label']]} for uid, info in ann.items() if info['label'] in selected_labels}
    selected_ann_idx_path = os.path.join(root, f'selected_{preserve_freq}_idx_' + split_name)
    with open(selected_ann_idx_path, 'w') as file:
        json.dump(selected_ann_idx, file)

