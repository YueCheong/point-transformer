import json

preserve_freq = 4

ann_path = '/home/hhfan/code/pc/process/label/uid_path_label_dict.json'
selected_ann_path = f'/home/hhfan/code/pc/process/label/selected_{preserve_freq}_uid_path_label_dict.json'
selected_ann_idx_path = f'/home/hhfan/code/pc/process/label/selected_{preserve_freq}_uid_path_idx_dict.json'
label2idx_map_path = f'/home/hhfan/code/pc/process/label/selected_{preserve_freq}_label2idx_map.json'
with open(ann_path, 'r') as f:
    ann = json.load(f)

# count label frequency
label_count = {}
for uid in ann:
    label = ann[uid]['label']
    if label in label_count:
        label_count[label] += 1
    else:
        label_count[label] = 1

# print the occurance frequence of label
# python count_label.py >> count_label.log
'''
sorted_labels = sorted(label_count.items(), key=lambda x: x[1], reverse=True)
for label, count in sorted_labels:
    print(f'label: {label}, occurs: {count} times')
print(f'there are {len(ann)} objects and {len(label_count)} kinds of label in total')
'''

# select label frequency > 4
'''
selected_ann = {uid: info for uid, info in ann.items() if label_count[info['label']] > 4}
print(f'ann: {len(ann)}\nselected_ann: {len(selected_ann)}')
with open(selected_ann_path, 'w') as file:
    json.dump(selected_ann, file)    
'''

# replace the label with idx
selected_labels = [label for label, count in label_count.items() if count > 4]
selected_labels.sort(key=lambda x: label_count[x], reverse=True)
label_to_id = {label: idx for idx, label in enumerate(selected_labels)}

selected_ann = {uid: {'path': info['path'], 'label': label_to_id[info['label']]} for uid, info in ann.items() if info['label'] in selected_labels}



with open(selected_ann_idx_path, 'w') as file:
    json.dump(selected_ann, file)

with open(label2idx_map_path, 'w') as file:
    json.dump(label_to_id, file)

    
    