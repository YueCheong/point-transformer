import objaverse
from collections import Counter
from skimage import io
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import os
import time
import csv

BASE_PATH = os.path.join(os.path.expanduser("~"), "alisa/code/objaverse")
FIG_PATH = os.path.join("/data_zju/alisa/objaverse/figs/") # test_figs
TXT_PATH = os.path.join("/data_zju/alisa/objaverse/unaccess_img.txt")
LABEL_PATH = os.path.join("/data_zju/alisa/objaverse/label.csv")


def count_one_tag(list, k):
    counter = Counter(list)
    res = counter.most_common(k)
    return res

def count_all_tag(list, k):
    flatten_list = [tag for sublist in list for tag in sublist]
    counter = Counter(flatten_list)
    res = counter.most_common(k)
    return res

def load_annotations():
    uids = objaverse.load_uids()
    # print("uids", uids)
    annotations = objaverse.load_annotations(uids[:798759]) # total 798759
    # print("annotations", annotations)
    tags = []
    categories = []
    descriptions = []
    img_paths = []
    labels = []
    
    for uid in annotations:
        tag = annotations[uid]['tags']
        category = annotations[uid]['categories']
        description = annotations[uid]['description'] 
        img_path = annotations[uid]['thumbnails']['images'][0]['url']       
        tags.append(tag)
        categories.append(category)
        descriptions.append(description)
        img_paths.append(img_path)

    # count every name in object tags
    for object in tags:
        label_list = []
        for names in object:
            label_list.append(names['name'])                  
        labels.append(label_list)
    # res = count_all_tag(labels, 100)
    # print("res", res)    
        
    return uids, annotations, img_paths, labels




def download_img(uids, img_paths):
    except_uids = []
    for index, img_path in enumerate(img_paths):
        print("uid {}, img_path {}".format(uids[index], img_path))
        # time.sleep(0.5)    
        try:
            img = io.imread(img_path) #(1080, 1920, 3)
            fig_name  = FIG_PATH + uids[index] + '.jpg'
            if not os.path.exists(fig_name):
                io.imsave(fig_name, img)
                print("img {} saved !".format(uids[index]))
            else:
                print("img has existed !")
        except:
            except_uids.append(uids[index])    
            print("img {} cannot access !".format(uids[index]))        
        continue    
    print("{} imgs cannot be download !".format(len(except_uids)))
    f = open(TXT_PATH, "w")
    for except_uid in except_uids:
        f.write(except_uid + '\n')
    f.close()



def clip_answer(model, inputs, text):
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)
    index = probs.argmax()
    answer = text[index]
    return answer

   

def construct_label_with_clip():
    
    uids = objaverse.load_uids()
    annotations = objaverse.load_annotations(uids[:798759]) # total 798759
   
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")  
    clip_labels = []
        
    for img_name in os.listdir(FIG_PATH):
        # images
        img = Image.open(FIG_PATH + img_name)
        # text
        uid = img_name[: -4]
        annotation = annotations[uid]
        tags = annotation['tags']
        tag_list = []
        for tag in tags: # one object has many tags
            tag_list.append(tag['name'])
        # CLIP predict label        
        if tag_list:
            inputs = processor(text=tag_list, images=img, return_tensors="pt", padding=True)
            answer = clip_answer(model, inputs, tag_list)
        else: # tags are None
            answer = ''
        print(answer)
        # dic = {uid:answer}
        clip_labels.append({uid:answer})
    print(clip_labels)
    
    with open(LABEL_PATH, "w") as csv_file:
        writer = csv.writer(csv_file)
        for object in clip_labels:
            print("row in clip_labels", object)
            for key, value in object.items():
                writer.writerow([key, value])
        
    

     
    
if __name__ == "__main__":
    # uids, annotations, img_paths, labels  = load_annotations()  
    construct_label_with_clip()

    
