from sentence_transformers import SentenceTransformer, util
from PIL import Image
import argparse
import sys
import numpy as np
import torch
import os
from tqdm import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', default="trial", type=str, help="text prompt")
    parser.add_argument('--clip', default="clip-ViT-B-32", type=str, help="CLIP model to encode the img and prompt")

    opt = parser.parse_args()

    #Load CLIP model
    model = SentenceTransformer(f'{opt.clip}')

    #Encode an image:
    avg_scores = []
    with torch.no_grad():
        for image_name in tqdm(os.listdir(opt.image_folder)):
            image_path = os.path.join(opt.image_folder, image_name)

            #Encode text descriptions
            text_prompt = image_name[:-4]
            text_emb = model.encode([text_prompt])
            
            image = np.array(Image.open(image_path))
            h, w = image.shape[:2]
            crop_images = [image[:, h * i: h * (i+1)] for i in range(w//h)]

            for crop_image in crop_images:
                pil_image = Image.fromarray(crop_image)
                img_emb = model.encode(pil_image)

                #Compute cosine similarities
                cos_scores = util.cos_sim(img_emb, text_emb)
                avg_scores.append(cos_scores[0][0].cpu().numpy())


    print("The final CLIP R-Precision is:", sum(avg_scores)/len(avg_scores))

                # pil_image.save("test.png")
                # breakpoint()