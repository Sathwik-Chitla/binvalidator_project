import pickle
import numpy as np

emb = pickle.load(open("data/asin_clip_embeddings.pkl", "rb"))
asin_list = np.load("data/asin_list.npy")

print("Embeddings:", len(emb))
print("ASIN list:", len(asin_list))

print("First 5 keys in dict:", list(emb.keys())[:5])
print("First 5 items in asin_list:", asin_list[:5])
