from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def read_embeddings(filename):
    with open(filename) as f:
        lines = f.readlines()[1:]
        data = [list(map(float, line.strip().split()[1:])) for line in lines]
    return np.array(data)

embedding_file = './results/link_prediction/cora/4-dis_s.emb'
X = read_embeddings(embedding_file)
X_reduced = TSNE(n_components=2, random_state=42).fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], s=5)
plt.title("Visualization of Node Embeddings (Discriminator, Epoch 4)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.grid(True)
plt.show()
