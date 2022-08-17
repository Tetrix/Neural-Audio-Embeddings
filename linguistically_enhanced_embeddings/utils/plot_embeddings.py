import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP

# apply TSNE
perplexity = 15


def collect_embeddings(encoder, text, features, word, device):
    words = []
    embeddings = []
    for line in text:
        line = line.split()
        for i in line:
            words.append(i.rstrip())

    indices = [i for i, x in enumerate(words) if x == word]

    for i in range(len(indices)):
        for j in range(i+1, len(indices)):
            word_feat = features[i]
            word_feat = word_feat.unsqueeze(1).to(device)

            _, _, encoder_output, encoder_hidden = encoder(word_feat, [int(word_feat.size(0))])
            embedding = encoder_hidden[0].sum(0, keepdim=True).squeeze().detach().cpu().numpy()
            embeddings.append(embedding)
    embeddings = np.array(embeddings)
    embeddings = np.vstack(embeddings)
    embeddings = np.mean(embeddings, axis=0)
    return embeddings
    

def plot_embeddings(encoder, text, features, device):
    embeddings_poor = collect_embeddings(encoder, text, features, "poor", device)
    embeddings_horrible = collect_embeddings(encoder, text, features, "horrible", device)
    embeddings_bad = collect_embeddings(encoder, text, features, "bad", device)
    embeddings_terrible = collect_embeddings(encoder, text, features, "terrible", device)
    embeddings_awful = collect_embeddings(encoder, text, features, "awful", device)
    embeddings_sick = collect_embeddings(encoder, text, features, "sick", device)

    embeddings_rich = collect_embeddings(encoder, text, features, "rich", device)
    embeddings_beautiful = collect_embeddings(encoder, text, features, "beautiful", device)
    embeddings_good = collect_embeddings(encoder, text, features, "good", device)
    embeddings_excellent = collect_embeddings(encoder, text, features, "excellent", device)
    embeddings_wonderful = collect_embeddings(encoder, text, features, "wonderful", device)
    embeddings_charming = collect_embeddings(encoder, text, features, "charming", device)

    
    embeddings = np.vstack((embeddings_poor, embeddings_horrible, embeddings_bad, embeddings_terrible, embeddings_awful, embeddings_sick, embeddings_rich, embeddings_beautiful, embeddings_good, embeddings_excellent, embeddings_wonderful, embeddings_charming))

    #embeddings = PCA(n_components=12).fit_transform(embeddings)
    embeddings = TSNE(n_components=2, learning_rate="auto", init="pca", perplexity=perplexity, method="exact").fit_transform(embeddings)
    #embeddings = UMAP(n_neighbors=50, min_dist=0.2).fit_transform(embeddings)

    df = pd.DataFrame(data=embeddings, columns=["dim1", "dim2"])
    
    df["Word"] = ["poor", "horrible", "bad", "terrible", "awful", "sick", "rich", "beautiful", "good", "excellent", "wonderful", "charming"]
    df["Category"] = ["negative", "negative", "negative", "negative", "negative", "negative", "positive", "positive", "positive", "positive", "positive", "positive"]

    plot = sns.scatterplot(data=df, x="dim1", y="dim2", hue="Category")
    
    for line in range(0, len(df["Word"])):
        plot.text(df.loc[line]["dim1"]+0.01, df.loc[line]["dim2"], df.loc[line]["Word"], horizontalalignment='left', size='small', color='black', weight='semibold')

    # plot
    plt.savefig("output/embeddings.png", dpi=300)
