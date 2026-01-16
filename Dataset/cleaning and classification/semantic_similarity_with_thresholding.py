import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report, f1_score
)

# from huggingface_hub import login
# from google.colab import userdata

# load datasets

df = pd.read_csv('indian_media.csv')               # full dataset
df_labeled = pd.read_csv('./news_categorized_manually.csv')      # 425 manually labeled samples (columns : headline,news,link,category)

# df_labeled = df_labeled[df_labeled["category"] != "Other"]


# event keywords
event_keywords = {
    "Ayodhya Ram Mandir": [
        "ayodhya ram mandir", "ram temple","ram mandir inauguration","ayodhya temple consecration",
        "ayodhya temple","ram janmabhoomi","lord ram temple","ram idol installation","ram mandir opening",
        "ayodhya historic temple","ram mandir verdict", "supreme court ayodhya","ram rajya",
        "hindu temple ayodhya","ram mandir celebrations","ram mandir supreme court decision"
    ],
    "Operation Sindoor": [
        "operation sindoor","rescue sindoor","indian rescue operation","sindoor evacuation",
        "pahalgam attack",
    ],
    "Covaxin": [
        "covaxin","bharat biotech","indian covid vaccine","covid immunization india","vaccine drive india",
        "indigenous covid vaccine","india vaccination campaign","covaxin approval","covid inoculation india",
        "vaccine booster india","covid-19 vaccine rollout","aiims covid vaccine", "covaxin side effects",
        "vaccine registration india","covid third wave india"
    ],
    "Chandrayaan-3": [
        "chandrayaan 3","isro lunar mission","india moon landing","isro spacecraft","chandrayaan mission",
        "indian moon rover","vikram lander","pragyaan rover","lunar south pole india","ch3 moon landing",
        "chandrayaan 2023","isro moon mission success","lunar surface exploration india","isro space achievements",
        "lunar polar region","india space research"
    ]
}


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# event embeddings from keywords  
event_embeddings = {}
for event, keywords in event_keywords.items():
    emb = model.encode(keywords, convert_to_tensor=True)   # dimension : [keywords,384]
    event_embeddings[event] = emb.mean(dim=0)  # centroid embedding


# giving more weight to headline than content
def combine_text(headline, content):
    return (headline + " ") * 2 + str(content)


# classification of news based on similarity using embeddings of keywords
# if max_similarity > threshold then article useful else categorize as other.

def classify_news(headline, content, threshold):
    text = combine_text(str(headline), str(content))
    news_emb = model.encode(text, convert_to_tensor=True)

    scores = {}
    for event, emb in event_embeddings.items():
        scores[event] = util.cos_sim(news_emb, emb).item()

    best_event, best_score = max(scores.items(), key=lambda x: x[1])

    if best_score >= threshold:
        return best_event, best_score
    else:
        return "Other", best_score


# finding best threshold using manually labelled western media dataset (here add all manual data after rama completes)
def evaluate_threshold(df, thresholds):
    best_f1 = 0
    best_t = 0
    for t in thresholds:
        preds = []
        for _, row in df.iterrows():
            pred, _ = classify_news(row["headline"], row["news"], threshold=t)
            preds.append(pred)

        f1 = f1_score(df["category"], preds, average="weighted", zero_division=0)
        print(f"Threshold {t:.2f} → Weighted F1 = {f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t, best_f1


# tuning threshold value using labelled data
if "category" in df_labeled.columns:
    thresholds = np.arange(0.3, 0.81, 0.05)
    best_t, best_f1 = evaluate_threshold(df_labeled, thresholds)
    print(f"\n✅ Best Threshold: {best_t:.2f} with F1 = {best_f1:.3f}")

    # evalute labeled data using best threshold
    preds, sims = [], []
    for _, row in df_labeled.iterrows():
        pred, score = classify_news(row["headline"], row["news"], threshold=best_t)
        preds.append(pred)
        sims.append(score)

    df_labeled["predicted_category"] = preds
    df_labeled["similarity_score"] = sims

    # Metrics
    accuracy = accuracy_score(df_labeled["category"], df_labeled["predicted_category"])
    precision, recall, f1, _ = precision_recall_fscore_support(
        df_labeled["category"], df_labeled["predicted_category"], average="weighted", zero_division=0
    )

    print("\n--- Evaluation on Labeled Data ---")
    print(f"Accuracy  : {accuracy:.3f}")
    print(f"Precision : {precision:.3f}")
    print(f"Recall    : {recall:.3f}")
    print(f"F1 Score  : {f1:.3f}")

    print("\n🔹 Detailed per-class report:")
    print(classification_report(df_labeled["category"], df_labeled["predicted_category"], zero_division=0))

    # Save predictions
    df_labeled.to_csv("news_labeled_eval.csv", index=False)

    # now will classify unlabeled data (indian media) (df is indian media)
    predicted, scores = [], []
    for _, row in df.iterrows():
        label, score = classify_news(row["headline"], row["news"], threshold=best_t)
        predicted.append(label)
        scores.append(score)

    df["predicted_category"] = predicted
    df["similarity_score"] = scores

    # Keep only relevant events
    df_cleaned = df[df["predicted_category"] != "Other"]
    df_cleaned.to_csv("indian_news_dataset_cleaned.csv",index=False)

    print("\n✅ Classification complete.")
    print("➡ Cleaned dataset saved as indian_news_dataset_cleaned.csv")
    print("➡ Evaluation results saved as news_labeled_eval.csv")

else:
    print("Error: 'category' column not found in df_labeled. Cannot evaluate threshold or classify data.")
    print("Please ensure your labeled dataset has a 'category' column with the ground truth labels.")