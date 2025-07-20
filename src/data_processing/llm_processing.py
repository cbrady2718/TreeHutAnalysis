
import logging
from collections import Counter

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers import pipeline

# LLM pipeline initializations
embedder = SentenceTransformer('all-MiniLM-L6-v2')
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
zero_shot_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


def extract_pricing_discussions(text):
    """Identify pricing discussions using zero-shot classification."""
    try:
        if not isinstance(text, str) or not text.strip():
            logging.warning(f"Skipping empty or invalid input in pricing extraction: {text}")
            return False, 0.0
        candidate_labels = ['pricing discussion', 'no pricing discussion']
        result = zero_shot_pipeline(text, candidate_labels, multi_label=False)
        is_pricing = result['labels'][0] == 'pricing discussion'
        sentiment = get_sentiment(text) if is_pricing else 0.0
        return is_pricing, sentiment
    except Exception as e:
        logging.error(f"Error extracting pricing discussions: {e}, input: {text}")
        return False, 0.0

def cluster_comments(df, n_clusters=5):
    """Cluster comments by semantic similarity to identify product lines."""
    try:
        valid_comments = df['comment_text'][df['comment_text'].str.strip() != '']
        if len(valid_comments) < n_clusters:
            logging.warning(f"Not enough valid comments for clustering: {len(valid_comments)}")
            return {}, []
        embeddings = embedder.encode(valid_comments.tolist(), show_progress_bar=False)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        # Map clusters to representative products using NER
        cluster_products = {}
        for cluster_id in range(n_clusters):
            cluster_comments = valid_comments[cluster_labels == cluster_id]
            products = []
            for comment in cluster_comments[:10]:  # Sample top comments for efficiency
                ents, _ = extract_products_and_deals(comment)
                products.extend(ents)
            if products:
                most_common = Counter(products).most_common(1)
                if most_common:
                    cluster_products[cluster_id] = most_common[0][0]
                else:
                    cluster_products[cluster_id] = f"Cluster {cluster_id}"
            else:
                cluster_products[cluster_id] = f"Cluster {cluster_id}"
        return cluster_products, cluster_labels
    except Exception as e:
        logging.error(f"Error clustering comments: {e}")
        return {}, []


def extract_products_and_deals(text):
    """Extract products and deals using LLM-based NER."""
    try:
        if not isinstance(text, str) or not text.strip():
            logging.warning(f"Skipping empty or invalid input in NER: {text}")
            return [], []
        entities = ner_pipeline(text)
        products = []
        deals = []
        skincare_keywords = ['scrub', 'cream', 'serum', 'mask', 'lotion']
        deal_keywords = ['bogo', 'deal', 'offer', 'discount', 'sale', 'bundle']
        for entity in entities:
            entity_text = entity['word'].lower()
            if entity['entity_group'] in ['PRODUCT', 'ORG'] and any(keyword in entity_text for keyword in skincare_keywords):
                products.append(entity_text)
            elif any(keyword in entity_text for keyword in deal_keywords):
                deals.append(entity_text)
        return products, deals
    except Exception as e:
        logging.error(f"Error extracting products/deals with LLM: {e}")
        return [], []

def extract_requests(text, intent_pipeline, logger):
    """Extract customer requests for flavors or sizes using LLM-based classification."""
    try:
        result = intent_pipeline(f"Classify request: {text}", return_all_scores=True)[0]
        max_score_label = max(result, key=lambda x: x['score'])['label']
        if max_score_label == 'LABEL_1':
            return [('flavor', text)]
        elif max_score_label == 'LABEL_2':
            return [('size', text)]
        return []
    except Exception as e:
        logger.error(f"Error extracting requests with LLM: {e}")
        return []

def get_sentiment(text, sentiment_pipeline):
    """Calculate sentiment score using LLM."""
    try:
        result = sentiment_pipeline(text)[0]
        score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
        return score
    except:
        return 0.0
