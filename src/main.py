
# Modular imports
import logging

from src.analysis.engagement_analysis import (analyze_engagement_and_requests,
                                              generate_report)
from src.data_processing.llm_processing import (cluster_comments,
                                                extract_pricing_discussions,
                                                extract_products_and_deals,
                                                extract_requests,
                                                get_sentiment)
from src.data_processing.load_data import load_data
from src.data_processing.text_preprocessing import preprocess_text

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(file_path='data/engagements.csv'):
    try:
        # Load data
        df = load_data(file_path, logger)
        # Preprocess comments
        df['comment_text_clean'] = df['comment_text'].apply(preprocess_text)
        # Analyze engagement, requests, pricing, and clusters
        product_sentiment, deal_sentiment, request_counts, request_examples, pricing_counts, pricing_sentiment_avg, cluster_sentiments, cluster_requests = analyze_engagement_and_requests(
            df,
            cluster_comments,
            extract_products_and_deals,
            get_sentiment,
            extract_requests,
            extract_pricing_discussions
        )
        # Generate report
        generate_report(
            product_sentiment,
            deal_sentiment,
            request_counts,
            request_examples,
            pricing_counts,
            pricing_sentiment_avg,
            cluster_sentiments,
            cluster_requests
        )
    except Exception as e:
        logger.error(f"Error in main pipeline: {e}")
        raise

if __name__ == '__main__':
    main()