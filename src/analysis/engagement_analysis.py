import logging
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns


def analyze_engagement_and_requests(df, cluster_comments, extract_products_and_deals, get_sentiment, extract_requests, extract_pricing_discussions):
    """Analyze product/deal sentiment, customer requests, pricing discussions, and clusters."""
    try:
        df['products'], df['deals'] = zip(*df['comment_text'].apply(extract_products_and_deals))
        df['sentiment'] = df['comment_text_clean'].apply(get_sentiment)
        df['requests'] = df['comment_text'].apply(extract_requests)
        df['is_pricing'], df['pricing_sentiment'] = zip(*df['comment_text'].apply(extract_pricing_discussions))
        cluster_products, cluster_labels = cluster_comments(df)
        df['cluster'] = pd.Series(cluster_labels, index=df.index[df['comment_text'].str.strip() != ''], dtype='Int64')
        product_sentiments = {}
        for _, row in df.iterrows():
            for product in row['products']:
                if product not in product_sentiments:
                    product_sentiments[product] = []
                product_sentiments[product].append(row['sentiment'])
        deal_sentiments = {}
        for _, row in df.iterrows():
            for deal in row['deals']:
                if deal not in deal_sentiments:
                    deal_sentiments[deal] = []
                deal_sentiments[deal].append(row['sentiment'])
        cluster_sentiments = {}
        cluster_requests = {}
        for cluster_id, product_name in cluster_products.items():
            cluster_comments_df = df[df['cluster'] == cluster_id]
            if not cluster_comments_df.empty:
                cluster_sentiments[product_name] = cluster_comments_df['sentiment'].mean()
                cluster_request_counts = Counter()
                for requests in cluster_comments_df['requests']:
                    for req_type, _ in requests:
                        cluster_request_counts[req_type] += 1
                cluster_requests[product_name] = cluster_request_counts
        product_sentiment_avg = {k: sum(v)/len(v) for k, v in product_sentiments.items() if v}
        deal_sentiment_avg = {k: sum(v)/len(v) for k, v in deal_sentiments.items() if v}
        request_counts = Counter()
        request_examples = {'flavor': [], 'size': [], 'product-specific': []}
        for requests in df['requests']:
            for req_type, req_text in requests:
                request_counts[req_type] += 1
                if len(request_examples[req_type]) < 3:
                    request_examples[req_type].append(req_text)
        pricing_comments = df[df['is_pricing']]
        pricing_sentiment_avg = pricing_comments['pricing_sentiment'].mean() if not pricing_comments.empty else 0.0
        pricing_counts = len(pricing_comments)
        plt.figure(figsize=(15, 12))
        if product_sentiment_avg:
            plt.subplot(2, 3, 1)
            sns.barplot(x=list(product_sentiment_avg.values()), y=list(product_sentiment_avg.keys()))
            plt.title('Average Sentiment by Product (LLM)')
            plt.xlabel('Sentiment Score')
            plt.ylabel('Product')
        if deal_sentiment_avg:
            plt.subplot(2, 3, 2)
            sns.barplot(x=list(deal_sentiment_avg.values()), y=list(deal_sentiment_avg.keys()))
            plt.title('Average Sentiment by Deal (LLM)')
            plt.xlabel('Sentiment Score')
            plt.ylabel('Deal')
        if request_counts:
            plt.subplot(2, 3, 3)
            sns.barplot(x=list(request_counts.values()), y=list(request_counts.keys()))
            plt.title('Frequency of Customer Requests (LLM)')
            plt.xlabel('Count')
            plt.ylabel('Request Type')
        df['week'] = df['timestamp'].dt.isocalendar().week
        weekly_sentiment = df.groupby('week')['sentiment'].mean()
        plt.subplot(2, 3, 4)
        weekly_sentiment.plot(kind='line', marker='o', color='green')
        plt.title('Average Sentiment by Week (March 2025)')
        plt.xlabel('Week')
        plt.ylabel('Sentiment Score')
        if pricing_counts > 0:
            plt.subplot(2, 3, 5)
            sns.histplot(pricing_comments['pricing_sentiment'], bins=20, kde=True)
            plt.title('Sentiment of Pricing Discussions')
            plt.xlabel('Sentiment Score')
            plt.ylabel('Count')
        if cluster_sentiments:
            plt.subplot(2, 3, 6)
            sns.barplot(x=list(cluster_sentiments.values()), y=list(cluster_sentiments.keys()))
            plt.title('Average Sentiment by Product Cluster')
            plt.xlabel('Sentiment Score')
            plt.ylabel('Cluster/Product')
        plt.tight_layout()
        plt.savefig('engagement_analysis.png')
        plt.close()
        logging.info("Engagement and request visualizations generated")
        return product_sentiment_avg, deal_sentiment_avg, request_counts, request_examples, pricing_counts, pricing_sentiment_avg, cluster_sentiments, cluster_requests
    except Exception as e:
        logging.error(f"Error in engagement analysis: {e}")
        return {}, {}, Counter(), {}, 0, 0.0, {}, {}

def generate_report(product_sentiment, deal_sentiment, request_counts, request_examples, pricing_counts, pricing_sentiment_avg, cluster_sentiments, cluster_requests):
    """Generate markdown report from analysis results."""
    report = "# @treehut Instagram Comments Analysis - March 2025\n\n"
    report += "## Key Insights\n"
    report += "- **Product Engagement**: LLM identifies high-sentiment products driving consumer interest.\n"
    report += "- **Deal Engagement**: Promotions like BOGO show strong positive sentiment, indicating effective marketing.\n"
    report += "- **Customer Requests**: High request volumes for specific products/sizes (e.g., full-size versions) indicate demand for new offerings.\n"
    report += f"- **Pricing Discussions**: {pricing_counts} comments discuss pricing (e.g., mini vs. full-size), with an average sentiment of {pricing_sentiment_avg:.2f}.\n"
    report += "- **Product Clusters**: Clustered comments reveal trending product lines with high engagement.\n"
    report += "\n## Top Products by Sentiment\n"
    for product, sentiment in sorted(product_sentiment.items(), key=lambda x: x[1], reverse=True)[:5]:
        report += f"- {product}: {sentiment:.2f}\n"
    report += "\n## Top Deals by Sentiment\n"
    for deal, sentiment in sorted(deal_sentiment.items(), key=lambda x: x[1], reverse=True)[:5]:
        report += f"- {deal}: {sentiment:.2f}\n"
    report += "\n## Customer Requests\n"
    for req_type, count in request_counts.items():
        report += f"- {req_type.capitalize()} requests: {count}\n"
        report += "  Examples:\n"
        for example in request_examples.get(req_type, [])[:3]:
            report += f"    - {example}\n"
    report += "\n## Pricing Discussions\n"
    report += f"- Total comments: {pricing_counts}\n"
    report += f"- Average sentiment: {pricing_sentiment_avg:.2f}\n"
    report += "\n## Trending Product Clusters\n"
    for cluster_name, sentiment in sorted(cluster_sentiments.items(), key=lambda x: x[1], reverse=True)[:5]:
        req_counts = cluster_requests.get(cluster_name, Counter())
        total_requests = sum(req_counts.values())
        report += f"- {cluster_name}: Sentiment {sentiment:.2f}, Requests: {total_requests}\n"
        for req_type, count in req_counts.items():
            report += f"  - {req_type.capitalize()} requests: {count}\n"
    report += "\n## Recommendations\n"
    report += "- **Expand Trending Product Lines**: Develop products from high-request clusters (e.g., full-size versions of popular lines).\n"
    report += "- **Address Pricing Concerns**: Review pricing strategies for mini vs. full-size products based on sentiment analysis.\n"
    report += "- **Promote Top Products**: Highlight high-sentiment products in campaigns and Stories.\n"
    report += "- **Extend Deals**: Continue successful promotions like BOGO, tailoring based on LLM insights.\n"
    report += "- **Real-Time Monitoring**: Use LLM to track emerging trends in real-time.\n"
    with open('treehut_report.md', 'w') as f:
        f.write(report)
    logging.info("Report generated successfully")
    return report
