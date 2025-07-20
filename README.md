# 🧴 TreeHut Instagram Engagement Analysis

**Author:** Chris Brady

---

## 🚀 Quick Start

### 1. Install Dependencies
```sh
pip install -r requirements.txt
```

### 2. Run the Analysis
```sh
python -m src.main
```
- Processes: `data/engagements.csv`
- Outputs: `treehut_report.md` (insights) and `engagement_analysis.png` (visualizations)

---

## 📈 Extension Proposal

In this limited time frame, our main goal was to identify which products, deals, or posts were driving the most discussion and engagement.

**Next Steps:**
- **Product Library Mapping:**
  - More granular entity recognition for products and deals.
  - Track each product/deal as a unique entity for deeper insights.
- **Post-to-Product Mapping:**
  - Use NLP to link comments and captions to specific products, deals, or flavors.
  - Analyze which post styles drive the most engagement for each product/deal.
- **Image Analysis:**
  - Extend analysis to post images (e.g., color scheme, key features) using computer vision.
  - Redefine Instagram strategy based on visual trends.
- **User Engagement:**
  - Identify loyal users and those frequently tagged in comments.
  - Use entity resolution to distinguish existing customers from prospects.
  - Trigger AI-driven custom DMs and offers for high-value users and prospects.

---

## 🗺️ Mapping Products/Deals to Posts

- NLP identifies when a post or comment refers to a specific product, deal, or flavor.
- Hard data mapping would enable:
  - Better analysis of post types driving engagement.
  - Comparison of posts promoting the same product/deal.
  - Data-driven recommendations for future post styles.

---

## 🖼️ Image & User Analysis (Future Directions)

- **Image Analysis:**
  - Use computer vision to extract features from post images.
  - Analyze color schemes and visual elements for engagement trends.
- **User Tagging & Loyalty:**
  - Find users most frequently tagged in comments.
  - Combine with customer data for targeted outreach.
  - Use AI agents for personalized DMs and offers.

---

## 💡 Recommendations
- Expand trending product lines (e.g., full-size versions of popular products).
- Address pricing concerns (mini vs. full-size sentiment).
- Promote top products in campaigns and Stories.
- Extend successful deals (e.g., BOGO) based on LLM insights.
- Implement real-time monitoring for emerging trends.

---

## External tools 
I used some pre-trained LLM's specifically bert-large-mini

I also used copilot, primarily for debugging, code organization and some aesthetic things like formatting markdowns and reports.