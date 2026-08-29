# InfluenceIQ — Social Media Influencer Marketing Optimization

## Project Overview
Full-stack ML application implementing 3 machine learning models for influencer marketing optimization.

## ML Models
| Model | Task | Metric |
|-------|------|--------|
| Linear Regression | Predict ROI % | R² = 0.50 |
| Logistic Regression | Classify High/Low Performer | Accuracy = 70% |
| K-Means (k=3) | Segment Micro/Macro/Premium | Inertia = 30.39 |

## Features
- **Dashboard**: Platform distribution, ROI buckets, cluster sizes, engagement scatter plot
- **Influencer Table**: Search, filter by platform/segment, sortable columns, pagination
- **ROI Prediction**: Live Linear Regression prediction + feature contribution chart
- **Classifier**: Logistic Regression with probability bars + coefficient analysis
- **Cluster Analysis**: K-Means assignment + distance-to-centroid chart + strategy recommendations
- **Model Metrics**: R², RMSE, accuracy, confusion matrix, actual vs predicted chart

## Setup
```bash
pip install -r requirements.txt
python app.py
# Open index.html in browser (or visit http://localhost:5000)
```

## API Endpoints
- GET  /api/dashboard       — Summary stats + chart data
- GET  /api/influencers     — Paginated table with filters
- GET  /api/top-performers  — Top ROI influencers
- POST /api/predict/roi     — Linear Regression ROI prediction
- POST /api/predict/classify — Logistic Regression classification
- POST /api/predict/cluster  — K-Means cluster assignment
- GET  /api/metrics         — All model performance metrics

## Input Features
follower_count, engagement_rate, avg_likes, avg_comments,
post_frequency, campaign_cost, conversion_rate
