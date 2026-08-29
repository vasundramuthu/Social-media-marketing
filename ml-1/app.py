"""
Social Media – Influencer Marketing Optimization
Flask Backend  |  3 ML Models

Run:
    pip install flask flask-cors scikit-learn pandas numpy
    python app.py
"""

import os, json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from sentiment_engine import SentimentEngine
from optimizer import CampaignOptimizer

try:
    from flask_cors import CORS
    app = Flask(__name__)
    CORS(app)
except ImportError:
    app = Flask(__name__)
    @app.after_request
    def add_cors(r):
        r.headers["Access-Control-Allow-Origin"]  = "*"
        r.headers["Access-Control-Allow-Headers"] = "Content-Type"
        r.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        return r

from sklearn.linear_model    import LinearRegression, LogisticRegression
from sklearn.cluster         import KMeans
from sklearn.preprocessing   import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (mean_squared_error, r2_score,
                                     accuracy_score, confusion_matrix)

# ─────────────────────────────────────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(42)
N         = 300
PLATFORMS = ['Instagram', 'YouTube', 'TikTok', 'Twitter', 'Facebook']
NICHES    = ['Fashion', 'Tech', 'Food', 'Travel', 'Fitness', 'Beauty', 'Gaming', 'Finance']

def generate_dataset():
    follower_count  = np.random.randint(5_000, 5_000_000, N)
    engagement_rate = np.clip(np.random.normal(3.5, 2.0, N), 0.5, 15.0)
    avg_likes       = (follower_count * engagement_rate / 100 * np.random.uniform(0.7,1.1,N)).astype(int)
    avg_comments    = (avg_likes * np.random.uniform(0.02, 0.10, N)).astype(int)
    post_frequency  = np.random.randint(1, 30, N)
    campaign_cost   = (follower_count / 1000 * np.random.uniform(3, 15, N)).astype(int)
    conversion_rate = np.clip(np.random.normal(1.0, 0.5, N), 0.1, 5.0)   # %
    avg_order       = np.random.uniform(30, 150, N)
    reach           = follower_count * 0.10
    conversions     = reach * conversion_rate / 100
    roi             = ((conversions * avg_order - campaign_cost) / campaign_cost * 100).round(2)
    is_high         = (roi > roi.mean()).astype(int)

    return pd.DataFrame({
        'name':              [f"Influencer_{i:03d}" for i in range(N)],
        'platform':          np.random.choice(PLATFORMS, N),
        'niche':             np.random.choice(NICHES, N),
        'follower_count':    follower_count,
        'engagement_rate':   engagement_rate.round(2),
        'avg_likes':         avg_likes,
        'avg_comments':      avg_comments,
        'post_frequency':    post_frequency,
        'campaign_cost':     campaign_cost,
        'conversion_rate':   conversion_rate.round(2),
        'roi':               roi,
        'is_high_performer': is_high,
    })

DF = generate_dataset()

# ─────────────────────────────────────────────────────────────────────────────
#  PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
FEAT = ['follower_count','engagement_rate','avg_likes','avg_comments',
        'post_frequency','campaign_cost','conversion_rate']
FEAT_CLU = ['follower_count','engagement_rate','avg_likes','conversion_rate']

sc_reg = StandardScaler(); sc_clf = StandardScaler(); sc_clu = MinMaxScaler()
X_reg  = sc_reg.fit_transform(DF[FEAT])
X_clf  = sc_clf.fit_transform(DF[FEAT])
X_clu  = sc_clu.fit_transform(DF[FEAT_CLU])
y_reg  = DF['roi'].values
y_clf  = DF['is_high_performer'].values

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL 1  —  Linear Regression (ROI prediction)
# ─────────────────────────────────────────────────────────────────────────────
Xtr_r,Xte_r,ytr_r,yte_r = train_test_split(X_reg,y_reg,test_size=0.2,random_state=42)
lin = LinearRegression()
lin.fit(Xtr_r, ytr_r)
yp_r = lin.predict(Xte_r)
LIN_M = dict(mse=round(mean_squared_error(yte_r,yp_r),2),
             rmse=round(float(np.sqrt(mean_squared_error(yte_r,yp_r))),2),
             r2=round(r2_score(yte_r,yp_r),4),
             train_size=int(len(Xtr_r)), test_size=int(len(Xte_r)))

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL 2  —  Logistic Regression (high/low performer)
# ─────────────────────────────────────────────────────────────────────────────
Xtr_c,Xte_c,ytr_c,yte_c = train_test_split(X_clf,y_clf,test_size=0.2,
                                             random_state=42,stratify=y_clf)
log = LogisticRegression(max_iter=1000, random_state=42)
log.fit(Xtr_c, ytr_c)
yp_c = log.predict(Xte_c)
cm   = confusion_matrix(yte_c, yp_c).tolist()
LOG_M = dict(accuracy=round(accuracy_score(yte_c,yp_c),4),
             confusion_matrix=cm,
             train_size=int(len(Xtr_c)), test_size=int(len(Xte_c)),
             class_dist=dict(high=int(y_clf.sum()), low=int(len(y_clf)-y_clf.sum())))

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL 3  —  K-Means Clustering (micro / macro / premium)
# ─────────────────────────────────────────────────────────────────────────────
km = KMeans(n_clusters=3, random_state=42, n_init=10)
km.fit(X_clu)
labels = km.labels_
cf = {i: float(DF['follower_count'][labels==i].median()) for i in range(3)}
sc = sorted(cf, key=cf.get)
CMAP = {
    sc[0]: dict(label='Micro',   color='#10b981', range_='<100K followers'),
    sc[1]: dict(label='Macro',   color='#f59e0b', range_='100K–1M followers'),
    sc[2]: dict(label='Premium', color='#ef4444', range_='>1M followers'),
}
DF['cluster']       = labels
DF['cluster_label'] = DF['cluster'].map(lambda c: CMAP[c]['label'])
KM_M = dict(
    inertia=round(float(km.inertia_),2),
    cluster_sizes={CMAP[i]['label']:int((labels==i).sum()) for i in range(3)},
)

print(f"  Linear Regression  R²: {LIN_M['r2']}  |  Logistic Acc: {LOG_M['accuracy']}  |  KMeans inertia: {KM_M['inertia']}")

# ─────────────────────────────────────────────────────────────────────────────
#  HELPER
# ─────────────────────────────────────────────────────────────────────────────
def clean(records):
    out = []
    for r in records:
        row = {}
        for k,v in r.items():
            if isinstance(v, (np.integer,np.int64,np.int32)): row[k]=int(v)
            elif isinstance(v, (np.floating,np.float64,np.float32)): row[k]=round(float(v),2)
            else: row[k]=v
        out.append(row)
    return out

# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/')
def root():
    return send_from_directory('.', 'index.html')

@app.route('/api/dashboard')
def dashboard():
    total  = len(DF)
    sample = DF.sample(120, random_state=1)[
        ['name','follower_count','engagement_rate','roi','cluster_label','platform','niche','is_high_performer']
    ].to_dict('records')
    return jsonify({
        'total_influencers':  total,
        'high_performers':    int(DF['is_high_performer'].sum()),
        'low_performers':     int(total - DF['is_high_performer'].sum()),
        'avg_roi':            round(float(DF['roi'].mean()),2),
        'avg_engagement':     round(float(DF['engagement_rate'].mean()),2),
        'avg_cost':           round(float(DF['campaign_cost'].mean()),2),
        'avg_conversion':     round(float(DF['conversion_rate'].mean()),2),
        'platform_dist':      DF['platform'].value_counts().to_dict(),
        'niche_dist':         DF['niche'].value_counts().to_dict(),
        'cluster_sizes':      KM_M['cluster_sizes'],
        'roi_buckets':        {
            'Negative (<0)':    int((DF['roi']<0).sum()),
            'Low (0–500)':      int(((DF['roi']>=0)&(DF['roi']<500)).sum()),
            'Medium (500–1500)':int(((DF['roi']>=500)&(DF['roi']<1500)).sum()),
            'High (1500+)':     int((DF['roi']>=1500).sum()),
        },
        'scatter_data': clean(sample),
    })

@app.route('/api/influencers')
def influencers():
    page     = int(request.args.get('page',1))
    per_page = int(request.args.get('per_page',20))
    search   = request.args.get('search','').lower()
    platform = request.args.get('platform','')
    cluster  = request.args.get('cluster','')
    sort_by  = request.args.get('sort','roi')
    asc      = request.args.get('dir','desc') == 'asc'

    df = DF.copy()
    if search:   df = df[df['name'].str.lower().str.contains(search)|df['niche'].str.lower().str.contains(search)]
    if platform: df = df[df['platform']==platform]
    if cluster:  df = df[df['cluster_label']==cluster]
    if sort_by in df.columns: df = df.sort_values(sort_by, ascending=asc)

    total   = len(df)
    start   = (page-1)*per_page
    records = clean(df.iloc[start:start+per_page].to_dict('records'))
    return jsonify({'data':records,'total':total,'page':page,
                    'pages':(total+per_page-1)//per_page,
                    'platforms':PLATFORMS,'clusters':['Micro','Macro','Premium']})

@app.route('/api/predict/roi', methods=['POST','OPTIONS'])
def pred_roi():
    if request.method=='OPTIONS': return '',204
    b = request.get_json()
    try:
        row = [[float(b['follower_count']),float(b['engagement_rate']),
                float(b['avg_likes']),float(b['avg_comments']),
                float(b['post_frequency']),float(b['campaign_cost']),float(b['conversion_rate'])]]
        roi = round(float(lin.predict(sc_reg.transform(pd.DataFrame(row, columns=FEAT)))[0]),2)
        if roi>1500:   advice,col='Excellent ROI — scale budget aggressively.','green'
        elif roi>500:  advice,col='Good ROI — maintain and optimize content.','teal'
        elif roi>0:    advice,col='Low ROI — refine targeting strategy.','yellow'
        else:          advice,col='Negative ROI — reconsider this influencer.','red'
        # Feature contributions (coefficients × scaled input)
        x_s = sc_reg.transform(pd.DataFrame(row, columns=FEAT))[0]
        contribs = {f:round(float(c*x),3) for f,c,x in zip(FEAT,lin.coef_,x_s)}
        return jsonify({'predicted_roi':roi,'advice':advice,'advice_color':col,
                        'contributions':contribs,'metrics':LIN_M})
    except Exception as e:
        return jsonify({'error':str(e)}),400

@app.route('/api/predict/classify', methods=['POST','OPTIONS'])
def pred_classify():
    if request.method=='OPTIONS': return '',204
    b = request.get_json()
    try:
        row = [[float(b['follower_count']),float(b['engagement_rate']),
                float(b['avg_likes']),float(b['avg_comments']),
                float(b['post_frequency']),float(b['campaign_cost']),float(b['conversion_rate'])]]
        pred  = int(log.predict(sc_clf.transform(pd.DataFrame(row, columns=FEAT)))[0])
        proba = log.predict_proba(sc_clf.transform(pd.DataFrame(row, columns=FEAT)))[0]
        coef  = log.coef_[0]
        fi    = sorted(zip(FEAT,coef), key=lambda x:abs(x[1]),reverse=True)
        return jsonify({
            'prediction': pred,
            'label':      'High Performer' if pred==1 else 'Low Performer',
            'confidence': round(float(max(proba))*100,1),
            'probability': {'high':round(float(proba[1])*100,1),'low':round(float(proba[0])*100,1)},
            'top_features': [{'feature':f,'coefficient':round(float(c),4)} for f,c in fi[:5]],
            'metrics': LOG_M,
        })
    except Exception as e:
        return jsonify({'error':str(e)}),400

@app.route('/api/predict/cluster', methods=['POST','OPTIONS'])
def pred_cluster():
    if request.method=='OPTIONS': return '',204
    b = request.get_json()
    try:
        row = [[float(b['follower_count']),float(b['engagement_rate']),
                float(b['avg_likes']),float(b['conversion_rate'])]]
        xs  = sc_clu.transform(pd.DataFrame(row, columns=FEAT_CLU))
        cid = int(km.predict(xs)[0])
        info = CMAP[cid]
        strategies = {
            'Micro':   ['High authenticity and trust with niche audiences.',
                        'Lower cost per engagement — ideal for tight budgets.',
                        'Best for long-term brand ambassador programs.'],
            'Macro':   ['Balanced reach + engagement for brand awareness.',
                        'Suitable for product reviews and sponsored posts.',
                        'Multi-post campaigns yield best results.'],
            'Premium': ['Maximum reach — ideal for mass-market launches.',
                        'Higher cost; ensure compelling creative briefs.',
                        'Focus on awareness, not direct conversion.'],
        }
        dists = {CMAP[i]['label']:round(float(np.linalg.norm(xs-km.cluster_centers_[i])),4) for i in range(3)}
        return jsonify({'cluster_id':cid,'cluster_label':info['label'],
                        'cluster_color':info['color'],'cluster_range':info['range_'],
                        'distances':dists,'strategies':strategies[info['label']],
                        'metrics':KM_M})
    except Exception as e:
        return jsonify({'error':str(e)}),400

@app.route('/api/metrics')
def metrics():
    return jsonify({'linear':LIN_M,'logistic':LOG_M,'kmeans':KM_M,'n':N,'features':FEAT})

@app.route('/api/top-performers')
def top_perf():
    n  = int(request.args.get('n',10))
    tp = DF[DF['is_high_performer']==1].nlargest(n,'roi')[
         ['name','platform','niche','follower_count','engagement_rate',
          'roi','campaign_cost','conversion_rate','cluster_label']].to_dict('records')
    return jsonify(clean(tp))

@app.route('/api/analyze/sentiment', methods=['POST'])
def analyze_text():
    data = request.json
    comments = data.get('comments', [])
    result = SentimentEngine.calculate_safety_score(comments)
    return jsonify(result)


if __name__ == '__main__':
    print("=" * 55)
    print("  Influencer Marketing Optimization — Running")
    print("=" * 55)
    app.run(debug=True, port=5000)

