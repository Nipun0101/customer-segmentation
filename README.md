# 🛒 Customer Segmentation — Interactive Visual Analytics (DSE3150)

An interactive Streamlit dashboard that explores a retail customer dataset, groups customers with **K-Means**, and lets a user explore the result with linked, filterable visualizations including a **customer similarity network**.

**Live app:** https://customer-segmentation-a4ndp9ybh8hhdsuqvqge8f.streamlit.app/

## Questions the tool answers
1. Which distinct customer groups exist?
2. What defines each group (income, spending, channel, recency)?
3. Which groups are most valuable, and which are drifting away?

## Dataset
`customer_segmentation.csv` — about 2,200 supermarket customers, 29 columns (demographics, product spending, purchase channels, campaign responses). Seven features are used for clustering: Age, Income, Total_Spending, NumWebPurchases, NumStorePurchases, NumWebVisitsMonth, Recency.

## Preprocessing (in `segmentation.py`)
- Drop rows with missing income; remove impossible ages and the income outlier; merge/remove junk marital values.
- Engineered features: Age, Total_Spending, Total_Children, AcceptedAny.
- Standardise features → K-Means (`random_state=42`) → PCA (2D) → k-nearest-neighbour similarity graph.

## Visualization pipeline
Raw CSV → cleaning and feature engineering → data tables → scaling, K-Means, PCA, k-NN → visual structures (points, lines, nodes/edges, rectangles) → Plotly views → user interaction feeds back into the tables.

## Features
| Tab | Content |
|---|---|
| Data | Dataset summary and cleaning notes |
| Explore | Histogram, box plot, bar, donut, correlation heatmap, scatter with trend line |
| Clusters | Elbow plot, PCA scatter, segment sizes, segment profile heatmap |
| Predict | Enter a customer, get a segment and see it on the PCA plot |
| Interactive Lab | Filters, parallel coordinates, brushing and linking, search and details-on-demand, similarity network, treemap |
| Design and Pipeline | Gestalt principles, visual mapping, overload control, evaluation plan |
| How to present | Presentation script |

Every chart has a "Why this chart / How to read it" note.

## Setup
```bash
git clone https://github.com/Nipun0101/customer-segmentation.git
cd customer-segmentation
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run segmentation.py
```

## Files
- `segmentation.py` — the app (preprocessing, models, all visualizations)
- `customer_segmentation.csv` — dataset
- `Analysis_Model.ipynb` — original exploratory analysis and model training
- `requirements.txt` — dependencies

## Tech
Python, Streamlit, Plotly, scikit-learn, NetworkX, pandas.

## Author
Nipun Bansal
