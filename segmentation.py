import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Customer Segmentation", page_icon="🛒", layout="wide")

FEATURES = ["Age", "Income", "Total_Spending", "NumWebPurchases",
            "NumStorePurchases", "NumWebVisitsMonth", "Recency"]
MNT = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts",
       "MntSweetProducts", "MntGoldProds"]


@st.cache_data
def load():
    df = pd.read_csv("customer_segmentation.csv")
    df = df.dropna(subset=["Income"])
    df["Marital_Status"] = df["Marital_Status"].replace({"Alone": "Single"})
    df = df[~df["Marital_Status"].isin(["YOLO", "Absurd"])]
    df["Age"] = 2025 - df["Year_Birth"]
    df = df[(df["Age"] < 100) & (df["Income"] < 200000)].copy()  # remove outliers
    df["Total_Spending"] = df[MNT].sum(axis=1)
    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]
    df["AcceptedAny"] = (df[[c for c in df if c.startswith("AcceptedCmp")]
                            + ["Response"]].sum(axis=1) > 0).astype(int)
    return df.reset_index(drop=True)


@st.cache_data
def elbow(X, kmax=9):
    return [KMeans(k, n_init=10, random_state=42).fit(X).inertia_ for k in range(2, kmax + 1)]


df = load()
scaler = StandardScaler().fit(df[FEATURES])
X = scaler.transform(df[FEATURES])

k = st.sidebar.slider("Number of clusters (k)", 3, 8, 6)
km = KMeans(k, n_init=10, random_state=42).fit(X)
df["Cluster"] = km.labels_
pca = PCA(2, random_state=42).fit(X)
pts = pca.transform(X)
df["PC1"], df["PC2"] = pts[:, 0], pts[:, 1]

# --- auto-name segments from their profile (spend tier + channel + recency) ---
prof = df.groupby("Cluster")[FEATURES].mean()
sp_rank = prof["Total_Spending"].rank(method="first")
names = {}
for c, r in prof.iterrows():
    tier = "High" if sp_rank[c] > 2 * k / 3 else "Mid" if sp_rank[c] > k / 3 else "Low"
    ch = "Web" if r.NumWebPurchases > r.NumStorePurchases else "Store"
    act = "Active" if r.Recency < prof["Recency"].median() else "Lapsing"
    names[c] = f"{c}: {tier} spend · {ch} · {act}"
df["Segment"] = df["Cluster"].map(names)
order = [names[c] for c in sorted(names)]

st.sidebar.markdown("**Dataset:** 2,200+ supermarket customers, 7 features used for clustering.")
st.title("🛒 Customer Segmentation Dashboard")
st.caption("K-Means clustering + visual analytics · every chart explains why it was chosen")


def why(chart, purpose, read):
    st.info(f"**Why {chart}?** {purpose}\n\n**How to read it:** {read}")


t1, t2, t3, t4, t5, t6, t7 = st.tabs(["1 · Data", "2 · Explore", "3 · Clusters", "4 · Predict", "5 · Interactive Lab", "6 · Design & Pipeline", "🎤 How to present"])

# ---------------- TAB 1 ----------------
with t1:
    a, b, c, d = st.columns(4)
    a.metric("Customers", f"{len(df):,}")
    b.metric("Avg income", f"{df.Income.mean():,.0f}")
    c.metric("Avg total spend", f"{df.Total_Spending.mean():,.0f}")
    d.metric("Accepted any campaign", f"{df.AcceptedAny.mean():.0%}")
    st.markdown("""
**One row = one customer.** Column groups: *Who they are* (age, income, education, marital status, children) ·
*What they buy* (6 product spend columns) · *How they buy* (web / store / catalog / deals) · *Marketing* (recency, campaigns).

**Cleaning done:** removed rows with missing income, impossible ages (birth year 1893), income outlier (666,666)
and junk marital values ("YOLO", "Absurd"); merged "Alone" into "Single".
**Engineered features:** Age, Total_Spending, Total_Children, AcceptedAny.
""")
    st.dataframe(df[FEATURES + ["Education", "Marital_Status", "Total_Children"]].head(50), width='stretch')

# ---------------- TAB 2 ----------------
with t2:
    st.subheader("A. Distributions — what does a typical customer look like?")
    col = st.selectbox("Variable", ["Income", "Age", "Total_Spending"])
    st.plotly_chart(px.histogram(df, x=col, nbins=40, marginal="box", color_discrete_sequence=["#4F46E5"]),
                    width='stretch')
    why("a histogram + box plot", "To see the shape of ONE numeric variable: where most values sit, how spread out they are, and whether there are outliers.",
        "Tall bars = many customers in that range. Spending is right-skewed: most customers spend little, a few spend a lot. Dots outside the box are outliers.")

    st.subheader("B. Spending by education — do groups differ?")
    st.plotly_chart(px.box(df, x="Education", y="Total_Spending", color="Education"), width='stretch')
    why("a box plot", "To compare the distribution of a number across categories, not just the average.",
        "Line inside the box = median; box = middle 50% of customers; whiskers = normal range. Higher box = higher spenders.")

    st.subheader("C. Campaign acceptance by marital status")
    acc = df.groupby("Marital_Status")["AcceptedAny"].mean().mul(100).round(1).reset_index()
    st.plotly_chart(px.bar(acc.sort_values("AcceptedAny", ascending=False), x="Marital_Status", y="AcceptedAny",
                           text="AcceptedAny", labels={"AcceptedAny": "% accepted a campaign"},
                           color_discrete_sequence=["#0891B2"]), width='stretch')
    why("a bar chart", "To compare one value (a percentage) across a few categories — the easiest chart for the eye to compare.",
        "Taller bar = higher acceptance rate. Compare bar heights directly.")

    st.subheader("D. Spending mix — where does the money go?")
    mix = df[MNT].sum().rename(lambda s: s.replace("Mnt", "").replace("Prods", "")).reset_index()
    mix.columns = ["Category", "Total"]
    st.plotly_chart(px.pie(mix, names="Category", values="Total", hole=0.45), width='stretch')
    why("a donut chart", "To show parts of a whole (share of total revenue by product category). Works because there are only 6 slices.",
        "Bigger slice = bigger share of total spend. Wine and meat dominate.")

    st.subheader("E. Correlation heatmap — which variables move together?")
    num = df[FEATURES + ["Total_Children"]].corr().round(2)
    st.plotly_chart(px.imshow(num, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1),
                    width='stretch')
    why("a heatmap", "To check many variable pairs at once. It guides which features matter for clustering and shows relationships.",
        "+1 (dark red) = rise together, −1 (dark blue) = one rises as the other falls, 0 = no link. Income and spending are strongly positive; web visits and income are negative.")

    st.subheader("F. Income vs spending")
    st.plotly_chart(px.scatter(df, x="Income", y="Total_Spending", opacity=0.5, trendline="ols",
                               color_discrete_sequence=["#E11D48"]), width='stretch')
    why("a scatter plot with trend line", "To see the relationship between TWO numeric variables, one dot per customer.",
        "Upward-sloping cloud = higher income → higher spending. The line is the average trend.")

# ---------------- TAB 3 ----------------
with t3:
    st.subheader("A. Elbow plot — how many clusters?")
    w = elbow(X)
    st.plotly_chart(px.line(x=list(range(2, 10)), y=w, markers=True,
                            labels={"x": "k (number of clusters)", "y": "WCSS (inertia)"}), width='stretch')
    why("an elbow (line) plot", "To choose k. WCSS measures how tight clusters are; it always falls as k grows, so we look for where the drop slows down.",
        "Pick the 'bend'. Here the bend is gradual, so k = 6 is a balance between detail and interpretability. Use the sidebar slider to change k.")

    st.subheader("B. PCA scatter — can we see the clusters?")
    st.plotly_chart(px.scatter(df, x="PC1", y="PC2", color="Segment", category_orders={"Segment": order},
                               hover_data=FEATURES, opacity=0.75), width='stretch')
    why("a PCA scatter plot", "Our 7 features can't be drawn directly. PCA compresses them into 2 axes that keep most of the variation, so every customer becomes one dot.",
        f"Each dot is a customer, colour = segment. Separate colour blobs = distinct groups. The 2 axes explain {pca.explained_variance_ratio_.sum():.0%} of the variance.")

    st.subheader("C. Segment size")
    sz = df["Segment"].value_counts().reindex(order).reset_index()
    sz.columns = ["Segment", "Customers"]
    st.plotly_chart(px.bar(sz, x="Customers", y="Segment", orientation="h", color="Segment"), width='stretch')
    why("a horizontal bar chart", "To compare segment sizes. Horizontal bars leave room for long labels.", "Longer bar = more customers in that segment.")

    st.subheader("D. Segment profile heatmap — what defines each segment?")
    z = (prof - prof.mean()) / prof.std()
    z.index = order
    st.plotly_chart(px.imshow(z.round(2), text_auto=True, color_continuous_scale="RdYlGn", aspect="auto"),
                    width='stretch')
    why("a heatmap of standardised averages", "This is the key chart for naming segments: it shows how each segment differs from the others on every feature, on one common scale.",
        "Green = higher than other segments, red = lower. Read each row to describe that segment (e.g. high income + high spend + low web visits = premium store shopper).")

    st.subheader("E. Segment table")
    st.dataframe(df.groupby("Segment")[FEATURES].mean().round(1).loc[order], width='stretch')

# ---------------- TAB 4 ----------------
with t4:
    st.subheader("Predict a customer's segment")
    c1, c2, c3, c4 = st.columns(4)
    age = c1.number_input("Age", 18, 100, 45)
    inc = c2.number_input("Income", 0, 200000, 60000, step=1000)
    spend = c3.number_input("Total spending", 0, 3000, 800)
    rec = c4.number_input("Recency (days)", 0, 99, 30)
    c5, c6, c7 = st.columns(3)
    wp = c5.number_input("Web purchases", 0, 30, 4)
    sp = c6.number_input("Store purchases", 0, 30, 6)
    wv = c7.number_input("Web visits / month", 0, 20, 5)
    if st.button("Predict segment", type="primary"):
        row = pd.DataFrame([[age, inc, spend, wp, sp, wv, rec]], columns=FEATURES)
        xs = scaler.transform(row)
        cl = int(km.predict(xs)[0])
        st.success(f"Predicted segment: **{names[cl]}**")
        p = pca.transform(xs)[0]
        fig = px.scatter(df, x="PC1", y="PC2", color="Segment", category_orders={"Segment": order}, opacity=0.35)
        fig.add_trace(go.Scatter(x=[p[0]], y=[p[1]], mode="markers", name="This customer",
                                 marker=dict(size=18, color="black", symbol="star")))
        st.plotly_chart(fig, width='stretch')
        st.caption("The black star shows where this customer lands among all existing customers.")

# ---------------- TAB 5 ----------------
with t7:
    st.subheader("Presentation script (8–10 minutes)")
    st.markdown("""
| # | Show | Say |
|---|------|-----|
| 1 | Data tab | "This is supermarket customer data: who they are, what they buy, how they buy. Goal: group similar customers so marketing can be targeted." |
| 2 | Histogram | "I first checked the shape of the data. Spending is skewed: few big spenders, many small." |
| 3 | Box plot (education) | "Higher education → higher spending, and the box plot shows the whole spread, not just the average." |
| 4 | Bar (campaigns) | "Singles accept campaigns more than couples. A bar chart is best for comparing categories." |
| 5 | Heatmap (correlation) | "Income and spending move together, so these are strong clustering features." |
| 6 | Elbow plot | "WCSS drops as k grows; the bend is gradual, so I chose k = 6 for interpretability." |
| 7 | PCA scatter | "7 features can't be drawn, so PCA gives 2 axes. Each dot is a customer, each colour a segment." |
| 8 | Profile heatmap | "This explains each segment. Green = above others, red = below. This is how I named them." |
| 9 | Predict tab | **Live demo:** enter a new customer, show the segment and the black star on the PCA plot. |
| 10 | Close | "Business use: loyalty offers for high spenders, win-back campaigns for lapsing customers, deals for browsers." |

**Chart-choice rule to say aloud:** distribution → histogram · compare groups → box / bar · parts of a whole → donut ·
two numbers → scatter · many pairs → heatmap · trend / choosing k → line · many dimensions → PCA scatter.

**Likely questions:** Why K-Means? (simple, fast, numeric data) · Why scale? (income is far larger than recency, K-Means uses distance) ·
Why k = 6? (elbow was gradual; silhouette score is a next step) · Limitations? (PCA loses information, K-Means assumes round clusters, outliers were removed).
""")

# ---------------- TAB 5: INTERACTIVE LAB ----------------
with t5:
    st.subheader("Interactive Lab — filter · brush & link · search · details-on-demand")
    f1, f2, f3 = st.columns(3)
    edus, mars = sorted(df.Education.unique()), sorted(df.Marital_Status.unique())
    edu = f1.multiselect("Education", edus, default=edus)
    mar = f2.multiselect("Marital status", mars, default=mars)
    seg = f3.multiselect("Segment", order, default=order)
    lo, hi = int(df.Income.min()), int(df.Income.max())
    ir = st.slider("Income range", lo, hi, (lo, hi), step=1000)
    v = df[df.Education.isin(edu) & df.Marital_Status.isin(mar) & df.Segment.isin(seg) & df.Income.between(*ir)]
    st.caption(f"Showing {len(v):,} of {len(df):,} customers (filtering = overload control)")
    if len(v) < 10:
        st.warning("Too few customers match these filters.")
        st.stop()

    st.markdown("#### A. Parallel coordinates (multi-dimensional data)")
    st.plotly_chart(px.parallel_coordinates(v, dimensions=FEATURES, color="Cluster",
                                            color_continuous_scale="Turbo"),
                    width="stretch")
    why("parallel coordinates", "Our data has 7 dimensions; a 2D scatter shows only two. Each customer becomes one line across all 7 axes. Gestalt: similar customers form bundles (similarity, continuity).",
        "Drag on any axis to brush a range; lines outside it fade. Bundles of same-coloured lines = a segment. Crossing lines between two axes = negative relation.")

    st.markdown("#### B. Brushing & linking")
    fig = px.scatter(v, x="Income", y="Total_Spending", color="Segment", custom_data=["ID"],
                     category_orders={"Segment": order}, opacity=0.7)
    fig.update_layout(dragmode="select")
    ev = st.plotly_chart(fig, on_select="rerun", selection_mode=("box", "lasso"), key="brush", width="stretch")
    ids = [p["customdata"][0] for p in ev.selection.points] if ev and ev.selection.points else []
    sel = v[v.ID.isin(ids)] if ids else v
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(sel["Segment"].value_counts().reindex(order).fillna(0).reset_index(), x="count", y="Segment",
                           orientation="h", title=f"Linked view: {'selected' if ids else 'all'} customers by segment"),
                    width="stretch")
    c2.plotly_chart(px.histogram(sel, x="Age", nbins=20, title="Linked view: age of the same customers"), width="stretch")
    why("brushing & linking", "One selection updates several views at once, so you can ask 'who are the high-income, low-spending customers?' and instantly see their segments and ages. Gestalt: proximity groups nearby dots.",
        "Use the box/lasso tool on the scatter. The two charts below recompute for only the selected points. Double-click the chart to clear.")

    st.markdown("#### C. Search & details-on-demand")
    cid = st.selectbox("Search customer ID (type to search)", v.ID.tolist())
    r = v[v.ID == cid].iloc[0]
    st.write(f"**Customer {cid}** — {r.Segment} · {r.Education} · {r.Marital_Status}")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Age", int(r.Age)); d2.metric("Income", f"{r.Income:,.0f}")
    d3.metric("Total spending", int(r.Total_Spending)); d4.metric("Recency (days)", int(r.Recency))

    st.markdown("#### D. Customer similarity network (graph view)")
    n1, n2 = st.columns(2)
    n = n1.slider("Customers sampled (nodes)", 80, 400, 250, step=10)
    nn = n2.slider("Nearest neighbours per customer", 1, 6, 3)
    sm = v.sample(min(n, len(v)), random_state=1)
    Xs = scaler.transform(sm[FEATURES])
    idx = NearestNeighbors(n_neighbors=nn + 1).fit(Xs).kneighbors(Xs)[1]
    G = nx.Graph()
    G.add_nodes_from(range(len(sm)))
    G.add_edges_from((i, j) for i, row in enumerate(idx) for j in row[1:])
    pos = nx.spring_layout(G, seed=7, k=0.35)
    ex, ey = [], []
    for a_, b_ in G.edges:
        ex += [pos[a_][0], pos[b_][0], None]; ey += [pos[a_][1], pos[b_][1], None]
    g = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=0.4, color="#bbb"), hoverinfo="skip", showlegend=False))
    for sname in order:
        m = (sm.Segment == sname).values
        ii = np.where(m)[0]
        g.add_trace(go.Scatter(x=[pos[i][0] for i in ii], y=[pos[i][1] for i in ii], mode="markers", name=sname,
                               marker=dict(size=5 + sm.Total_Spending.values[ii] / 150),
                               text=[f"ID {sm.ID.values[i]} · spend {sm.Total_Spending.values[i]} · income {sm.Income.values[i]:,.0f}" for i in ii]))
    g.update_layout(xaxis_visible=False, yaxis_visible=False, height=550)
    st.plotly_chart(g, width="stretch")
    deg = dict(G.degree())
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Vertices |V|", G.number_of_nodes()); s2.metric("Edges |E|", G.number_of_edges())
    s3.metric("Density", f"{nx.density(G):.3f}"); s4.metric("Components", nx.number_connected_components(G))
    s5.metric("Pendant / isolated", f"{sum(d == 1 for d in deg.values())} / {len(list(nx.isolates(G)))}")
    why("a node-link network", "Customers are vertices; an edge joins a customer to their most similar customers (nearest neighbours in the 7 scaled features). This turns tabular data into a graph. Gestalt: connectedness + proximity make communities visible; colour (similarity) shows segments agree with the graph.",
        "Node colour = segment, size = total spending. Tight same-colour clumps = homogeneous groups; nodes linking two colours = customers between segments. Hover a node for details; zoom/pan with the toolbar.")

    st.markdown("#### E. Treemap — visualization of groups (hierarchy)")
    st.plotly_chart(px.treemap(v.assign(n=1), path=["Segment", "Education"], values="n", color="Total_Spending",
                               color_continuous_scale="Viridis"), width="stretch")
    why("a treemap", "Shows groups inside groups (segment → education) with area = number of customers and colour = average spending, using all the space with no overlap. This is the 'visualization of groups' technique and uses aggregation to reduce overload.",
        "Bigger rectangle = more customers. Brighter colour = higher average spending. Click a rectangle to zoom into it.")

# ---------------- TAB 6: DESIGN & PIPELINE ----------------
with t6:
    st.subheader("How this project maps to the guidelines")
    st.markdown("""
**1 · Problem framing** — *Questions:* Which customer groups exist? What defines them? Which are most valuable and which are drifting away?
Dataset: ~2,200 customers × 7 numeric features (+ education, marital status). Tabular/multi-dimensional theme, converted to a **graph** by linking each customer to their *k* nearest neighbours.
Graph terms (live values in Lab → D): vertices = customers, edges = similarity links, degree = number of links, pendant = degree 1, isolated = degree 0, finite, undirected, density = edges / possible edges.

**2 · Gestalt principles used**

| Principle | Where |
|---|---|
| Proximity | PCA scatter and network: similar customers sit close |
| Similarity | Same colour = same segment in every chart |
| Connectedness | Edges in the network link similar customers |
| Continuity | Lines in parallel coordinates flow across axes |
| Closure | Segment clouds read as one group even without a border |

**Avoiding overload:** filters (education, marital, segment, income), aggregation (treemap, heatmap, bars), sampling + k-neighbour limit in the network, details only on hover/search.

**3 · Pipeline (reference model)**
`Raw CSV` → *cleaning + feature engineering* → `Data tables` (customers, segment profiles) → *scaling, K-Means, PCA, k-NN* → `Visual structures` (points, lines, nodes/edges, rectangles) → *Plotly views* → `Interactive views` (user filters/brushes feed back into the tables).

**4 · Visual mapping**

| Data attribute | Type | Visual variable | Why |
|---|---|---|---|
| Segment | Nominal | Colour hue | Hue is best for unordered categories |
| Total spending | Quantitative | Node size / position / colour lightness | Size and position are accurate for numbers |
| Income, Age | Quantitative | Position (axes) | Position is the most accurate visual channel |
| Cluster membership | Nominal | Spatial grouping + colour | Redundant encoding aids reading |
| Education → segment | Hierarchical | Treemap area nesting | Shows parts-of-whole in groups |

**5 · Interaction included:** zoom/pan, filtering, brushing-and-linking, details-on-demand, search, dynamic layout (change k, neighbours, sample size).

**6 · Domain technique:** parallel coordinates (multi-dimensional), node-link network (graph/cluster), treemap (groups).

**Evaluation (small usability test, 3–5 classmates):** give 3 tasks — "find the highest-spending segment", "find a customer with income above 70k and low spending", "which segment has most web visitors?" — record time and correctness, then report results on your last slide.
""")
