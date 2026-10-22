import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import pandas as pd

# 1. Page Configuration & Professional Styling
st.set_page_config(
    page_title="Global Intelligence | News Topic Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, enterprise look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #1a1c24;
        border: 1px solid #2e3440;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .highlight-text {
        color: #4f8bf9;
        font-weight: 600;
    }
    .stTextArea textarea {
        background-color: #161922;
        border: 1px solid #333a4d;
        color: #f1f2f6;
        border-radius: 8px;
        font-size: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Complete Reuters 46 Topic Mapping (Human Readable Industry Labels)
# 2. Complete Reuters 46 Topic Mapping (Exact Keras Dataset Indexed)
REUTERS_LABELS = {
    0: "Cocoa & Commodities",
    1: "Grain & Agriculture",
    2: "Vegetable Oil (veg-oil)",
    3: "Earnings & Financial Reports (earn)",
    4: "Mergers & Acquisitions (acq)",
    5: "Wheat & Grain Supplies",
    6: "Copper & Base Metals",
    7: "Housing & Construction",
    8: "Money Supply (dlr)",
    9: "Coffee Markets",
    10: "Shipping & Maritime",
    11: "Sugar Trade",
    12: "Trade & Tariffs",
    13: "National Reserves",
    14: "Crude Petroleum",
    15: "Industrial Production Index (ipi)",
    16: "Interest Rates & Central Banks",
    17: "GNP / GDP Economic Growth",
    18: "Strategic Metals",
    19: "Foreign Exchange (money-fx)",
    20: "Natural Gas (gas)",
    21: "Livestock & Cattle",
    22: "Retail Sales Growth",
    23: "Consumer Price Index (cpi)",
    24: "Jobs & Labor Market",
    25: "Wholesale Price Inflation (wpi)",
    26: "Tin Mining",
    27: "Lead Commodities",
    28: "Zinc Metallurgy",
    29: "Rubber Trade",
    30: "Aluminium Metallurgy",
    31: "Petrochemicals",
    32: "Fuel & Heat Energy",
    33: "Gas Infrastructure",
    34: "Commercial Loans",
    35: "Soya & Oilseed Products",
    36: "Cattle Inventories",
    37: "Corn Harvests",
    38: "Iron & Steel Production",
    39: "Platinum & Precious Metals",
    40: "Strategic Oil Reserves",
    41: "Institutional Reserves",
    42: "National Production Indices",
    43: "National Debt Instruments",
    44: "Domestic Inventories",
    45: "Sunflower Oil Byproducts"
}

# 3. Model & Tokenizer Loader
@st.cache_resource
def load_deep_learning_artifacts():
    model = tf.keras.models.load_model("reuters_rnn_model.h5")
    with open("reuters_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

try:
    model, tokenizer = load_deep_learning_artifacts()
except Exception as e:
    st.error("⚠️ Model Artifacts Not Detected. Please make sure `reuters_rnn_model.h5` and `reuters_tokenizer.pkl` exist in the root directory.")
    st.stop()

# 4. Sidebar Panel Configuration
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/news.png", width=70)
    st.title("Settings & Insights")
    st.markdown("---")
    
    st.markdown("### ⚙️ Engine Parameters")
    st.markdown("- **Model:** `Bidirectional LSTM + MaxPool`")
    st.markdown("- **Input Sequence Limit:** `250 Tokens`")
    st.markdown("- **Vocabulary Capacity:** `10,000 Words`")
    st.markdown("- **Output Space:** `46 Industrial Verticals`")
    
    st.markdown("---")
    st.markdown("### 🧪 Quick Presets")
    sample_choice = st.selectbox(
        "Load Production Sample:",
        [
            "Select a pre-loaded headline...",
            "Federal Reserve Cuts Key Lending Rate",
            "ExxonMobil Expands Offshore Drilling Operations",
            "Wheat Harvest Falls Due to Severe Drought",
            "Tech Giant In Final Talks to Acquire AI Startup"
        ]
    )

    preset_texts = {
        "Federal Reserve Cuts Key Lending Rate": "The Federal Reserve voted unanimously to reduce interest rates by 25 basis points today, pointing to steady economic growth and moderated inflation indices across commercial banking sectors.",
        "ExxonMobil Expands Offshore Drilling Operations": "Crude oil output reached new monthly peaks as major energy consortiums expanded deepwater drilling rigs and pipeline transit terminals following increased global fuel demand.",
        "Wheat Harvest Falls Due to Severe Drought": "Global grain supplies faced fresh supply bottlenecks as prolonged regional drought slashed winter wheat harvest projections, triggering spikes in agricultural futures trading.",
        "Tech Giant In Final Talks to Acquire AI Startup": "The multinational enterprise confirmed it is closing an agreement to acquire the machine learning venture for a cash valuation of 2.4 billion dollars, expanding its enterprise cloud footprint."
    }

# 5. Dashboard Header
st.title("🌐 Newsroom Intelligence & Multi-Class Topic Routing")
st.markdown("Empowering automated media classification through deep sequential context models. Submit real-time wire dispatches for topic inference.")
st.markdown("---")

# 6. Main Workspace Layout
col_left, col_right = st.columns([1.1, 0.9], gap="large")

with col_left:
    st.subheader("📝 Dispatch Input Console")
    
    # Auto-fill preset or use blank
    default_text = preset_texts.get(sample_choice, "")
    user_input = st.text_area(
        "Paste News Narrative or Editorial Body:",
        value=default_text,
        height=220,
        placeholder="Type or paste real-time news reporting here..."
    )

    # Real-time Text Stats Bar
    word_count = len(user_input.split())
    char_count = len(user_input)
    st.caption(f"📊 **Metrics:** `{word_count}` Words | `{char_count}` Characters | Engine Max Limit: `250 Tokens`")
    
    col_btn, _ = st.columns([0.4, 0.6])
    with col_btn:
        submit_btn = st.button("🚀 Analyze & Route Dispatch", use_container_width=True)

with col_right:
    st.subheader("🎯 Classification Results")
    
    if submit_btn:
        if user_input.strip() == "":
            st.warning("Input buffer empty. Please provide article content to initialize classification.")
        else:
            with st.spinner("Processing semantics through neural embedding space..."):
                # Data Pipeline Vectorization
                sequence = tokenizer.texts_to_sequences([user_input])
                padded_tensor = pad_sequences(sequence, maxlen=250)
                
                # Inference
                prediction_probs = model.predict(padded_tensor, verbose=0)[0]
                pred_label = int(np.argmax(prediction_probs))
                confidence_score = float(prediction_probs[pred_label]) * 100
                category_name = REUTERS_LABELS.get(pred_label, f"Unknown (ID: {pred_label})")

            # Metrics Cards
            st.markdown(f"""
            <div class="metric-card">
                <span style="color: #8b92a2; font-size: 13px; text-transform: uppercase; letter-spacing: 1px;">Top Identified Sector</span>
                <h3 style="color: #2ed573; margin-top: 5px; margin-bottom: 0;">{category_name}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric(label="Inference Confidence", value=f"{confidence_score:.2f}%")
            with m_col2:
                status_eval = "High Reliability" if confidence_score > 60 else "Review Recommended"
                st.metric(label="Model Status", value=status_eval)

            # Analytics Tab Section
            st.markdown("#### 📈 Posterior Probability Distribution")
            
            # Extract Top 5 Predictions
            top_indices = np.argsort(prediction_probs)[-5:][::-1]
            top_data = [
                {
                    "Sector Category": REUTERS_LABELS.get(idx, f"ID #{idx}"),
                    "Confidence Probability (%)": float(prediction_probs[idx] * 100)
                }
                for idx in top_indices
            ]
            df_chart = pd.DataFrame(top_data)
            
            # Interactive Bar Chart
            st.bar_chart(
                data=df_chart.set_index("Sector Category"),
                use_container_width=True
            )
            
            # Expandable Breakdown Table
            with st.expander("🔍 View Raw Probability Ledger"):
                st.dataframe(df_chart, use_container_width=True, hide_index=True)
    else:
        st.info("Awaiting dispatch input. Provide text on the left console or select a preset from the sidebar to view inference outputs.")