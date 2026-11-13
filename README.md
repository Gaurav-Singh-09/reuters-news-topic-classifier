# Reuters News Topic Classifier

Deep learning text classifier using Bidirectional LSTM and Streamlit on Reuters-21578 dataset.


# 📰 Global Intelligence — News Topic Classifier

An interactive **Streamlit** application powered by a **TensorFlow / Keras Bidirectional LSTM** neural network trained on the Reuters newswire dataset to classify and route real-time news dispatches across **46 industrial and financial verticals**.

---

## ✨ Key Features

- **Real-Time Dispatch Routing:** Paste any news article, headline, or editorial narrative to immediately identify its primary sector category.
- **46 Reuters Industrial Verticals:** Maps raw class indices to human-readable industry labels (e.g., *Interest Rates & Central Banks*, *Crude Petroleum*, *Mergers & Acquisitions*, *Grain & Agriculture*).
- **Interactive Probability Analytics:** Displays top-1 confidence scores, reliability status indicators (`> 60%` High Reliability threshold), a Top-5 posterior probability bar chart, and an expandable raw probability ledger.
- **One-Click Production Presets:** Test model behavior immediately using built-in sidebar scenarios across monetary policy, energy, agriculture, and tech M&A.
- **Live Input Telemetry:** Tracks real-time word and character counts against the model's `250`-token sequence window.

---

## 🛠️ Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **Deep Learning:** [TensorFlow](https://www.tensorflow.org/) / Keras (`Bidirectional LSTM + MaxPool`)
- **Data Processing:** [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/)
- **Serialization:** Python `pickle` & Keras HDF5 (`.h5`)

---

## 📁 Project Structure

```text
├── app.py                     # Main Streamlit application
├── reuters_rnn_model.h5       # Trained Keras Bidirectional LSTM model
├── reuters_tokenizer.pkl      # Fitted Keras text tokenizer (10,000-word vocabulary)
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
