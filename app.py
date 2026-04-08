import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from llm_connectors import build_connectors
from analysis import count_brand_mentions, sentiment_score, extract_themes, classify_tone

load_dotenv()

st.set_page_config(page_title="LLM Brand Presence Analyzer", layout="wide")

st.title("LLM Brand Presence Analyzer")
st.caption("Benchmark brand salience, narratives, and risk framing across multiple LLMs. Analysis is deterministic (no LLM-based scoring).")

with st.sidebar:
    st.header("Analysis Settings")

    brand = st.text_input("Brand name", value="Lloyds Banking Group")
competitors = st.multiselect(
    "Competitors",
    options=[
        "HSBC",
        "Nationwide",
        "Santander",
        "Monzo",
        "Revolut",
        "NatWest",
        "Barclays"
    ],
    default=[
        "HSBC",
        "Nationwide",
        "Santander",
        "Monzo",
        "Revolut"
    ]
)

competitors = ", ".join(competitors)

    category = st.text_input("Category", value="UK banks")
    topic = st.text_input("Topic", value="sustainability")

    st.divider()
    st.subheader("Query types")
    query_flags = {
        "unprompted_recall": st.checkbox("Unprompted recall", value=True),
        "comparative": st.checkbox("Comparative analysis", value=True),
        "brand_strengths": st.checkbox("Brand strengths", value=True),
        "risk_criticisms": st.checkbox("Risks & criticisms", value=True),
        "purchase_consideration": st.checkbox("Consideration drivers", value=False),
    }

    st.divider()
    st.subheader("Live API keys (optional)")
    st.caption("Leave blank to run in Offline mode.")

    openai_key = st.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    anthropic_key = st.text_input("ANTHROPIC_API_KEY", type="password", value=os.getenv("ANTHROPIC_API_KEY", ""))
    google_key = st.text_input("GOOGLE_API_KEY", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
    mistral_key = st.text_input("MISTRAL_API_KEY", type="password", value=os.getenv("MISTRAL_API_KEY", ""))

    st.subheader("Models (optional)")
    openai_model = st.text_input("OPENAI_MODEL", value=os.getenv("OPENAI_MODEL", "gpt-4.1"))
    anthropic_model = st.text_input("ANTHROPIC_MODEL", value=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"))
    gemini_model = st.text_input("GEMINI_MODEL", value=os.getenv("GEMINI_MODEL", "gemini-1.5-pro"))
    mistral_model = st.text_input("MISTRAL_MODEL", value=os.getenv("MISTRAL_MODEL", "mistral-large-latest"))

    run = st.button("Run analysis", type="primary")

BASE_INSTRUCTION = (
    "You are an analyst. Answer in 150–200 words. "
    "Use neutral, evidence-aware language. Avoid marketing copy. "
    "If you are unsure, say so. Include both positives and criticisms where relevant."
)

TEMPLATES = {
    "unprompted_recall": "Which {category} organisations are leaders in {topic}?",
    "comparative": "Compare {brand} with {competitors} on {topic}. Provide key differences.",
    "brand_strengths": "What is {brand} best known for in the context of {topic}?",
    "risk_criticisms": "What criticisms or risks are associated with {brand} regarding {topic}?",
    "purchase_consideration": "If someone is choosing between {brand} and {competitors}, what factors might influence the decision?",
}

LABELS = {
    "unprompted_recall": "Unprompted recall",
    "comparative": "Comparative analysis",
    "brand_strengths": "Brand strengths",
    "risk_criticisms": "Risks & criticisms",
    "purchase_consideration": "Consideration drivers",
}


def build_prompt(qid: str) -> str:
    body = TEMPLATES[qid].format(
        brand=brand.strip(),
        competitors=competitors.strip(),
        category=category.strip(),
        topic=topic.strip(),
    )

    prompt = BASE_INSTRUCTION + "\n\n" + body
    return prompt


def get_connectors():
    keys = {
        "OPENAI_API_KEY": openai_key.strip() or None,
        "ANTHROPIC_API_KEY": anthropic_key.strip() or None,
        "GOOGLE_API_KEY": google_key.strip() or None,
        "MISTRAL_API_KEY": mistral_key.strip() or None,
    }
    models = {
        "OPENAI_MODEL": openai_model.strip() or None,
        "ANTHROPIC_MODEL": anthropic_model.strip() or None,
        "GEMINI_MODEL": gemini_model.strip() or None,
        "MISTRAL_MODEL": mistral_model.strip() or None,
    }
    return build_connectors(keys=keys, models=models, include_offline=True)


tab1, tab2, tab3 = st.tabs(["Dashboard", "Detailed Results", "Export"])

if run:
    connectors = get_connectors()
    selected_queries = [qid for qid, enabled in query_flags.items() if enabled]

    if not selected_queries:
        st.warning("Select at least one query type in the sidebar.")
        st.stop()

    brand_list = [brand.strip()] + [c.strip() for c in competitors.split(",") if c.strip()]

    rows = []
    with st.spinner("Querying models..."):
        for model_label, connector in connectors.items():
            for qid in selected_queries:
                prompt = build_prompt(qid)
                try:
                    response = connector.generate(prompt)
                    error = ""
                except Exception as e:
                    response = ""
                    error = str(e)

                row = {
                    "model": model_label,
                    "query_id": qid,
                    "query_label": LABELS.get(qid, qid),
                    "prompt": prompt,
                    "response": response,
                    "error": error,
                }

                for b in brand_list:
                    row[f"mentions::{b}"] = count_brand_mentions(response, b) if response else 0

                s = sentiment_score(response) if response else 0.0
                row["sentiment_score"] = s
                row["tone"] = classify_tone(s)
                rows.append(row)

    df = pd.DataFrame(rows)
    st.session_state["results_df"] = df

    corpus = [r for r in df["response"].tolist() if isinstance(r, str) and r.strip()]
    st.session_state["themes"] = extract_themes(corpus, top_k=12) if corpus else []

    st.success("Done.")

if "results_df" in st.session_state:
    df = st.session_state["results_df"]
    themes = st.session_state.get("themes", [])

    with tab1:
        st.subheader("Brand Presence Overview")
        brand_list = [brand.strip()] + [c.strip() for c in competitors.split(",") if c.strip()]
        mention_cols = [f"mentions::{b}" for b in brand_list]

        overview = df.groupby("model")[mention_cols].sum().reset_index()
        st.dataframe(overview, use_container_width=True)

        colA, colB = st.columns([1, 1])
        with colA:
            st.subheader("Tone distribution")
            tone_counts = df["tone"].value_counts().reset_index()
            tone_counts.columns = ["tone", "count"]
            st.bar_chart(tone_counts.set_index("tone"))

        with colB:
            st.subheader("Top themes (corpus-wide)")
            if themes:
                st.write(", ".join([f"{t} ({round(s,2)})" for t, s in themes]))
            else:
                st.caption("No themes available.")

        err = df[df["error"].astype(bool)]
        if len(err):
            st.warning("Some model calls failed. Check Detailed Results for error messages.")

    with tab2:
        st.subheader("Model-by-model responses")
        for (m, q), block in df.groupby(["model", "query_label"], sort=False):
            with st.expander(f"{m} — {q}"):
                bcols = [c for c in block.columns if c.startswith("mentions::")]
                st.write("**Mention counts:**")
                st.json({c.replace("mentions::", ""): int(block.iloc[0][c]) for c in bcols})
                st.write("**Tone:**", block.iloc[0]["tone"], f"(score {block.iloc[0]['sentiment_score']:.2f})")
                if block.iloc[0]["error"]:
                    st.error(block.iloc[0]["error"])
                st.write("**Response:**")
                st.write(block.iloc[0]["response"] or "(empty)")
                st.write("**Prompt used:**")
                st.code(block.iloc[0]["prompt"], language="text")

    with tab3:
        st.subheader("Export")
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download results CSV", data=csv, file_name="llm_brand_presence_results.csv", mime="text/csv")

else:
    with tab1:
        st.info("Set your inputs in the sidebar and click **Run analysis**. If you don't add keys, the tool will run in **Offline** mode.")
