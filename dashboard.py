"""Streamlit dashboard for Neural Flight School. Run: streamlit run dashboard.py"""
import streamlit as st
from attention_lab import run
from pathlib import Path
st.set_page_config(page_title="Neural Flight School",layout="wide")
st.title("Neural Flight School")
st.info("This educational transformer has random, untrained weights. The attention maps demonstrate tensor mechanics only.")
text=st.text_area("Toy input text","attention is a computation")
seed=st.number_input("Random seed",min_value=0,max_value=100000,value=7)
device=st.selectbox("Device",["auto","cuda","mps","cpu"])
if st.button("Build attention maps",type="primary"):
    with st.spinner("Computing attention tensors…"):
        report=run(text,seed=int(seed),device=device)
    st.json({"device":report["device"],"token_count":report["token_count"],"logits_shape":report["logits_shape"]})
    cols=st.columns(min(2,len(report["layers"])))
    for i,layer in enumerate(report["layers"]):
        with cols[i%len(cols)]:
            st.subheader(f"Layer {layer['layer']}")
            st.image(str(Path("outputs/attention-lab")/f"attention-layer-{layer['layer']}.png"),use_container_width=True)
            st.caption(f"Mean attention entropy: {layer['mean_attention_entropy']:.3f}")
