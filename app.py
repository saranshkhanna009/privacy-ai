import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from models.classical_statistics import CopulaGenerativeModel
from core.base_model import BaseGenerativeModel
from models.deep_generative import DeepGenerativeModel
from validation.privacy_auditor import PrivacyAuditor
from validation.utility_auditor import UtilityAuditor

st.set_page_config(page_title="Privacy AI Dashboard", layout="wide")

st.title("🔒 Privacy-Preserving AI Framework")
st.subheader("Generate HIPAA-compliant, synthetic tabular records using ML & Deep Learning")

# --- SIDEBAR CONFIGURATIONS ---
st.sidebar.header("Pipeline Configuration")
model_choice = st.sidebar.selectbox("Select Generative Engine", ["Statistical Copula", "PyTorch Autoencoder"])
num_records_to_gen = st.sidebar.slider("Records to Generate", 100, 2000, 500, step=100)

if model_choice == "PyTorch Autoencoder":
    epochs = st.sidebar.slider("Training Epochs", 10, 200, 60)
    lr = st.sidebar.slider("Learning Rate", 0.001, 0.05, 0.01, step=0.001)
    config = {"latent_dim": 2, "epochs": epochs, "batch_size": 32, "lr": lr}
else:
    config = {}

# --- SAMPLE GENERATION FOR LACK OF UPLOAD ---
uploaded_file = st.sidebar.file_uploader("Upload Sensitive CSV File", type=["csv"])

if uploaded_file is not None:
    real_df = pd.read_csv(uploaded_file)
else:
    np.random.seed(42)
    real_df = pd.DataFrame({
        'Age': np.random.randint(20, 80, size=1000),
        'Blood_Pressure': np.random.normal(120, 15, size=1000),
        'Cholesterol': np.random.normal(200, 30, size=1000)
    })

# --- DATA VIEW LAYOUT ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Original Sensitive Data (Restricted)")
    st.dataframe(real_df.head(10), use_container_width=True)

# --- RUN COMPUTATION TRIGGER ---
if st.sidebar.button("🚀 Run Generative Pipeline & Audit"):
    
    with st.spinner(f"Executing {model_choice} model algorithms..."):
        if model_choice == "Statistical Copula":
            generator = CopulaGenerativeModel(config=config)
        else:
            generator = DeepGenerativeModel(config=config)
            
        generator.fit(real_df)
        synth_df = generator.generate(num_records_to_gen)
        
        # Auditing Execution Block
        auditor = PrivacyAuditor()
        audit_results = auditor.audit_leakage(real_df, synth_df)
        
        utility_auditor = UtilityAuditor()
        corr_diff = utility_auditor.calculate_correlation_similarity(real_df, synth_df)
        
        # Fallback index fallback check for custom files
        target_column = "Cholesterol" if "Cholesterol" in real_df.columns else real_df.columns[-1]
        tstr_score = utility_auditor.run_tstr_test(real_df, synth_df, target_col=target_column)

    with col2:
        st.markdown("### 🧬 Generated Synthetic Data (Safe)")
        st.dataframe(synth_df.head(10), use_container_width=True)
        
        csv_bytes = synth_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Safe Synthetic CSV",
            data=csv_bytes,
            file_name="synthetic_safe_data.csv",
            mime="text/csv"
        )

    st.markdown("---")
    st.markdown("<h2>🔍 Security Compliance & Verification Report</h2>", unsafe_allow_html=True)
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        status_color = "Green" if audit_results["passed_audit"] else "Red"
        st.markdown(f"**Audit Status:** <span style='color:{status_color}; font-size:20px; font-weight:bold;'>{'PASSED' if audit_results['passed_audit'] else 'FAILED'}</span>", unsafe_allow_html=True)
    with m2:
        st.metric(label="Exact Duplicates Leaked", value=audit_results["exact_duplicates_found"])
    with m3:
        st.metric(label="Min Distance to Real Record", value=f"{audit_results['min_distance_to_real']:.4f}")
    with m4:
        st.metric(label="Average Proximity (DCR)", value=f"{audit_results['mean_distance_to_real']:.2f}")

    st.markdown("---")
    st.markdown("### 📊 Data Utility & Predictive Quality Report")
    ut1, ut2 = st.columns(2)
    with ut1:
        st.metric(label="Correlation Matrix Deviation (Lower is Better)", value=f"{corr_diff:.4f}")
    with ut2:
        st.metric(label="TSTR Score (Train on Synthetic, Test on Real R²)", value=f"{tstr_score:.2f}")

    # --- PLOT CHART VIA CLEAN SHALLOW CLONES ---
    st.markdown("### 📈 Distribution Overlaps (Real vs Synthetic)")
    plot_real = real_df.copy()
    plot_synth = synth_df.copy()

    plot_real['Dataset'] = 'Real (Sensitive)'
    plot_synth['Dataset'] = 'Synthetic (Safe)'
    combined_df = pd.concat([plot_real, plot_synth])
    
    plot_feature = "Blood_Pressure" if "Blood_Pressure" in real_df.columns else real_df.columns[0]
    fig = px.histogram(combined_df, x=plot_feature, color="Dataset", barmode="overlay", marginal="box", title=f"{plot_feature.replace('_', ' ')} Overlay Analysis")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("💡 Adjust your pipeline settings in the left sidebar and click 'Run Generative Pipeline' to spin up your engine components.")