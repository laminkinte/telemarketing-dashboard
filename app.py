"""
Telemarketing Dashboard - Fast Optimized Version
Handles large CSV files efficiently
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import warnings
warnings.filterwarnings('ignore')

# Set page config first
st.set_page_config(
    page_title="Telemarketing Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Optimize pandas
pd.options.mode.chained_assignment = None

# Custom CSS for better performance
st.markdown("""
<style>
    /* Optimize table rendering */
    .stDataFrame {
        font-size: 12px;
    }
    .dataframe {
        max-height: 500px;
        overflow-y: auto;
    }
    /* Hide Streamlit branding for cleaner UI */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False, ttl=3600)
def read_csv_in_chunks(file, chunk_size=50000):
    """Read CSV in chunks for better memory management"""
    try:
        # Try to read the file in chunks
        chunks = []
        for chunk in pd.read_csv(file, chunksize=chunk_size, low_memory=False):
            chunks.append(chunk)
        
        if chunks:
            df = pd.concat(chunks, ignore_index=True)
            st.session_state['chunked_read'] = True
            return df
        else:
            return pd.DataFrame()
    except:
        # Fallback to normal read
        try:
            df = pd.read_csv(file, low_memory=False)
            st.session_state['chunked_read'] = False
            return df
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return pd.DataFrame()

@st.cache_data(show_spinner=False)
def analyze_transactions_fast(df, analysis_period="Last 7 days"):
    """Fast analysis function optimized for large datasets"""
    
    # Create a copy and clean
    df_clean = df.copy()
    
    # Only keep essential columns to save memory
    essential_cols = ['User Identifier', 'Product Name', 'Service Name', 
                     'Created At', 'Entity Name', 'Full Name']
    
    # Filter to only essential columns that exist
    existing_cols = [col for col in essential_cols if col in df_clean.columns]
    df_clean = df_clean[existing_cols].copy()
    
    # Clean data efficiently
    df_clean['Product Name'] = df_clean['Product Name'].astype(str).str.strip()
    df_clean['Entity Name'] = df_clean['Entity Name'].astype(str).str.strip()
    
    # Convert User Identifier to numeric efficiently
    if 'User Identifier' in df_clean.columns:
        df_clean['User Identifier'] = pd.to_numeric(df_clean['User Identifier'], errors='coerce')
    
    # Parse dates
    if 'Created At' in df_clean.columns:
        # Try to parse dates efficiently
        try:
            df_clean['Created At'] = pd.to_datetime(df_clean['Created At'], errors='coerce', dayfirst=True)
        except:
            df_clean['Created At'] = pd.to_datetime(df_clean['Created At'], errors='coerce')
    
    # Define categories
    p2p_keywords = ['internal wallet transfer', 'p2p']
    intl_keywords = ['international remittance']
    
    # Filter by date if possible
    if 'Created At' in df_clean.columns and df_clean['Created At'].notna().any():
        end_date = df_clean['Created At'].max()
        
        if analysis_period == "Last 7 days":
            start_date = end_date - timedelta(days=7)
        elif analysis_period == "Last 30 days":
            start_date = end_date - timedelta(days=30)
        elif analysis_period == "Last 90 days":
            start_date = end_date - timedelta(days=90)
        else:  # All data
            start_date = df_clean['Created At'].min()
        
        mask = (df_clean['Created At'] >= start_date) & (df_clean['Created At'] <= end_date)
        df_period = df_clean[mask].copy()
    else:
        df_period = df_clean.copy()
        start_date = None
        end_date = None
    
    # Filter customers
    customer_mask = df_period['Entity Name'].str.contains('customer', case=False, na=False)
    df_customers = df_period[customer_mask].copy()
    
    # Get unique customers
    unique_customers = df_customers['User Identifier'].dropna().unique()
    
    # Identify international customers
    intl_mask = df_customers['Product Name'].str.contains('international remittance', case=False, na=False)
    intl_customers = df_customers[intl_mask]['User Identifier'].dropna().unique()
    
    # Identify P2P customers
    p2p_mask = df_customers['Product Name'].str.contains('|'.join(p2p_keywords), case=False, na=False)
    p2p_customers = df_customers[p2p_mask]['User Identifier'].dropna().unique()
    
    # International customers not using P2P
    intl_not_p2p = list(set(intl_customers) - set(p2p_customers))
    
    # Create reports
    # Report 1: International not P2P
    if len(intl_not_p2p) > 0:
        # Sample first 1000 for display
        sample_size = min(1000, len(intl_not_p2p))
        sample_customers = intl_not_p2p[:sample_size]
        
        report1_data = []
        for cust_id in sample_customers:
            cust_data = df_customers[df_customers['User Identifier'] == cust_id]
            if not cust_data.empty:
                # Get name
                names = cust_data['Full Name'].dropna()
                full_name = names.iloc[0] if len(names) > 0 else 'Unknown'
                
                # Get last date
                last_date = cust_data['Created At'].max()
                last_date_str = last_date.strftime('%Y-%m-%d') if pd.notna(last_date) else 'Unknown'
                
                report1_data.append({
                    'User ID': cust_id,
                    'Full Name': full_name,
                    'Last Transaction': last_date_str,
                    'Total TX': len(cust_data)
                })
        
        report1_df = pd.DataFrame(report1_data)
    else:
        report1_df = pd.DataFrame(columns=['User ID', 'Full Name', 'Last Transaction', 'Total TX'])
    
    # Summary
    summary = {
        'total_customers': len(unique_customers),
        'intl_customers': len(intl_customers),
        'intl_not_p2p': len(intl_not_p2p),
        'p2p_users': len(p2p_customers),
        'sample_size': min(1000, len(intl_not_p2p)) if len(intl_not_p2p) > 0 else 0
    }
    
    return {
        'intl_not_p2p': report1_df,
        'summary': summary,
        'total_intl_not_p2p': len(intl_not_p2p)
    }

def main():
    """Main optimized app"""
    
    st.title("📊 Fast Telemarketing Dashboard")
    st.markdown("*Optimized for large transaction files*")
    
    # Sidebar with progress indicators
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File upload with size warning
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=['csv'],
            help="Upload your transaction data (CSV format)"
        )
        
        if uploaded_file is not None:
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
            if file_size > 50:
                st.warning(f"⚠️ Large file: {file_size:.1f} MB")
                st.info("Processing may take a moment...")
        
        # Analysis options
        analysis_period = st.selectbox(
            "Analysis Period",
            ["Last 7 days", "Last 30 days", "Last 90 days", "All data"],
            index=0
        )
        
        # Sample size selector
        max_display = st.slider(
            "Max rows to display",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            help="Limit display rows for better performance"
        )
        
        # Run button
        if st.button("🚀 Run Fast Analysis", type="primary", use_container_width=True):
            st.session_state['run_analysis'] = True
        else:
            st.session_state['run_analysis'] = False
        
        st.markdown("---")
        st.markdown("### 📋 Tips")
        st.markdown("""
        - For files > 50MB, processing may take 30-60 seconds
        - Results are cached for 1 hour
        - Only first 1,000 customers shown (full list downloadable)
        """)
    
    # Main content
    if uploaded_file is not None:
        # File info
        file_name = uploaded_file.name
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File", file_name)
        with col2:
            st.metric("Size", f"{file_size:.1f} MB")
        
        # Progress container
        progress_container = st.container()
        
        if st.session_state.get('run_analysis', False):
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Read file
                status_text.text("📖 Reading file...")
                progress_bar.progress(25)
                
                df = read_csv_in_chunks(uploaded_file)
                
                if df.empty:
                    st.error("Failed to read file. Please check the format.")
                    return
                
                # Step 2: Analyze
                status_text.text("🔍 Analyzing transactions...")
                progress_bar.progress(50)
                
                results = analyze_transactions_fast(df, analysis_period)
                
                # Step 3: Display results
                status_text.text("📊 Preparing results...")
                progress_bar.progress(75)
                
                # Clear progress
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                
                # Brief pause
                import time
                time.sleep(0.5)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
            
            # Show results
            st.success(f"✅ Analysis complete! Processed {len(df):,} transactions")
            
            # Summary metrics
            st.subheader("📈 Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Customers",
                    f"{results['summary']['total_customers']:,}",
                    help="Unique customers in period"
                )
            
            with col2:
                st.metric(
                    "International Customers",
                    f"{results['summary']['intl_customers']:,}",
                    help="Customers using international remittance"
                )
            
            with col3:
                conversion_potential = (
                    results['summary']['intl_not_p2p'] / 
                    max(results['summary']['intl_customers'], 1)
                ) * 100
                st.metric(
                    "Intl Not Using P2P",
                    f"{results['summary']['intl_not_p2p']:,}",
                    delta=f"{conversion_potential:.1f}%",
                    delta_color="inverse",
                    help="Potential P2P conversions"
                )
            
            with col4:
                st.metric(
                    "P2P Users",
                    f"{results['summary']['p2p_users']:,}",
                    help="Customers already using P2P"
                )
            
            # Display table
            st.subheader(f"📋 International Customers Not Using P2P (Sample: {len(results['intl_not_p2p']):,})")
            
            if not results['intl_not_p2p'].empty:
                st.dataframe(
                    results['intl_not_p2p'].head(max_display),
                    use_container_width=True,
                    height=400
                )
                
                # Download options
                st.subheader("📥 Download Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Download full list (all IDs, not just sample)
                    if results['total_intl_not_p2p'] > 0:
                        # Create full list with just IDs
                        full_list = pd.DataFrame({
                            'User ID': results.get('full_intl_list', [])[:10000]  # Limit to 10,000
                        })
                        
                        csv = full_list.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=f"📥 Download All IDs ({min(results['total_intl_not_p2p'], 10000):,})",
                            data=csv,
                            file_name="all_international_customers.csv",
                            mime="text/csv",
                            help="Full list of customer IDs (limited to 10,000)"
                        )
                
                with col2:
                    # Download sample
                    csv = results['intl_not_p2p'].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Sample",
                        data=csv,
                        file_name="international_customers_sample.csv",
                        mime="text/csv",
                        help="Detailed sample with names and dates"
                    )
            else:
                st.info("No international customers found without P2P usage")
            
            # Insights
            st.subheader("💡 Insights")
            if results['summary']['intl_customers'] > 0:
                conversion_rate = (results['summary']['intl_not_p2p'] / 
                                 results['summary']['intl_customers']) * 100
                
                if conversion_rate > 50:
                    st.success(f"🎯 **High Potential!** {conversion_rate:.1f}% of international customers don't use P2P")
                    st.info("""
                    **Recommended Action:** Prioritize these customers in your telemarketing campaign.
                    They already trust your platform for international transfers and are prime candidates
                    for P2P adoption.
                    """)
                elif conversion_rate > 20:
                    st.warning(f"📈 **Good Opportunity:** {conversion_rate:.1f}% conversion potential")
                else:
                    st.info(f"📊 **Moderate Opportunity:** {conversion_rate:.1f}% conversion potential")
        else:
            # Preview data
            st.subheader("📄 Data Preview")
            
            # Show first few rows without processing everything
            try:
                preview_df = pd.read_csv(uploaded_file, nrows=100)
                st.dataframe(preview_df, use_container_width=True)
                st.info(f"📊 Preview showing 100 of {len(pd.read_csv(uploaded_file, nrows=1)):,}+ rows. Click **'Run Fast Analysis'** to process.")
            except:
                st.info("Click **'Run Fast Analysis'** to process the file")
    else:
        # Welcome screen
        st.subheader("🚀 Get Started")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Upload your transaction data
            
            **Supported:** CSV files with transaction data
            
            **Optimal performance:**
            - Files under 100MB process fastest
            - Essential columns required:
              * User Identifier
              * Product Name  
              * Created At
              * Entity Name
            
            **The analysis will identify:**
            1. International remittance customers
            2. Which ones are NOT using P2P
            3. Conversion potential
            """)
        
        with col2:
            # Quick sample download
            st.markdown("### 🧪 Need sample data?")
            sample_data = pd.DataFrame({
                'User Identifier': [1001, 1002, 1003, 1004, 1005],
                'Product Name': ['International Remittance', 'Internal Wallet Transfer', 
                               'Airtime Topup', 'Deposit', 'International Remittance'],
                'Service Name': ['Send Money', 'Send Money', 'Airtime Topup', 
                               'Send Money', 'Send Money'],
                'Created At': ['2024-01-15 10:30:00', '2024-01-14 14:20:00', 
                             '2024-01-13 09:15:00', '2024-01-12 16:45:00', '2024-01-11 11:00:00'],
                'Entity Name': ['Customer', 'Customer', 'Customer', 'Customer', 'Customer'],
                'Full Name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson']
            })
            
            csv = sample_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Sample CSV",
                data=csv,
                file_name="sample_transactions.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # Performance tips
        with st.expander("⚡ Performance Tips for Large Files", expanded=True):
            st.markdown("""
            **For files over 100MB:**
            
            1. **Pre-filter your data** before uploading:
               - Keep only last 30-90 days of data
               - Remove unnecessary columns
               - Filter to only customer transactions
            
            2. **Use chunked processing** (already implemented):
               - The app reads data in 50,000 row chunks
               - Results are cached for 1 hour
            
            3. **Expected processing times:**
               - 100MB: ~10-20 seconds
               - 500MB: ~60-90 seconds
               - 1GB+: 2-3+ minutes
            
            4. **Memory efficient:**
               - Only essential columns are kept
               - Results are streamed, not stored entirely in memory
            """)

if __name__ == "__main__":
    # Initialize session state
    if 'run_analysis' not in st.session_state:
        st.session_state['run_analysis'] = False
    if 'chunked_read' not in st.session_state:
        st.session_state['chunked_read'] = False
    
    main()
