"""
Telemarketing Dashboard with Excel Sheet-wise Export
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import warnings
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

warnings.filterwarnings('ignore')

# Try to import analysis functions
try:
    from utils.analysis import (
        analyze_telemarketing_data,
        generate_excel_report,
        create_detailed_reports
    )
    ANALYSIS_AVAILABLE = True
except ImportError as e:
    st.warning(f"Analysis module import error: {e}. Using simplified version.")
    ANALYSIS_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Telemarketing Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #3B82F6;
        color: white;
        font-weight: bold;
    }
    .success-message {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def display_summary_metrics(summary):
    """Display summary metrics"""
    if not summary:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Total Customers",
            value=f"{summary.get('total_customers', 0):,}",
            help="Unique customers in analysis period"
        )
    
    with col2:
        st.metric(
            label="🌍 International Customers",
            value=f"{summary.get('intl_customers', 0):,}",
            help="Customers using international remittance"
        )
    
    with col3:
        intl_not_p2p = summary.get('intl_not_p2p', 0)
        intl_customers = summary.get('intl_customers', 1)
        conversion_rate = (intl_not_p2p / max(intl_customers, 1)) * 100
        st.metric(
            label="🎯 Intl Not Using P2P",
            value=f"{intl_not_p2p:,}",
            delta=f"{conversion_rate:.1f}%",
            delta_color="inverse",
            help="International customers not using P2P (conversion opportunity)"
        )
    
    with col4:
        st.metric(
            label="🏠 Domestic Not Using P2P",
            value=f"{summary.get('domestic_not_p2p', 0):,}",
            help="Domestic customers using other services but not P2P"
        )

def display_reports_in_tabs(results):
    """Display reports in tabs"""
    if not results:
        return
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌍 International Customers",
        "🏠 Domestic Customers", 
        "📊 Summary",
        "📈 Impact Template"
    ])
    
    with tab1:
        if 'intl_not_p2p' in results and not results['intl_not_p2p'].empty:
            st.subheader("International Remittance Customers Not Using P2P")
            st.dataframe(results['intl_not_p2p'], use_container_width=True, height=400)
            st.caption(f"Showing {len(results['intl_not_p2p'])} customers")
        else:
            st.info("No international customers found without P2P usage")
    
    with tab2:
        if 'domestic_other_not_p2p' in results and not results['domestic_other_not_p2p'].empty:
            st.subheader("Domestic Customers Using Other Services But Not P2P")
            st.dataframe(results['domestic_other_not_p2p'], use_container_width=True, height=400)
            st.caption(f"Showing {len(results['domestic_other_not_p2p'])} customers")
        else:
            st.info("No domestic customers found using other services without P2P")
    
    with tab3:
        if 'summary' in results and results['summary']:
            st.subheader("Analysis Summary")
            summary_df = pd.DataFrame({
                'Metric': list(results['summary'].keys()),
                'Value': list(results['summary'].values())
            })
            st.dataframe(summary_df, use_container_width=True, height=400)
            
            # Insights
            st.subheader("💡 Insights")
            intl_customers = results['summary'].get('intl_customers', 0)
            intl_not_p2p = results['summary'].get('intl_not_p2p', 0)
            
            if intl_customers > 0:
                conversion_potential = (intl_not_p2p / intl_customers) * 100
                if conversion_potential > 50:
                    st.success(f"**High Conversion Potential!** {conversion_potential:.1f}% of international customers don't use P2P")
                elif conversion_potential > 25:
                    st.warning(f"**Good Opportunity:** {conversion_potential:.1f}% conversion potential")
                else:
                    st.info(f"**Moderate Opportunity:** {conversion_potential:.1f}% conversion potential")
    
    with tab4:
        if 'impact_template' in results and not results['impact_template'].empty:
            st.subheader("Daily Impact Tracking Template")
            st.info("Use this template to track daily calling results and P2P adoption")
            st.dataframe(results['impact_template'], use_container_width=True, height=400)

def create_excel_download_button(results):
    """Create Excel download button with sheet-wise output"""
    
    if not results:
        return
    
    st.markdown("---")
    st.subheader("📥 Download Complete Excel Report")
    
    # Create two columns for download options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Full Analysis Report")
        st.markdown("""
        Includes all reports in separate sheets:
        - International Customers Not Using P2P
        - Domestic Customers Not Using P2P  
        - Detailed Analysis
        - Summary Statistics
        - Daily Impact Template
        """)
        
        if st.button("⬇️ Download Full Excel Report", key="full_excel", use_container_width=True):
            try:
                if ANALYSIS_AVAILABLE:
                    # Generate Excel with all sheets
                    excel_data = generate_excel_report(results)
                else:
                    # Fallback: Create Excel manually
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        # Sheet 1: International Customers
                        if 'intl_not_p2p' in results and not results['intl_not_p2p'].empty:
                            results['intl_not_p2p'].to_excel(
                                writer, 
                                sheet_name='International_Not_P2P', 
                                index=False
                            )
                        
                        # Sheet 2: Domestic Customers
                        if 'domestic_other_not_p2p' in results and not results['domestic_other_not_p2p'].empty:
                            results['domestic_other_not_p2p'].to_excel(
                                writer, 
                                sheet_name='Domestic_Not_P2P', 
                                index=False
                            )
                        
                        # Sheet 3: Summary
                        if 'summary' in results:
                            summary_df = pd.DataFrame({
                                'Metric': list(results['summary'].keys()),
                                'Value': list(results['summary'].values())
                            })
                            summary_df.to_excel(
                                writer, 
                                sheet_name='Summary', 
                                index=False
                            )
                        
                        # Sheet 4: Impact Template
                        if 'impact_template' in results:
                            results['impact_template'].to_excel(
                                writer, 
                                sheet_name='Impact_Template', 
                                index=False
                            )
                        
                        # Sheet 5: Raw Data Sample
                        if 'sample_data' in results:
                            results['sample_data'].to_excel(
                                writer, 
                                sheet_name='Data_Sample', 
                                index=False
                            )
                    
                    excel_data = output.getvalue()
                
                # Create download button
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="📥 Click to Download Excel File",
                    data=excel_data,
                    file_name=f"Telemarketing_Analysis_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_excel"
                )
                
            except Exception as e:
                st.error(f"Error creating Excel file: {str(e)}")
    
    with col2:
        st.markdown("### 📋 Individual CSV Downloads")
        
        col2a, col2b = st.columns(2)
        
        with col2a:
            if 'intl_not_p2p' in results and not results['intl_not_p2p'].empty:
                csv_intl = results['intl_not_p2p'].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="🌍 International CSV",
                    data=csv_intl,
                    file_name="international_customers.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2b:
            if 'domestic_other_not_p2p' in results and not results['domestic_other_not_p2p'].empty:
                csv_dom = results['domestic_other_not_p2p'].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="🏠 Domestic CSV",
                    data=csv_dom,
                    file_name="domestic_customers.csv",
                    mime="text/csv",
                    use_container_width=True
                )

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Telemarketing Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("Analyze transaction data to identify P2P conversion opportunities and track telemarketing campaign effectiveness.")
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'df_uploaded' not in st.session_state:
        st.session_state.df_uploaded = None
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Transaction CSV",
            type=['csv'],
            help="Upload your transaction data file (CSV format)"
        )
        
        if uploaded_file is not None:
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
            if file_size > 50:
                st.warning(f"Large file detected: {file_size:.1f} MB")
                st.info("Processing may take a moment...")
        
        # Analysis options
        st.markdown("### 📅 Analysis Period")
        analysis_period = st.selectbox(
            "Select time period",
            ["Last 7 days", "Last 30 days", "Last 90 days", "All available data"],
            index=0
        )
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            min_transactions = st.number_input(
                "Minimum transactions per customer",
                min_value=1,
                value=1,
                help="Filter customers with fewer transactions"
            )
            
            include_sample = st.checkbox(
                "Include data sample in report",
                value=True,
                help="Include sample of raw data in Excel output"
            )
        
        # Run analysis button
        analyze_clicked = st.button(
            "🚀 Run Analysis", 
            type="primary", 
            use_container_width=True,
            disabled=uploaded_file is None
        )
        
        if uploaded_file is None:
            st.info("Please upload a CSV file to begin analysis")
        
        st.markdown("---")
        st.markdown("### 📋 About")
        st.markdown("""
        **Reports Generated:**
        1. International remittance customers not using P2P
        2. Domestic customers using other services but not P2P
        3. Daily impact tracking template
        4. Summary statistics
        
        **Output:** Excel file with separate sheets for each report
        """)
    
    # Main content
    if uploaded_file is not None:
        # Read and cache the data
        if st.session_state.df_uploaded is None or analyze_clicked:
            with st.spinner("Reading and processing data..."):
                try:
                    # Read CSV with optimized settings
                    df = pd.read_csv(uploaded_file, low_memory=False)
                    st.session_state.df_uploaded = df
                    
                    # Show data preview
                    with st.expander("📄 Data Preview", expanded=False):
                        st.write(f"**Total rows:** {len(df):,}")
                        st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
                        st.dataframe(df.head(), use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
                    return
        
        # Run analysis when button is clicked
        if analyze_clicked and st.session_state.df_uploaded is not None:
            with st.spinner("Analyzing data and generating reports..."):
                try:
                    # Run analysis
                    results = analyze_telemarketing_data(
                        st.session_state.df_uploaded, 
                        analysis_period
                    )
                    
                    # Add sample data to results
                    if 'sample_data' not in results:
                        results['sample_data'] = st.session_state.df_uploaded.head(100)
                    
                    # Store results in session state
                    st.session_state.analysis_results = results
                    
                    # Show success
                    st.success("✅ Analysis completed successfully!")
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    return
        
        # Display results if available
        if st.session_state.analysis_results is not None:
            results = st.session_state.analysis_results
            
            # Summary metrics
            if 'summary' in results:
                display_summary_metrics(results['summary'])
            
            # Reports in tabs
            display_reports_in_tabs(results)
            
            # Excel download
            create_excel_download_button(results)
            
            # Additional insights
            st.markdown("---")
            st.markdown("### 🎯 Telemarketing Strategy")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **📞 Calling Priority:**
                1. **High Value International Customers**
                   - Recent international remittance users
                   - Higher transaction frequency/amounts
                
                2. **Active Domestic Users**
                   - Regular users of other services
                   - Demonstrated platform engagement
                
                3. **Inactive International Users**
                   - Historical international users
                   - Recent inactivity
                """)
            
            with col2:
                st.markdown("""
                **📊 Tracking Metrics:**
                - **Daily:** Calls made, customers reached
                - **Weekly:** P2P adoption rate
                - **Monthly:** Revenue impact
                
                **💡 Talking Points:**
                - "I see you use our international service..."
                - "Have you tried our instant P2P transfer?"
                - "It's faster and has lower fees..."
                """)
        
        elif not analyze_clicked:
            # Show preview and prompt for analysis
            st.info("""
            **📊 Data loaded successfully!**
            
            Click the **"Run Analysis"** button in the sidebar to:
            1. Identify international customers not using P2P
            2. Find domestic customers with cross-selling potential
            3. Generate Excel report with separate sheets
            
            **File ready for analysis:** Your CSV file has been loaded and is ready for processing.
            """)
            
            # Quick stats
            if st.session_state.df_uploaded is not None:
                df = st.session_state.df_uploaded
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", f"{len(df):,}")
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    if 'Created At' in df.columns:
                        try:
                            df['Created At'] = pd.to_datetime(df['Created At'], errors='coerce')
                            date_range = f"{df['Created At'].min().strftime('%Y-%m-%d')} to {df['Created At'].max().strftime('%Y-%m-%d')}"
                            st.metric("Date Range", date_range)
                        except:
                            st.metric("Date Info", "Check format")
    
    else:
        # Welcome screen
        st.markdown("""
        <div class="info-box">
            <h3>📁 Welcome to the Telemarketing Dashboard</h3>
            <p>Upload your transaction CSV file to generate targeted calling lists and track campaign effectiveness.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 📋 How it Works
            
            1. **Upload** your transaction CSV file
            2. **Configure** analysis settings
            3. **Run Analysis** to generate reports
            4. **Download** Excel file with separate sheets
            
            ### 📊 Reports Generated
            
            **Excel File with Sheets:**
            - `International_Not_P2P`: International remittance customers not using P2P
            - `Domestic_Not_P2P`: Domestic customers using other services but not P2P
            - `Detailed_Analysis`: Comprehensive analysis with insights
            - `Summary`: Key metrics and statistics
            - `Impact_Template`: Daily tracking template
            - `Data_Sample`: Sample of raw data
            
            ### 🔍 Required Data Format
            
            Your CSV should contain:
            - `User Identifier` (customer ID)
            - `Product Name` (e.g., "International Remittance", "Internal Wallet Transfer")
            - `Service Name` (e.g., "Send Money", "Bill Payment")
            - `Created At` (transaction date/time)
            - `Entity Name` (should contain "Customer")
            - `Full Name` (customer name)
            """)
        
        with col2:
            # Sample data download
            st.markdown("### 🧪 Need Sample Data?")
            
            sample_data = pd.DataFrame({
                'User Identifier': [1001, 1002, 1003, 1004, 1005],
                'Product Name': ['International Remittance', 'Internal Wallet Transfer', 
                               'Airtime Topup', 'Deposit', 'International Remittance'],
                'Service Name': ['Send Money', 'Send Money', 'Airtime Topup', 
                               'Send Money', 'Send Money'],
                'Created At': ['2024-01-15 10:30:00', '2024-01-14 14:20:00', 
                             '2024-01-13 09:15:00', '2024-01-12 16:45:00', '2024-01-11 11:00:00'],
                'Entity Name': ['Customer', 'Customer', 'Customer', 'Customer', 'Customer'],
                'Full Name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson'],
                'Amount': [1000, 500, 25, 200, 1500],
                'Status': ['SUCCESS', 'SUCCESS', 'SUCCESS', 'SUCCESS', 'SUCCESS']
            })
            
            csv = sample_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Sample CSV",
                data=csv,
                file_name="sample_transaction_data.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            st.markdown("---")
            st.markdown("### ⚡ Quick Start")
            st.info("""
            1. Download sample CSV
            2. Upload it to test
            3. Click "Run Analysis"
            4. Download Excel report
            """)

if __name__ == "__main__":
    main()
