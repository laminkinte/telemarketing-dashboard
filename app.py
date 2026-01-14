"""
Telemarketing Dashboard - Streamlit App
Analyze transaction data for P2P conversion opportunities
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import warnings
warnings.filterwarnings('ignore')

# Try to import plotly, but provide fallback
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly is not installed. Charts will be limited.")

# Import analysis functions
try:
    from utils.analysis import analyze_telemarketing_data
except ImportError:
    # Define a simple version if module not found
    def analyze_telemarketing_data(df, analysis_period="Last 7 days"):
        """Simplified analysis function for testing"""
        return {
            'intl_not_p2p': pd.DataFrame({'Message': ['Analysis module not loaded']}),
            'domestic_other_not_p2p': pd.DataFrame({'Message': ['Analysis module not loaded']}),
            'impact_template': pd.DataFrame({'Date': [datetime.now().strftime('%Y-%m-%d')]}),
            'summary': {}
        }

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
        width: 100%;
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
    .dataframe {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

def display_summary_stats(summary):
    """Display summary statistics in metrics cards"""
    if not summary:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Customers",
            value=f"{summary.get('total_customers', 0):,}",
            help="Total unique customers in the period"
        )
    
    with col2:
        st.metric(
            label="International Customers",
            value=f"{summary.get('intl_customers', 0):,}",
            help="Customers who used international remittance"
        )
    
    with col3:
        intl_not_p2p = summary.get('intl_not_p2p', 0)
        intl_customers = summary.get('intl_customers', 1)
        conversion_rate = (intl_not_p2p / max(intl_customers, 1)) * 100
        st.metric(
            label="Intl Not Using P2P",
            value=f"{intl_not_p2p:,}",
            delta=f"{conversion_rate:.1f}%",
            delta_color="inverse",
            help="International customers not using P2P"
        )
    
    with col4:
        st.metric(
            label="Domestic Not Using P2P",
            value=f"{summary.get('domestic_not_p2p', 0):,}",
            help="Domestic customers using other services but not P2P"
        )

def create_simple_charts(summary):
    """Create simple charts using Streamlit's built-in charts"""
    
    if not summary:
        return
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Distribution")
        
        # Create data for bar chart
        chart_data = pd.DataFrame({
            'Category': ['P2P Users', 'Intl Not P2P', 'Domestic Not P2P'],
            'Count': [
                summary.get('p2p_users', 0),
                summary.get('intl_not_p2p', 0),
                summary.get('domestic_not_p2p', 0)
            ]
        })
        
        st.bar_chart(chart_data.set_index('Category'))
    
    with col2:
        if summary.get('intl_customers', 0) > 0:
            st.subheader("International Customers: P2P Usage")
            
            intl_p2p_users = summary.get('intl_customers', 0) - summary.get('intl_not_p2p', 0)
            intl_not_p2p = summary.get('intl_not_p2p', 0)
            
            pie_data = pd.DataFrame({
                'Status': ['Using P2P', 'Not Using P2P'],
                'Count': [intl_p2p_users, intl_not_p2p]
            })
            
            st.dataframe(pie_data)

def display_data_tables(results):
    """Display data tables with download options"""
    
    if not results:
        return
    
    # Tab for different reports
    tab1, tab2, tab3 = st.tabs([
        "📞 International Customers",
        "🏠 Domestic Customers", 
        "📈 Impact Template"
    ])
    
    with tab1:
        if 'intl_not_p2p' in results and not results['intl_not_p2p'].empty:
            st.subheader("International Remittance Customers Not Using P2P")
            st.dataframe(results['intl_not_p2p'], use_container_width=True)
            
            # Download button
            if st.button("📥 Download International Customers CSV", key="dl_intl"):
                csv = results['intl_not_p2p'].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Click to Download",
                    data=csv,
                    file_name="international_not_p2p.csv",
                    mime="text/csv",
                    key="dl_intl_final"
                )
        else:
            st.info("No international customers found without P2P usage")
    
    with tab2:
        if 'domestic_other_not_p2p' in results and not results['domestic_other_not_p2p'].empty:
            st.subheader("Domestic Customers Using Other Services But Not P2P")
            st.dataframe(results['domestic_other_not_p2p'], use_container_width=True)
            
            # Download button
            if st.button("📥 Download Domestic Customers CSV", key="dl_dom"):
                csv = results['domestic_other_not_p2p'].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Click to Download",
                    data=csv,
                    file_name="domestic_not_p2p.csv",
                    mime="text/csv",
                    key="dl_dom_final"
                )
        else:
            st.info("No domestic customers found using other services without P2P")
    
    with tab3:
        if 'impact_template' in results and not results['impact_template'].empty:
            st.subheader("Daily Impact Tracking Template")
            st.info("Use this template to track daily calling results and P2P adoption")
            st.dataframe(results['impact_template'], use_container_width=True)
            
            # Download button
            if st.button("📥 Download Impact Template CSV", key="dl_impact"):
                csv = results['impact_template'].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Click to Download",
                    data=csv,
                    file_name="daily_impact_template.csv",
                    mime="text/csv",
                    key="dl_impact_final"
                )

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Telemarketing Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("""
    Analyze transaction data to identify P2P conversion opportunities and track telemarketing campaign effectiveness.
    """)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Transaction CSV",
            type=['csv'],
            help="Upload your transaction data file"
        )
        
        # Date range selector
        st.markdown("### 📅 Date Range")
        analysis_period = st.selectbox(
            "Analysis Period",
            ["Last 7 days", "Last 30 days", "Last 90 days", "All available data"],
            index=0
        )
        
        # Run analysis button
        analyze_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📋 About")
        st.markdown("""
        This dashboard helps identify:
        - International remittance customers not using P2P
        - Domestic customers using other services but not P2P
        - Track telemarketing campaign effectiveness
        """)
    
    # Main content area
    if uploaded_file is not None:
        if analyze_button:
            with st.spinner("Analyzing data..."):
                try:
                    # Read uploaded file
                    df = pd.read_csv(uploaded_file)
                    
                    # Show data preview
                    with st.expander("Data Preview", expanded=False):
                        st.write(f"**Total rows:** {len(df):,}")
                        st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
                        st.dataframe(df.head(), use_container_width=True)
                    
                    # Run analysis
                    results = analyze_telemarketing_data(df, analysis_period)
                    
                    # Display success message
                    st.success("✅ Analysis completed successfully!")
                    
                    # Display summary metrics
                    if 'summary' in results and results['summary']:
                        st.markdown('<h2 class="sub-header">📈 Summary Statistics</h2>', unsafe_allow_html=True)
                        display_summary_stats(results['summary'])
                        
                        # Show insights
                        st.markdown("### 💡 Insights")
                        if results['summary'].get('intl_customers', 0) > 0:
                            conversion_rate = (results['summary'].get('intl_not_p2p', 0) / 
                                            results['summary'].get('intl_customers', 1)) * 100
                            st.info(f"**{conversion_rate:.1f}% of international customers don't use P2P**")
                            if conversion_rate > 30:
                                st.success("🎯 **High Potential**: This represents a significant cross-selling opportunity!")
                    
                    # Data tables
                    st.markdown('<h2 class="sub-header">📋 Detailed Reports</h2>', unsafe_allow_html=True)
                    display_data_tables(results)
                    
                    # Download all reports as Excel
                    st.markdown("---")
                    st.markdown("### 📥 Download Full Report")
                    
                    if st.button("📊 Download Complete Excel Report"):
                        # Create Excel file in memory
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            if 'intl_not_p2p' in results and not results['intl_not_p2p'].empty:
                                results['intl_not_p2p'].to_excel(writer, sheet_name='International_Not_P2P', index=False)
                            if 'domestic_other_not_p2p' in results and not results['domestic_other_not_p2p'].empty:
                                results['domestic_other_not_p2p'].to_excel(writer, sheet_name='Domestic_Not_P2P', index=False)
                            if 'impact_template' in results:
                                results['impact_template'].to_excel(writer, sheet_name='Daily_Impact_Template', index=False)
                            if 'summary' in results and results['summary']:
                                summary_df = pd.DataFrame({
                                    'Metric': list(results['summary'].keys()),
                                    'Value': list(results['summary'].values())
                                })
                                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                        
                        excel_data = output.getvalue()
                        
                        st.download_button(
                            label="Click to Download Excel File",
                            data=excel_data,
                            file_name=f"telemarketing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.error("Please check your data format and try again.")
                    st.code(f"Error details: {e}", language="python")
        else:
            # Preview uploaded data
            st.markdown("### 📄 Data Preview")
            try:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head(100), use_container_width=True)
                st.info(f"📊 **{len(df):,}** transactions loaded. Click **'Run Analysis'** to process the data.")
            except Exception as e:
                st.error(f"Error reading file: {e}")
    else:
        # Welcome screen
        col1, col2 = st.columns([1, 3])
        with col2:
            st.markdown("""
            <div style='text-align: center; padding: 2rem;'>
                <h3>📁 Upload your transaction data to get started</h3>
                <p>Supported format: CSV with transaction data</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Requirements in expander
        with st.expander("📋 Required Data Format", expanded=True):
            st.markdown("""
            Your CSV file should contain these columns:
            
            **Required Columns:**
            - `User Identifier` - Unique customer ID
            - `Product Name` - Product/service name
            - `Service Name` - Service category
            - `Created At` - Transaction date and time
            - `Entity Name` - Entity type (should contain "Customer")
            - `Full Name` - Customer's full name
            
            **Optional but helpful:**
            - `Amount` - Transaction amount
            - `Status` - Transaction status
            
            **Example Product Names:**
            - International Remittance
            - Internal Wallet Transfer (P2P)
            - Deposit
            - Airtime Topup
            - Bill Payment
            """)
            
            # Create and show sample data
            sample_data = pd.DataFrame({
                'User Identifier': [1001, 1002, 1003],
                'Product Name': ['International Remittance', 'Internal Wallet Transfer', 'Airtime Topup'],
                'Service Name': ['Send Money', 'Send Money', 'Airtime Topup'],
                'Created At': ['2024-01-15 10:30:00', '2024-01-14 14:20:00', '2024-01-13 09:15:00'],
                'Entity Name': ['Customer', 'Customer', 'Customer'],
                'Full Name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
                'Amount': [1000, 500, 25],
                'Status': ['SUCCESS', 'SUCCESS', 'SUCCESS']
            })
            
            st.dataframe(sample_data, use_container_width=True)
            
            # Download sample button
            csv = sample_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Sample CSV",
                data=csv,
                file_name="sample_transaction_data.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
