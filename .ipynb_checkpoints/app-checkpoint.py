"""
Telemarketing Dashboard - Streamlit App
Analyze transaction data for P2P conversion opportunities
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import io
from utils.analysis import analyze_telemarketing_data
import warnings
warnings.filterwarnings('ignore')

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
</style>
""", unsafe_allow_html=True)

def display_summary_stats(summary):
    """Display summary statistics in metrics cards"""
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
        conversion_rate = (summary.get('intl_not_p2p', 0) / max(summary.get('intl_customers', 1), 1)) * 100
        st.metric(
            label="Intl Not Using P2P",
            value=f"{summary.get('intl_not_p2p', 0):,}",
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

def create_visualizations(results, summary):
    """Create interactive visualizations"""
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Customer distribution chart
        labels = ['P2P Users', 'Intl Not P2P', 'Domestic Not P2P', 'Other']
        values = [
            summary.get('p2p_users', 0),
            summary.get('intl_not_p2p', 0),
            summary.get('domestic_not_p2p', 0),
            summary.get('total_customers', 0) - 
            (summary.get('p2p_users', 0) + 
             summary.get('intl_not_p2p', 0) + 
             summary.get('domestic_not_p2p', 0))
        ]
        
        fig1 = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.3,
            marker_colors=['#10B981', '#3B82F6', '#F59E0B', '#EF4444']
        )])
        fig1.update_layout(
            title="Customer Distribution by Category",
            showlegend=True,
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Potential conversion chart
        if 'intl_customers' in summary and summary['intl_customers'] > 0:
            labels = ['Using P2P', 'Not Using P2P']
            intl_p2p_users = summary['intl_customers'] - summary.get('intl_not_p2p', 0)
            values = [intl_p2p_users, summary.get('intl_not_p2p', 0)]
            
            fig2 = px.bar(
                x=labels,
                y=values,
                text=values,
                color=labels,
                color_discrete_map={'Using P2P': '#10B981', 'Not Using P2P': '#EF4444'}
            )
            fig2.update_layout(
                title="International Customers: P2P Usage",
                xaxis_title="",
                yaxis_title="Number of Customers",
                showlegend=False,
                height=400
            )
            fig2.update_traces(texttemplate='%{text}', textposition='outside')
            st.plotly_chart(fig2, use_container_width=True)

def display_data_tables(results):
    """Display data tables with download options"""
    
    # Tab for different reports
    tab1, tab2, tab3, tab4 = st.tabs([
        "📞 International Customers Not Using P2P",
        "🏠 Domestic Customers Not Using P2P",
        "📈 Daily Impact Template",
        "📊 Summary Report"
    ])
    
    with tab1:
        if not results['intl_not_p2p'].empty:
            st.subheader("International Remittance Customers Not Using P2P")
            st.dataframe(results['intl_not_p2p'], use_container_width=True)
            
            # Download button
            csv = results['intl_not_p2p'].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name="international_not_p2p.csv",
                mime="text/csv"
            )
        else:
            st.info("No international customers found without P2P usage")
    
    with tab2:
        if not results['domestic_other_not_p2p'].empty:
            st.subheader("Domestic Customers Using Other Services But Not P2P")
            st.dataframe(results['domestic_other_not_p2p'], use_container_width=True)
            
            # Download button
            csv = results['domestic_other_not_p2p'].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name="domestic_not_p2p.csv",
                mime="text/csv"
            )
        else:
            st.info("No domestic customers found using other services without P2P")
    
    with tab3:
        st.subheader("Daily Impact Tracking Template")
        st.info("Use this template to track daily calling results and P2P adoption")
        st.dataframe(results['impact_template'], use_container_width=True)
        
        # Download button
        csv = results['impact_template'].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Template",
            data=csv,
            file_name="daily_impact_template.csv",
            mime="text/csv"
        )
    
    with tab4:
        if 'summary' in results:
            st.subheader("Summary Metrics")
            summary_df = pd.DataFrame({
                'Metric': list(results['summary'].keys()),
                'Value': list(results['summary'].values())
            })
            st.dataframe(summary_df, use_container_width=True)

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
            ["Last 7 days", "Last 30 days", "Last 90 days", "All available data", "Custom range"],
            index=0
        )
        
        # Advanced options
        with st.expander("Advanced Options"):
            include_vendors = st.checkbox("Include vendor transactions", value=False)
            min_transactions = st.number_input("Minimum transactions per customer", 
                                              min_value=1, value=1, step=1)
        
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
                    # Save uploaded file temporarily
                    df = pd.read_csv(uploaded_file)
                    
                    # Run analysis
                    results = analyze_telemarketing_data(df, analysis_period)
                    
                    # Display results
                    st.success("✅ Analysis completed successfully!")
                    
                    # Display summary metrics
                    st.markdown('<h2 class="sub-header">📈 Summary Statistics</h2>', unsafe_allow_html=True)
                    display_summary_stats(results['summary'])
                    
                    # Visualizations
                    st.markdown('<h2 class="sub-header">📊 Visualizations</h2>', unsafe_allow_html=True)
                    create_visualizations(results, results['summary'])
                    
                    # Data tables
                    st.markdown('<h2 class="sub-header">📋 Detailed Reports</h2>', unsafe_allow_html=True)
                    display_data_tables(results)
                    
                    # Download all reports as Excel
                    st.markdown("---")
                    st.markdown("### 📥 Download Full Report")
                    
                    # Create Excel file in memory
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        if not results['intl_not_p2p'].empty:
                            results['intl_not_p2p'].to_excel(writer, sheet_name='International_Not_P2P', index=False)
                        if not results['domestic_other_not_p2p'].empty:
                            results['domestic_other_not_p2p'].to_excel(writer, sheet_name='Domestic_Not_P2P', index=False)
                        results['impact_template'].to_excel(writer, sheet_name='Daily_Impact_Template', index=False)
                        summary_df = pd.DataFrame({
                            'Metric': list(results['summary'].keys()),
                            'Value': list(results['summary'].values())
                        })
                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    excel_data = output.getvalue()
                    
                    st.download_button(
                        label="📥 Download Full Excel Report",
                        data=excel_data,
                        file_name=f"telemarketing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.error("Please check your data format and try again.")
        else:
            # Preview uploaded data
            st.markdown("### 📄 Data Preview")
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(100), use_container_width=True)
            st.info(f"📊 **{len(df)}** transactions loaded. Click 'Run Analysis' to process the data.")
    else:
        # Welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style='text-align: center; padding: 2rem;'>
                <h3>📁 Upload your transaction data to get started</h3>
                <p>Supported format: CSV with transaction data</p>
                <div style='margin-top: 2rem; padding: 1rem; background-color: #F0F9FF; border-radius: 10px;'>
                    <h4>📋 Required Columns:</h4>
                    <ul style='text-align: left;'>
                        <li>User Identifier</li>
                        <li>Product Name</li>
                        <li>Service Name</li>
                        <li>Created At</li>
                        <li>Entity Name</li>
                        <li>Full Name</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Sample data download
            st.markdown("---")
            st.markdown("### 🧪 Need sample data?")
            
            # Create sample data
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
                mime="text/csv"
            )

if __name__ == "__main__":
    main()