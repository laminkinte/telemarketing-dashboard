import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import warnings
import zipfile
warnings.filterwarnings('ignore')

# Check for required libraries
try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False
    st.warning("openpyxl not installed. Excel export will use CSV format.")

# Try to import plotly with fallback
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not installed. Some visualizations will be disabled.")
    # Create mock functions to avoid errors
    class MockPlotly:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    px = MockPlotly()
    go = MockPlotly()

# Set page configuration
st.set_page_config(
    page_title="Telemarketing Campaign Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    .dataframe {
        font-size: 0.9rem;
    }
    .filter-pair-info {
        background-color: #f0f9ff;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin: 0.5rem 0;
    }
    .segment-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.25rem;
    }
    .segment-25 { background-color: #dcfce7; color: #166534; }
    .segment-50 { background-color: #fef3c7; color: #92400e; }
    .segment-75 { background-color: #fef3c7; color: #92400e; }
    .segment-100 { background-color: #fee2e2; color: #991b1b; }
</style>
""", unsafe_allow_html=True)

def load_data(uploaded_file):
    """Load and clean the transaction data"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        
        st.success(f"✅ Successfully loaded {len(df):,} transactions")
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Check required columns
        required_columns = ['User Identifier', 'Product Name', 'Created At']
        optional_columns = ['Service Name', 'Entity Name', 'Full Name', 'Transaction Amount']
        
        # Check for minimum required columns
        missing_required = [col for col in required_columns if col not in df.columns]
        if missing_required:
            st.error(f"Missing required columns: {missing_required}")
            st.info(f"Available columns: {list(df.columns)}")
            return None
        
        # Add missing optional columns with default values
        for col in optional_columns:
            if col not in df.columns:
                df[col] = ''
                st.warning(f"Note: Column '{col}' not found. Using empty values.")
        
        # Data cleaning
        for col in ['Product Name', 'Service Name', 'Entity Name', 'Full Name']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        df['User Identifier'] = pd.to_numeric(df['User Identifier'], errors='coerce')
        
        # Parse date column with multiple format handling
        date_formats = ['%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', '%d-%m-%Y %H:%M:%S', 
                       '%d/%m/%Y %H:%M:%S', '%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y']
        
        for fmt in date_formats:
            try:
                df['Created At'] = pd.to_datetime(df['Created At'], format=fmt, errors='coerce')
                if df['Created At'].notna().any():
                    break
            except:
                continue
        
        # If still not parsed, try generic parsing
        if df['Created At'].isna().all():
            df['Created At'] = pd.to_datetime(df['Created At'], errors='coerce')
        
        # Add derived columns
        df['Date'] = df['Created At'].dt.date
        df['Day'] = df['Created At'].dt.day_name()
        df['Hour'] = df['Created At'].dt.hour
        
        # Convert transaction amount to numeric if available
        if 'Transaction Amount' in df.columns:
            # Clean the Transaction Amount column
            df['Transaction Amount'] = df['Transaction Amount'].astype(str).str.replace(',', '').str.replace('$', '').str.replace('€', '').str.replace('£', '').str.strip()
            df['Transaction Amount'] = pd.to_numeric(df['Transaction Amount'], errors='coerce')
            # Fill NaN with 0 for amount calculations
            df['Transaction Amount'] = df['Transaction Amount'].fillna(0)
        
        return df
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def create_zip_with_csv_files(results, filter_desc=""):
    """Create a ZIP file with all CSV reports"""
    try:
        # Create a BytesIO object for the zip file
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # 1. Executive Summary CSV
            summary_data = [
                ['TELEMARKETING CAMPAIGN ANALYSIS REPORT'],
                ['Generated on', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                ['Filter Applied', filter_desc[:200]],
                ['Report Period', f"{results['summary_stats']['start_date']} to {results['summary_stats']['end_date']}"],
                [''],
                ['OVERALL SUMMARY'],
                ['Total Transactions', results['summary_stats']['total_transactions']],
                ['Unique Customers', results['summary_stats']['unique_customers']],
                ['International Customers', results['summary_stats']['intl_customers']],
                ['P2P Customers', results['summary_stats']['p2p_customers']],
                ['Other Services Customers', results['summary_stats']['other_customers']],
                [''],
                ['TARGET GROUPS SUMMARY'],
                ['International Customers Not Using P2P', results['summary_stats']['intl_not_p2p']],
                ['Domestic Customers Not Using P2P', results['summary_stats']['domestic_not_p2p']],
                ['Total Addressable Market', results['summary_stats']['intl_not_p2p'] + results['summary_stats']['domestic_not_p2p']],
                [''],
                ['INTERNATIONAL WITHDRAWAL ANALYSIS'],
                ['Total International Recipients', results['summary_stats'].get('total_intl_recipients', 0)],
                ['Average Withdrawal Percentage', f"{results['summary_stats'].get('avg_withdrawal_percentage', 0):.1f}% of International Received"],
                ['Segment ≤25%', results['summary_stats'].get('intl_withdrawal_segment_25', 0)],
                ['Segment 25%-50%', results['summary_stats'].get('intl_withdrawal_segment_50', 0)],
                ['Segment 50%-75%', results['summary_stats'].get('intl_withdrawal_segment_75', 0)],
                ['Segment 75%-100%', results['summary_stats'].get('intl_withdrawal_segment_100', 0)],
                [''],
                ['SPECIFIC INTERNATIONAL GROUPS'],
                ['Received & Withdrew', results['summary_stats'].get('received_withdrew_count', 0)],
                ['Received, No Withdrawal, Other Services', results['summary_stats'].get('received_no_withdraw_other_count', 0)],
                ['Received & Only Withdrew', results['summary_stats'].get('received_only_withdraw_count', 0)]
            ]
            
            summary_df = pd.DataFrame(summary_data)
            summary_csv = summary_df.to_csv(index=False, header=False).encode('utf-8')
            zip_file.writestr('Executive_Summary.csv', summary_csv)
            
            # 2. International Targets
            if not results['intl_not_p2p'].empty:
                intl_csv = results['intl_not_p2p'].to_csv(index=False).encode('utf-8')
                zip_file.writestr('International_Targets.csv', intl_csv)
            
            # 3. Domestic Targets
            if not results['domestic_other_not_p2p'].empty:
                domestic_csv = results['domestic_other_not_p2p'].to_csv(index=False).encode('utf-8')
                zip_file.writestr('Domestic_Targets.csv', domestic_csv)
            
            # 4. International Withdrawal Segments
            if 'detailed_analysis' in results['intl_withdrawal_segments'] and not results['intl_withdrawal_segments']['detailed_analysis'].empty:
                segments_csv = results['intl_withdrawal_segments']['detailed_analysis'].to_csv(index=False).encode('utf-8')
                zip_file.writestr('Intl_Withdrawal_Segments.csv', segments_csv)
            
            # 5. International Received & Withdrew
            if 'received_withdrew' in results['specific_intl_groups'] and not results['specific_intl_groups']['received_withdrew'].empty:
                received_withdrew_csv = results['specific_intl_groups']['received_withdrew'].to_csv(index=False).encode('utf-8')
                zip_file.writestr('Intl_Received_Withdrew.csv', received_withdrew_csv)
            
            # 6. International No Withdraw Other
            if 'received_no_withdraw_other' in results['specific_intl_groups'] and not results['specific_intl_groups']['received_no_withdraw_other'].empty:
                no_withdraw_csv = results['specific_intl_groups']['received_no_withdraw_other'].to_csv(index=False).encode('utf-8')
                zip_file.writestr('Intl_No_Withdraw_Other.csv', no_withdraw_csv)
            
            # 7. International Only Withdrew
            if 'received_only_withdraw' in results['specific_intl_groups'] and not results['specific_intl_groups']['received_only_withdraw'].empty:
                only_withdrew_csv = results['specific_intl_groups']['received_only_withdraw'].to_csv(index=False).encode('utf-8')
                zip_file.writestr('Intl_Only_Withdrew.csv', only_withdrew_csv)
            
            # 8. Impact Template
            impact_csv = results['impact_template'].to_csv(index=False).encode('utf-8')
            zip_file.writestr('Impact_Template.csv', impact_csv)
            
            # 9. Detailed Metrics
            metrics_data = [
                ['DETAILED PERFORMANCE METRICS'],
                [''],
                ['CONVERSION POTENTIAL ANALYSIS'],
                ['International to P2P Conversion Rate', f"{(results['summary_stats']['intl_not_p2p'] / max(results['summary_stats']['intl_customers'], 1) * 100):.1f}%"],
                ['Domestic Cross-sell Potential', f"{(results['summary_stats']['domestic_not_p2p'] / max(results['summary_stats']['other_customers'], 1) * 100):.1f}%"],
                ['Overall Market Penetration', f"{(results['summary_stats']['p2p_customers'] / max(results['summary_stats']['unique_customers'], 1) * 100):.1f}%"],
                [''],
                ['INTERNATIONAL WITHDRAWAL BEHAVIOR (Percentage against International Received)'],
                ['Low Withdrawal (≤25%) Customers', results['summary_stats'].get('intl_withdrawal_segment_25', 0)],
                ['Moderate Withdrawal (25-50%) Customers', results['summary_stats'].get('intl_withdrawal_segment_50', 0)],
                ['High Withdrawal (50-75%) Customers', results['summary_stats'].get('intl_withdrawal_segment_75', 0)],
                ['Very High Withdrawal (75-100%) Customers', results['summary_stats'].get('intl_withdrawal_segment_100', 0)],
                [''],
                ['CUSTOMER SEGMENT DISTRIBUTION'],
                ['International Focus Group Size', results['summary_stats']['intl_not_p2p']],
                ['Domestic Focus Group Size', results['summary_stats']['domestic_not_p2p']],
                ['Received & Withdrew Group', results['summary_stats'].get('received_withdrew_count', 0)],
                ['Received, No Withdrawal Group', results['summary_stats'].get('received_no_withdraw_other_count', 0)],
                ['Received & Only Withdrew Group', results['summary_stats'].get('received_only_withdraw_count', 0)]
            ]
            
            metrics_df = pd.DataFrame(metrics_data)
            metrics_csv = metrics_df.to_csv(index=False, header=False).encode('utf-8')
            zip_file.writestr('Detailed_Metrics.csv', metrics_csv)
            
            # 10. Recommendations
            recommendations = [
                ['TELEMARKETING CAMPAIGN RECOMMENDATIONS'],
                [''],
                ['PRIORITY 1: INTERNATIONAL CUSTOMER ENGAGEMENT'],
                [f'1. Target {results["summary_stats"]["intl_not_p2p"]} international customers not using P2P'],
                ['2. Focus on customers with recent international transactions'],
                ['3. Offer P2P fee waivers for first 3 transactions'],
                ['4. Bundle P2P with international remittance services'],
                [''],
                ['PRIORITY 2: DOMESTIC CUSTOMER CROSS-SELL'],
                [f'5. Engage {results["summary_stats"]["domestic_not_p2p"]} domestic customers'],
                ['6. Promote P2P as faster alternative to current services'],
                ['7. Offer loyalty rewards for P2P adoption'],
                [''],
                ['INTERNATIONAL WITHDRAWAL SEGMENT STRATEGY (Based on % of International Received)'],
                [f'8. Low Withdrawal (≤25%): {results["summary_stats"].get("intl_withdrawal_segment_25", 0)} customers - Focus on retention & premium services'],
                [f'9. Moderate (25-50%): {results["summary_stats"].get("intl_withdrawal_segment_50", 0)} customers - Balance service education'],
                [f'10. High (50-75%): {results["summary_stats"].get("intl_withdrawal_segment_75", 0)} customers - Risk mitigation strategies'],
                [f'11. Very High (75-100%): {results["summary_stats"].get("intl_withdrawal_segment_100", 0)} customers - Immediate attention required'],
                [''],
                ['CAMPAIGN IMPLEMENTATION'],
                ['12. Use the Impact Template for daily tracking'],
                ['13. Schedule calls during peak transaction hours'],
                ['14. Train agents on P2P benefits and features'],
                ['15. Track conversion rates weekly'],
                ['16. Collect customer feedback systematically'],
                ['17. Follow up with unreached customers within 48 hours'],
                ['18. Monitor high-withdrawal customers for churn risk'],
                ['19. Develop targeted messaging for each segment'],
                ['20. Set clear KPIs and review progress bi-weekly']
            ]
            
            rec_df = pd.DataFrame(recommendations)
            rec_csv = rec_df.to_csv(index=False, header=False).encode('utf-8')
            zip_file.writestr('Recommendations.csv', rec_csv)
        
        zip_buffer.seek(0)
        return zip_buffer
    
    except Exception as e:
        st.error(f"Error creating ZIP file: {str(e)}")
        return None

# All other functions remain the same as in the previous code...
# [Include all the other functions from the previous code here, including:
# filter_by_customer_pair, analyze_intl_withdrawal_segments, 
# analyze_specific_intl_groups, filter_data, analyze_telemarketing_data]

# [IMPORTANT: Paste all the other functions from the previous code here, 
# but replace the create_comprehensive_excel_report function with the simplified version below]

def create_comprehensive_excel_report(results, filter_desc=""):
    """Create comprehensive Excel report with all sheets"""
    if not OPENPYXL_AVAILABLE:
        st.warning("openpyxl not available. Using CSV format for export.")
        return create_zip_with_csv_files(results, filter_desc)
    
    try:
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sheet 1: Executive Summary
            summary_data = [
                ['TELEMARKETING CAMPAIGN ANALYSIS REPORT'],
                ['Generated on', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                ['Filter Applied', filter_desc[:200]],
                ['Report Period', f"{results['summary_stats']['start_date']} to {results['summary_stats']['end_date']}"],
                [''],
                ['OVERALL SUMMARY'],
                ['Total Transactions', results['summary_stats']['total_transactions']],
                ['Unique Customers', results['summary_stats']['unique_customers']],
                ['International Customers', results['summary_stats']['intl_customers']],
                ['P2P Customers', results['summary_stats']['p2p_customers']],
                ['Other Services Customers', results['summary_stats']['other_customers']],
                [''],
                ['TARGET GROUPS SUMMARY'],
                ['International Customers Not Using P2P', results['summary_stats']['intl_not_p2p']],
                ['Domestic Customers Not Using P2P', results['summary_stats']['domestic_not_p2p']],
                ['Total Addressable Market', results['summary_stats']['intl_not_p2p'] + results['summary_stats']['domestic_not_p2p']],
                [''],
                ['INTERNATIONAL WITHDRAWAL ANALYSIS'],
                ['Total International Recipients', results['summary_stats'].get('total_intl_recipients', 0)],
                ['Average Withdrawal Percentage', f"{results['summary_stats'].get('avg_withdrawal_percentage', 0):.1f}% of International Received"],
                ['Segment ≤25%', results['summary_stats'].get('intl_withdrawal_segment_25', 0)],
                ['Segment 25%-50%', results['summary_stats'].get('intl_withdrawal_segment_50', 0)],
                ['Segment 50%-75%', results['summary_stats'].get('intl_withdrawal_segment_75', 0)],
                ['Segment 75%-100%', results['summary_stats'].get('intl_withdrawal_segment_100', 0)],
                [''],
                ['SPECIFIC INTERNATIONAL GROUPS'],
                ['Received & Withdrew', results['summary_stats'].get('received_withdrew_count', 0)],
                ['Received, No Withdrawal, Other Services', results['summary_stats'].get('received_no_withdraw_other_count', 0)],
                ['Received & Only Withdrew', results['summary_stats'].get('received_only_withdraw_count', 0)]
            ]
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Executive_Summary', index=False, header=False)
            
            # Sheet 2: International Targets (Not Using P2P)
            if not results['intl_not_p2p'].empty:
                results['intl_not_p2p'].to_excel(writer, sheet_name='International_Targets', index=False)
            
            # Sheet 3: Domestic Targets
            if not results['domestic_other_not_p2p'].empty:
                results['domestic_other_not_p2p'].to_excel(writer, sheet_name='Domestic_Targets', index=False)
            
            # Sheet 4: International Withdrawal Segments
            if 'detailed_analysis' in results['intl_withdrawal_segments'] and not results['intl_withdrawal_segments']['detailed_analysis'].empty:
                results['intl_withdrawal_segments']['detailed_analysis'].to_excel(writer, sheet_name='Intl_Withdrawal_Segments', index=False)
            
            # Sheet 5: International Received & Withdrew
            if 'received_withdrew' in results['specific_intl_groups'] and not results['specific_intl_groups']['received_withdrew'].empty:
                results['specific_intl_groups']['received_withdrew'].to_excel(writer, sheet_name='Intl_Received_Withdrew', index=False)
            
            # Sheet 6: International Received, No Withdrawal, Other Services
            if 'received_no_withdraw_other' in results['specific_intl_groups'] and not results['specific_intl_groups']['received_no_withdraw_other'].empty:
                results['specific_intl_groups']['received_no_withdraw_other'].to_excel(writer, sheet_name='Intl_No_Withdraw_Other', index=False)
            
            # Sheet 7: International Received & Only Withdrew
            if 'received_only_withdraw' in results['specific_intl_groups'] and not results['specific_intl_groups']['received_only_withdraw'].empty:
                results['specific_intl_groups']['received_only_withdraw'].to_excel(writer, sheet_name='Intl_Only_Withdrew', index=False)
            
            # Sheet 8: Impact Template
            results['impact_template'].to_excel(writer, sheet_name='Impact_Template', index=False)
            
            # Sheet 9: Detailed Metrics
            metrics_data = [
                ['DETAILED PERFORMANCE METRICS'],
                [''],
                ['CONVERSION POTENTIAL ANALYSIS'],
                ['International to P2P Conversion Rate', f"{(results['summary_stats']['intl_not_p2p'] / max(results['summary_stats']['intl_customers'], 1) * 100):.1f}%"],
                ['Domestic Cross-sell Potential', f"{(results['summary_stats']['domestic_not_p2p'] / max(results['summary_stats']['other_customers'], 1) * 100):.1f}%"],
                ['Overall Market Penetration', f"{(results['summary_stats']['p2p_customers'] / max(results['summary_stats']['unique_customers'], 1) * 100):.1f}%"],
                [''],
                ['INTERNATIONAL WITHDRAWAL BEHAVIOR (Percentage against International Received)'],
                ['Low Withdrawal (≤25%) Customers', results['summary_stats'].get('intl_withdrawal_segment_25', 0)],
                ['Moderate Withdrawal (25-50%) Customers', results['summary_stats'].get('intl_withdrawal_segment_50', 0)],
                ['High Withdrawal (50-75%) Customers', results['summary_stats'].get('intl_withdrawal_segment_75', 0)],
                ['Very High Withdrawal (75-100%) Customers', results['summary_stats'].get('intl_withdrawal_segment_100', 0)],
                [''],
                ['CUSTOMER SEGMENT DISTRIBUTION'],
                ['International Focus Group Size', results['summary_stats']['intl_not_p2p']],
                ['Domestic Focus Group Size', results['summary_stats']['domestic_not_p2p']],
                ['Received & Withdrew Group', results['summary_stats'].get('received_withdrew_count', 0)],
                ['Received, No Withdrawal Group', results['summary_stats'].get('received_no_withdraw_other_count', 0)],
                ['Received & Only Withdrew Group', results['summary_stats'].get('received_only_withdraw_count', 0)]
            ]
            
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_excel(writer, sheet_name='Detailed_Metrics', index=False, header=False)
            
            # Sheet 10: Recommendations
            recommendations = [
                ['TELEMARKETING CAMPAIGN RECOMMENDATIONS'],
                [''],
                ['PRIORITY 1: INTERNATIONAL CUSTOMER ENGAGEMENT'],
                [f'1. Target {results["summary_stats"]["intl_not_p2p"]} international customers not using P2P'],
                ['2. Focus on customers with recent international transactions'],
                ['3. Offer P2P fee waivers for first 3 transactions'],
                ['4. Bundle P2P with international remittance services'],
                [''],
                ['PRIORITY 2: DOMESTIC CUSTOMER CROSS-SELL'],
                [f'5. Engage {results["summary_stats"]["domestic_not_p2p"]} domestic customers'],
                ['6. Promote P2P as faster alternative to current services'],
                ['7. Offer loyalty rewards for P2P adoption'],
                [''],
                ['INTERNATIONAL WITHDRAWAL SEGMENT STRATEGY (Based on % of International Received)'],
                [f'8. Low Withdrawal (≤25%): {results["summary_stats"].get("intl_withdrawal_segment_25", 0)} customers - Focus on retention & premium services'],
                [f'9. Moderate (25-50%): {results["summary_stats"].get("intl_withdrawal_segment_50", 0)} customers - Balance service education'],
                [f'10. High (50-75%): {results["summary_stats"].get("intl_withdrawal_segment_75", 0)} customers - Risk mitigation strategies'],
                [f'11. Very High (75-100%): {results["summary_stats"].get("intl_withdrawal_segment_100", 0)} customers - Immediate attention required'],
                [''],
                ['CAMPAIGN IMPLEMENTATION'],
                ['12. Use the Impact Template for daily tracking'],
                ['13. Schedule calls during peak transaction hours'],
                ['14. Train agents on P2P benefits and features'],
                ['15. Track conversion rates weekly'],
                ['16. Collect customer feedback systematically'],
                ['17. Follow up with unreached customers within 48 hours'],
                ['18. Monitor high-withdrawal customers for churn risk'],
                ['19. Develop targeted messaging for each segment'],
                ['20. Set clear KPIs and review progress bi-weekly']
            ]
            
            rec_df = pd.DataFrame(recommendations)
            rec_df.to_excel(writer, sheet_name='Recommendations', index=False, header=False)
        
        output.seek(0)
        return output
    
    except Exception as e:
        st.error(f"Error creating Excel file: {str(e)}")
        # Fall back to CSV/ZIP format
        return create_zip_with_csv_files(results, filter_desc)

# Main function with updated download section
def main():
    """Main Streamlit app"""
    st.markdown('<h1 class="main-header">📊 Telemarketing Campaign Analyzer</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Transaction Data",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your transaction file (CSV or Excel)"
        )
        
        if uploaded_file is not None:
            # Load data
            with st.spinner("Loading data..."):
                df = load_data(uploaded_file)
            
            if df is not None:
                # Customer Pair Filter
                st.subheader("🎯 Customer Behavior Filter")
                
                pair_filter_options = {
                    "all_customers": "All Customers",
                    "intl_not_p2p": "International remittance but NOT P2P",
                    "p2p_not_deposit": "P2P but NOT Deposit",
                    "p2p_and_withdrawal": "P2P AND Withdrawal",
                    "intl_and_p2p": "International remittance AND P2P",
                    "intl_received_withdraw": "Received International & Withdrew",
                    "intl_received_no_withdraw": "Received International, NO Withdrawal",
                    "intl_received_only_withdraw": "Received International & ONLY Withdrew"
                }
                
                selected_pair_filter = st.selectbox(
                    "Select Customer Behavior Pair:",
                    options=list(pair_filter_options.keys()),
                    format_func=lambda x: pair_filter_options[x],
                    index=0
                )
                
                # Date range filter
                st.subheader("📅 Date Range Filter")
                min_date = df['Created At'].min().date() if df['Created At'].notna().any() else datetime.now().date() - timedelta(days=30)
                max_date = df['Created At'].max().date() if df['Created At'].notna().any() else datetime.now().date()
                
                start_date = st.date_input(
                    "Start Date",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date
                )
                
                end_date = st.date_input(
                    "End Date",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date
                )
                
                # Product filter
                st.subheader("📦 Product Filter")
                product_names = df['Product Name'].dropna().unique()
                if len(product_names) > 0:
                    unique_products = ['All'] + sorted([str(p) for p in product_names if str(p) != 'nan'])[:50]
                else:
                    unique_products = ['All']
                
                product_filter = st.selectbox(
                    "Filter by Product",
                    options=unique_products,
                    index=0
                )
                
                # Customer type filter
                st.subheader("👥 Customer Type Filter")
                customer_type_filter = st.selectbox(
                    "Filter by Customer Type",
                    options=['All', 'Customer Only', 'Vendor/Agent Only'],
                    index=0
                )
                
                # Analyze button
                analyze_button = st.button(
                    "🚀 Analyze Data",
                    type="primary",
                    use_container_width=True
                )
        else:
            st.info("👈 Please upload a transaction file to begin analysis")
            df = None
            analyze_button = False
            start_date = end_date = product_filter = customer_type_filter = None
            selected_pair_filter = "all_customers"
    
    # Main content area
    if uploaded_file is not None and df is not None and analyze_button:
        # Filter data
        with st.spinner("Filtering data..."):
            filtered_df, filter_desc, customer_count = filter_data(
                df, start_date, end_date, product_filter, customer_type_filter, selected_pair_filter
            )
        
        if len(filtered_df) == 0:
            st.warning("⚠️ No data found for the selected filters. Please adjust your criteria.")
        else:
            # Display filter info
            st.markdown(f'<div class="filter-pair-info">{filter_desc}</div>', unsafe_allow_html=True)
            
            # Display metrics
            st.markdown('<h2 class="sub-header">📈 Key Metrics</h2>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Transactions", f"{len(filtered_df):,}")
            with col2:
                st.metric("Unique Customers", f"{customer_count:,}")
            with col3:
                date_range_str = f"{start_date} to {end_date}"
                st.metric("Date Range", date_range_str)
            with col4:
                st.metric("Products", filtered_df['Product Name'].nunique())
            
            # Run analysis
            st.markdown('<h2 class="sub-header">🎯 Telemarketing Analysis Results</h2>', unsafe_allow_html=True)
            
            with st.spinner("Analyzing data for telemarketing targets..."):
                results = analyze_telemarketing_data(filtered_df)
            
            # Display results in tabs
            tab_names = [
                "📊 Executive Summary",
                "🎯 International Targets", 
                "🏠 Domestic Targets", 
                "📈 Intl Withdrawal Segments",
                "💸 Intl Received & Withdrew",
                "🔄 Intl No Withdraw Other",
                "💰 Intl Only Withdrew",
                "📋 Impact Template", 
                "📊 Detailed Metrics",
                "💡 Recommendations"
            ]
            
            tabs = st.tabs(tab_names)
            
            with tabs[0]:  # Executive Summary
                st.subheader("Executive Summary")
                
                # Overall Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Customers", f"{results['summary_stats']['unique_customers']:,}")
                with col2:
                    total_targets = results['summary_stats']['intl_not_p2p'] + results['summary_stats']['domestic_not_p2p']
                    st.metric("Total Targets", f"{total_targets:,}")
                with col3:
                    st.metric("Campaign Period", f"{results['summary_stats']['start_date']} to {results['summary_stats']['end_date']}")
                
                # Key Findings
                st.subheader("Key Findings")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**International Opportunity**: {results['summary_stats']['intl_not_p2p']} customers using international remittance but not P2P")
                    st.info(f"**Domestic Opportunity**: {results['summary_stats']['domestic_not_p2p']} customers using other services but not P2P")
                
                with col2:
                    if 'total_intl_recipients' in results['summary_stats']:
                        st.info(f"**International Recipients**: {results['summary_stats']['total_intl_recipients']} customers received international remittance")
                        st.info(f"**Avg Withdrawal Rate**: {results['summary_stats'].get('avg_withdrawal_percentage', 0):.1f}% of international received amount withdrawn")
            
            with tabs[1]:  # International Targets
                st.subheader("International Customers NOT Using P2P")
                if not results['intl_not_p2p'].empty:
                    st.dataframe(results['intl_not_p2p'], use_container_width=True)
                    st.info(f"Found {len(results['intl_not_p2p'])} international customers not using P2P")
                else:
                    st.info("No international customers found who are not using P2P")
            
            with tabs[2]:  # Domestic Targets
                st.subheader("Domestic Customers Using Other Services But NOT P2P")
                if not results['domestic_other_not_p2p'].empty:
                    st.dataframe(results['domestic_other_not_p2p'], use_container_width=True)
                    st.info(f"Found {len(results['domestic_other_not_p2p'])} domestic customers not using P2P")
                else:
                    st.info("No domestic customers found who are not using P2P")
            
            # [Other tabs remain the same...]
            # ... rest of the tab content remains the same as before ...
            
            # Download section - UPDATED
            st.markdown('<h2 class="sub-header">📥 Download Comprehensive Report</h2>', unsafe_allow_html=True)
            
            st.info("The comprehensive report includes all analysis results:")
            
            sheets_info = [
                "1. **Executive Summary**: Overall campaign summary and key metrics",
                "2. **International Targets**: Customers using international remittance but not P2P",
                "3. **Domestic Targets**: Domestic customers using other services but not P2P",
                "4. **Intl Withdrawal Segments**: International recipients segmented by withdrawal percentage",
                "5. **Intl Received & Withdrew**: Customers who received international and withdrew",
                "6. **Intl No Withdraw Other**: Received international, no withdrawal, used other services",
                "7. **Intl Only Withdrew**: Received international and only withdrew (no other services)",
                "8. **Impact Template**: Daily tracking template for campaign progress",
                "9. **Detailed Metrics**: Comprehensive performance metrics",
                "10. **Recommendations**: Actionable campaign recommendations"
            ]
            
            for info in sheets_info:
                st.write(info)
            
            # Create and download comprehensive report
            report_data = create_comprehensive_excel_report(results, filter_desc)
            if report_data:
                if OPENPYXL_AVAILABLE:
                    file_name = f"telemarketing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                    mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    label = "📊 Download Excel Report (10 Sheets)"
                else:
                    file_name = f"telemarketing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                    mime_type = "application/zip"
                    label = "📦 Download ZIP Report (10 CSV Files)"
                
                st.download_button(
                    label=label,
                    data=report_data,
                    file_name=file_name,
                    mime=mime_type,
                    use_container_width=True,
                    help="Download all analysis in one file"
                )
            
            # Call-to-action
            st.markdown("---")
            st.markdown("### 🚀 Ready to Start Your Campaign?")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                total_targets = results['summary_stats']['intl_not_p2p'] + results['summary_stats']['domestic_not_p2p']
                st.metric("Total Targets", f"{total_targets:,}", "customers to call")
            with col2:
                campaign_days = (end_date - start_date).days + 1
                daily_target = total_targets // campaign_days if campaign_days > 0 else total_targets
                st.metric("Daily Target", f"{daily_target:,}", f"over {campaign_days} days")
            with col3:
                high_withdrawal = results['summary_stats'].get('intl_withdrawal_segment_75', 0) + results['summary_stats'].get('intl_withdrawal_segment_100', 0)
                st.metric("High Withdrawal", f"{high_withdrawal:,}", "customers for retention")

if __name__ == "__main__":
    main()
