import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import warnings
warnings.filterwarnings('ignore')

# Try to import required libraries
try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False
    st.warning("openpyxl not installed. Excel export will not be available.")

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
        
        # Parse date column
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
            df['Transaction Amount'] = pd.to_numeric(df['Transaction Amount'], errors='coerce')
            df['Transaction Amount'] = df['Transaction Amount'].fillna(0)
        
        return df
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def filter_by_customer_pair(df, filter_pair):
    """Filter customers based on transaction behavior pairs"""
    
    # Define product categories
    p2p_products = ['Internal Wallet Transfer (P2P)', 'Internal Wallet Transfer', 'P2P Transfer', 'Wallet Transfer', 'P2P']
    international_remittance = ['International Remittance', 'International Transfer', 'Remittance', 'International']
    deposit_products = ['Deposit', 'Cash In', 'Deposit Customer', 'Deposit Agent']
    withdrawal_products = ['Withdrawal', 'Scan To Withdraw Agent', 'Scan To Withdraw Customer', 'Cash Out', 'Withdraw']
    
    # Get unique customers
    unique_customers = df['User Identifier'].dropna().unique()
    
    # Helper function to get customers for a product category
    def get_customers_for_product(product_list):
        mask = df['Product Name'].str.contains('|'.join(product_list), case=False, na=False)
        return set(df[mask]['User Identifier'].dropna().unique())
    
    # Get customer sets for each product category
    p2p_customers = get_customers_for_product(p2p_products)
    intl_customers = get_customers_for_product(international_remittance)
    deposit_customers = get_customers_for_product(deposit_products)
    withdrawal_customers = get_customers_for_product(withdrawal_products)
    
    # Apply the selected filter pair
    filtered_customers = set()
    filter_description = ""
    
    if filter_pair == "intl_not_p2p":
        # Customers who did International remittance but not P2P
        filtered_customers = intl_customers - p2p_customers
        filter_description = f"Customers who did International remittance but NOT P2P\n\n• International Customers: {len(intl_customers):,}\n• P2P Customers: {len(p2p_customers):,}\n• Result: {len(filtered_customers):,} customers"
    
    elif filter_pair == "p2p_not_deposit":
        # Customers who did P2P but not Deposit
        filtered_customers = p2p_customers - deposit_customers
        filter_description = f"Customers who did P2P but NOT Deposit\n\n• P2P Customers: {len(p2p_customers):,}\n• Deposit Customers: {len(deposit_customers):,}\n• Result: {len(filtered_customers):,} customers"
    
    elif filter_pair == "p2p_and_withdrawal":
        # Customers who did P2P and withdrawal
        filtered_customers = p2p_customers & withdrawal_customers
        filter_description = f"Customers who did BOTH P2P and Withdrawal\n\n• P2P Customers: {len(p2p_customers):,}\n• Withdrawal Customers: {len(withdrawal_customers):,}\n• Result: {len(filtered_customers):,} customers"
    
    elif filter_pair == "intl_and_p2p":
        # Customers who did international remittances and P2P
        filtered_customers = intl_customers & p2p_customers
        filter_description = f"Customers who did BOTH International remittance and P2P\n\n• International Customers: {len(intl_customers):,}\n• P2P Customers: {len(p2p_customers):,}\n• Result: {len(filtered_customers):,} customers"
    
    elif filter_pair == "all_customers":
        # All customers (no filter)
        filtered_customers = set(unique_customers)
        filter_description = f"All Customers\n\n• Total Unique Customers: {len(filtered_customers):,}"
    
    # Filter the dataframe to only include transactions from the selected customers
    if filtered_customers:
        filtered_df = df[df['User Identifier'].isin(filtered_customers)].copy()
    else:
        filtered_df = pd.DataFrame(columns=df.columns)
    
    return filtered_df, filter_description, len(filtered_customers)

def analyze_telemarketing_data(filtered_df):
    """Main analysis function with all reports"""
    # Define product categories
    p2p_products = ['Internal Wallet Transfer (P2P)', 'Internal Wallet Transfer', 'P2P Transfer', 'Wallet Transfer']
    international_remittance = ['International Remittance', 'International Transfer', 'Remittance']
    other_services = [
        'Deposit', 'Scan To Withdraw Agent', 'Scan To Withdraw Customer', 
        'Scan To Send', 'Ticket', 'Cash In', 'Cash Out', 'Withdrawal'
    ]
    bill_payment_services = ['Bill Payment', 'Airtime Topup', 'Utility Payment', 'Topup']
    
    # Get unique customers
    unique_customers = filtered_df['User Identifier'].dropna().unique()
    
    # REPORT 1: International remittance customers not using P2P
    intl_pattern = '|'.join(international_remittance)
    intl_mask = filtered_df['Product Name'].str.contains(intl_pattern, case=False, na=False)
    intl_customers = filtered_df[intl_mask]['User Identifier'].dropna().unique()
    
    p2p_mask = filtered_df['Product Name'].str.contains('|'.join(p2p_products), case=False, na=False)
    p2p_customers = filtered_df[p2p_mask]['User Identifier'].dropna().unique()
    
    intl_not_p2p = [cust for cust in intl_customers if cust not in p2p_customers]
    
    # Create detailed DataFrame for international customers not using P2P
    intl_details = []
    for cust_id in intl_not_p2p[:1000]:
        cust_records = filtered_df[filtered_df['User Identifier'] == cust_id]
        if not cust_records.empty:
            # Get customer name
            name_record = cust_records[cust_records['Full Name'].notna() & (cust_records['Full Name'] != 'nan')]
            full_name = name_record.iloc[0]['Full Name'] if not name_record.empty else 'Name not available'
            
            # Calculate metrics
            total_transactions = len(cust_records)
            intl_count = len(cust_records[intl_mask])
            p2p_count = len(cust_records[p2p_mask])
            
            intl_details.append({
                'User_ID': int(cust_id) if pd.notna(cust_id) else 0,
                'Full_Name': full_name,
                'International_Transaction_Count': intl_count,
                'P2P_Transaction_Count': p2p_count,
                'Total_Transactions': total_transactions,
                'Last_Transaction_Date': cust_records['Created At'].max().strftime('%Y-%m-%d') if pd.notna(cust_records['Created At'].max()) else 'Unknown',
                'First_Transaction_Date': cust_records['Created At'].min().strftime('%Y-%m-%d') if pd.notna(cust_records['Created At'].min()) else 'Unknown',
                'Customer_Since': cust_records['Created At'].min().strftime('%Y-%m-%d') if pd.notna(cust_records['Created At'].min()) else 'Unknown'
            })
    
    report1_df = pd.DataFrame(intl_details) if intl_details else pd.DataFrame(
        columns=['User_ID', 'Full_Name', 'International_Transaction_Count', 'P2P_Transaction_Count',
                'Total_Transactions', 'Last_Transaction_Date', 'First_Transaction_Date', 'Customer_Since']
    )
    
    # REPORT 2: Non-international customers using other services but not P2P
    other_mask = filtered_df['Product Name'].str.contains('|'.join(other_services + bill_payment_services), case=False, na=False)
    other_customers = filtered_df[other_mask]['User Identifier'].dropna().unique()
    
    domestic_other_not_p2p = [
        cust for cust in other_customers 
        if (cust not in intl_customers) and (cust not in p2p_customers)
    ]
    
    domestic_details = []
    for cust_id in domestic_other_not_p2p[:1000]:
        cust_records = filtered_df[filtered_df['User Identifier'] == cust_id]
        if not cust_records.empty:
            # Get customer name
            name_record = cust_records[cust_records['Full Name'].notna() & (cust_records['Full Name'] != 'nan')]
            full_name = name_record.iloc[0]['Full Name'] if not name_record.empty else 'Name not available'
            
            # Get services used
            services_used = cust_records['Product Name'].dropna().unique()
            other_services_list = [str(s) for s in services_used if any(
                service.lower() in str(s).lower() for service in other_services + bill_payment_services
            )]
            
            domestic_details.append({
                'User_ID': int(cust_id) if pd.notna(cust_id) else 0,
                'Full_Name': full_name,
                'Total_Transactions': len(cust_records),
                'Other_Services_Count': len(other_services_list),
                'Other_Services_Used': ', '.join(other_services_list[:5]) if other_services_list else 'None',
                'Last_Transaction_Date': cust_records['Created At'].max().strftime('%Y-%m-%d') if pd.notna(cust_records['Created At'].max()) else 'Unknown',
                'First_Transaction_Date': cust_records['Created At'].min().strftime('%Y-%m-%d') if pd.notna(cust_records['Created At'].min()) else 'Unknown',
                'Customer_Since': cust_records['Created At'].min().strftime('%Y-%m-%d') if pd.notna(cust_records['Created At'].min()) else 'Unknown'
            })
    
    report2_df = pd.DataFrame(domestic_details) if domestic_details else pd.DataFrame(
        columns=['User_ID', 'Full_Name', 'Total_Transactions', 'Other_Services_Count',
                'Other_Services_Used', 'Last_Transaction_Date', 'First_Transaction_Date', 'Customer_Since']
    )
    
    # REPORT 3: Daily Impact Report Template
    if len(filtered_df) > 0 and filtered_df['Created At'].notna().any():
        start_date = filtered_df['Created At'].min().date()
        end_date = filtered_df['Created At'].max().date()
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
    else:
        dates = pd.date_range(end=datetime.now().date(), periods=7, freq='D')
    
    impact_template = []
    for date in dates:
        impact_template.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Total_Customers_Called': 0,
            'Customers_Reached': 0,
            'Customers_Not_Reached': 0,
            'P2P_Adoption_After_Call': 0,
            'Conversion_Rate_Percent': 0.0,
            'Average_Transaction_Value': 0.0,
            'Customer_Feedback': '',
            'Follow_up_Required': '',
            'Notes': ''
        })
    
    impact_df = pd.DataFrame(impact_template)
    
    # Summary statistics
    summary_stats = {
        'total_transactions': len(filtered_df),
        'unique_customers': len(unique_customers),
        'intl_customers': len(intl_customers),
        'intl_not_p2p': len(intl_not_p2p),
        'p2p_customers': len(p2p_customers),
        'other_customers': len(other_customers),
        'domestic_not_p2p': len(domestic_other_not_p2p),
        'start_date': filtered_df['Created At'].min().strftime('%Y-%m-%d') if len(filtered_df) > 0 and filtered_df['Created At'].notna().any() else 'N/A',
        'end_date': filtered_df['Created At'].max().strftime('%Y-%m-%d') if len(filtered_df) > 0 and filtered_df['Created At'].notna().any() else 'N/A'
    }
    
    return {
        'intl_not_p2p': report1_df,
        'domestic_other_not_p2p': report2_df,
        'impact_template': impact_df,
        'summary_stats': summary_stats
    }

def create_excel_workbook(results, filter_desc=""):
    """Create Excel workbook with all reports"""
    try:
        # Create a BytesIO object for the Excel file
        output = io.BytesIO()
        
        # Create Excel writer
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
                ['Total Addressable Market', results['summary_stats']['intl_not_p2p'] + results['summary_stats']['domestic_not_p2p']]
            ]
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Executive_Summary', index=False, header=False)
            
            # Sheet 2: International Targets
            if not results['intl_not_p2p'].empty:
                results['intl_not_p2p'].to_excel(writer, sheet_name='International_Targets', index=False)
            
            # Sheet 3: Domestic Targets
            if not results['domestic_other_not_p2p'].empty:
                results['domestic_other_not_p2p'].to_excel(writer, sheet_name='Domestic_Targets', index=False)
            
            # Sheet 4: Impact Template
            results['impact_template'].to_excel(writer, sheet_name='Impact_Template', index=False)
            
            # Sheet 5: Recommendations
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
                ['CAMPAIGN IMPLEMENTATION'],
                ['8. Use the Impact Template for daily tracking'],
                ['9. Schedule calls during peak transaction hours'],
                ['10. Train agents on P2P benefits and features'],
                ['11. Track conversion rates weekly'],
                ['12. Collect customer feedback systematically'],
                ['13. Follow up with unreached customers within 48 hours']
            ]
            
            rec_df = pd.DataFrame(recommendations)
            rec_df.to_excel(writer, sheet_name='Recommendations', index=False, header=False)
        
        output.seek(0)
        return output
    
    except Exception as e:
        st.error(f"Error creating Excel file: {str(e)}")
        return None

def filter_data(df, start_date, end_date, product_filter, customer_type_filter, pair_filter="all_customers"):
    """Filter data based on selected criteria"""
    filtered_df = df.copy()
    
    # First apply customer pair filter if not "all_customers"
    if pair_filter != "all_customers":
        filtered_df, filter_desc, customer_count = filter_by_customer_pair(filtered_df, pair_filter)
    else:
        filter_desc = "Showing all customers"
        customer_count = filtered_df['User Identifier'].nunique()
    
    # Date filter
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['Created At'] >= pd.Timestamp(start_date)) & 
            (filtered_df['Created At'] <= pd.Timestamp(end_date + timedelta(days=1)))
        ]
    
    # Product filter
    if product_filter and product_filter != 'All':
        filtered_df = filtered_df[filtered_df['Product Name'].str.contains(product_filter, case=False, na=False)]
    
    # Customer type filter
    if customer_type_filter and customer_type_filter != 'All':
        if customer_type_filter == 'Customer Only':
            filtered_df = filtered_df[
                (filtered_df['Entity Name'].str.contains('Customer', case=False, na=False)) |
                (filtered_df['Entity Name'].isna()) |
                (filtered_df['Entity Name'] == '')
            ]
        elif customer_type_filter == 'Vendor/Agent Only':
            filtered_df = filtered_df[
                (filtered_df['Entity Name'].str.contains('Vendor|Agent', case=False, na=False))
            ]
    
    return filtered_df, filter_desc, customer_count

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
                    "intl_and_p2p": "International remittance AND P2P"
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
                
                # Add sample data preview
                with st.expander("📋 Data Preview"):
                    st.dataframe(df.head(50), use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Rows", f"{len(df):,}")
                    with col2:
                        date_range = f"{min_date} to {max_date}" if df['Created At'].notna().any() else "No dates"
                        st.metric("Date Range", date_range)
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
                "📋 Impact Template", 
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
                    st.info(f"**P2P Penetration**: {results['summary_stats']['p2p_customers']} customers already using P2P ({results['summary_stats']['p2p_customers']/max(results['summary_stats']['unique_customers'],1)*100:.1f}%)")
            
            with tabs[1]:  # International Targets
                st.subheader("International Customers NOT Using P2P")
                if not results['intl_not_p2p'].empty:
                    st.dataframe(results['intl_not_p2p'], use_container_width=True)
                    st.info(f"Found {len(results['intl_not_p2p'])} international customers not using P2P")
                    
                    # Additional insights
                    col1, col2 = st.columns(2)
                    with col1:
                        avg_intl_tx = results['intl_not_p2p']['International_Transaction_Count'].mean()
                        st.metric("Avg Intl Transactions", f"{avg_intl_tx:.1f}")
                    with col2:
                        recent_customers = len(results['intl_not_p2p'][results['intl_not_p2p']['Last_Transaction_Date'] >= '2024-01-01'])
                        st.metric("Recent Activity", f"{recent_customers}", "Since 2024")
                else:
                    st.info("No international customers found who are not using P2P")
            
            with tabs[2]:  # Domestic Targets
                st.subheader("Domestic Customers Using Other Services But NOT P2P")
                if not results['domestic_other_not_p2p'].empty:
                    st.dataframe(results['domestic_other_not_p2p'], use_container_width=True)
                    st.info(f"Found {len(results['domestic_other_not_p2p'])} domestic customers not using P2P")
                    
                    # Additional insights
                    col1, col2 = st.columns(2)
                    with col1:
                        avg_other_services = results['domestic_other_not_p2p']['Other_Services_Count'].mean()
                        st.metric("Avg Other Services", f"{avg_other_services:.1f}")
                    with col2:
                        active_customers = len(results['domestic_other_not_p2p'][results['domestic_other_not_p2p']['Total_Transactions'] > 5])
                        st.metric("Active Customers", f"{active_customers}", "5+ transactions")
                else:
                    st.info("No domestic customers found who are not using P2P")
            
            with tabs[3]:  # Impact Template
                st.subheader("Daily Impact Report Template")
                st.dataframe(results['impact_template'], use_container_width=True)
                st.info("Use this template to track daily calling campaign results")
            
            with tabs[4]:  # Recommendations
                st.subheader("Campaign Recommendations")
                
                st.info("**Priority 1: International Customer Engagement**")
                st.write(f"1. Target {results['summary_stats']['intl_not_p2p']} international customers not using P2P")
                st.write("2. Focus on customers with recent international transactions")
                st.write("3. Offer P2P fee waivers for first 3 transactions")
                st.write("4. Bundle P2P with international remittance services")
                
                st.info("**Priority 2: Domestic Customer Cross-Sell**")
                st.write(f"5. Engage {results['summary_stats']['domestic_not_p2p']} domestic customers")
                st.write("6. Promote P2P as faster alternative to current services")
                st.write("7. Offer loyalty rewards for P2P adoption")
                
                st.info("**Campaign Implementation**")
                st.write("8. Use the Impact Template for daily tracking")
                st.write("9. Schedule calls during peak transaction hours")
                st.write("10. Train agents on P2P benefits and features")
                st.write("11. Track conversion rates weekly")
                st.write("12. Collect customer feedback systematically")
                st.write("13. Follow up with unreached customers within 48 hours")
            
            # Download section
            st.markdown('<h2 class="sub-header">📥 Download Comprehensive Report</h2>', unsafe_allow_html=True)
            
            if not OPENPYXL_AVAILABLE:
                st.warning("⚠️ openpyxl is not installed. Excel export is not available.")
                st.info("To enable Excel export, install openpyxl by running: `pip install openpyxl`")
            else:
                st.info("The comprehensive Excel report includes 5 sheets with all analysis results:")
                
                sheets_info = [
                    "1. **Executive_Summary**: Overall campaign summary and key metrics",
                    "2. **International_Targets**: Customers using international remittance but not P2P",
                    "3. **Domestic_Targets**: Domestic customers using other services but not P2P",
                    "4. **Impact_Template**: Daily tracking template for campaign progress",
                    "5. **Recommendations**: Actionable campaign recommendations"
                ]
                
                for info in sheets_info:
                    st.write(info)
                
                # Create and download comprehensive Excel report
                excel_data = create_excel_workbook(results, filter_desc)
                if excel_data:
                    st.download_button(
                        label="📊 Download Excel Report (5 Sheets)",
                        data=excel_data,
                        file_name=f"telemarketing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        help="Download all analysis in one Excel file with 5 detailed sheets"
                    )
            
            # Individual download buttons for convenience
            st.markdown("### Individual Downloads")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if not results['intl_not_p2p'].empty:
                    csv_intl = results['intl_not_p2p'].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="🎯 International Targets (CSV)",
                        data=csv_intl,
                        file_name="international_targets.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col2:
                if not results['domestic_other_not_p2p'].empty:
                    csv_domestic = results['domestic_other_not_p2p'].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="🏠 Domestic Targets (CSV)",
                        data=csv_domestic,
                        file_name="domestic_targets.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col3:
                csv_impact = results['impact_template'].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📋 Impact Template (CSV)",
                    data=csv_impact,
                    file_name="impact_template.csv",
                    mime="text/csv",
                    use_container_width=True
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
                st.metric("Success Rate Goal", "20%", "minimum conversion target")

if __name__ == "__main__":
    main()
