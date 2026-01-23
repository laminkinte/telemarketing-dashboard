import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import warnings
warnings.filterwarnings('ignore')

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

def filter_by_customer_pair(df, filter_pair):
    """Filter customers based on transaction behavior pairs"""
    
    # Define product categories
    p2p_products = ['Internal Wallet Transfer (P2P)', 'Internal Wallet Transfer', 'P2P Transfer', 'Wallet Transfer', 'P2P']
    international_remittance = ['International Remittance', 'International Transfer', 'Remittance', 'International', 'Receive International']
    deposit_products = ['Deposit', 'Cash In', 'Deposit Customer', 'Deposit Agent']
    withdrawal_products = ['Withdrawal', 'Scan To Withdraw Agent', 'Scan To Withdraw Customer', 'Cash Out', 'Withdraw']
    receive_products = ['Receive', 'Credit', 'Incoming', 'Received']
    
    # Combine receive with international for receiving international remittance
    receive_intl_products = international_remittance + [f"{r} {i}" for r in receive_products for i in international_remittance]
    
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
    receive_intl_customers = get_customers_for_product(receive_intl_products)
    
    # Get customers using other services (excluding receive intl and withdrawal)
    other_services_products = p2p_products + deposit_products + ['Bill Payment', 'Airtime', 'Utility', 'Topup', 'Ticket', 'Scan', 'Send']
    other_services_customers = get_customers_for_product(other_services_products)
    
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
    
    elif filter_pair == "intl_received_withdraw":
        # Customers who Received International Remittance and followed by withdrawal
        filtered_customers = receive_intl_customers & withdrawal_customers
        filter_description = f"Customers who Received International Remittance AND Withdrew\n\n• International Recipients: {len(receive_intl_customers):,}\n• Withdrawal Customers: {len(withdrawal_customers):,}\n• Result: {len(filtered_customers):,} customers"
    
    elif filter_pair == "intl_received_no_withdraw":
        # Customers who Received International Remittance did not withdraw and use other services
        filtered_customers = receive_intl_customers - withdrawal_customers
        filter_description = f"Customers who Received International Remittance but NO Withdrawal\n\n• International Recipients: {len(receive_intl_customers):,}\n• Withdrawal Customers: {len(withdrawal_customers):,}\n• Result: {len(filtered_customers):,} customers"
    
    elif filter_pair == "intl_received_only_withdraw":
        # Customers who Received International Remittance and only withdraw and did not use any other services
        filtered_customers = receive_intl_customers & withdrawal_customers - other_services_customers
        filter_description = f"Customers who Received International Remittance and ONLY Withdrew\n\n• International Recipients: {len(receive_intl_customers):,}\n• Pure Withdrawal Only: {len(filtered_customers):,} customers"
    
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

def analyze_intl_withdrawal_segments(df):
    """Analyze withdrawal behavior segmentation for international remittance recipients"""
    
    # Define product patterns - looking for RECEIVE international transactions
    receive_intl_keywords = ['Receive International', 'International Receive', 'International Credit', 
                           'Remittance Receive', 'Receive Remittance', 'International Incoming']
    receive_intl_pattern = '|'.join(receive_intl_keywords)
    
    # Also consider general international transactions if receive specific not found
    intl_pattern = '|'.join(['International Remittance', 'International Transfer', 'Remittance', 'International'])
    
    withdrawal_pattern = '|'.join(['Withdrawal', 'Scan To Withdraw', 'Cash Out', 'Withdraw'])
    
    # Get all international recipients - first try receive-specific, then general
    receive_mask = df['Product Name'].str.contains(receive_intl_pattern, case=False, na=False)
    if receive_mask.sum() == 0:  # If no receive-specific transactions found
        receive_mask = df['Product Name'].str.contains(intl_pattern, case=False, na=False)
    
    receive_intl_customers = df[receive_mask]['User Identifier'].dropna().unique()
    
    results = {
        'withdrawal_segments': {},
        'detailed_analysis': pd.DataFrame(),
        'summary_stats': {},
        'segment_data': pd.DataFrame()
    }
    
    if len(receive_intl_customers) == 0:
        return results
    
    # Analyze each international recipient
    segment_analysis = []
    detailed_records = []
    
    for cust_id in receive_intl_customers[:2000]:  # Increased limit for better analysis
        cust_data = df[df['User Identifier'] == cust_id].copy()
        
        # Get customer name
        name_record = cust_data[cust_data['Full Name'].notna() & (cust_data['Full Name'] != 'nan')]
        full_name = name_record.iloc[0]['Full Name'] if not name_record.empty else 'Name not available'
        
        # Get receive international transactions
        receive_transactions = cust_data[receive_mask]
        if len(receive_transactions) == 0:
            continue  # Skip if no receive transactions found
        
        # Get withdrawal transactions
        withdrawal_mask = cust_data['Product Name'].str.contains(withdrawal_pattern, case=False, na=False)
        withdrawal_transactions = cust_data[withdrawal_mask]
        
        # Calculate amounts - use transaction amounts if available, otherwise count transactions
        if 'Transaction Amount' in df.columns and df['Transaction Amount'].notna().any():
            total_received = receive_transactions['Transaction Amount'].sum()
            total_withdrawn = withdrawal_transactions['Transaction Amount'].sum()
        else:
            # If no amount data, use transaction counts
            total_received = len(receive_transactions)
            total_withdrawn = len(withdrawal_transactions)
        
        # Calculate withdrawal percentage AGAINST INTERNATIONAL REMITTANCE RECEIVED
        if total_received > 0:
            withdrawal_percentage = (total_withdrawn / total_received) * 100
        else:
            withdrawal_percentage = 0
        
        # Determine segment based on accurate percentage ranges
        if withdrawal_percentage <= 25:
            segment = '≤25%'
            segment_desc = 'Withdrawal ≤ 25% of International Received'
        elif withdrawal_percentage <= 50:
            segment = '25%-50%'
            segment_desc = '25% < Withdrawal ≤ 50% of International Received'
        elif withdrawal_percentage <= 75:
            segment = '50%-75%'
            segment_desc = '50% < Withdrawal ≤ 75% of International Received'
        else:
            segment = '75%-100%'
            segment_desc = '75% < Withdrawal ≤ 100% of International Received'
        
        # Get other services used (excluding receive intl and withdrawal)
        other_mask = ~receive_mask & ~withdrawal_mask
        other_services = cust_data[other_mask]
        other_services_list = other_services['Product Name'].dropna().unique()
        
        # FIXED: Handle case where other_services_list might be empty or None
        if other_services_list is not None and len(other_services_list) > 0:
            other_services_str = ', '.join([str(s) for s in other_services_list[:5]])
        else:
            other_services_str = 'None'
        
        # Prepare customer record for segment analysis
        segment_analysis.append({
            'User_ID': cust_id,
            'Full_Name': full_name,
            'Total_International_Received': total_received,
            'Total_Withdrawn': total_withdrawn,
            'Withdrawal_Percentage': withdrawal_percentage,
            'Segment': segment,
            'Segment_Description': segment_desc,
            'International_Receive_Count': len(receive_transactions),
            'Withdrawal_Count': len(withdrawal_transactions),
            'Other_Services_Count': len(other_services_list) if other_services_list is not None else 0,
            'First_Transaction_Date': cust_data['Created At'].min().strftime('%Y-%m-%d') if pd.notna(cust_data['Created At'].min()) else 'Unknown',
            'Last_Transaction_Date': cust_data['Created At'].max().strftime('%Y-%m-%d') if pd.notna(cust_data['Created At'].max()) else 'Unknown'
        })
        
        # Detailed records for export
        detailed_records.append({
            'User_ID': cust_id,
            'Full_Name': full_name,
            'Segment': segment,
            'Segment_Description': segment_desc,
            'Withdrawal_Percentage': f"{withdrawal_percentage:.1f}%",
            'Withdrawal_Percentage_Raw': withdrawal_percentage,
            'Total_International_Received_Amount': total_received,
            'Total_Withdrawn_Amount': total_withdrawn,
            'International_Receive_Count': len(receive_transactions),
            'Withdrawal_Count': len(withdrawal_transactions),
            'Other_Services_Used': other_services_str,
            'Other_Services_Count': len(other_services_list) if other_services_list is not None else 0,
            'First_International_Date': receive_transactions['Created At'].min().strftime('%Y-%m-%d') if len(receive_transactions) > 0 else 'None',
            'Last_International_Date': receive_transactions['Created At'].max().strftime('%Y-%m-%d') if len(receive_transactions) > 0 else 'None',
            'First_Withdrawal_Date': withdrawal_transactions['Created At'].min().strftime('%Y-%m-%d') if len(withdrawal_transactions) > 0 else 'None',
            'Last_Withdrawal_Date': withdrawal_transactions['Created At'].max().strftime('%Y-%m-%d') if len(withdrawal_transactions) > 0 else 'None',
            'Customer_Since': cust_data['Created At'].min().strftime('%Y-%m-%d') if pd.notna(cust_data['Created At'].min()) else 'Unknown'
        })
    
    # Convert to DataFrames
    if segment_analysis:
        segment_df = pd.DataFrame(segment_analysis)
        detailed_df = pd.DataFrame(detailed_records)
        
        # Calculate segment distribution
        segment_distribution = segment_df['Segment'].value_counts()
        
        # Reindex to ensure all segments are present
        all_segments = ['≤25%', '25%-50%', '50%-75%', '75%-100%']
        segment_distribution = segment_distribution.reindex(all_segments, fill_value=0)
        
        # Calculate percentages
        total_customers = len(segment_df)
        if total_customers > 0:
            segment_percentages = (segment_distribution / total_customers * 100).round(2)
        else:
            segment_percentages = pd.Series([0, 0, 0, 0], index=all_segments)
        
        results['withdrawal_segments'] = {
            'distribution': segment_distribution,
            'percentages': segment_percentages,
            'total_customers': total_customers
        }
        
        results['detailed_analysis'] = detailed_df
        results['segment_data'] = segment_df
        
        # Summary statistics
        withdrawal_percentages = segment_df['Withdrawal_Percentage']
        
        results['summary_stats'] = {
            'total_intl_recipients': len(receive_intl_customers),
            'analyzed_customers': total_customers,
            'avg_withdrawal_percentage': withdrawal_percentages.mean(),
            'median_withdrawal_percentage': withdrawal_percentages.median(),
            'max_withdrawal_percentage': withdrawal_percentages.max(),
            'min_withdrawal_percentage': withdrawal_percentages.min(),
            'std_withdrawal_percentage': withdrawal_percentages.std(),
            'total_intl_received_amount': segment_df['Total_International_Received'].sum(),
            'total_withdrawn_amount': segment_df['Total_Withdrawn'].sum(),
            'overall_withdrawal_percentage': (segment_df['Total_Withdrawn'].sum() / segment_df['Total_International_Received'].sum() * 100) if segment_df['Total_International_Received'].sum() > 0 else 0,
            'segment_25_count': len(segment_df[segment_df['Segment'] == '≤25%']),
            'segment_50_count': len(segment_df[segment_df['Segment'] == '25%-50%']),
            'segment_75_count': len(segment_df[segment_df['Segment'] == '50%-75%']),
            'segment_100_count': len(segment_df[segment_df['Segment'] == '75%-100%'])
        }
    
    return results

def analyze_specific_intl_groups(df):
    """Analyze specific international remittance recipient groups"""
    
    # Define product patterns
    receive_intl_keywords = ['Receive International', 'International Receive', 'International Credit', 
                           'Remittance Receive', 'Receive Remittance', 'International Incoming']
    receive_intl_pattern = '|'.join(receive_intl_keywords)
    
    # Also consider general international transactions if receive specific not found
    intl_pattern = '|'.join(['International Remittance', 'International Transfer', 'Remittance', 'International'])
    
    withdrawal_pattern = '|'.join(['Withdrawal', 'Scan To Withdraw', 'Cash Out', 'Withdraw'])
    
    # Get all international recipients
    receive_mask = df['Product Name'].str.contains(receive_intl_pattern, case=False, na=False)
    if receive_mask.sum() == 0:  # If no receive-specific transactions found
        receive_mask = df['Product Name'].str.contains(intl_pattern, case=False, na=False)
    
    receive_intl_customers = df[receive_mask]['User Identifier'].dropna().unique()
    
    # Define other services patterns
    other_services_patterns = [
        'P2P', 'Deposit', 'Bill Payment', 'Airtime', 'Utility', 'Topup', 
        'Ticket', 'Scan', 'Send', 'Payment', 'Transfer', 'Wallet'
    ]
    other_services_pattern = '|'.join(other_services_patterns)
    
    groups = {
        'received_withdrew': [],      # Group 1: Received AND Withdrew
        'received_no_withdraw_other': [],  # Group 2: Received, NO Withdrawal, but Other Services
        'received_only_withdraw': []  # Group 3: Received and ONLY Withdrew (no other services)
    }
    
    for cust_id in receive_intl_customers[:2000]:  # Increased limit
        cust_data = df[df['User Identifier'] == cust_id].copy()
        
        # Get customer name
        name_record = cust_data[cust_data['Full Name'].notna() & (cust_data['Full Name'] != 'nan')]
        full_name = name_record.iloc[0]['Full Name'] if not name_record.empty else 'Name not available'
        
        # Check transaction types
        has_receive_intl = receive_mask.any()
        has_withdrawal = cust_data['Product Name'].str.contains(withdrawal_pattern, case=False, na=False).any()
        
        # Check for other services (excluding receive intl and withdrawal)
        other_mask = cust_data['Product Name'].str.contains(other_services_pattern, case=False, na=False)
        # Exclude receive intl and withdrawal from other services
        receive_intl_mask = cust_data['Product Name'].str.contains(receive_intl_pattern + '|' + intl_pattern, case=False, na=False)
        withdrawal_mask = cust_data['Product Name'].str.contains(withdrawal_pattern, case=False, na=False)
        other_services_mask = other_mask & ~receive_intl_mask & ~withdrawal_mask
        has_other_services = other_services_mask.any()
        
        # Calculate metrics
        receive_count = len(cust_data[receive_mask])
        withdrawal_count = len(cust_data[cust_data['Product Name'].str.contains(withdrawal_pattern, case=False, na=False)])
        other_count = len(cust_data[other_services_mask])
        
        # Calculate amounts if available
        if 'Transaction Amount' in df.columns and df['Transaction Amount'].notna().any():
            total_received = cust_data[receive_mask]['Transaction Amount'].sum()
            total_withdrawn = cust_data[cust_data['Product Name'].str.contains(withdrawal_pattern, case=False, na=False)]['Transaction Amount'].sum()
        else:
            total_received = receive_count
            total_withdrawn = withdrawal_count
        
        # Calculate withdrawal percentage AGAINST INTERNATIONAL REMITTANCE RECEIVED
        withdrawal_percentage = (total_withdrawn / total_received * 100) if total_received > 0 else 0
        
        # Prepare customer record
        customer_record = {
            'User_ID': cust_id,
            'Full_Name': full_name,
            'International_Receive_Count': receive_count,
            'Withdrawal_Count': withdrawal_count,
            'Other_Services_Count': other_count,
            'Total_International_Received': total_received,
            'Total_Withdrawn': total_withdrawn,
            'Withdrawal_Percentage': withdrawal_percentage,  # Percentage against international received
            'First_Transaction_Date': cust_data['Created At'].min().strftime('%Y-%m-%d') if pd.notna(cust_data['Created At'].min()) else 'Unknown',
            'Last_Transaction_Date': cust_data['Created At'].max().strftime('%Y-%m-%d') if pd.notna(cust_data['Created At'].max()) else 'Unknown',
            'Customer_Since': cust_data['Created At'].min().strftime('%Y-%m-%d') if pd.notna(cust_data['Created At'].min()) else 'Unknown'
        }
        
        # Categorize customer based on ACCURATE logic
        if has_receive_intl and has_withdrawal:
            # Group 1: Received AND Withdrew
            groups['received_withdrew'].append(customer_record)
        
        if has_receive_intl and not has_withdrawal and has_other_services:
            # Group 2: Received, NO Withdrawal, but has Other Services
            groups['received_no_withdraw_other'].append(customer_record)
        
        if has_receive_intl and has_withdrawal and not has_other_services:
            # Group 3: Received and ONLY Withdrew (no other services at all)
            groups['received_only_withdraw'].append(customer_record)
    
    # Convert to DataFrames with proper column names
    results = {}
    for group_name, records in groups.items():
        if records:
            df_group = pd.DataFrame(records)
            
            # Add calculated columns
            if not df_group.empty:
                # Add ratio columns
                df_group['Withdrawal_Ratio'] = df_group.apply(
                    lambda x: f"{x['Withdrawal_Count']}/{x['International_Receive_Count']}" if x['International_Receive_Count'] > 0 else "0/0",
                    axis=1
                )
                
                df_group['Withdrawal_Percentage_Formatted'] = df_group['Withdrawal_Percentage'].apply(lambda x: f"{x:.1f}%")
                
                # Add activity level based on transaction counts
                df_group['Activity_Level'] = df_group.apply(
                    lambda x: 'High' if (x['International_Receive_Count'] + x['Withdrawal_Count'] + x['Other_Services_Count']) > 10 
                    else 'Medium' if (x['International_Receive_Count'] + x['Withdrawal_Count'] + x['Other_Services_Count']) > 5 
                    else 'Low', axis=1
                )
            
            results[group_name] = df_group
        else:
            results[group_name] = pd.DataFrame()
    
    return results

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
            
            # Get transaction amounts if available
            if 'Transaction Amount' in filtered_df.columns:
                intl_amount = cust_records[intl_mask]['Transaction Amount'].sum()
            else:
                intl_amount = intl_count
            
            intl_details.append({
                'User_ID': int(cust_id) if pd.notna(cust_id) else 0,
                'Full_Name': full_name,
                'International_Transaction_Count': intl_count,
                'International_Transaction_Amount': intl_amount,
                'P2P_Transaction_Count': p2p_count,
                'Total_Transactions': total_transactions,
                'Last_Transaction_Date': cust_records['Created At'].max().strftime('%Y-%m-%d') if pd.notna(cust_records['Created At'].max()) else 'Unknown',
                'First_Transaction_Date': cust_records['Created At'].min().strftime('%Y-%m-%d') if pd.notna(cust_records['Created At'].min()) else 'Unknown',
                'Customer_Since': cust_records['Created At'].min().strftime('%Y-%m-%d') if pd.notna(cust_records['Created At'].min()) else 'Unknown'
            })
    
    report1_df = pd.DataFrame(intl_details) if intl_details else pd.DataFrame(
        columns=['User_ID', 'Full_Name', 'International_Transaction_Count', 'International_Transaction_Amount',
                'P2P_Transaction_Count', 'Total_Transactions', 'Last_Transaction_Date', 
                'First_Transaction_Date', 'Customer_Since']
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
            
            # Get transaction amounts if available
            if 'Transaction Amount' in filtered_df.columns:
                total_amount = cust_records['Transaction Amount'].sum()
                other_amount = cust_records[other_mask]['Transaction Amount'].sum()
            else:
                total_amount = len(cust_records)
                other_amount = len(cust_records[other_mask])
            
            domestic_details.append({
                'User_ID': int(cust_id) if pd.notna(cust_id) else 0,
                'Full_Name': full_name,
                'Total_Transactions': len(cust_records),
                'Other_Services_Count': len(other_services_list),
                'Other_Services_Used': ', '.join(other_services_list[:5]) if other_services_list else 'None',
                'Total_Transaction_Amount': total_amount,
                'Other_Services_Amount': other_amount,
                'Last_Transaction_Date': cust_records['Created At'].max().strftime('%Y-%m-%d') if pd.notna(cust_records['Created At'].max()) else 'Unknown',
                'First_Transaction_Date': cust_records['Created At'].min().strftime('%Y-%m-%d') if pd.notna(cust_records['Created At'].min()) else 'Unknown',
                'Customer_Since': cust_records['Created At'].min().strftime('%Y-%m-%d') if pd.notna(cust_records['Created At'].min()) else 'Unknown'
            })
    
    report2_df = pd.DataFrame(domestic_details) if domestic_details else pd.DataFrame(
        columns=['User_ID', 'Full_Name', 'Total_Transactions', 'Other_Services_Count',
                'Other_Services_Used', 'Total_Transaction_Amount', 'Other_Services_Amount',
                'Last_Transaction_Date', 'First_Transaction_Date', 'Customer_Since']
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
    
    # New reports for international remittance recipients
    intl_withdrawal_segments = analyze_intl_withdrawal_segments(filtered_df)
    specific_intl_groups = analyze_specific_intl_groups(filtered_df)
    
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
    
    # Add international withdrawal segment stats
    if 'summary_stats' in intl_withdrawal_segments:
        summary_stats.update({
            'intl_withdrawal_segment_25': intl_withdrawal_segments['summary_stats'].get('segment_25_count', 0),
            'intl_withdrawal_segment_50': intl_withdrawal_segments['summary_stats'].get('segment_50_count', 0),
            'intl_withdrawal_segment_75': intl_withdrawal_segments['summary_stats'].get('segment_75_count', 0),
            'intl_withdrawal_segment_100': intl_withdrawal_segments['summary_stats'].get('segment_100_count', 0),
            'total_intl_recipients': intl_withdrawal_segments['summary_stats'].get('analyzed_customers', 0),
            'avg_withdrawal_percentage': intl_withdrawal_segments['summary_stats'].get('avg_withdrawal_percentage', 0),
            'total_intl_received_amount': intl_withdrawal_segments['summary_stats'].get('total_intl_received_amount', 0)
        })
    
    # Add specific intl group counts
    summary_stats.update({
        'received_withdrew_count': len(specific_intl_groups.get('received_withdrew', [])),
        'received_no_withdraw_other_count': len(specific_intl_groups.get('received_no_withdraw_other', [])),
        'received_only_withdraw_count': len(specific_intl_groups.get('received_only_withdraw', []))
    })
    
    return {
        'intl_not_p2p': report1_df,
        'domestic_other_not_p2p': report2_df,
        'impact_template': impact_df,
        'intl_withdrawal_segments': intl_withdrawal_segments,
        'specific_intl_groups': specific_intl_groups,
        'summary_stats': summary_stats
    }

def create_comprehensive_excel_report(results, filter_desc=""):
    """Create comprehensive Excel report with all sheets"""
    output = io.BytesIO()
    
    try:
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
                ['Total International Amount Received', f"{results['summary_stats'].get('total_intl_received_amount', 0):,.2f}"],
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
                # Add summary header
                intl_summary = [
                    ['INTERNATIONAL CUSTOMERS NOT USING P2P'],
                    ['Total Customers', len(results['intl_not_p2p'])],
                    ['Average International Transactions per Customer', f"{results['intl_not_p2p']['International_Transaction_Count'].mean():.1f}"],
                    [''],
                    ['Detailed Customer List:']
                ]
                
                intl_summary_df = pd.DataFrame(intl_summary)
                startrow = len(intl_summary_df) + 1
                intl_summary_df.to_excel(writer, sheet_name='International_Targets', index=False, header=False)
                results['intl_not_p2p'].to_excel(writer, sheet_name='International_Targets', startrow=startrow, index=False)
            
            # Sheet 3: Domestic Targets
            if not results['domestic_other_not_p2p'].empty:
                domestic_summary = [
                    ['DOMESTIC CUSTOMERS NOT USING P2P'],
                    ['Total Customers', len(results['domestic_other_not_p2p'])],
                    ['Average Other Services per Customer', f"{results['domestic_other_not_p2p']['Other_Services_Count'].mean():.1f}"],
                    [''],
                    ['Detailed Customer List:']
                ]
                
                domestic_summary_df = pd.DataFrame(domestic_summary)
                startrow = len(domestic_summary_df) + 1
                domestic_summary_df.to_excel(writer, sheet_name='Domestic_Targets', index=False, header=False)
                results['domestic_other_not_p2p'].to_excel(writer, sheet_name='Domestic_Targets', startrow=startrow, index=False)
            
            # Sheet 4: International Withdrawal Segments
            if 'detailed_analysis' in results['intl_withdrawal_segments'] and not results['intl_withdrawal_segments']['detailed_analysis'].empty:
                seg_data = results['intl_withdrawal_segments']
                detailed_df = seg_data['detailed_analysis']
                
                segment_summary = [
                    ['INTERNATIONAL RECIPIENTS - WITHDRAWAL SEGMENT ANALYSIS'],
                    ['IMPORTANT: Withdrawal Percentage is calculated as (Total Withdrawn / Total International Received) * 100%'],
                    [''],
                    ['Total Customers Analyzed', seg_data['summary_stats'].get('analyzed_customers', 0)],
                    ['Total International Amount Received', f"{seg_data['summary_stats'].get('total_intl_received_amount', 0):,.2f}"],
                    ['Total Amount Withdrawn', f"{seg_data['summary_stats'].get('total_withdrawn_amount', 0):,.2f}"],
                    ['Overall Withdrawal Percentage', f"{seg_data['summary_stats'].get('overall_withdrawal_percentage', 0):.1f}% of International Received"],
                    [''],
                    ['SEGMENT DISTRIBUTION (Based on Withdrawal % of International Received)'],
                    ['Segment', 'Count', 'Percentage', 'Description'],
                    ['≤25%', seg_data['summary_stats'].get('segment_25_count', 0), f"{seg_data['withdrawal_segments']['percentages'].get('≤25%', 0):.1f}%", 'Withdrew ≤25% of International Received'],
                    ['25%-50%', seg_data['summary_stats'].get('segment_50_count', 0), f"{seg_data['withdrawal_segments']['percentages'].get('25%-50%', 0):.1f}%", 'Withdrew 25%-50% of International Received'],
                    ['50%-75%', seg_data['summary_stats'].get('segment_75_count', 0), f"{seg_data['withdrawal_segments']['percentages'].get('50%-75%', 0):.1f}%", 'Withdrew 50%-75% of International Received'],
                    ['75%-100%', seg_data['summary_stats'].get('segment_100_count', 0), f"{seg_data['withdrawal_segments']['percentages'].get('75%-100%', 0):.1f}%", 'Withdrew 75%-100% of International Received'],
                    [''],
                    ['DETAILED CUSTOMER ANALYSIS:']
                ]
                
                segment_summary_df = pd.DataFrame(segment_summary)
                startrow = len(segment_summary_df) + 1
                segment_summary_df.to_excel(writer, sheet_name='Intl_Withdrawal_Segments', index=False, header=False)
                detailed_df.to_excel(writer, sheet_name='Intl_Withdrawal_Segments', startrow=startrow, index=False)
            
            # Sheet 5: International Received & Withdrew
            if 'received_withdrew' in results['specific_intl_groups'] and not results['specific_intl_groups']['received_withdrew'].empty:
                group_df = results['specific_intl_groups']['received_withdrew']
                
                group_summary = [
                    ['GROUP 1: RECEIVED INTERNATIONAL REMITTANCE AND WITHDREW'],
                    ['IMPORTANT: Withdrawal Percentage = (Total Withdrawn / Total International Received) * 100%'],
                    [''],
                    ['Total Customers', len(group_df)],
                    ['Average International Receive Count', f"{group_df['International_Receive_Count'].mean():.1f}"],
                    ['Average Withdrawal Count', f"{group_df['Withdrawal_Count'].mean():.1f}"],
                    ['Average Withdrawal Percentage', f"{group_df['Withdrawal_Percentage'].mean():.1f}% of International Received"],
                    [''],
                    ['Customer Details:']
                ]
                
                group_summary_df = pd.DataFrame(group_summary)
                startrow = len(group_summary_df) + 1
                group_summary_df.to_excel(writer, sheet_name='Intl_Received_Withdrew', index=False, header=False)
                
                # Select and order columns for export
                export_cols = ['User_ID', 'Full_Name', 'International_Receive_Count', 'Withdrawal_Count', 
                             'Withdrawal_Percentage_Formatted', 'Total_International_Received', 'Total_Withdrawn',
                             'First_Transaction_Date', 'Last_Transaction_Date', 'Customer_Since', 'Activity_Level']
                export_df = group_df[[col for col in export_cols if col in group_df.columns]]
                export_df.to_excel(writer, sheet_name='Intl_Received_Withdrew', startrow=startrow, index=False)
            
            # Sheet 6: International Received, No Withdrawal, Other Services
            if 'received_no_withdraw_other' in results['specific_intl_groups'] and not results['specific_intl_groups']['received_no_withdraw_other'].empty:
                group_df = results['specific_intl_groups']['received_no_withdraw_other']
                
                group_summary = [
                    ['GROUP 2: RECEIVED INTERNATIONAL, NO WITHDRAWAL, USED OTHER SERVICES'],
                    ['Total Customers', len(group_df)],
                    ['Average International Receive Count', f"{group_df['International_Receive_Count'].mean():.1f}"],
                    ['Average Other Services Count', f"{group_df['Other_Services_Count'].mean():.1f}"],
                    [''],
                    ['Customer Details:']
                ]
                
                group_summary_df = pd.DataFrame(group_summary)
                startrow = len(group_summary_df) + 1
                group_summary_df.to_excel(writer, sheet_name='Intl_No_Withdraw_Other', index=False, header=False)
                
                # Select and order columns for export
                export_cols = ['User_ID', 'Full_Name', 'International_Receive_Count', 'Other_Services_Count', 
                             'Total_International_Received', 'First_Transaction_Date', 'Last_Transaction_Date', 
                             'Customer_Since', 'Activity_Level']
                export_df = group_df[[col for col in export_cols if col in group_df.columns]]
                export_df.to_excel(writer, sheet_name='Intl_No_Withdraw_Other', startrow=startrow, index=False)
            
            # Sheet 7: International Received & Only Withdrew
            if 'received_only_withdraw' in results['specific_intl_groups'] and not results['specific_intl_groups']['received_only_withdraw'].empty:
                group_df = results['specific_intl_groups']['received_only_withdraw']
                
                group_summary = [
                    ['GROUP 3: RECEIVED INTERNATIONAL AND ONLY WITHDREW (NO OTHER SERVICES)'],
                    ['IMPORTANT: Withdrawal Percentage = (Total Withdrawn / Total International Received) * 100%'],
                    [''],
                    ['Total Customers', len(group_df)],
                    ['Average International Receive Count', f"{group_df['International_Receive_Count'].mean():.1f}"],
                    ['Average Withdrawal Count', f"{group_df['Withdrawal_Count'].mean():.1f}"],
                    ['Average Withdrawal Percentage', f"{group_df['Withdrawal_Percentage'].mean():.1f}% of International Received"],
                    [''],
                    ['Customer Details:']
                ]
                
                group_summary_df = pd.DataFrame(group_summary)
                startrow = len(group_summary_df) + 1
                group_summary_df.to_excel(writer, sheet_name='Intl_Only_Withdrew', index=False, header=False)
                
                # Select and order columns for export
                export_cols = ['User_ID', 'Full_Name', 'International_Receive_Count', 'Withdrawal_Count', 
                             'Withdrawal_Percentage_Formatted', 'Total_International_Received', 'Total_Withdrawn',
                             'First_Transaction_Date', 'Last_Transaction_Date', 'Customer_Since', 'Activity_Level']
                export_df = group_df[[col for col in export_cols if col in group_df.columns]]
                export_df.to_excel(writer, sheet_name='Intl_Only_Withdrew', startrow=startrow, index=False)
            
            # Sheet 8: Impact Template
            impact_summary = [
                ['DAILY IMPACT REPORT TEMPLATE'],
                ['Instructions: Fill this template daily to track campaign progress'],
                ['Date Range', f"{results['summary_stats']['start_date']} to {results['summary_stats']['end_date']}"],
                [''],
                ['Template:']
            ]
            
            impact_summary_df = pd.DataFrame(impact_summary)
            startrow = len(impact_summary_df) + 1
            impact_summary_df.to_excel(writer, sheet_name='Impact_Template', index=False, header=False)
            results['impact_template'].to_excel(writer, sheet_name='Impact_Template', startrow=startrow, index=False)
            
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
            
            # Adjust column widths for all sheets
            workbook = writer.book
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
        
        output.seek(0)
        return output
    
    except Exception as e:
        st.error(f"Error creating Excel file: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

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
                
                # Show filter description
                if selected_pair_filter != "all_customers":
                    with st.expander("🔍 Filter Description"):
                        if selected_pair_filter == "intl_not_p2p":
                            st.info("Shows customers who have performed International remittance transactions but have NEVER used P2P services.")
                        elif selected_pair_filter == "p2p_not_deposit":
                            st.info("Shows customers who have used P2P services but have NEVER made any deposits.")
                        elif selected_pair_filter == "p2p_and_withdrawal":
                            st.info("Shows customers who have used BOTH P2P and Withdrawal services at least once.")
                        elif selected_pair_filter == "intl_and_p2p":
                            st.info("Shows customers who have used BOTH International remittance and P2P services.")
                        elif selected_pair_filter == "intl_received_withdraw":
                            st.info("Shows customers who have RECEIVED International remittance AND made withdrawals.")
                        elif selected_pair_filter == "intl_received_no_withdraw":
                            st.info("Shows customers who have RECEIVED International remittance but NO withdrawals (but used other services).")
                        elif selected_pair_filter == "intl_received_only_withdraw":
                            st.info("Shows customers who have RECEIVED International remittance and ONLY made withdrawals (no other services).")
                
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
                    st.info(f"**P2P Penetration**: {results['summary_stats']['p2p_customers']} customers already using P2P ({results['summary_stats']['p2p_customers']/max(results['summary_stats']['unique_customers'],1)*100:.1f}%)")
                
                with col2:
                    if 'total_intl_recipients' in results['summary_stats']:
                        st.info(f"**International Recipients**: {results['summary_stats']['total_intl_recipients']} customers received international remittance")
                        st.info(f"**Avg Withdrawal Rate**: {results['summary_stats'].get('avg_withdrawal_percentage', 0):.1f}% of international received amount withdrawn")
                
                # Segment Distribution
                st.subheader("International Withdrawal Segments")
                if 'intl_withdrawal_segments' in results and results['intl_withdrawal_segments']['withdrawal_segments']:
                    seg_data = results['intl_withdrawal_segments']['withdrawal_segments']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("≤25%", 
                                 f"{seg_data['distribution'].get('≤25%', 0):,}",
                                 f"{seg_data['percentages'].get('≤25%', 0):.1f}%")
                    with col2:
                        st.metric("25%-50%", 
                                 f"{seg_data['distribution'].get('25%-50%', 0):,}",
                                 f"{seg_data['percentages'].get('25%-50%', 0):.1f}%")
                    with col3:
                        st.metric("50%-75%", 
                                 f"{seg_data['distribution'].get('50%-75%', 0):,}",
                                 f"{seg_data['percentages'].get('50%-75%', 0):.1f}%")
                    with col4:
                        st.metric("75%-100%", 
                                 f"{seg_data['distribution'].get('75%-100%', 0):,}",
                                 f"{seg_data['percentages'].get('75%-100%', 0):.1f}%")
            
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
            
            with tabs[3]:  # International Withdrawal Segments
                st.subheader("International Recipients - Withdrawal Percentage Segments")
                st.info("**IMPORTANT**: Withdrawal Percentage = (Total Withdrawn / Total International Received) × 100%")
                
                if 'withdrawal_segments' in results['intl_withdrawal_segments'] and results['intl_withdrawal_segments']['withdrawal_segments']:
                    seg_data = results['intl_withdrawal_segments']
                    
                    # Segment metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("≤25%", 
                                 f"{seg_data['summary_stats'].get('segment_25_count', 0):,}",
                                 "Withdrew ≤25% of International")
                    with col2:
                        st.metric("25%-50%", 
                                 f"{seg_data['summary_stats'].get('segment_50_count', 0):,}",
                                 "Withdrew 25-50% of International")
                    with col3:
                        st.metric("50%-75%", 
                                 f"{seg_data['summary_stats'].get('segment_75_count', 0):,}",
                                 "Withdrew 50-75% of International")
                    with col4:
                        st.metric("75%-100%", 
                                 f"{seg_data['summary_stats'].get('segment_100_count', 0):,}",
                                 "Withdrew 75-100% of International")
                    
                    # Overall statistics
                    st.subheader("Overall Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Avg Withdrawal %", f"{seg_data['summary_stats'].get('avg_withdrawal_percentage', 0):.1f}%")
                        st.caption("Of International Received")
                    with col2:
                        st.metric("Median Withdrawal %", f"{seg_data['summary_stats'].get('median_withdrawal_percentage', 0):.1f}%")
                        st.caption("Of International Received")
                    with col3:
                        overall_percentage = seg_data['summary_stats'].get('overall_withdrawal_percentage', 0)
                        st.metric("Overall Withdrawal %", f"{overall_percentage:.1f}%")
                        st.caption("Of International Received")
                    
                    # Total amounts
                    st.subheader("Total Amounts")
                    col1, col2 = st.columns(2)
                    with col1:
                        total_received = seg_data['summary_stats'].get('total_intl_received_amount', 0)
                        st.metric("Total International Received", f"{total_received:,.2f}")
                    with col2:
                        total_withdrawn = seg_data['summary_stats'].get('total_withdrawn_amount', 0)
                        st.metric("Total Withdrawn", f"{total_withdrawn:,.2f}")
                    
                    # Detailed analysis
                    if 'detailed_analysis' in seg_data and not seg_data['detailed_analysis'].empty:
                        st.subheader("Detailed Customer Analysis")
                        st.dataframe(seg_data['detailed_analysis'], use_container_width=True)
                else:
                    st.info("No international recipients found for withdrawal segmentation analysis")
            
            with tabs[4]:  # International Received & Withdrew
                st.subheader("Customers who Received International Remittance AND Withdrew")
                st.info("**IMPORTANT**: Withdrawal Percentage = (Total Withdrawn / Total International Received) × 100%")
                
                if 'received_withdrew' in results['specific_intl_groups'] and not results['specific_intl_groups']['received_withdrew'].empty:
                    group_df = results['specific_intl_groups']['received_withdrew']
                    
                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Customers", f"{len(group_df):,}")
                    with col2:
                        avg_withdrawal_pct = group_df['Withdrawal_Percentage'].mean()
                        st.metric("Avg Withdrawal %", f"{avg_withdrawal_pct:.1f}%")
                        st.caption("Of International Received")
                    with col3:
                        high_withdrawal = len(group_df[group_df['Withdrawal_Percentage'] > 75])
                        st.metric("High Withdrawal", f"{high_withdrawal}", ">75% of International")
                    
                    # Data display
                    st.dataframe(group_df, use_container_width=True)
                else:
                    st.info("No customers found who both received international remittance and made withdrawals")
            
            with tabs[5]:  # International No Withdraw Other
                st.subheader("Customers who Received International, NO Withdrawal, but used Other Services")
                
                if 'received_no_withdraw_other' in results['specific_intl_groups'] and not results['specific_intl_groups']['received_no_withdraw_other'].empty:
                    group_df = results['specific_intl_groups']['received_no_withdraw_other']
                    
                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Customers", f"{len(group_df):,}")
                    with col2:
                        avg_other_services = group_df['Other_Services_Count'].mean()
                        st.metric("Avg Other Services", f"{avg_other_services:.1f}")
                    with col3:
                        active_customers = len(group_df[group_df['Other_Services_Count'] > 3])
                        st.metric("Active Users", f"{active_customers}", "3+ services")
                    
                    # Data display
                    st.dataframe(group_df, use_container_width=True)
                else:
                    st.info("No customers found who received international remittance without withdrawals but used other services")
            
            with tabs[6]:  # International Only Withdrew
                st.subheader("Customers who Received International and ONLY Withdrew")
                st.info("**IMPORTANT**: Withdrawal Percentage = (Total Withdrawn / Total International Received) × 100%")
                
                if 'received_only_withdraw' in results['specific_intl_groups'] and not results['specific_intl_groups']['received_only_withdraw'].empty:
                    group_df = results['specific_intl_groups']['received_only_withdraw']
                    
                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Customers", f"{len(group_df):,}")
                    with col2:
                        avg_withdrawal_pct = group_df['Withdrawal_Percentage'].mean()
                        st.metric("Avg Withdrawal %", f"{avg_withdrawal_pct:.1f}%")
                        st.caption("Of International Received")
                    with col3:
                        pure_withdrawal = len(group_df[group_df['Withdrawal_Percentage'] == 100])
                        st.metric("Pure Withdrawal", f"{pure_withdrawal}", "100% of International")
                    
                    # Data display
                    st.dataframe(group_df, use_container_width=True)
                else:
                    st.info("No customers found who only received international remittance and made withdrawals (no other services)")
            
            with tabs[7]:  # Impact Template
                st.subheader("Daily Impact Report Template")
                st.dataframe(results['impact_template'], use_container_width=True)
                st.info("Use this template to track daily calling campaign results")
            
            with tabs[8]:  # Detailed Metrics
                st.subheader("Detailed Performance Metrics")
                
                # Conversion metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    intl_conversion = (results['summary_stats']['intl_not_p2p'] / max(results['summary_stats']['intl_customers'], 1) * 100)
                    st.metric("Intl to P2P Potential", f"{intl_conversion:.1f}%")
                with col2:
                    domestic_potential = (results['summary_stats']['domestic_not_p2p'] / max(results['summary_stats']['other_customers'], 1) * 100)
                    st.metric("Domestic Cross-sell", f"{domestic_potential:.1f}%")
                with col3:
                    market_penetration = (results['summary_stats']['p2p_customers'] / max(results['summary_stats']['unique_customers'], 1) * 100)
                    st.metric("Market Penetration", f"{market_penetration:.1f}%")
                
                # Customer segmentation
                st.subheader("Customer Segmentation")
                seg_data = {
                    'Segment': ['International Targets', 'Domestic Targets', 'Received & Withdrew', 
                               'Received, No Withdrawal', 'Received & Only Withdrew'],
                    'Count': [results['summary_stats']['intl_not_p2p'], 
                             results['summary_stats']['domestic_not_p2p'],
                             results['summary_stats'].get('received_withdrew_count', 0),
                             results['summary_stats'].get('received_no_withdraw_other_count', 0),
                             results['summary_stats'].get('received_only_withdraw_count', 0)]
                }
                seg_df = pd.DataFrame(seg_data)
                st.dataframe(seg_df, use_container_width=True)
            
            with tabs[9]:  # Recommendations
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
                
                st.info("**International Withdrawal Segment Strategy**")
                if 'intl_withdrawal_segments' in results and results['intl_withdrawal_segments']['withdrawal_segments']:
                    seg_data = results['intl_withdrawal_segments']['summary_stats']
                    st.write(f"8. Low Withdrawal (≤25%): {seg_data.get('segment_25_count', 0)} customers - Focus on retention & premium services")
                    st.write(f"9. Moderate (25-50%): {seg_data.get('segment_50_count', 0)} customers - Balance service education")
                    st.write(f"10. High (50-75%): {seg_data.get('segment_75_count', 0)} customers - Risk mitigation strategies")
                    st.write(f"11. Very High (75-100%): {seg_data.get('segment_100_count', 0)} customers - Immediate attention required")
            
            # Download section
            st.markdown('<h2 class="sub-header">📥 Download Comprehensive Report</h2>', unsafe_allow_html=True)
            
            st.info("The comprehensive Excel report includes 10 sheets with all analysis results:")
            
            sheets_info = [
                "1. **Executive_Summary**: Overall campaign summary and key metrics",
                "2. **International_Targets**: Customers using international remittance but not P2P",
                "3. **Domestic_Targets**: Domestic customers using other services but not P2P",
                "4. **Intl_Withdrawal_Segments**: International recipients segmented by withdrawal percentage",
                "5. **Intl_Received_Withdrew**: Customers who received international and withdrew",
                "6. **Intl_No_Withdraw_Other**: Received international, no withdrawal, used other services",
                "7. **Intl_Only_Withdrew**: Received international and only withdrew (no other services)",
                "8. **Impact_Template**: Daily tracking template for campaign progress",
                "9. **Detailed_Metrics**: Comprehensive performance metrics",
                "10. **Recommendations**: Actionable campaign recommendations"
            ]
            
            for info in sheets_info:
                st.write(info)
            
            # Create and download comprehensive Excel report
            excel_data = create_comprehensive_excel_report(results, filter_desc)
            if excel_data:
                st.download_button(
                    label="📊 Download Comprehensive Excel Report",
                    data=excel_data,
                    file_name=f"telemarketing_comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    help="Download all analysis in one Excel file with 10 detailed sheets"
                )
            
            # Individual download buttons for convenience
            st.markdown("### Individual Downloads")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if not results['intl_not_p2p'].empty:
                    csv_intl = results['intl_not_p2p'].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="🎯 International Targets",
                        data=csv_intl,
                        file_name="international_targets.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col2:
                if 'detailed_analysis' in results['intl_withdrawal_segments'] and not results['intl_withdrawal_segments']['detailed_analysis'].empty:
                    csv_segments = results['intl_withdrawal_segments']['detailed_analysis'].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📈 Withdrawal Segments",
                        data=csv_segments,
                        file_name="withdrawal_segments.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col3:
                csv_impact = results['impact_template'].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📋 Impact Template",
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
                high_withdrawal = results['summary_stats'].get('intl_withdrawal_segment_75', 0) + results['summary_stats'].get('intl_withdrawal_segment_100', 0)
                st.metric("High Withdrawal", f"{high_withdrawal:,}", "customers for retention")

if __name__ == "__main__":
    main()
