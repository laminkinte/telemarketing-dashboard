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
    .segmentation-box {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e2e8f0;
        margin: 0.5rem 0;
    }
    .percentage-indicator {
        font-size: 1.2rem;
        font-weight: bold;
        color: #3B82F6;
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
        optional_columns = ['Service Name', 'Entity Name', 'Full Name', 'Amount', 'Transaction Type', 'Direction']
        
        # Check for minimum required columns
        missing_required = [col for col in required_columns if col not in df.columns]
        if missing_required:
            st.error(f"Missing required columns: {missing_required}")
            st.info(f"Available columns: {list(df.columns)}")
            return None
        
        # Add missing optional columns with default values
        for col in optional_columns:
            if col not in df.columns:
                if col == 'Amount':
                    df[col] = 1.0  # Default amount if not available
                elif col == 'Direction':
                    df[col] = 'Credit'  # Default direction
                else:
                    df[col] = ''
                st.warning(f"Note: Column '{col}' not found. Using default values.")
        
        # Data cleaning
        for col in ['Product Name', 'Service Name', 'Entity Name', 'Full Name', 'Transaction Type', 'Direction']:
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
        df['Month'] = df['Created At'].dt.month
        df['Year'] = df['Created At'].dt.year
        
        return df
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def filter_by_customer_pair(df, filter_pair, segmentation_threshold=None):
    """Filter customers based on transaction behavior pairs with segmentation"""
    
    # Define product categories
    p2p_products = ['Internal Wallet Transfer (P2P)', 'Internal Wallet Transfer', 'P2P Transfer', 'Wallet Transfer', 'P2P']
    international_remittance = ['International Remittance', 'International Transfer', 'Remittance', 'International', 'Receive Remittance']
    deposit_products = ['Deposit', 'Cash In', 'Deposit Customer', 'Deposit Agent', 'Bank Deposit']
    withdrawal_products = ['Withdrawal', 'Scan To Withdraw Agent', 'Scan To Withdraw Customer', 'Cash Out', 'Withdraw', 'ATM Withdrawal']
    other_services = [
        'Bill Payment', 'Airtime Topup', 'Utility Payment', 'Topup', 
        'Merchant Payment', 'Ticket', 'Scan To Send', 'Loan', 'Investment'
    ]
    
    # Get unique customers
    unique_customers = df['User Identifier'].dropna().unique()
    
    # Helper function to get customers for a product category
    def get_customers_for_product(product_list):
        mask = df['Product Name'].str.contains('|'.join(product_list), case=False, na=False)
        return df[mask]['User Identifier'].dropna().unique()
    
    # Get customer sets for each product category
    p2p_customers = set(get_customers_for_product(p2p_products))
    intl_customers = set(get_customers_for_product(international_remittance))
    deposit_customers = set(get_customers_for_product(deposit_products))
    withdrawal_customers = set(get_customers_for_product(withdrawal_products))
    other_service_customers = set(get_customers_for_product(other_services))
    
    # Get customers who RECEIVED international remittance
    # This requires looking at transaction direction
    if 'Direction' in df.columns:
        # Look for incoming/received international transactions
        intl_received_mask = (
            df['Product Name'].str.contains('|'.join(international_remittance), case=False, na=False) &
            df['Direction'].str.contains('In|Receive|Credit', case=False, na=False)
        )
    else:
        # If no direction column, assume all international transactions are received
        intl_received_mask = df['Product Name'].str.contains('|'.join(international_remittance), case=False, na=False)
    
    intl_received_customers = set(df[intl_received_mask]['User Identifier'].dropna().unique())
    
    # Apply the selected filter pair
    filtered_customers = set()
    filter_description = ""
    segmentation_results = None
    
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
    
    elif filter_pair == "intl_received_withdrawal_segmented":
        # Customers who Received International Remittance and followed by withdrawal
        # This is a complex analysis with segmentation
        filtered_customers, segmentation_results = analyze_intl_received_withdrawal_segmentation(df)
        filter_description = f"Customers who Received International Remittance and Withdrew\n\n• Total International Recipients: {len(intl_received_customers):,}\n• Customers with Withdrawal: {len(filtered_customers):,}"
    
    elif filter_pair == "intl_received_no_withdrawal_other":
        # Customers who Received International Remittance did not withdraw and use other services
        filtered_customers = analyze_intl_received_no_withdrawal_with_other(df)
        filter_description = f"Customers who Received International Remittance, NO Withdrawal, but use Other Services\n\n• Total International Recipients: {len(intl_received_customers):,}\n• Result: {len(filtered_customers):,} customers"
    
    elif filter_pair == "intl_received_only_withdrawal":
        # Customers who Received International Remittance and only withdraw and did not use any other services
        filtered_customers = analyze_intl_received_only_withdrawal(df)
        filter_description = f"Customers who Received International Remittance and ONLY Withdrew\n\n• Total International Recipients: {len(intl_received_customers):,}\n• Result: {len(filtered_customers):,} customers"
    
    elif filter_pair == "all_customers":
        # All customers (no filter)
        filtered_customers = set(unique_customers)
        filter_description = f"All Customers\n\n• Total Unique Customers: {len(filtered_customers):,}"
    
    # Filter the dataframe to only include transactions from the selected customers
    filtered_df = df[df['User Identifier'].isin(filtered_customers)].copy()
    
    return filtered_df, filter_description, len(filtered_customers), segmentation_results

def analyze_intl_received_withdrawal_segmentation(df):
    """Analyze customers who received international remittance and withdrew, with segmentation"""
    
    # Define product categories
    international_remittance = ['International Remittance', 'International Transfer', 'Remittance', 'International', 'Receive Remittance']
    withdrawal_products = ['Withdrawal', 'Scan To Withdraw Agent', 'Scan To Withdraw Customer', 'Cash Out', 'Withdraw', 'ATM Withdrawal']
    other_services = [
        'Bill Payment', 'Airtime Topup', 'Utility Payment', 'Topup', 
        'Merchant Payment', 'Ticket', 'Scan To Send', 'Loan', 'Investment',
        'Deposit', 'P2P'
    ]
    
    # Identify international received transactions
    if 'Direction' in df.columns:
        intl_received_mask = (
            df['Product Name'].str.contains('|'.join(international_remittance), case=False, na=False) &
            df['Direction'].str.contains('In|Receive|Credit', case=False, na=False)
        )
    else:
        intl_received_mask = df['Product Name'].str.contains('|'.join(international_remittance), case=False, na=False)
    
    # Group by customer
    intl_received_customers = {}
    withdrawal_customers = set()
    
    for user_id in df['User Identifier'].dropna().unique():
        user_df = df[df['User Identifier'] == user_id].sort_values('Created At')
        
        # Find international received transactions
        intl_transactions = user_df[intl_received_mask]
        if len(intl_transactions) > 0:
            # Find withdrawal transactions AFTER international receipt
            withdrawal_mask = user_df['Product Name'].str.contains('|'.join(withdrawal_products), case=False, na=False)
            withdrawal_transactions = user_df[withdrawal_mask]
            
            if len(withdrawal_transactions) > 0:
                # Check if withdrawals happened after first international receipt
                first_intl_date = intl_transactions['Created At'].min()
                withdrawals_after = withdrawal_transactions[withdrawal_transactions['Created At'] > first_intl_date]
                
                if len(withdrawals_after) > 0:
                    withdrawal_customers.add(user_id)
                    
                    # Calculate total received amount
                    total_received = intl_transactions['Amount'].sum() if 'Amount' in intl_transactions.columns else len(intl_transactions)
                    
                    # Calculate total withdrawn amount
                    total_withdrawn = withdrawals_after['Amount'].sum() if 'Amount' in withdrawals_after.columns else len(withdrawals_after)
                    
                    # Calculate withdrawal percentage
                    if total_received > 0:
                        withdrawal_percentage = (total_withdrawn / total_received) * 100
                    else:
                        withdrawal_percentage = 0
                    
                    # Get other services used
                    other_mask = user_df['Product Name'].str.contains('|'.join(other_services), case=False, na=False)
                    other_services_count = user_df[other_mask]['Product Name'].nunique()
                    
                    intl_received_customers[user_id] = {
                        'total_received': total_received,
                        'total_withdrawn': total_withdrawn,
                        'withdrawal_percentage': withdrawal_percentage,
                        'withdrawal_count': len(withdrawals_after),
                        'intl_count': len(intl_transactions),
                        'other_services_count': other_services_count,
                        'first_intl_date': first_intl_date,
                        'last_withdrawal_date': withdrawals_after['Created At'].max()
                    }
    
    # Create segmentation
    segmentation_results = {
        'segment_25': {'customers': [], 'count': 0, 'range': '≤25%'},
        'segment_50': {'customers': [], 'count': 0, 'range': '≤50% and >25%'},
        'segment_75': {'customers': [], 'count': 0, 'range': '≤75% and >50%'},
        'segment_100': {'customers': [], 'count': 0, 'range': '≤100% and >75%'}
    }
    
    for user_id, data in intl_received_customers.items():
        percentage = data['withdrawal_percentage']
        
        if percentage <= 25:
            segmentation_results['segment_25']['customers'].append(user_id)
            segmentation_results['segment_25']['count'] += 1
        elif percentage <= 50:
            segmentation_results['segment_50']['customers'].append(user_id)
            segmentation_results['segment_50']['count'] += 1
        elif percentage <= 75:
            segmentation_results['segment_75']['customers'].append(user_id)
            segmentation_results['segment_75']['count'] += 1
        elif percentage <= 100:
            segmentation_results['segment_100']['customers'].append(user_id)
            segmentation_results['segment_100']['count'] += 1
    
    # All customers who withdrew after receiving
    all_customers = list(intl_received_customers.keys())
    
    return set(all_customers), segmentation_results

def analyze_intl_received_no_withdrawal_with_other(df):
    """Analyze customers who received international remittance, no withdrawal, but use other services"""
    
    # Define product categories
    international_remittance = ['International Remittance', 'International Transfer', 'Remittance', 'International', 'Receive Remittance']
    withdrawal_products = ['Withdrawal', 'Scan To Withdraw Agent', 'Scan To Withdraw Customer', 'Cash Out', 'Withdraw', 'ATM Withdrawal']
    other_services = [
        'Bill Payment', 'Airtime Topup', 'Utility Payment', 'Topup', 
        'Merchant Payment', 'Ticket', 'Scan To Send', 'Loan', 'Investment',
        'Deposit', 'P2P'
    ]
    
    # Identify international received transactions
    if 'Direction' in df.columns:
        intl_received_mask = (
            df['Product Name'].str.contains('|'.join(international_remittance), case=False, na=False) &
            df['Direction'].str.contains('In|Receive|Credit', case=False, na=False)
        )
    else:
        intl_received_mask = df['Product Name'].str.contains('|'.join(international_remittance), case=False, na=False)
    
    # Get unique customers
    unique_customers = df['User Identifier'].dropna().unique()
    filtered_customers = set()
    
    for user_id in unique_customers:
        user_df = df[df['User Identifier'] == user_id]
        
        # Check if customer received international remittance
        intl_transactions = user_df[intl_received_mask]
        if len(intl_transactions) == 0:
            continue  # Skip customers who didn't receive international
        
        # Check if customer made any withdrawals
        withdrawal_mask = user_df['Product Name'].str.contains('|'.join(withdrawal_products), case=False, na=False)
        withdrawal_transactions = user_df[withdrawal_mask]
        
        if len(withdrawal_transactions) > 0:
            continue  # Skip customers who made withdrawals
        
        # Check if customer used other services
        other_mask = user_df['Product Name'].str.contains('|'.join(other_services), case=False, na=False)
        other_transactions = user_df[other_mask]
        
        if len(other_transactions) > 0:
            filtered_customers.add(user_id)
    
    return filtered_customers

def analyze_intl_received_only_withdrawal(df):
    """Analyze customers who received international remittance and only withdrew (no other services)"""
    
    # Define product categories
    international_remittance = ['International Remittance', 'International Transfer', 'Remittance', 'International', 'Receive Remittance']
    withdrawal_products = ['Withdrawal', 'Scan To Withdraw Agent', 'Scan To Withdraw Customer', 'Cash Out', 'Withdraw', 'ATM Withdrawal']
    other_services = [
        'Bill Payment', 'Airtime Topup', 'Utility Payment', 'Topup', 
        'Merchant Payment', 'Ticket', 'Scan To Send', 'Loan', 'Investment',
        'Deposit', 'P2P'
    ]
    
    # Identify international received transactions
    if 'Direction' in df.columns:
        intl_received_mask = (
            df['Product Name'].str.contains('|'.join(international_remittance), case=False, na=False) &
            df['Direction'].str.contains('In|Receive|Credit', case=False, na=False)
        )
    else:
        intl_received_mask = df['Product Name'].str.contains('|'.join(international_remittance), case=False, na=False)
    
    # Get unique customers
    unique_customers = df['User Identifier'].dropna().unique()
    filtered_customers = set()
    
    for user_id in unique_customers:
        user_df = df[df['User Identifier'] == user_id]
        
        # Check if customer received international remittance
        intl_transactions = user_df[intl_received_mask]
        if len(intl_transactions) == 0:
            continue  # Skip customers who didn't receive international
        
        # Check if customer made withdrawals
        withdrawal_mask = user_df['Product Name'].str.contains('|'.join(withdrawal_products), case=False, na=False)
        withdrawal_transactions = user_df[withdrawal_mask]
        
        if len(withdrawal_transactions) == 0:
            continue  # Skip customers with no withdrawals
        
        # Check if customer used ANY other services
        other_mask = user_df['Product Name'].str.contains('|'.join(other_services), case=False, na=False)
        other_transactions = user_df[other_mask]
        
        # Count unique product types excluding international and withdrawal
        all_products = set(user_df['Product Name'].unique())
        intl_products = set([p for p in international_remittance if any(p.lower() in str(prod).lower() for prod in all_products)])
        withdrawal_products_set = set([p for p in withdrawal_products if any(p.lower() in str(prod).lower() for prod in all_products)])
        
        # Only include if international and withdrawal are the ONLY services used
        if len(all_products - intl_products - withdrawal_products_set) == 0:
            filtered_customers.add(user_id)
    
    return filtered_customers

def filter_data(df, start_date, end_date, product_filter, customer_type_filter, pair_filter="all_customers"):
    """Filter data based on selected criteria"""
    filtered_df = df.copy()
    
    # First apply customer pair filter if not "all_customers"
    if pair_filter != "all_customers":
        filtered_df, filter_desc, customer_count, segmentation_results = filter_by_customer_pair(filtered_df, pair_filter)
    else:
        filter_desc = "Showing all customers"
        customer_count = filtered_df['User Identifier'].nunique()
        segmentation_results = None
    
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
    
    return filtered_df, filter_desc, customer_count, segmentation_results

def create_visualizations(filtered_df, start_date, end_date):
    """Create all visualizations"""
    viz_data = {}
    
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly not available. Using basic charts.")
        return viz_data
    
    try:
        # 1. Daily Transaction Volume
        daily_volume = filtered_df.groupby('Date').size().reset_index(name='Count')
        fig1 = px.line(daily_volume, x='Date', y='Count',
                      title='Daily Transaction Volume',
                      labels={'Count': 'Number of Transactions', 'Date': 'Date'},
                      template='plotly_white')
        fig1.update_traces(line=dict(width=3))
        viz_data['daily_volume'] = fig1
        
        # 2. Top Products Distribution
        top_products = filtered_df['Product Name'].value_counts().head(10)
        if len(top_products) > 0:
            fig2 = px.bar(x=top_products.values, y=top_products.index,
                         orientation='h',
                         title='Top 10 Products by Transaction Count',
                         labels={'x': 'Transaction Count', 'y': 'Product'},
                         template='plotly_white')
            fig2.update_traces(marker_color='#3B82F6')
            viz_data['top_products'] = fig2
        
        # 3. Hourly Distribution
        hourly_dist = filtered_df['Hour'].value_counts().sort_index()
        if len(hourly_dist) > 0:
            fig3 = px.line(x=hourly_dist.index, y=hourly_dist.values,
                          title='Hourly Transaction Distribution',
                          labels={'x': 'Hour of Day', 'y': 'Transaction Count'},
                          template='plotly_white')
            fig3.update_traces(line=dict(width=3, color='#10B981'))
            viz_data['hourly_dist'] = fig3
        
        # 4. Customer Segmentation
        if 'Entity Name' in filtered_df.columns:
            customer_types = filtered_df['Entity Name'].fillna('Unknown')
            customer_counts = customer_types.value_counts()
            if len(customer_counts) > 0:
                fig4 = px.pie(values=customer_counts.values,
                             names=customer_counts.index,
                             title='Customer Type Distribution',
                             template='plotly_white')
                viz_data['customer_segmentation'] = fig4
        
        # 5. Weekday Analysis
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_data = filtered_df['Day'].value_counts().reindex(weekday_order)
        if len(weekday_data) > 0:
            fig5 = px.bar(x=weekday_data.index, y=weekday_data.values,
                         title='Transaction Volume by Weekday',
                         labels={'x': 'Day', 'y': 'Transaction Count'},
                         template='plotly_white')
            fig5.update_traces(marker_color='#8B5CF6')
            viz_data['weekday_analysis'] = fig5
            
    except Exception as e:
        st.warning(f"Could not create some visualizations: {str(e)}")
    
    return viz_data

def create_basic_visualizations(filtered_df):
    """Create basic visualizations using Streamlit native charts"""
    viz_data = {}
    
    try:
        # Daily volume as line chart
        daily_volume = filtered_df.groupby('Date').size().reset_index(name='Count')
        if len(daily_volume) > 0:
            viz_data['daily_volume'] = daily_volume
        
        # Top products
        top_products = filtered_df['Product Name'].value_counts().head(10)
        if len(top_products) > 0:
            viz_data['top_products'] = top_products
        
        # Hourly distribution
        hourly_dist = filtered_df['Hour'].value_counts().sort_index()
        if len(hourly_dist) > 0:
            viz_data['hourly_dist'] = hourly_dist
        
        # Customer segmentation
        if 'Entity Name' in filtered_df.columns:
            customer_counts = filtered_df['Entity Name'].fillna('Unknown').value_counts()
            if len(customer_counts) > 0:
                viz_data['customer_segmentation'] = customer_counts
        
        # Weekday analysis
        weekday_data = filtered_df['Day'].value_counts()
        if len(weekday_data) > 0:
            viz_data['weekday_analysis'] = weekday_data
            
    except Exception as e:
        st.warning(f"Could not create basic visualizations: {str(e)}")
    
    return viz_data

def analyze_telemarketing_data(filtered_df):
    """Main analysis function"""
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
    
    p2p_customers = filtered_df[filtered_df['Product Name'].isin(p2p_products)]['User Identifier'].dropna().unique()
    
    intl_not_p2p = [cust for cust in intl_customers if cust not in p2p_customers]
    
    # Create detailed DataFrame for international customers not using P2P
    intl_details = []
    for cust_id in intl_not_p2p[:500]:  # Limit for performance
        cust_records = filtered_df[filtered_df['User Identifier'] == cust_id]
        if not cust_records.empty:
            # Get customer name
            name_record = cust_records[cust_records['Full Name'].notna() & (cust_records['Full Name'] != 'nan')]
            if not name_record.empty:
                full_name = name_record.iloc[0]['Full Name']
            else:
                full_name = 'Name not available'
            
            # Get last transaction date
            last_date = cust_records['Created At'].max()
            last_date_str = last_date.strftime('%Y-%m-%d') if pd.notna(last_date) else 'Unknown'
            
            # Calculate metrics
            total_transactions = len(cust_records)
            intl_count = len(cust_records[intl_mask])
            
            intl_details.append({
                'User_ID': cust_id,
                'Full_Name': full_name,
                'Last_Transaction_Date': last_date_str,
                'Total_Transactions': total_transactions,
                'International_Transactions': intl_count,
                'Customer_Since': cust_records['Created At'].min().strftime('%Y-%m-%d') if pd.notna(cust_records['Created At'].min()) else 'Unknown'
            })
    
    report1_df = pd.DataFrame(intl_details) if intl_details else pd.DataFrame(
        columns=['User_ID', 'Full_Name', 'Last_Transaction_Date', 'Total_Transactions', 
                'International_Transactions', 'Customer_Since']
    )
    
    # REPORT 2: Non-international customers using other services but not P2P
    other_mask = filtered_df['Product Name'].isin(other_services) | \
                filtered_df['Service Name'].isin(bill_payment_services)
    
    other_customers = filtered_df[other_mask]['User Identifier'].dropna().unique()
    
    domestic_other_not_p2p = [
        cust for cust in other_customers 
        if (cust not in intl_customers) and (cust not in p2p_customers)
    ]
    
    domestic_details = []
    for cust_id in domestic_other_not_p2p[:500]:  # Limit for performance
        cust_records = filtered_df[filtered_df['User Identifier'] == cust_id]
        if not cust_records.empty:
            # Get customer name
            name_record = cust_records[cust_records['Full Name'].notna() & (cust_records['Full Name'] != 'nan')]
            if not name_record.empty:
                full_name = name_record.iloc[0]['Full Name']
            else:
                full_name = 'Name not available'
            
            # Get last transaction date
            last_date = cust_records['Created At'].max()
            last_date_str = last_date.strftime('%Y-%m-%d') if pd.notna(last_date) else 'Unknown'
            
            # Get services used
            services_used = cust_records['Product Name'].dropna().unique()
            other_services_list = [str(s) for s in services_used if any(service in str(s).lower() for service in 
                                                                       [o.lower() for o in other_services + bill_payment_services])]
            
            domestic_details.append({
                'User_ID': cust_id,
                'Full_Name': full_name,
                'Last_Transaction_Date': last_date_str,
                'Total_Transactions': len(cust_records),
                'Other_Services_Used': ', '.join(other_services_list[:3]),
                'Other_Services_Count': len(other_services_list),
                'Customer_Since': cust_records['Created At'].min().strftime('%Y-%m-%d') if pd.notna(cust_records['Created At'].min()) else 'Unknown'
            })
    
    report2_df = pd.DataFrame(domestic_details) if domestic_details else pd.DataFrame(
        columns=['User_ID', 'Full_Name', 'Last_Transaction_Date', 'Total_Transactions',
                'Other_Services_Used', 'Other_Services_Count', 'Customer_Since']
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

def create_excel_download(results, filter_desc="", segmentation_results=None):
    """Create Excel file with all reports"""
    output = io.BytesIO()
    
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sheet 1: International customers not using P2P
            if not results['intl_not_p2p'].empty:
                results['intl_not_p2p'].to_excel(writer, sheet_name='International_Targets', index=False)
            
            # Sheet 2: Domestic customers not using P2P
            if not results['domestic_other_not_p2p'].empty:
                results['domestic_other_not_p2p'].to_excel(writer, sheet_name='Domestic_Targets', index=False)
            
            # Sheet 3: Impact template
            results['impact_template'].to_excel(writer, sheet_name='Impact_Template', index=False)
            
            # Sheet 4: Summary
            summary_df = pd.DataFrame({
                'Metric': [
                    'Filter Applied',
                    'Report Period Start',
                    'Report Period End',
                    'Total Transactions',
                    'Total Unique Customers',
                    'International Customers',
                    'International Customers Not Using P2P',
                    'P2P Users',
                    'Other Services Users',
                    'Domestic Customers Not Using P2P',
                    'Total Addressable Market'
                ],
                'Value': [
                    filter_desc[:100],  # Truncate if too long
                    results['summary_stats']['start_date'],
                    results['summary_stats']['end_date'],
                    results['summary_stats']['total_transactions'],
                    results['summary_stats']['unique_customers'],
                    results['summary_stats']['intl_customers'],
                    results['summary_stats']['intl_not_p2p'],
                    results['summary_stats']['p2p_customers'],
                    results['summary_stats']['other_customers'],
                    results['summary_stats']['domestic_not_p2p'],
                    results['summary_stats']['intl_not_p2p'] + results['summary_stats']['domestic_not_p2p']
                ]
            })
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 5: Recommendations
            recommendations = [
                f"PRIORITY 1: Target {results['summary_stats']['intl_not_p2p']} international customers first",
                f"PRIORITY 2: Engage {results['summary_stats']['domestic_not_p2p']} domestic customers",
                "STRATEGY: Bundle P2P with their existing services",
                "TIMING: Call during their typical transaction hours",
                "INCENTIVE: Offer fee waiver for first 3 P2P transactions",
                "TRACKING: Use the impact template daily",
                "FOLLOW-UP: Schedule callbacks for unreached customers",
                "TRAINING: Ensure agents understand P2P benefits",
                "METRICS: Track conversion rate weekly",
                "FEEDBACK: Collect customer feedback during calls"
            ]
            rec_df = pd.DataFrame({'Recommendations': recommendations})
            rec_df.to_excel(writer, sheet_name='Recommendations', index=False)
            
            # Sheet 6: Segmentation Results (if available)
            if segmentation_results:
                seg_data = []
                for seg_key, seg_info in segmentation_results.items():
                    seg_data.append({
                        'Segment': seg_info['range'],
                        'Customer_Count': seg_info['count'],
                        'Customer_Ids': ', '.join(map(str, seg_info['customers'][:100]))  # Limit to first 100
                    })
                
                seg_df = pd.DataFrame(seg_data)
                seg_df.to_excel(writer, sheet_name='Segmentation', index=False)
        
        output.seek(0)
        return output
    except Exception as e:
        st.error(f"Error creating Excel file: {str(e)}")
        return None

def display_segmentation_results(segmentation_results):
    """Display segmentation results in a visual format"""
    if not segmentation_results:
        return
    
    st.markdown('<h3 class="sub-header">📊 Withdrawal Percentage Segmentation</h3>', unsafe_allow_html=True)
    
    # Create columns for segmentation display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        seg25 = segmentation_results['segment_25']
        st.markdown(f'<div class="segmentation-box">', unsafe_allow_html=True)
        st.markdown(f'<div class="percentage-indicator">≤25%</div>', unsafe_allow_html=True)
        st.metric("Customers", seg25['count'])
        st.caption(f"Withdrew up to 25% of received amount")
        st.markdown(f'</div>', unsafe_allow_html=True)
    
    with col2:
        seg50 = segmentation_results['segment_50']
        st.markdown(f'<div class="segmentation-box">', unsafe_allow_html=True)
        st.markdown(f'<div class="percentage-indicator">26-50%</div>', unsafe_allow_html=True)
        st.metric("Customers", seg50['count'])
        st.caption(f"Withdrew 26-50% of received amount")
        st.markdown(f'</div>', unsafe_allow_html=True)
    
    with col3:
        seg75 = segmentation_results['segment_75']
        st.markdown(f'<div class="segmentation-box">', unsafe_allow_html=True)
        st.markdown(f'<div class="percentage-indicator">51-75%</div>', unsafe_allow_html=True)
        st.metric("Customers", seg75['count'])
        st.caption(f"Withdrew 51-75% of received amount")
        st.markdown(f'</div>', unsafe_allow_html=True)
    
    with col4:
        seg100 = segmentation_results['segment_100']
        st.markdown(f'<div class="segmentation-box">', unsafe_allow_html=True)
        st.markdown(f'<div class="percentage-indicator">76-100%</div>', unsafe_allow_html=True)
        st.metric("Customers", seg100['count'])
        st.caption(f"Withdrew 76-100% of received amount")
        st.markdown(f'</div>', unsafe_allow_html=True)
    
    # Create a bar chart for segmentation
    seg_counts = {
        '≤25%': seg25['count'],
        '26-50%': seg50['count'],
        '51-75%': seg75['count'],
        '76-100%': seg100['count']
    }
    
    if PLOTLY_AVAILABLE:
        fig = px.bar(
            x=list(seg_counts.keys()),
            y=list(seg_counts.values()),
            title="Withdrawal Amount Segmentation",
            labels={'x': 'Withdrawal Percentage', 'y': 'Number of Customers'},
            color=list(seg_counts.keys()),
            color_discrete_sequence=['#3B82F6', '#10B981', '#F59E0B', '#EF4444']
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Fallback to Streamlit bar chart
        seg_df = pd.DataFrame({
            'Segment': list(seg_counts.keys()),
            'Count': list(seg_counts.values())
        })
        st.bar_chart(seg_df.set_index('Segment'))

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
                    "intl_received_withdrawal_segmented": "Received International & Withdrew (Segmented)",
                    "intl_received_no_withdrawal_other": "Received International, NO Withdrawal, Uses Other Services",
                    "intl_received_only_withdrawal": "Received International & ONLY Withdrew"
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
                        elif selected_pair_filter == "intl_received_withdrawal_segmented":
                            st.info("Shows customers who received International remittance and then withdrew, segmented by withdrawal percentage.")
                        elif selected_pair_filter == "intl_received_no_withdrawal_other":
                            st.info("Shows customers who received International remittance, did NOT withdraw, but used other services.")
                        elif selected_pair_filter == "intl_received_only_withdrawal":
                            st.info("Shows customers who received International remittance and ONLY withdrew (no other services).")
                
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
            filtered_df, filter_desc, customer_count, segmentation_results = filter_data(
                df, start_date, end_date, product_filter, customer_type_filter, selected_pair_filter
            )
        
        if len(filtered_df) == 0:
            st.warning("⚠️ No data found for the selected filters. Please adjust your criteria.")
        else:
            # Display filter info
            st.markdown(f'<div class="filter-pair-info">{filter_desc}</div>', unsafe_allow_html=True)
            
            # Display segmentation results if available
            if segmentation_results:
                display_segmentation_results(segmentation_results)
            
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
            
            # Show current filter info
            with st.expander("📋 Current Filter Information", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Applied Filters:**")
                    st.write(f"- Customer Pair: {pair_filter_options[selected_pair_filter]}")
                    st.write(f"- Date Range: {start_date} to {end_date}")
                    st.write(f"- Product Filter: {product_filter}")
                    st.write(f"- Customer Type: {customer_type_filter}")
                
                with col2:
                    st.write("**Filter Results:**")
                    st.write(f"- Transactions after filter: {len(filtered_df):,}")
                    st.write(f"- Customers after filter: {customer_count:,}")
                    if len(df) > 0:
                        coverage = len(filtered_df)/len(df)*100
                        st.write(f"- Data coverage: {coverage:.1f}% of original data")
            
            # Create visualizations
            st.markdown('<h2 class="sub-header">📊 Data Visualizations</h2>', unsafe_allow_html=True)
            
            if PLOTLY_AVAILABLE:
                viz_data = create_visualizations(filtered_df, start_date, end_date)
                
                # Display charts in grid if available
                if viz_data:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'daily_volume' in viz_data:
                            st.plotly_chart(viz_data['daily_volume'], use_container_width=True)
                        if 'hourly_dist' in viz_data:
                            st.plotly_chart(viz_data['hourly_dist'], use_container_width=True)
                    
                    with col2:
                        if 'top_products' in viz_data:
                            st.plotly_chart(viz_data['top_products'], use_container_width=True)
                        if 'customer_segmentation' in viz_data:
                            st.plotly_chart(viz_data['customer_segmentation'], use_container_width=True)
                    
                    if 'weekday_analysis' in viz_data:
                        st.plotly_chart(viz_data['weekday_analysis'], use_container_width=True)
            else:
                # Use basic visualizations
                basic_viz = create_basic_visualizations(filtered_df)
                
                if basic_viz:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'daily_volume' in basic_viz:
                            st.subheader("Daily Transaction Volume")
                            st.line_chart(basic_viz['daily_volume'].set_index('Date'))
                        
                        if 'hourly_dist' in basic_viz:
                            st.subheader("Hourly Distribution")
                            st.bar_chart(basic_viz['hourly_dist'])
                    
                    with col2:
                        if 'top_products' in basic_viz:
                            st.subheader("Top Products")
                            st.bar_chart(basic_viz['top_products'])
                        
                        if 'customer_segmentation' in basic_viz:
                            st.subheader("Customer Types")
                            st.write(basic_viz['customer_segmentation'])
            
            # Run analysis
            st.markdown('<h2 class="sub-header">🎯 Telemarketing Analysis Results</h2>', unsafe_allow_html=True)
            
            with st.spinner("Analyzing data for telemarketing targets..."):
                results = analyze_telemarketing_data(filtered_df)
            
            # Display results in tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "🎯 International Targets", 
                "🏠 Domestic Targets", 
                "📋 Impact Template", 
                "📊 Summary"
            ])
            
            with tab1:
                st.subheader("International Remittance Customers NOT Using P2P")
                if not results['intl_not_p2p'].empty:
                    st.dataframe(results['intl_not_p2p'], use_container_width=True)
                    st.info(f"Found {len(results['intl_not_p2p'])} international customers not using P2P")
                else:
                    st.info("No international customers found who are not using P2P")
                
            with tab2:
                st.subheader("Domestic Customers Using Other Services But NOT P2P")
                if not results['domestic_other_not_p2p'].empty:
                    st.dataframe(results['domestic_other_not_p2p'], use_container_width=True)
                    st.info(f"Found {len(results['domestic_other_not_p2p'])} domestic customers not using P2P")
                else:
                    st.info("No domestic customers found who are not using P2P")
                
            with tab3:
                st.subheader("Daily Impact Report Template")
                st.dataframe(results['impact_template'], use_container_width=True)
                st.info("Use this template to track daily calling campaign results")
                
            with tab4:
                st.subheader("Campaign Summary")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📊 Key Statistics")
                    stats_data = {
                        'Metric': [
                            'International Customers',
                            'International Not Using P2P',
                            'Domestic Not Using P2P',
                            'P2P Users',
                            'Total Addressable Market'
                        ],
                        'Count': [
                            results['summary_stats']['intl_customers'],
                            results['summary_stats']['intl_not_p2p'],
                            results['summary_stats']['domestic_not_p2p'],
                            results['summary_stats']['p2p_customers'],
                            results['summary_stats']['intl_not_p2p'] + results['summary_stats']['domestic_not_p2p']
                        ]
                    }
                    stats_df = pd.DataFrame(stats_data)
                    st.dataframe(stats_df, use_container_width=True)
                
                with col2:
                    st.markdown("### 🎯 Campaign Focus")
                    
                    # Create simple priority indicator
                    priorities = pd.DataFrame({
                        'Segment': ['International', 'Domestic'],
                        'Count': [results['summary_stats']['intl_not_p2p'], 
                                 results['summary_stats']['domestic_not_p2p']]
                    })
                    
                    if not priorities.empty:
                        st.bar_chart(priorities.set_index('Segment'))
                    
                    # Quick insights
                    st.markdown("#### 💡 Quick Insights")
                    if results['summary_stats']['intl_customers'] > 0:
                        conversion_potential = (results['summary_stats']['intl_not_p2p'] / 
                                              max(results['summary_stats']['intl_customers'], 1)) * 100
                        st.write(f"- **{conversion_potential:.1f}%** of international customers don't use P2P")
                    
                    total_targets = results['summary_stats']['intl_not_p2p'] + results['summary_stats']['domestic_not_p2p']
                    if total_targets > 0:
                        st.write(f"- **{total_targets}** total customers to target")
                
                # Recommendations
                st.markdown("### 💡 Recommendations")
                
                recommendations = [
                    f"**Priority 1**: Target {results['summary_stats']['intl_not_p2p']} international customers first (highest potential)",
                    f"**Priority 2**: Engage {results['summary_stats']['domestic_not_p2p']} domestic customers (cross-sell opportunity)",
                    "**Strategy**: Bundle P2P promotion with their most used service",
                    "**Timing**: Analyze their transaction patterns for optimal calling times",
                    "**Incentive**: Consider offering incentives for P2P adoption",
                    "**Tracking**: Use the impact template to measure campaign success"
                ]
                
                for rec in recommendations:
                    st.markdown(f"✅ {rec}")
            
            # Download section
            st.markdown('<h2 class="sub-header">📥 Download Reports</h2>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Download Excel report
                excel_data = create_excel_download(results, filter_desc, segmentation_results)
                if excel_data:
                    st.download_button(
                        label="📊 Download Full Report",
                        data=excel_data,
                        file_name=f"telemarketing_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
            
            with col2:
                # Download CSV for international targets
                if not results['intl_not_p2p'].empty:
                    csv_intl = results['intl_not_p2p'].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="🎯 International Targets",
                        data=csv_intl,
                        file_name="international_targets.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col3:
                # Download CSV for domestic targets
                if not results['domestic_other_not_p2p'].empty:
                    csv_domestic = results['domestic_other_not_p2p'].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="🏠 Domestic Targets",
                        data=csv_domestic,
                        file_name="domestic_targets.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col4:
                # Download impact template
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
                if results['summary_stats']['intl_customers'] > 0:
                    opportunity = (results['summary_stats']['intl_not_p2p'] / 
                                 results['summary_stats']['intl_customers'] * 100)
                    st.metric("Conversion Opportunity", f"{opportunity:.1f}%", "international market")

if __name__ == "__main__":
    main()
