import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import warnings
from collections import defaultdict
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
            df['Transaction Amount'] = pd.to_numeric(df['Transaction Amount'], errors='coerce')
        
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
        return df[mask]['User Identifier'].dropna().unique()
    
    # Get customer sets for each product category
    p2p_customers = set(get_customers_for_product(p2p_products))
    intl_customers = set(get_customers_for_product(international_remittance))
    deposit_customers = set(get_customers_for_product(deposit_products))
    withdrawal_customers = set(get_customers_for_product(withdrawal_products))
    receive_intl_customers = set(get_customers_for_product(receive_intl_products))
    
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
        # Get all customers who used other services (excluding receive intl and withdrawal)
        other_services = p2p_products + deposit_products + ['Bill Payment', 'Airtime', 'Utility', 'Topup', 'Ticket', 'Scan']
        other_customers = set(get_customers_for_product(other_services))
        filtered_customers = receive_intl_customers & withdrawal_customers - other_customers
        filter_description = f"Customers who Received International Remittance and ONLY Withdrew\n\n• International Recipients: {len(receive_intl_customers):,}\n• Pure Withdrawal Only: {len(filtered_customers):,} customers"
    
    elif filter_pair == "all_customers":
        # All customers (no filter)
        filtered_customers = set(unique_customers)
        filter_description = f"All Customers\n\n• Total Unique Customers: {len(filtered_customers):,}"
    
    # Filter the dataframe to only include transactions from the selected customers
    filtered_df = df[df['User Identifier'].isin(filtered_customers)].copy()
    
    return filtered_df, filter_description, len(filtered_customers)

def analyze_intl_withdrawal_segments(df):
    """Analyze withdrawal behavior segmentation for international remittance recipients"""
    
    # Define product categories
    receive_intl_pattern = '|'.join(['International Remittance', 'International Transfer', 'Remittance', 'International', 'Receive'])
    withdrawal_pattern = '|'.join(['Withdrawal', 'Scan To Withdraw', 'Cash Out', 'Withdraw'])
    
    # Get all international recipients
    receive_mask = df['Product Name'].str.contains(receive_intl_pattern, case=False, na=False)
    receive_intl_customers = df[receive_mask]['User Identifier'].dropna().unique()
    
    results = {
        'withdrawal_segments': {},
        'detailed_analysis': {},
        'summary_stats': {}
    }
    
    if len(receive_intl_customers) == 0:
        return results
    
    # Analyze each international recipient
    segment_analysis = []
    detailed_records = []
    
    for cust_id in receive_intl_customers[:1000]:  # Limit for performance
        cust_data = df[df['User Identifier'] == cust_id].copy()
        
        # Get customer name
        name_record = cust_data[cust_data['Full Name'].notna() & (cust_data['Full Name'] != 'nan')]
        full_name = name_record.iloc[0]['Full Name'] if not name_record.empty else 'Name not available'
        
        # Separate receive and withdrawal transactions
        receive_transactions = cust_data[receive_mask]
        withdrawal_transactions = cust_data[cust_data['Product Name'].str.contains(withdrawal_pattern, case=False, na=False)]
        
        # Calculate amounts if available
        total_received = receive_transactions['Transaction Amount'].sum() if 'Transaction Amount' in df.columns else len(receive_transactions)
        total_withdrawn = withdrawal_transactions['Transaction Amount'].sum() if 'Transaction Amount' in df.columns else len(withdrawal_transactions)
        
        # Calculate withdrawal percentage
        if total_received > 0:
            withdrawal_percentage = (total_withdrawn / total_received) * 100
        else:
            withdrawal_percentage = 0
        
        # Determine segment
        if withdrawal_percentage <= 25:
            segment = '≤25%'
            segment_color = 'segment-25'
        elif withdrawal_percentage <= 50:
            segment = '25% < x ≤ 50%'
            segment_color = 'segment-50'
        elif withdrawal_percentage <= 75:
            segment = '50% < x ≤ 75%'
            segment_color = 'segment-75'
        else:
            segment = '75% < x ≤ 100%'
            segment_color = 'segment-100'
        
        # Get other services used
        other_services = cust_data[~receive_mask & ~cust_data['Product Name'].str.contains(withdrawal_pattern, case=False, na=False)]
        other_services_list = other_services['Product Name'].dropna().unique()
        
        segment_analysis.append({
            'User_ID': cust_id,
            'Full_Name': full_name,
            'Total_Received': total_received,
            'Total_Withdrawn': total_withdrawn,
            'Withdrawal_Percentage': withdrawal_percentage,
            'Segment': segment,
            'Segment_Color': segment_color,
            'First_Transaction_Date': cust_data['Created At'].min().strftime('%Y-%m-%d') if pd.notna(cust_data['Created At'].min()) else 'Unknown',
            'Last_Transaction_Date': cust_data['Created At'].max().strftime('%Y-%m-%d') if pd.notna(cust_data['Created At'].max()) else 'Unknown',
            'Other_Services_Count': len(other_services_list)
        })
        
        # Detailed records for export
        detailed_records.append({
            'User_ID': cust_id,
            'Full_Name': full_name,
            'Segment': segment,
            'Withdrawal_Percentage': f"{withdrawal_percentage:.1f}%",
            'Total_Received_Amount': total_received,
            'Total_Withdrawn_Amount': total_withdrawn,
            'Receive_Transaction_Count': len(receive_transactions),
            'Withdrawal_Transaction_Count': len(withdrawal_transactions),
            'Other_Services_Used': ', '.join([str(s) for s in other_services_list[:5]]),
            'First_International_Date': receive_transactions['Created At'].min().strftime('%Y-%m-%d') if len(receive_transactions) > 0 else 'None',
            'Last_International_Date': receive_transactions['Created At'].max().strftime('%Y-%m-%d') if len(receive_transactions) > 0 else 'None',
            'First_Withdrawal_Date': withdrawal_transactions['Created At'].min().strftime('%Y-%m-%d') if len(withdrawal_transactions) > 0 else 'None',
            'Last_Withdrawal_Date': withdrawal_transactions['Created At'].max().strftime('%Y-%m-%d') if len(withdrawal_transactions) > 0 else 'None'
        })
    
    # Convert to DataFrames
    segment_df = pd.DataFrame(segment_analysis)
    detailed_df = pd.DataFrame(detailed_records)
    
    # Calculate segment distribution
    if not segment_df.empty:
        segment_distribution = segment_df['Segment'].value_counts().reindex(['≤25%', '25% < x ≤ 50%', '50% < x ≤ 75%', '75% < x ≤ 100%'], fill_value=0)
        segment_percentages = (segment_distribution / len(segment_df) * 100).round(1)
        
        results['withdrawal_segments'] = {
            'distribution': segment_distribution,
            'percentages': segment_percentages,
            'total_customers': len(segment_df)
        }
    
    results['detailed_analysis'] = detailed_df
    results['segment_data'] = segment_df
    
    # Summary statistics
    if not detailed_df.empty:
        # Extract numeric values for summary
        withdrawal_percentages = segment_df['Withdrawal_Percentage']
        
        results['summary_stats'] = {
            'total_intl_recipients': len(receive_intl_customers),
            'analyzed_customers': len(segment_df),
            'avg_withdrawal_percentage': withdrawal_percentages.mean(),
            'median_withdrawal_percentage': withdrawal_percentages.median(),
            'max_withdrawal_percentage': withdrawal_percentages.max(),
            'min_withdrawal_percentage': withdrawal_percentages.min(),
            'total_received_amount': segment_df['Total_Received'].sum(),
            'total_withdrawn_amount': segment_df['Total_Withdrawn'].sum(),
            'segment_25_count': len(segment_df[segment_df['Segment'] == '≤25%']),
            'segment_50_count': len(segment_df[segment_df['Segment'] == '25% < x ≤ 50%']),
            'segment_75_count': len(segment_df[segment_df['Segment'] == '50% < x ≤ 75%']),
            'segment_100_count': len(segment_df[segment_df['Segment'] == '75% < x ≤ 100%'])
        }
    
    return results

def analyze_specific_intl_groups(df):
    """Analyze specific international remittance recipient groups"""
    
    # Define product patterns
    receive_intl_pattern = '|'.join(['International Remittance', 'International Transfer', 'Remittance', 'International', 'Receive'])
    withdrawal_pattern = '|'.join(['Withdrawal', 'Scan To Withdraw', 'Cash Out', 'Withdraw'])
    other_services_pattern = '|'.join(['P2P', 'Deposit', 'Bill Payment', 'Airtime', 'Utility', 'Topup', 'Ticket', 'Scan'])
    
    # Get all international recipients
    receive_mask = df['Product Name'].str.contains(receive_intl_pattern, case=False, na=False)
    receive_intl_customers = df[receive_mask]['User Identifier'].dropna().unique()
    
    groups = {
        'received_withdrew': [],
        'received_no_withdraw_other': [],
        'received_only_withdraw': []
    }
    
    for cust_id in receive_intl_customers[:1000]:  # Limit for performance
        cust_data = df[df['User Identifier'] == cust_id].copy()
        
        # Get customer name
        name_record = cust_data[cust_data['Full Name'].notna() & (cust_data['Full Name'] != 'nan')]
        full_name = name_record.iloc[0]['Full Name'] if not name_record.empty else 'Name not available'
        
        # Check transaction types
        has_receive_intl = receive_mask.any()
        has_withdrawal = cust_data['Product Name'].str.contains(withdrawal_pattern, case=False, na=False).any()
        has_other_services = cust_data['Product Name'].str.contains(other_services_pattern, case=False, na=False).any()
        
        # Calculate metrics
        receive_count = len(cust_data[receive_mask])
        withdrawal_count = len(cust_data[cust_data['Product Name'].str.contains(withdrawal_pattern, case=False, na=False)])
        other_count = len(cust_data[cust_data['Product Name'].str.contains(other_services_pattern, case=False, na=False)])
        
        # Prepare customer record
        customer_record = {
            'User_ID': cust_id,
            'Full_Name': full_name,
            'Receive_Count': receive_count,
            'Withdrawal_Count': withdrawal_count,
            'Other_Services_Count': other_count,
            'First_Transaction_Date': cust_data['Created At'].min().strftime('%Y-%m-%d') if pd.notna(cust_data['Created At'].min()) else 'Unknown',
            'Last_Transaction_Date': cust_data['Created At'].max().strftime('%Y-%m-%d') if pd.notna(cust_data['Created At'].max()) else 'Unknown'
        }
        
        # Categorize customer
        if has_receive_intl and has_withdrawal:
            groups['received_withdrew'].append(customer_record)
        
        if has_receive_intl and not has_withdrawal and has_other_services:
            groups['received_no_withdraw_other'].append(customer_record)
        
        if has_receive_intl and has_withdrawal and not has_other_services:
            groups['received_only_withdraw'].append(customer_record)
    
    # Convert to DataFrames
    results = {}
    for group_name, records in groups.items():
        if records:
            df = pd.DataFrame(records)
            
            # Add summary metrics
            if not df.empty:
                df['Withdrawal_Ratio'] = df.apply(
                    lambda x: f"{x['Withdrawal_Count']}/{x['Receive_Count']}" if x['Receive_Count'] > 0 else "0/0",
                    axis=1
                )
                df['Activity_Ratio'] = df.apply(
                    lambda x: f"{x['Other_Services_Count']}/{x['Receive_Count']}" if x['Receive_Count'] > 0 else "0/0",
                    axis=1
                )
            
            results[group_name] = df
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
            'intl_withdrawal_segment_100': intl_withdrawal_segments['summary_stats'].get('segment_100_count', 0)
        })
    
    return {
        'intl_not_p2p': report1_df,
        'domestic_other_not_p2p': report2_df,
        'impact_template': impact_df,
        'intl_withdrawal_segments': intl_withdrawal_segments,
        'specific_intl_groups': specific_intl_groups,
        'summary_stats': summary_stats
    }

def create_excel_download(results, filter_desc=""):
    """Create Excel file with all reports including new international withdrawal analysis"""
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
            
            # Sheet 4: New - International Withdrawal Segments
            if 'detailed_analysis' in results['intl_withdrawal_segments'] and not results['intl_withdrawal_segments']['detailed_analysis'].empty:
                # Add summary at the top
                summary_data = []
                
                if 'summary_stats' in results['intl_withdrawal_segments']:
                    stats = results['intl_withdrawal_segments']['summary_stats']
                    summary_data = [
                        ['International Withdrawal Analysis Summary'],
                        [''],
                        ['Total International Recipients Analyzed', stats.get('analyzed_customers', 0)],
                        ['Average Withdrawal Percentage', f"{stats.get('avg_withdrawal_percentage', 0):.1f}%"],
                        ['Median Withdrawal Percentage', f"{stats.get('median_withdrawal_percentage', 0):.1f}%"],
                        ['Total Received Amount', stats.get('total_received_amount', 0)],
                        ['Total Withdrawn Amount', stats.get('total_withdrawn_amount', 0)],
                        [''],
                        ['Segment Distribution:'],
                        ['≤25% Withdrawal', stats.get('segment_25_count', 0)],
                        ['25% < x ≤ 50%', stats.get('segment_50_count', 0)],
                        ['50% < x ≤ 75%', stats.get('segment_75_count', 0)],
                        ['75% < x ≤ 100%', stats.get('segment_100_count', 0)],
                        [''],
                        ['Detailed Customer Analysis:']
                    ]
                
                # Create DataFrame for summary
                summary_df = pd.DataFrame(summary_data)
                
                # Write summary and data
                startrow = len(summary_df) + 1
                summary_df.to_excel(writer, sheet_name='Intl_Withdrawal_Segments', index=False, header=False)
                results['intl_withdrawal_segments']['detailed_analysis'].to_excel(
                    writer, sheet_name='Intl_Withdrawal_Segments', 
                    startrow=startrow, index=False
                )
            
            # Sheet 5: New - International Received & Withdrew
            if 'received_withdrew' in results['specific_intl_groups'] and not results['specific_intl_groups']['received_withdrew'].empty:
                summary_data = [
                    ['Customers who Received International Remittance AND Withdrew'],
                    [''],
                    ['Total Customers', len(results['specific_intl_groups']['received_withdrew'])],
                    ['Average Receive Count', results['specific_intl_groups']['received_withdrew']['Receive_Count'].mean().round(1)],
                    ['Average Withdrawal Count', results['specific_intl_groups']['received_withdrew']['Withdrawal_Count'].mean().round(1)],
                    ['Average Other Services Count', results['specific_intl_groups']['received_withdrew']['Other_Services_Count'].mean().round(1)],
                    [''],
                    ['Customer Details:']
                ]
                
                summary_df = pd.DataFrame(summary_data)
                startrow = len(summary_df) + 1
                summary_df.to_excel(writer, sheet_name='Intl_Received_Withdrew', index=False, header=False)
                results['specific_intl_groups']['received_withdrew'].to_excel(
                    writer, sheet_name='Intl_Received_Withdrew', 
                    startrow=startrow, index=False
                )
            
            # Sheet 6: New - International Received, No Withdrawal, Other Services
            if 'received_no_withdraw_other' in results['specific_intl_groups'] and not results['specific_intl_groups']['received_no_withdraw_other'].empty:
                summary_data = [
                    ['Customers who Received International Remittance, NO Withdrawal, but used Other Services'],
                    [''],
                    ['Total Customers', len(results['specific_intl_groups']['received_no_withdraw_other'])],
                    ['Average Receive Count', results['specific_intl_groups']['received_no_withdraw_other']['Receive_Count'].mean().round(1)],
                    ['Average Other Services Count', results['specific_intl_groups']['received_no_withdraw_other']['Other_Services_Count'].mean().round(1)],
                    [''],
                    ['Customer Details:']
                ]
                
                summary_df = pd.DataFrame(summary_data)
                startrow = len(summary_df) + 1
                summary_df.to_excel(writer, sheet_name='Intl_No_Withdraw_Other', index=False, header=False)
                results['specific_intl_groups']['received_no_withdraw_other'].to_excel(
                    writer, sheet_name='Intl_No_Withdraw_Other', 
                    startrow=startrow, index=False
                )
            
            # Sheet 7: New - International Received & Only Withdrew
            if 'received_only_withdraw' in results['specific_intl_groups'] and not results['specific_intl_groups']['received_only_withdraw'].empty:
                summary_data = [
                    ['Customers who Received International Remittance and ONLY Withdrew (no other services)'],
                    [''],
                    ['Total Customers', len(results['specific_intl_groups']['received_only_withdraw'])],
                    ['Average Receive Count', results['specific_intl_groups']['received_only_withdraw']['Receive_Count'].mean().round(1)],
                    ['Average Withdrawal Count', results['specific_intl_groups']['received_only_withdraw']['Withdrawal_Count'].mean().round(1)],
                    [''],
                    ['Customer Details:']
                ]
                
                summary_df = pd.DataFrame(summary_data)
                startrow = len(summary_df) + 1
                summary_df.to_excel(writer, sheet_name='Intl_Only_Withdrew', index=False, header=False)
                results['specific_intl_groups']['received_only_withdraw'].to_excel(
                    writer, sheet_name='Intl_Only_Withdrew', 
                    startrow=startrow, index=False
                )
            
            # Sheet 8: Summary
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
                    'Total Addressable Market',
                    '',
                    'International Withdrawal Analysis:',
                    '  ≤25% Withdrawal Segment',
                    '  25% < x ≤ 50% Segment',
                    '  50% < x ≤ 75% Segment',
                    '  75% < x ≤ 100% Segment',
                    '  Received & Withdrew',
                    '  Received, No Withdraw, Other Services',
                    '  Received & Only Withdrew'
                ],
                'Value': [
                    filter_desc[:100],
                    results['summary_stats']['start_date'],
                    results['summary_stats']['end_date'],
                    results['summary_stats']['total_transactions'],
                    results['summary_stats']['unique_customers'],
                    results['summary_stats']['intl_customers'],
                    results['summary_stats']['intl_not_p2p'],
                    results['summary_stats']['p2p_customers'],
                    results['summary_stats']['other_customers'],
                    results['summary_stats']['domestic_not_p2p'],
                    results['summary_stats']['intl_not_p2p'] + results['summary_stats']['domestic_not_p2p'],
                    '',
                    '',
                    results['summary_stats'].get('intl_withdrawal_segment_25', 0),
                    results['summary_stats'].get('intl_withdrawal_segment_50', 0),
                    results['summary_stats'].get('intl_withdrawal_segment_75', 0),
                    results['summary_stats'].get('intl_withdrawal_segment_100', 0),
                    len(results['specific_intl_groups'].get('received_withdrew', [])),
                    len(results['specific_intl_groups'].get('received_no_withdraw_other', [])),
                    len(results['specific_intl_groups'].get('received_only_withdraw', []))
                ]
            })
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 9: Recommendations
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
                "FEEDBACK: Collect customer feedback during calls",
                "",
                "INTERNATIONAL WITHDRAWAL SEGMENT STRATEGY:",
                f"  • ≤25% segment ({results['summary_stats'].get('intl_withdrawal_segment_25', 0)} customers): Focus on retention & cross-selling",
                f"  • 25-50% segment ({results['summary_stats'].get('intl_withdrawal_segment_50', 0)} customers): Balance services education",
                f"  • 50-75% segment ({results['summary_stats'].get('intl_withdrawal_segment_75', 0)} customers): Risk mitigation strategies",
                f"  • 75-100% segment ({results['summary_stats'].get('intl_withdrawal_segment_100', 0)} customers): High withdrawal behavior review"
            ]
            rec_df = pd.DataFrame({'Recommendations': recommendations})
            rec_df.to_excel(writer, sheet_name='Recommendations', index=False)
        
        output.seek(0)
        return output
    except Exception as e:
        st.error(f"Error creating Excel file: {str(e)}")
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
                    st.write(f"- Data coverage: {len(filtered_df)/len(df)*100:.1f}% of original data")
            
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
            
            # Display results in tabs - UPDATED WITH NEW TABS
            tab_names = [
                "🎯 International Targets", 
                "🏠 Domestic Targets", 
                "📊 Intl Withdrawal Segments",
                "💸 Intl Received & Withdrew",
                "🔄 Intl No Withdraw Other",
                "💰 Intl Only Withdrew",
                "📋 Impact Template", 
                "📈 Summary"
            ]
            
            tabs = st.tabs(tab_names)
            
            with tabs[0]:  # International Targets
                st.subheader("International Remittance Customers NOT Using P2P")
                if not results['intl_not_p2p'].empty:
                    st.dataframe(results['intl_not_p2p'], use_container_width=True)
                    st.info(f"Found {len(results['intl_not_p2p'])} international customers not using P2P")
                else:
                    st.info("No international customers found who are not using P2P")
                
            with tabs[1]:  # Domestic Targets
                st.subheader("Domestic Customers Using Other Services But NOT P2P")
                if not results['domestic_other_not_p2p'].empty:
                    st.dataframe(results['domestic_other_not_p2p'], use_container_width=True)
                    st.info(f"Found {len(results['domestic_other_not_p2p'])} domestic customers not using P2P")
                else:
                    st.info("No domestic customers found who are not using P2P")
            
            with tabs[2]:  # International Withdrawal Segments
                st.subheader("International Recipients - Withdrawal Percentage Segments")
                
                if 'withdrawal_segments' in results['intl_withdrawal_segments'] and results['intl_withdrawal_segments']['withdrawal_segments']:
                    segments = results['intl_withdrawal_segments']['withdrawal_segments']
                    
                    # Display segment distribution
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("≤25%", 
                                 f"{segments['distribution'].get('≤25%', 0):,}",
                                 f"{segments['percentages'].get('≤25%', 0):.1f}%")
                    with col2:
                        st.metric("25% < x ≤ 50%", 
                                 f"{segments['distribution'].get('25% < x ≤ 50%', 0):,}",
                                 f"{segments['percentages'].get('25% < x ≤ 50%', 0):.1f}%")
                    with col3:
                        st.metric("50% < x ≤ 75%", 
                                 f"{segments['distribution'].get('50% < x ≤ 75%', 0):,}",
                                 f"{segments['percentages'].get('50% < x ≤ 75%', 0):.1f}%")
                    with col4:
                        st.metric("75% < x ≤ 100%", 
                                 f"{segments['distribution'].get('75% < x ≤ 100%', 0):,}",
                                 f"{segments['percentages'].get('75% < x ≤ 100%', 0):.1f}%")
                    
                    # Show detailed analysis
                    if 'detailed_analysis' in results['intl_withdrawal_segments'] and not results['intl_withdrawal_segments']['detailed_analysis'].empty:
                        st.subheader("Detailed Customer Analysis")
                        st.dataframe(results['intl_withdrawal_segments']['detailed_analysis'], use_container_width=True)
                        
                        # Show insights
                        st.subheader("💡 Segment Insights")
                        insights = [
                            f"**≤25% Segment**: Low withdrawal behavior - good retention opportunity",
                            f"**25-50% Segment**: Moderate withdrawal - balanced service usage",
                            f"**50-75% Segment**: High withdrawal - monitor for churn risk",
                            f"**75-100% Segment**: Very high withdrawal - immediate attention needed"
                        ]
                        for insight in insights:
                            st.write(insight)
                else:
                    st.info("No international recipients found for withdrawal segmentation analysis")
            
            with tabs[3]:  # International Received & Withdrew
                st.subheader("Customers who Received International Remittance AND Withdrew")
                
                if 'received_withdrew' in results['specific_intl_groups'] and not results['specific_intl_groups']['received_withdrew'].empty:
                    df = results['specific_intl_groups']['received_withdrew']
                    
                    # Display summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Customers", f"{len(df):,}")
                    with col2:
                        avg_receive = df['Receive_Count'].mean().round(1)
                        st.metric("Avg Receive Count", avg_receive)
                    with col3:
                        avg_withdraw = df['Withdrawal_Count'].mean().round(1)
                        st.metric("Avg Withdrawal Count", avg_withdraw)
                    
                    # Show data
                    st.dataframe(df, use_container_width=True)
                    
                    # Insight
                    st.info(f"**Insight**: {len(df)} customers received international remittance and also made withdrawals. "
                           f"These customers show balanced usage of both receiving and withdrawal services.")
                else:
                    st.info("No customers found who both received international remittance and made withdrawals")
            
            with tabs[4]:  # International No Withdraw Other
                st.subheader("Customers who Received International Remittance, NO Withdrawal, but used Other Services")
                
                if 'received_no_withdraw_other' in results['specific_intl_groups'] and not results['specific_intl_groups']['received_no_withdraw_other'].empty:
                    df = results['specific_intl_groups']['received_no_withdraw_other']
                    
                    # Display summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Customers", f"{len(df):,}")
                    with col2:
                        avg_receive = df['Receive_Count'].mean().round(1)
                        st.metric("Avg Receive Count", avg_receive)
                    with col3:
                        avg_other = df['Other_Services_Count'].mean().round(1)
                        st.metric("Avg Other Services", avg_other)
                    
                    # Show data
                    st.dataframe(df, use_container_width=True)
                    
                    # Insight
                    st.info(f"**Insight**: {len(df)} customers received international remittance but didn't withdraw. "
                           f"They used other services instead - good candidates for cross-selling withdrawal services.")
                else:
                    st.info("No customers found who received international remittance without withdrawals but used other services")
            
            with tabs[5]:  # International Only Withdrew
                st.subheader("Customers who Received International Remittance and ONLY Withdrew")
                
                if 'received_only_withdraw' in results['specific_intl_groups'] and not results['specific_intl_groups']['received_only_withdraw'].empty:
                    df = results['specific_intl_groups']['received_only_withdraw']
                    
                    # Display summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Customers", f"{len(df):,}")
                    with col2:
                        avg_receive = df['Receive_Count'].mean().round(1)
                        st.metric("Avg Receive Count", avg_receive)
                    with col3:
                        avg_withdraw = df['Withdrawal_Count'].mean().round(1)
                        st.metric("Avg Withdrawal Count", avg_withdraw)
                    
                    # Show data
                    st.dataframe(df, use_container_width=True)
                    
                    # Insight
                    st.info(f"**Insight**: {len(df)} customers received international remittance and only made withdrawals. "
                           f"They show simple cash-out behavior - opportunity to introduce other financial services.")
                else:
                    st.info("No customers found who only received international remittance and made withdrawals (no other services)")
            
            with tabs[6]:  # Impact Template
                st.subheader("Daily Impact Report Template")
                st.dataframe(results['impact_template'], use_container_width=True)
                st.info("Use this template to track daily calling campaign results")
                
            with tabs[7]:  # Summary
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
                            'Total Addressable Market',
                            '',
                            'International Withdrawal Segments:',
                            '  ≤25%',
                            '  25% < x ≤ 50%',
                            '  50% < x ≤ 75%',
                            '  75% < x ≤ 100%',
                            '',
                            'Specific Intl Groups:',
                            '  Received & Withdrew',
                            '  Received, No Withdraw, Other',
                            '  Received & Only Withdrew'
                        ],
                        'Count': [
                            results['summary_stats']['intl_customers'],
                            results['summary_stats']['intl_not_p2p'],
                            results['summary_stats']['domestic_not_p2p'],
                            results['summary_stats']['p2p_customers'],
                            results['summary_stats']['intl_not_p2p'] + results['summary_stats']['domestic_not_p2p'],
                            '',
                            '',
                            results['summary_stats'].get('intl_withdrawal_segment_25', 0),
                            results['summary_stats'].get('intl_withdrawal_segment_50', 0),
                            results['summary_stats'].get('intl_withdrawal_segment_75', 0),
                            results['summary_stats'].get('intl_withdrawal_segment_100', 0),
                            '',
                            '',
                            len(results['specific_intl_groups'].get('received_withdrew', [])),
                            len(results['specific_intl_groups'].get('received_no_withdraw_other', [])),
                            len(results['specific_intl_groups'].get('received_only_withdraw', []))
                        ]
                    }
                    stats_df = pd.DataFrame(stats_data)
                    st.dataframe(stats_df, use_container_width=True, height=400)
                
                with col2:
                    st.markdown("### 🎯 Campaign Focus")
                    
                    # Create segment visualization
                    segment_data = pd.DataFrame({
                        'Segment': ['≤25%', '25-50%', '50-75%', '75-100%'],
                        'Count': [
                            results['summary_stats'].get('intl_withdrawal_segment_25', 0),
                            results['summary_stats'].get('intl_withdrawal_segment_50', 0),
                            results['summary_stats'].get('intl_withdrawal_segment_75', 0),
                            results['summary_stats'].get('intl_withdrawal_segment_100', 0)
                        ]
                    })
                    
                    if not segment_data.empty and segment_data['Count'].sum() > 0:
                        st.subheader("Withdrawal Segment Distribution")
                        st.bar_chart(segment_data.set_index('Segment'))
                    
                    # Quick insights
                    st.markdown("#### 💡 Quick Insights")
                    
                    total_intl_segments = sum([
                        results['summary_stats'].get('intl_withdrawal_segment_25', 0),
                        results['summary_stats'].get('intl_withdrawal_segment_50', 0),
                        results['summary_stats'].get('intl_withdrawal_segment_75', 0),
                        results['summary_stats'].get('intl_withdrawal_segment_100', 0)
                    ])
                    
                    if total_intl_segments > 0:
                        high_withdrawal = results['summary_stats'].get('intl_withdrawal_segment_75', 0) + results['summary_stats'].get('intl_withdrawal_segment_100', 0)
                        high_withdrawal_pct = (high_withdrawal / total_intl_segments * 100) if total_intl_segments > 0 else 0
                        st.write(f"- **{high_withdrawal_pct:.1f}%** of international recipients have >50% withdrawal rate")
                    
                    received_withdrew = len(results['specific_intl_groups'].get('received_withdrew', []))
                    if received_withdrew > 0:
                        st.write(f"- **{received_withdrew}** customers both receive and withdraw funds")
                
                # Recommendations
                st.markdown("### 💡 Recommendations")
                
                recommendations = [
                    f"**Priority 1**: Target {results['summary_stats']['intl_not_p2p']} international customers first (highest potential)",
                    f"**Priority 2**: Engage {results['summary_stats']['domestic_not_p2p']} domestic customers (cross-sell opportunity)",
                    f"**Withdrawal Strategy**: Focus on {results['summary_stats'].get('intl_withdrawal_segment_75', 0) + results['summary_stats'].get('intl_withdrawal_segment_100', 0)} high-withdrawal customers for retention",
                    f"**Cross-sell**: Engage {len(results['specific_intl_groups'].get('received_no_withdraw_other', []))} customers who receive but don't withdraw",
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
                excel_data = create_excel_download(results, filter_desc)
                if excel_data:
                    st.download_button(
                        label="📊 Download Full Report",
                        data=excel_data,
                        file_name=f"telemarketing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        help="Download all reports in one Excel file with separate sheets"
                    )
            
            with col2:
                # Download CSV for withdrawal segments
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
                # Download CSV for specific intl groups
                if 'received_withdrew' in results['specific_intl_groups'] and not results['specific_intl_groups']['received_withdrew'].empty:
                    csv_received = results['specific_intl_groups']['received_withdrew'].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="💸 Received & Withdrew",
                        data=csv_received,
                        file_name="received_withdrew.csv",
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
            
            # Additional download buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'received_no_withdraw_other' in results['specific_intl_groups'] and not results['specific_intl_groups']['received_no_withdraw_other'].empty:
                    csv_no_withdraw = results['specific_intl_groups']['received_no_withdraw_other'].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="🔄 No Withdraw Other",
                        data=csv_no_withdraw,
                        file_name="no_withdraw_other.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col2:
                if 'received_only_withdraw' in results['specific_intl_groups'] and not results['specific_intl_groups']['received_only_withdraw'].empty:
                    csv_only_withdraw = results['specific_intl_groups']['received_only_withdraw'].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="💰 Only Withdrew",
                        data=csv_only_withdraw,
                        file_name="only_withdrew.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col3:
                if not results['intl_not_p2p'].empty:
                    csv_intl = results['intl_not_p2p'].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="🎯 International Targets",
                        data=csv_intl,
                        file_name="international_targets.csv",
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
