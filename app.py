import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import warnings
from functools import lru_cache
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
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background-color: #3B82F6;
    }
    
    /* Optimize dataframe rendering */
    .stDataFrame {
        font-size: 0.85rem;
    }
    
    /* Cache indicator */
    .cache-info {
        background-color: #f0f9ff;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.875rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Global variables for caching
_CACHED_DATA = None
_CACHED_FILTERED_DATA = None
_CACHED_RESULTS = None

def load_data_optimized(uploaded_file, sample_size=None):
    """Load and clean the transaction data with optimizations"""
    try:
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Reading file...")
        progress_bar.progress(10)
        
        # Read file with optimizations
        if uploaded_file.name.endswith('.csv'):
            # For CSV, use optimized reading
            chunksize = 100000
            chunks = []
            for chunk in pd.read_csv(uploaded_file, chunksize=chunksize, low_memory=False):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            # For Excel, read with optimizations
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        
        progress_bar.progress(30)
        status_text.text(f"Loaded {len(df):,} rows. Cleaning data...")
        
        # Sample data if requested for faster testing
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            st.info(f"⚠️ Using sample of {sample_size:,} rows for faster processing. Uncheck 'Use Sample' for full analysis.")
        
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
        
        progress_bar.progress(50)
        status_text.text("Processing dates and amounts...")
        
        # Optimize data types
        df['User Identifier'] = pd.to_numeric(df['User Identifier'], errors='coerce')
        
        # Parse dates efficiently
        if df['Created At'].dtype == 'object':
            df['Created At'] = pd.to_datetime(df['Created At'], errors='coerce', infer_datetime_format=True)
        
        # Add derived columns efficiently
        df['Date'] = df['Created At'].dt.date
        df['Day'] = df['Created At'].dt.day_name()
        df['Hour'] = df['Created At'].dt.hour
        
        # Optimize text columns
        text_columns = ['Product Name', 'Service Name', 'Entity Name', 'Full Name']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().fillna('')
        
        # Convert transaction amount efficiently
        if 'Transaction Amount' in df.columns:
            # Use vectorized operations
            df['Transaction Amount'] = (
                df['Transaction Amount']
                .astype(str)
                .str.replace(r'[$,€£]', '', regex=True)
                .str.replace(',', '')
                .str.strip()
            )
            df['Transaction Amount'] = pd.to_numeric(df['Transaction Amount'], errors='coerce')
            df['Transaction Amount'] = df['Transaction Amount'].fillna(0)
        
        # Drop rows with missing essential data
        initial_count = len(df)
        df = df.dropna(subset=['User Identifier', 'Created At'])
        df = df[df['Created At'].notna()]
        
        if len(df) < initial_count:
            st.info(f"Removed {initial_count - len(df):,} rows with missing essential data")
        
        progress_bar.progress(80)
        status_text.text("Optimizing memory usage...")
        
        # Optimize memory usage
        df = optimize_dataframe_memory(df)
        
        progress_bar.progress(100)
        status_text.text("✅ Data loaded successfully!")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"✅ Successfully loaded {len(df):,} transactions")
        
        # Cache the loaded data
        global _CACHED_DATA
        _CACHED_DATA = df
        
        return df
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

def optimize_dataframe_memory(df):
    """Optimize dataframe memory usage"""
    # Convert object columns to category if they have limited unique values
    for col in df.select_dtypes(include=['object']).columns:
        num_unique = df[col].nunique()
        num_total = len(df)
        if num_unique / num_total < 0.5:  # If less than 50% unique values
            df[col] = df[col].astype('category')
    
    # Downcast numeric columns
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def filter_by_customer_pair_cached(df, filter_pair, start_date=None, end_date=None):
    """Cached version of filter_by_customer_pair"""
    return filter_by_customer_pair(df, filter_pair, start_date, end_date)

def filter_by_customer_pair(df, filter_pair, start_date=None, end_date=None):
    """Filter customers based on transaction behavior pairs - OPTIMIZED"""
    
    # Define product categories with compiled regex patterns
    p2p_pattern = re.compile('|'.join(['Internal Wallet Transfer (P2P)', 'Internal Wallet Transfer', 
                                      'P2P Transfer', 'Wallet Transfer', 'P2P']), re.IGNORECASE)
    
    intl_pattern = re.compile('|'.join(['International Remittance', 'International Transfer', 
                                       'Remittance', 'International']), re.IGNORECASE)
    
    deposit_pattern = re.compile('|'.join(['Deposit', 'Cash In', 'Deposit Customer', 
                                          'Deposit Agent']), re.IGNORECASE)
    
    withdrawal_pattern = re.compile('|'.join(['Withdrawal', 'Scan To Withdraw', 'Cash Out', 
                                            'Withdraw']), re.IGNORECASE)
    
    receive_pattern = re.compile('|'.join(['Receive', 'Credit', 'Incoming']), re.IGNORECASE)
    
    # Apply date filter first if provided
    filtered_df = df.copy()
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['Created At'] >= pd.Timestamp(start_date)) & 
            (filtered_df['Created At'] <= pd.Timestamp(end_date + timedelta(days=1)))
        ]
    
    # Get unique customers
    unique_customers = filtered_df['User Identifier'].dropna().unique()
    
    # Helper function to get customers for a product category - OPTIMIZED
    def get_customers_for_product(pattern):
        mask = filtered_df['Product Name'].apply(lambda x: bool(pattern.search(str(x))))
        return set(filtered_df.loc[mask, 'User Identifier'].dropna().unique())
    
    # Get customer sets for each product category
    p2p_customers = get_customers_for_product(p2p_pattern)
    intl_customers = get_customers_for_product(intl_pattern)
    deposit_customers = get_customers_for_product(deposit_pattern)
    withdrawal_customers = get_customers_for_product(withdrawal_pattern)
    
    # Get customers who received international remittance
    # Look for receive + international patterns
    receive_intl_mask = filtered_df['Product Name'].apply(
        lambda x: bool(receive_pattern.search(str(x))) and bool(intl_pattern.search(str(x)))
    )
    receive_intl_customers = set(filtered_df.loc[receive_intl_mask, 'User Identifier'].dropna().unique())
    
    # If no specific receive patterns, use all international as proxy
    if not receive_intl_customers:
        receive_intl_customers = intl_customers
    
    # Get customers using other services (excluding international and withdrawal)
    other_pattern = re.compile('|'.join(['P2P', 'Deposit', 'Bill', 'Airtime', 'Utility', 
                                        'Topup', 'Ticket', 'Scan', 'Send', 'Payment']), re.IGNORECASE)
    other_customers = get_customers_for_product(other_pattern)
    
    # Apply the selected filter pair
    filtered_customers = set()
    filter_description = ""
    
    if filter_pair == "intl_not_p2p":
        filtered_customers = intl_customers - p2p_customers
        filter_description = f"Customers who did International remittance but NOT P2P"
    
    elif filter_pair == "p2p_not_deposit":
        filtered_customers = p2p_customers - deposit_customers
        filter_description = f"Customers who did P2P but NOT Deposit"
    
    elif filter_pair == "p2p_and_withdrawal":
        filtered_customers = p2p_customers & withdrawal_customers
        filter_description = f"Customers who did BOTH P2P and Withdrawal"
    
    elif filter_pair == "intl_and_p2p":
        filtered_customers = intl_customers & p2p_customers
        filter_description = f"Customers who did BOTH International remittance and P2P"
    
    elif filter_pair == "intl_received_withdraw":
        filtered_customers = receive_intl_customers & withdrawal_customers
        filter_description = f"Customers who Received International Remittance AND Withdrew"
    
    elif filter_pair == "intl_received_no_withdraw":
        filtered_customers = receive_intl_customers - withdrawal_customers
        filter_description = f"Customers who Received International Remittance but NO Withdrawal"
    
    elif filter_pair == "intl_received_only_withdraw":
        filtered_customers = receive_intl_customers & withdrawal_customers - other_customers
        filter_description = f"Customers who Received International Remittance and ONLY Withdrew"
    
    elif filter_pair == "all_customers":
        filtered_customers = set(unique_customers)
        filter_description = f"All Customers"
    
    # Filter the dataframe
    if filtered_customers:
        result_df = filtered_df[filtered_df['User Identifier'].isin(filtered_customers)].copy()
    else:
        result_df = pd.DataFrame(columns=filtered_df.columns)
    
    # Add counts to description
    filter_description += f"\n\n• Result: {len(filtered_customers):,} customers\n• Transactions: {len(result_df):,}"
    
    return result_df, filter_description, len(filtered_customers)

@st.cache_data(ttl=3600, show_spinner=False)
def analyze_intl_withdrawal_segments_cached(df, max_customers=5000):
    """Cached version with customer limit"""
    return analyze_intl_withdrawal_segments(df, max_customers)

def analyze_intl_withdrawal_segments(df, max_customers=5000):
    """Analyze withdrawal behavior segmentation - OPTIMIZED with limit"""
    
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Analyzing international withdrawal segments...")
    
    # Define patterns with compiled regex
    intl_received_patterns = [
        'Receive.*International',
        'International.*Receive', 
        'International.*Credit',
        'Remittance.*Receive',
        'Receive.*Remittance',
        'International.*Incoming',
        'Incoming.*International'
    ]
    
    # Compile patterns
    receive_intl_regex = re.compile('|'.join(intl_received_patterns), re.IGNORECASE)
    general_intl_regex = re.compile('|'.join(['International Remittance', 'International Transfer', 
                                            'Remittance', 'International']), re.IGNORECASE)
    withdrawal_regex = re.compile('|'.join(['Withdrawal', 'Scan To Withdraw', 'Cash Out', 
                                          'Withdraw']), re.IGNORECASE)
    
    progress_bar.progress(20)
    
    # Find receive international transactions
    receive_mask = df['Product Name'].apply(lambda x: bool(receive_intl_regex.search(str(x))))
    
    # If no receive-specific transactions found, use general international
    if receive_mask.sum() == 0:
        receive_mask = df['Product Name'].apply(lambda x: bool(general_intl_regex.search(str(x))))
        if receive_mask.sum() == 0:
            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()
            return {
                'withdrawal_segments': {},
                'detailed_analysis': pd.DataFrame(),
                'summary_stats': {},
                'segment_data': pd.DataFrame()
            }
    
    # Get unique customers - limit for performance
    receive_intl_customers = df.loc[receive_mask, 'User Identifier'].dropna().unique()
    
    if len(receive_intl_customers) > max_customers:
        st.info(f"⚠️ Analyzing {max_customers:,} out of {len(receive_intl_customers):,} international recipients for performance.")
        receive_intl_customers = np.random.choice(receive_intl_customers, size=max_customers, replace=False)
    
    progress_bar.progress(40)
    status_text.text(f"Analyzing {len(receive_intl_customers):,} customers...")
    
    # Pre-compute masks for efficiency
    withdrawal_mask = df['Product Name'].apply(lambda x: bool(withdrawal_regex.search(str(x))))
    
    # Analyze in batches
    batch_size = 500
    segment_analysis = []
    detailed_records = []
    
    for i in range(0, len(receive_intl_customers), batch_size):
        batch_customers = receive_intl_customers[i:i+batch_size]
        batch_df = df[df['User Identifier'].isin(batch_customers)]
        
        for cust_id in batch_customers:
            cust_data = batch_df[batch_df['User Identifier'] == cust_id]
            if cust_data.empty:
                continue
            
            # Get customer name efficiently
            name_records = cust_data[cust_data['Full Name'].str.len() > 0]
            full_name = name_records.iloc[0]['Full Name'] if not name_records.empty else 'Name not available'
            
            # Get transactions efficiently
            receive_transactions = cust_data[receive_mask & (cust_data['User Identifier'] == cust_id)]
            withdrawal_transactions = cust_data[withdrawal_mask & (cust_data['User Identifier'] == cust_id)]
            
            if len(receive_transactions) == 0:
                continue
            
            # Calculate amounts
            if 'Transaction Amount' in df.columns:
                total_received = receive_transactions['Transaction Amount'].sum()
                total_withdrawn = withdrawal_transactions['Transaction Amount'].sum()
            else:
                total_received = len(receive_transactions)
                total_withdrawn = len(withdrawal_transactions)
            
            # Calculate withdrawal percentage
            withdrawal_percentage = (total_withdrawn / total_received * 100) if total_received > 0 else 0
            
            # Determine segment
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
            
            segment_analysis.append({
                'User_ID': cust_id,
                'Full_Name': full_name,
                'Total_International_Received': total_received,
                'Total_Withdrawn': total_withdrawn,
                'Withdrawal_Percentage': withdrawal_percentage,
                'Segment': segment
            })
            
            detailed_records.append({
                'User_ID': cust_id,
                'Full_Name': full_name,
                'Segment': segment,
                'Withdrawal_Percentage': f"{withdrawal_percentage:.2f}%",
                'Withdrawal_vs_International': f"{withdrawal_percentage:.2f}% of International Received",
                'Total_International_Received': total_received,
                'Total_Withdrawn': total_withdrawn,
                'International_Receive_Count': len(receive_transactions),
                'Withdrawal_Count': len(withdrawal_transactions),
                'First_International_Date': receive_transactions['Created At'].min().strftime('%Y-%m-%d') if len(receive_transactions) > 0 else 'None',
                'Last_International_Date': receive_transactions['Created At'].max().strftime('%Y-%m-%d') if len(receive_transactions) > 0 else 'None'
            })
        
        # Update progress
        progress = min(40 + (i / len(receive_intl_customers) * 60), 100)
        progress_bar.progress(int(progress))
    
    # Clear progress
    progress_bar.progress(100)
    status_text.empty()
    progress_bar.empty()
    
    # Process results
    results = {
        'withdrawal_segments': {},
        'detailed_analysis': pd.DataFrame(),
        'summary_stats': {},
        'segment_data': pd.DataFrame()
    }
    
    if segment_analysis:
        segment_df = pd.DataFrame(segment_analysis)
        detailed_df = pd.DataFrame(detailed_records)
        
        # Calculate segment distribution
        segment_distribution = segment_df['Segment'].value_counts()
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
            'total_international_received': segment_df['Total_International_Received'].sum(),
            'total_withdrawn': segment_df['Total_Withdrawn'].sum(),
            'overall_withdrawal_percentage': (segment_df['Total_Withdrawn'].sum() / segment_df['Total_International_Received'].sum() * 100) if segment_df['Total_International_Received'].sum() > 0 else 0,
            'segment_25_count': len(segment_df[segment_df['Segment'] == '≤25%']),
            'segment_50_count': len(segment_df[segment_df['Segment'] == '25%-50%']),
            'segment_75_count': len(segment_df[segment_df['Segment'] == '50%-75%']),
            'segment_100_count': len(segment_df[segment_df['Segment'] == '75%-100%'])
        }
    
    return results

@st.cache_data(ttl=3600, show_spinner=False)
def analyze_specific_intl_groups_cached(df, max_customers=5000):
    """Cached version with customer limit"""
    return analyze_specific_intl_groups(df, max_customers)

def analyze_specific_intl_groups(df, max_customers=5000):
    """Analyze specific international groups - OPTIMIZED"""
    
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Analyzing specific international groups...")
    
    # Compile regex patterns
    receive_intl_regex = re.compile('|'.join([
        'Receive.*International',
        'International.*Receive', 
        'International.*Credit',
        'Remittance.*Receive',
        'Receive.*Remittance'
    ]), re.IGNORECASE)
    
    general_intl_regex = re.compile('|'.join(['International Remittance', 'International Transfer', 
                                            'Remittance', 'International']), re.IGNORECASE)
    
    withdrawal_regex = re.compile('|'.join(['Withdrawal', 'Scan To Withdraw', 'Cash Out', 
                                          'Withdraw']), re.IGNORECASE)
    
    other_regex = re.compile('|'.join(['P2P', 'Deposit', 'Bill', 'Airtime', 'Utility', 
                                     'Topup', 'Ticket', 'Scan', 'Send', 'Payment']), re.IGNORECASE)
    
    progress_bar.progress(20)
    
    # Find receive international transactions
    receive_mask = df['Product Name'].apply(lambda x: bool(receive_intl_regex.search(str(x))))
    
    # If no receive-specific transactions found, use general international
    if receive_mask.sum() == 0:
        receive_mask = df['Product Name'].apply(lambda x: bool(general_intl_regex.search(str(x))))
        if receive_mask.sum() == 0:
            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()
            return {
                'received_withdrew': pd.DataFrame(),
                'received_no_withdraw_other': pd.DataFrame(),
                'received_only_withdraw': pd.DataFrame()
            }
    
    # Get unique customers - limit for performance
    receive_intl_customers = df.loc[receive_mask, 'User Identifier'].dropna().unique()
    
    if len(receive_intl_customers) > max_customers:
        st.info(f"⚠️ Analyzing {max_customers:,} out of {len(receive_intl_customers):,} international recipients for performance.")
        receive_intl_customers = np.random.choice(receive_intl_customers, size=max_customers, replace=False)
    
    progress_bar.progress(40)
    status_text.text(f"Analyzing {len(receive_intl_customers):,} customers...")
    
    # Pre-compute masks
    withdrawal_mask = df['Product Name'].apply(lambda x: bool(withdrawal_regex.search(str(x))))
    
    # Analyze in batches
    batch_size = 500
    groups = {
        'received_withdrew': [],
        'received_no_withdraw_other': [],
        'received_only_withdraw': []
    }
    
    for i in range(0, len(receive_intl_customers), batch_size):
        batch_customers = receive_intl_customers[i:i+batch_size]
        batch_df = df[df['User Identifier'].isin(batch_customers)]
        
        for cust_id in batch_customers:
            cust_data = batch_df[batch_df['User Identifier'] == cust_id]
            if cust_data.empty:
                continue
            
            # Get customer name
            name_records = cust_data[cust_data['Full Name'].str.len() > 0]
            full_name = name_records.iloc[0]['Full Name'] if not name_records.empty else 'Name not available'
            
            # Check transaction types efficiently
            has_receive_intl = receive_mask.any() and (cust_id in receive_intl_customers)
            has_withdrawal = withdrawal_mask.any() and cust_data[withdrawal_mask].shape[0] > 0
            
            # Check for other services
            other_services_mask = other_regex.search(cust_data['Product Name'].iloc[0]) if not cust_data.empty else False
            has_other_services = bool(other_services_mask)
            
            # Calculate metrics
            receive_count = cust_data[receive_mask].shape[0]
            withdrawal_count = cust_data[withdrawal_mask].shape[0]
            other_count = cust_data[other_services_mask].shape[0] if has_other_services else 0
            
            # Calculate amounts
            if 'Transaction Amount' in df.columns:
                total_received = cust_data[receive_mask]['Transaction Amount'].sum() if receive_count > 0 else 0
                total_withdrawn = cust_data[withdrawal_mask]['Transaction Amount'].sum() if withdrawal_count > 0 else 0
            else:
                total_received = receive_count
                total_withdrawn = withdrawal_count
            
            # Calculate withdrawal percentage
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
                'Withdrawal_Percentage': withdrawal_percentage,
                'First_Transaction_Date': cust_data['Created At'].min().strftime('%Y-%m-%d') if not cust_data.empty else 'Unknown',
                'Last_Transaction_Date': cust_data['Created At'].max().strftime('%Y-%m-%d') if not cust_data.empty else 'Unknown'
            }
            
            # Categorize customer
            if has_receive_intl and has_withdrawal:
                groups['received_withdrew'].append(customer_record)
            
            if has_receive_intl and not has_withdrawal and has_other_services:
                groups['received_no_withdraw_other'].append(customer_record)
            
            if has_receive_intl and has_withdrawal and not has_other_services:
                groups['received_only_withdraw'].append(customer_record)
        
        # Update progress
        progress = min(40 + (i / len(receive_intl_customers) * 60), 100)
        progress_bar.progress(int(progress))
    
    # Clear progress
    progress_bar.progress(100)
    status_text.empty()
    progress_bar.empty()
    
    # Convert to DataFrames
    results = {}
    for group_name, records in groups.items():
        if records:
            df_group = pd.DataFrame(records)
            
            if not df_group.empty:
                # Add calculated columns
                df_group['Withdrawal_Percentage_Formatted'] = df_group['Withdrawal_Percentage'].apply(lambda x: f"{x:.2f}%")
                df_group['Withdrawal_vs_International'] = df_group['Withdrawal_Percentage'].apply(lambda x: f"{x:.2f}% of International Received")
            
            results[group_name] = df_group
        else:
            results[group_name] = pd.DataFrame()
    
    return results

@st.cache_data(ttl=3600, show_spinner=False)
def analyze_telemarketing_data_cached(filtered_df, sample_size=None):
    """Cached version of main analysis"""
    return analyze_telemarketing_data(filtered_df, sample_size)

def analyze_telemarketing_data(filtered_df, sample_size=None):
    """Main analysis function with optimizations"""
    
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Starting analysis...")
    
    # Sample data if requested
    if sample_size and len(filtered_df) > sample_size:
        original_size = len(filtered_df)
        filtered_df = filtered_df.sample(n=min(sample_size, len(filtered_df)), random_state=42)
        st.info(f"⚠️ Using sample of {len(filtered_df):,} rows out of {original_size:,} for faster analysis")
    
    progress_bar.progress(10)
    status_text.text("Analyzing international customers...")
    
    # Define product categories with regex
    p2p_regex = re.compile('|'.join(['Internal Wallet Transfer (P2P)', 'Internal Wallet Transfer', 
                                    'P2P Transfer', 'Wallet Transfer']), re.IGNORECASE)
    
    intl_regex = re.compile('|'.join(['International Remittance', 'International Transfer', 
                                     'Remittance', 'International']), re.IGNORECASE)
    
    # Get unique customers
    unique_customers = filtered_df['User Identifier'].dropna().unique()
    
    # REPORT 1: International remittance customers not using P2P
    intl_mask = filtered_df['Product Name'].apply(lambda x: bool(intl_regex.search(str(x))))
    p2p_mask = filtered_df['Product Name'].apply(lambda x: bool(p2p_regex.search(str(x))))
    
    intl_customers = filtered_df[intl_mask]['User Identifier'].dropna().unique()
    p2p_customers = filtered_df[p2p_mask]['User Identifier'].dropna().unique()
    
    intl_not_p2p = [cust for cust in intl_customers if cust not in p2p_customers]
    
    progress_bar.progress(30)
    status_text.text(f"Found {len(intl_not_p2p):,} international targets...")
    
    # Create detailed DataFrame for international customers not using P2P - limit to 1000 for performance
    intl_details = []
    max_details = min(1000, len(intl_not_p2p))
    
    for idx, cust_id in enumerate(intl_not_p2p[:max_details]):
        cust_records = filtered_df[filtered_df['User Identifier'] == cust_id]
        if not cust_records.empty:
            # Get customer name efficiently
            name_record = cust_records[cust_records['Full Name'].str.len() > 0]
            full_name = name_record.iloc[0]['Full Name'] if not name_record.empty else 'Name not available'
            
            # Calculate metrics
            intl_count = cust_records[intl_mask].shape[0]
            p2p_count = cust_records[p2p_mask].shape[0]
            
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
                'Total_Transactions': len(cust_records),
                'Last_Transaction_Date': cust_records['Created At'].max().strftime('%Y-%m-%d') if pd.notna(cust_records['Created At'].max()) else 'Unknown',
                'Customer_Since': cust_records['Created At'].min().strftime('%Y-%m-%d') if pd.notna(cust_records['Created At'].min()) else 'Unknown'
            })
        
        # Update progress
        if idx % 100 == 0:
            progress = 30 + (idx / max_details * 20)
            progress_bar.progress(int(progress))
    
    report1_df = pd.DataFrame(intl_details) if intl_details else pd.DataFrame(
        columns=['User_ID', 'Full_Name', 'International_Transaction_Count', 'International_Transaction_Amount',
                'P2P_Transaction_Count', 'Total_Transactions', 'Last_Transaction_Date', 'Customer_Since']
    )
    
    progress_bar.progress(60)
    status_text.text("Analyzing domestic customers...")
    
    # Skip domestic analysis if we're sampling (for speed)
    report2_df = pd.DataFrame()
    if sample_size is None or sample_size > 10000:
        # REPORT 2: Non-international customers using other services but not P2P
        other_regex = re.compile('|'.join(['Deposit', 'Scan To Withdraw', 'Cash In', 'Cash Out', 
                                         'Bill Payment', 'Airtime', 'Utility', 'Topup', 'Ticket']), re.IGNORECASE)
        other_mask = filtered_df['Product Name'].apply(lambda x: bool(other_regex.search(str(x))))
        
        other_customers = filtered_df[other_mask]['User Identifier'].dropna().unique()
        
        domestic_other_not_p2p = [
            cust for cust in other_customers 
            if (cust not in intl_customers) and (cust not in p2p_customers)
        ]
        
        domestic_details = []
        max_domestic = min(1000, len(domestic_other_not_p2p))
        
        for idx, cust_id in enumerate(domestic_other_not_p2p[:max_domestic]):
            cust_records = filtered_df[filtered_df['User Identifier'] == cust_id]
            if not cust_records.empty:
                name_record = cust_records[cust_records['Full Name'].str.len() > 0]
                full_name = name_record.iloc[0]['Full Name'] if not name_record.empty else 'Name not available'
                
                other_services_list = list(cust_records[other_mask]['Product Name'].unique())[:3]
                
                domestic_details.append({
                    'User_ID': int(cust_id) if pd.notna(cust_id) else 0,
                    'Full_Name': full_name,
                    'Total_Transactions': len(cust_records),
                    'Other_Services_Used': ', '.join(other_services_list),
                    'Last_Transaction_Date': cust_records['Created At'].max().strftime('%Y-%m-%d') if pd.notna(cust_records['Created At'].max()) else 'Unknown',
                    'Customer_Since': cust_records['Created At'].min().strftime('%Y-%m-%d') if pd.notna(cust_records['Created At'].min()) else 'Unknown'
                })
            
            if idx % 100 == 0:
                progress = 60 + (idx / max_domestic * 10)
                progress_bar.progress(int(progress))
        
        report2_df = pd.DataFrame(domestic_details) if domestic_details else pd.DataFrame(
            columns=['User_ID', 'Full_Name', 'Total_Transactions', 'Other_Services_Used',
                    'Last_Transaction_Date', 'Customer_Since']
        )
    
    progress_bar.progress(80)
    status_text.text("Creating impact template...")
    
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
    
    progress_bar.progress(90)
    status_text.text("Analyzing international withdrawal segments...")
    
    # New reports for international remittance recipients
    intl_withdrawal_segments = analyze_intl_withdrawal_segments_cached(filtered_df, max_customers=5000)
    specific_intl_groups = analyze_specific_intl_groups_cached(filtered_df, max_customers=5000)
    
    progress_bar.progress(95)
    status_text.text("Finalizing results...")
    
    # Summary statistics
    summary_stats = {
        'total_transactions': len(filtered_df),
        'unique_customers': len(unique_customers),
        'intl_customers': len(intl_customers),
        'intl_not_p2p': len(intl_not_p2p),
        'p2p_customers': len(p2p_customers),
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
            'avg_withdrawal_percentage': intl_withdrawal_segments['summary_stats'].get('avg_withdrawal_percentage', 0)
        })
    
    # Add specific intl group counts
    summary_stats.update({
        'received_withdrew_count': len(specific_intl_groups.get('received_withdrew', [])),
        'received_no_withdraw_other_count': len(specific_intl_groups.get('received_no_withdraw_other', [])),
        'received_only_withdraw_count': len(specific_intl_groups.get('received_only_withdraw', []))
    })
    
    progress_bar.progress(100)
    status_text.empty()
    progress_bar.empty()
    
    return {
        'intl_not_p2p': report1_df,
        'domestic_other_not_p2p': report2_df,
        'impact_template': impact_df,
        'intl_withdrawal_segments': intl_withdrawal_segments,
        'specific_intl_groups': specific_intl_groups,
        'summary_stats': summary_stats
    }

def create_comprehensive_excel_report(results, filter_desc=""):
    """Create comprehensive Excel report with all sheets - OPTIMIZED"""
    output = io.BytesIO()
    
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sheet 1: Executive Summary (lightweight)
            summary_data = [
                ['TELEMARKETING CAMPAIGN ANALYSIS REPORT'],
                ['Generated on', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                ['Filter Applied', filter_desc[:200]],
                ['Report Period', f"{results['summary_stats']['start_date']} to {results['summary_stats']['end_date']}"],
                [''],
                ['KEY METRICS'],
                ['Total Transactions', results['summary_stats']['total_transactions']],
                ['Unique Customers', results['summary_stats']['unique_customers']],
                ['International Customers Not Using P2P', results['summary_stats']['intl_not_p2p']],
                ['International Recipients Analyzed', results['summary_stats'].get('total_intl_recipients', 0)],
                ['Average Withdrawal %', f"{results['summary_stats'].get('avg_withdrawal_percentage', 0):.2f}% of International Received"],
            ]
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Executive_Summary', index=False, header=False)
            
            # Only write data sheets if they have data
            sheets_written = 1
            
            # Sheet 2: International Targets
            if not results['intl_not_p2p'].empty:
                results['intl_not_p2p'].head(10000).to_excel(writer, sheet_name='International_Targets', index=False)
                sheets_written += 1
            
            # Sheet 3: Domestic Targets
            if not results['domestic_other_not_p2p'].empty:
                results['domestic_other_not_p2p'].head(10000).to_excel(writer, sheet_name='Domestic_Targets', index=False)
                sheets_written += 1
            
            # Sheet 4: International Withdrawal Segments
            if 'detailed_analysis' in results['intl_withdrawal_segments'] and not results['intl_withdrawal_segments']['detailed_analysis'].empty:
                results['intl_withdrawal_segments']['detailed_analysis'].head(10000).to_excel(
                    writer, sheet_name='Intl_Withdrawal_Segments', index=False)
                sheets_written += 1
            
            # Sheet 5-7: Specific Groups
            for group_name, sheet_name in [
                ('received_withdrew', 'Intl_Received_Withdrew'),
                ('received_no_withdraw_other', 'Intl_No_Withdraw_Other'),
                ('received_only_withdraw', 'Intl_Only_Withdrew')
            ]:
                if group_name in results['specific_intl_groups'] and not results['specific_intl_groups'][group_name].empty:
                    results['specific_intl_groups'][group_name].head(10000).to_excel(
                        writer, sheet_name=sheet_name, index=False)
                    sheets_written += 1
            
            # Sheet 8: Impact Template
            results['impact_template'].to_excel(writer, sheet_name='Impact_Template', index=False)
            sheets_written += 1
            
            st.success(f"✅ Excel report created with {sheets_written} sheets")
        
        output.seek(0)
        return output
    
    except Exception as e:
        st.error(f"Error creating Excel file: {str(e)}")
        return None

def main():
    """Main Streamlit app with performance optimizations"""
    st.markdown('<h1 class="main-header">📊 Telemarketing Campaign Analyzer</h1>', unsafe_allow_html=True)
    
    # Performance settings in sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Performance settings
        st.subheader("🚀 Performance Settings")
        
        # Option to use sample data for faster processing
        use_sample = st.checkbox("Use Sample Data (Faster)", value=False, 
                                help="Use a sample of data for faster processing during testing")
        
        sample_size = None
        if use_sample:
            sample_size = st.number_input("Sample Size", min_value=1000, max_value=100000, 
                                         value=10000, step=1000,
                                         help="Number of rows to use for analysis")
        
        # Cache indicator
        st.markdown('<div class="cache-info">🔁 Using cached results where possible</div>', 
                   unsafe_allow_html=True)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Transaction Data",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your transaction file (CSV or Excel)"
        )
        
        if uploaded_file is not None:
            # Load data with progress indication
            with st.spinner("🔄 Loading and optimizing data..."):
                df = load_data_optimized(uploaded_file, sample_size=sample_size)
            
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
                
                # Quick filter descriptions
                if selected_pair_filter != "all_customers":
                    with st.expander("🔍 Filter Description"):
                        desc_map = {
                            "intl_not_p2p": "Customers with international remittance but no P2P",
                            "p2p_not_deposit": "Customers with P2P but no deposits",
                            "p2p_and_withdrawal": "Customers using both P2P and withdrawal",
                            "intl_and_p2p": "Customers using both international and P2P",
                            "intl_received_withdraw": "Received international AND withdrew",
                            "intl_received_no_withdraw": "Received international, no withdrawal",
                            "intl_received_only_withdraw": "Received international, only withdrew"
                        }
                        st.info(desc_map.get(selected_pair_filter, ""))
                
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
                
                # Quick analysis button
                st.markdown("---")
                analyze_button = st.button(
                    "🚀 Start Analysis",
                    type="primary",
                    use_container_width=True,
                    help="Click to start the analysis (results are cached)"
                )
                
                # Data preview with performance warning
                with st.expander("📋 Quick Data Preview"):
                    preview_rows = st.slider("Preview rows", 10, 100, 20)
                    st.dataframe(df.head(preview_rows), use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Rows", f"{len(df):,}")
                    with col2:
                        st.metric("Date Range", f"{min_date} to {max_date}")
                        
                    if len(df) > 100000:
                        st.warning("⚠️ Large dataset detected. Analysis may take longer.")
        else:
            st.info("👈 Please upload a transaction file to begin analysis")
            df = None
            analyze_button = False
            start_date = end_date = None
            selected_pair_filter = "all_customers"
    
    # Main content area
    if uploaded_file is not None and df is not None and analyze_button:
        # Clear previous cache if filter changed
        cache_key = f"{selected_pair_filter}_{start_date}_{end_date}_{sample_size}"
        
        # Filter data with caching
        with st.spinner("🔄 Applying filters..."):
            filtered_df, filter_desc, customer_count = filter_by_customer_pair_cached(
                df, selected_pair_filter, start_date, end_date
            )
        
        if len(filtered_df) == 0:
            st.warning("⚠️ No data found for the selected filters. Please adjust your criteria.")
        else:
            # Display filter info
            st.markdown(f'<div class="filter-pair-info">{filter_desc}</div>', unsafe_allow_html=True)
            
            # Quick stats
            st.markdown('<h2 class="sub-header">📈 Quick Stats</h2>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Transactions", f"{len(filtered_df):,}")
            with col2:
                st.metric("Unique Customers", f"{customer_count:,}")
            with col3:
                st.metric("Date Range", f"{start_date} to {end_date}")
            
            # Run analysis with caching
            st.markdown('<h2 class="sub-header">🎯 Analysis Results</h2>', unsafe_allow_html=True)
            
            # Use tabs for better organization
            tab1, tab2, tab3 = st.tabs(["📊 Summary", "📁 Detailed Reports", "📥 Export"])
            
            with tab1:
                # Run analysis with progress
                results = analyze_telemarketing_data_cached(filtered_df, sample_size=sample_size)
                
                # Display summary metrics
                st.subheader("Campaign Summary")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Intl Targets", f"{results['summary_stats']['intl_not_p2p']:,}")
                with col2:
                    st.metric("Intl Recipients", f"{results['summary_stats'].get('total_intl_recipients', 0):,}")
                with col3:
                    avg_withdrawal = results['summary_stats'].get('avg_withdrawal_percentage', 0)
                    st.metric("Avg Withdrawal %", f"{avg_withdrawal:.1f}%")
                
                # Withdrawal segments
                if 'intl_withdrawal_segments' in results and results['intl_withdrawal_segments']['withdrawal_segments']:
                    st.subheader("Withdrawal Segments")
                    seg_data = results['intl_withdrawal_segments']['withdrawal_segments']
                    
                    cols = st.columns(4)
                    segments = ['≤25%', '25%-50%', '50%-75%', '75%-100%']
                    colors = ['#10B981', '#F59E0B', '#F59E0B', '#EF4444']
                    
                    for idx, (col, segment, color) in enumerate(zip(cols, segments, colors)):
                        with col:
                            count = seg_data['distribution'].get(segment, 0)
                            percentage = seg_data['percentages'].get(segment, 0)
                            st.markdown(f"""
                            <div style="text-align: center; padding: 1rem; border-radius: 0.5rem; 
                                        background-color: {color}20; border-left: 4px solid {color};">
                                <h3 style="margin: 0; color: {color};">{segment}</h3>
                                <p style="font-size: 1.5rem; margin: 0.5rem 0; font-weight: bold;">{count:,}</p>
                                <p style="margin: 0;">{percentage:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Quick recommendations
                st.subheader("💡 Quick Insights")
                
                insights = []
                if results['summary_stats']['intl_not_p2p'] > 0:
                    insights.append(f"**{results['summary_stats']['intl_not_p2p']:,}** international customers not using P2P - high potential")
                
                if results['summary_stats'].get('received_only_withdraw_count', 0) > 0:
                    insights.append(f"**{results['summary_stats']['received_only_withdraw_count']:,}** customers only withdraw received funds - target for cross-selling")
                
                if results['summary_stats'].get('intl_withdrawal_segment_100', 0) > 0:
                    insights.append(f"**{results['summary_stats']['intl_withdrawal_segment_100']:,}** customers withdraw 75-100% of funds - monitor for churn risk")
                
                for insight in insights:
                    st.write(f"• {insight}")
            
            with tab2:
                st.subheader("Detailed Reports")
                
                # Create subtabs for different reports
                report_tabs = st.tabs([
                    "🎯 International Targets",
                    "📈 Withdrawal Segments",
                    "💸 Specific Groups"
                ])
                
                with report_tabs[0]:
                    if not results['intl_not_p2p'].empty:
                        st.dataframe(results['intl_not_p2p'], use_container_width=True)
                        st.info(f"Showing {len(results['intl_not_p2p'])} international customers not using P2P")
                    else:
                        st.info("No international targets found")
                
                with report_tabs[1]:
                    if 'detailed_analysis' in results['intl_withdrawal_segments'] and not results['intl_withdrawal_segments']['detailed_analysis'].empty:
                        st.dataframe(results['intl_withdrawal_segments']['detailed_analysis'], 
                                   use_container_width=True, height=400)
                        st.info(f"Showing withdrawal segments for {len(results['intl_withdrawal_segments']['detailed_analysis'])} customers")
                    else:
                        st.info("No withdrawal segment data available")
                
                with report_tabs[2]:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'received_withdrew' in results['specific_intl_groups'] and not results['specific_intl_groups']['received_withdrew'].empty:
                            st.write("**Received & Withdrew**")
                            st.dataframe(results['specific_intl_groups']['received_withdrew'], 
                                       use_container_width=True, height=300)
                    
                    with col2:
                        if 'received_no_withdraw_other' in results['specific_intl_groups'] and not results['specific_intl_groups']['received_no_withdraw_other'].empty:
                            st.write("**Received, No Withdrawal**")
                            st.dataframe(results['specific_intl_groups']['received_no_withdraw_other'], 
                                       use_container_width=True, height=300)
                    
                    with col3:
                        if 'received_only_withdraw' in results['specific_intl_groups'] and not results['specific_intl_groups']['received_only_withdraw'].empty:
                            st.write("**Received & Only Withdrew**")
                            st.dataframe(results['specific_intl_groups']['received_only_withdraw'], 
                                       use_container_width=True, height=300)
            
            with tab3:
                st.subheader("Export Results")
                
                # Excel export
                st.info("Download all analysis results in a single Excel file:")
                
                excel_data = create_comprehensive_excel_report(results, filter_desc)
                if excel_data:
                    st.download_button(
                        label="📊 Download Excel Report",
                        data=excel_data,
                        file_name=f"telemarketing_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                
                # Individual CSV exports
                st.subheader("Individual CSV Exports")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if not results['intl_not_p2p'].empty:
                        csv_data = results['intl_not_p2p'].to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="🎯 International Targets",
                            data=csv_data,
                            file_name="international_targets.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                
                with col2:
                    if 'detailed_analysis' in results['intl_withdrawal_segments'] and not results['intl_withdrawal_segments']['detailed_analysis'].empty:
                        csv_data = results['intl_withdrawal_segments']['detailed_analysis'].to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📈 Withdrawal Segments",
                            data=csv_data,
                            file_name="withdrawal_segments.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                
                with col3:
                    csv_data = results['impact_template'].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📋 Impact Template",
                        data=csv_data,
                        file_name="impact_template.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                # Call-to-action
                st.markdown("---")
                st.markdown("### 🚀 Ready to Start Your Campaign?")
                
                col1, col2 = st.columns(2)
                with col1:
                    total_targets = results['summary_stats']['intl_not_p2p']
                    st.metric("Total Targets", f"{total_targets:,}")
                with col2:
                    if total_targets > 0:
                        daily_target = total_targets // 30  # Assume 30-day campaign
                        st.metric("Daily Target", f"{daily_target:,}", "over 30 days")

# Import regex module at the top
import re

if __name__ == "__main__":
    main()
