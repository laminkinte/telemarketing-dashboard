"""
Core analysis functions for telemarketing dashboard
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def clean_transaction_data(df):
    """Clean and prepare transaction data"""
    # Create a copy
    df = df.copy()
    
    # Clean column names
    df.columns = [col.strip() for col in df.columns]
    
    # Convert data types
    if 'User Identifier' in df.columns:
        df['User Identifier'] = pd.to_numeric(df['User Identifier'], errors='coerce')
    
    # Clean text columns
    text_columns = ['Product Name', 'Service Name', 'Entity Name', 'Full Name']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    # Parse dates
    if 'Created At' in df.columns:
        # Try multiple date formats
        for fmt in ['%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M', '%m/%d/%Y %H:%M', '%Y/%m/%d %H:%M:%S']:
            try:
                df['Created At'] = pd.to_datetime(df['Created At'], format=fmt, errors='coerce')
                if df['Created At'].notna().any():
                    break
            except:
                continue
        # Last resort
        if df['Created At'].isna().all():
            df['Created At'] = pd.to_datetime(df['Created At'], errors='coerce')
    
    return df

def analyze_telemarketing_data(df, analysis_period="Last 7 days"):
    """
    Analyze transaction data to generate telemarketing reports
    """
    try:
        # Clean the data
        df = clean_transaction_data(df)
        
        # Check required columns
        required_columns = ['User Identifier', 'Product Name', 'Service Name', 
                           'Created At', 'Entity Name', 'Full Name']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return {
                'error': f"Missing required columns: {', '.join(missing_columns)}",
                'intl_not_p2p': pd.DataFrame(),
                'domestic_other_not_p2p': pd.DataFrame(),
                'impact_template': pd.DataFrame(),
                'summary': {}
            }
        
        # Filter by date range
        if df['Created At'].notna().any():
            end_date = df['Created At'].max()
            
            if analysis_period == "Last 7 days":
                start_date = end_date - timedelta(days=7)
            elif analysis_period == "Last 30 days":
                start_date = end_date - timedelta(days=30)
            elif analysis_period == "Last 90 days":
                start_date = end_date - timedelta(days=90)
            else:  # All available data
                start_date = df['Created At'].min()
            
            filtered_df = df[(df['Created At'] >= start_date) & (df['Created At'] <= end_date)].copy()
        else:
            filtered_df = df.copy()
            start_date = None
            end_date = None
        
        # Define product categories
        p2p_products = ['Internal Wallet Transfer (P2P)', 'Internal Wallet Transfer']
        other_services = ['Deposit', 'Scan To Withdraw Agent', 'Scan To Withdraw Customer', 
                         'Scan To Send', 'Ticket']
        bill_payment_services = ['Bill Payment', 'Airtime Topup']
        
        # Filter for customers
        customer_mask = (
            filtered_df['Entity Name'].str.contains('Customer', case=False, na=False) |
            filtered_df['Entity Name'].isna() |
            (filtered_df['Entity Name'] == '')
        )
        customer_df = filtered_df[customer_mask].copy()
        
        # Get unique customers
        unique_customers = customer_df['User Identifier'].dropna().unique()
        
        # 1. International remittance customers not using P2P
        intl_mask = customer_df['Product Name'].str.contains('International Remittance', case=False, na=False)
        intl_customers = customer_df[intl_mask]['User Identifier'].dropna().unique()
        
        p2p_mask = customer_df['Product Name'].isin(p2p_products)
        p2p_customers = customer_df[p2p_mask]['User Identifier'].dropna().unique()
        
        intl_not_p2p = [cust for cust in intl_customers if cust not in p2p_customers]
        
        # Build report 1
        report1_data = []
        for cust_id in intl_not_p2p:
            cust_data = customer_df[customer_df['User Identifier'] == cust_id]
            if not cust_data.empty:
                # Get customer info
                name_records = cust_data[cust_data['Full Name'].notna() & (cust_data['Full Name'] != 'nan')]
                full_name = name_records['Full Name'].iloc[0] if not name_records.empty else 'Unknown'
                
                # Get dates
                last_date = cust_data['Created At'].max()
                last_date_str = last_date.strftime('%Y-%m-%d') if pd.notna(last_date) else 'Unknown'
                
                # Count transactions
                total_tx = len(cust_data)
                intl_tx = len(cust_data[cust_data['Product Name'].str.contains('International Remittance', case=False, na=False)])
                
                report1_data.append({
                    'User Identifier': cust_id,
                    'Full Name': full_name,
                    'Last Transaction Date': last_date_str,
                    'Total Transactions': total_tx,
                    'International Remittance Count': intl_tx
                })
        
        report1_df = pd.DataFrame(report1_data) if report1_data else pd.DataFrame(
            columns=['User Identifier', 'Full Name', 'Last Transaction Date', 
                    'Total Transactions', 'International Remittance Count']
        )
        
        # 2. Domestic customers using other services but not P2P
        other_mask = (
            customer_df['Product Name'].isin(other_services) |
            customer_df['Service Name'].isin(bill_payment_services)
        )
        other_customers = customer_df[other_mask]['User Identifier'].dropna().unique()
        
        domestic_not_p2p = [
            cust for cust in other_customers 
            if (cust not in intl_customers) and (cust not in p2p_customers)
        ]
        
        # Build report 2
        report2_data = []
        for cust_id in domestic_not_p2p:
            cust_data = customer_df[customer_df['User Identifier'] == cust_id]
            if not cust_data.empty:
                # Get customer info
                name_records = cust_data[cust_data['Full Name'].notna() & (cust_data['Full Name'] != 'nan')]
                full_name = name_records['Full Name'].iloc[0] if not name_records.empty else 'Unknown'
                
                # Get dates
                last_date = cust_data['Created At'].max()
                last_date_str = last_date.strftime('%Y-%m-%d') if pd.notna(last_date) else 'Unknown'
                
                # Get services used
                services = cust_data['Product Name'].unique()
                other_services_list = [s for s in services if str(s) in other_services]
                
                report2_data.append({
                    'User Identifier': cust_id,
                    'Full Name': full_name,
                    'Last Transaction Date': last_date_str,
                    'Total Transactions': len(cust_data),
                    'Other Services Used': ', '.join(other_services_list[:3]),
                    'Services Count': len(other_services_list)
                })
        
        report2_df = pd.DataFrame(report2_data) if report2_data else pd.DataFrame(
            columns=['User Identifier', 'Full Name', 'Last Transaction Date',
                    'Total Transactions', 'Other Services Used', 'Services Count']
        )
        
        # 3. Impact template
        if start_date and end_date:
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
        else:
            dates = pd.date_range(end=datetime.now(), periods=7, freq='D')
        
        impact_data = []
        for date in dates:
            impact_data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Customers Called': 0,
                'Customers Reached': 0,
                'P2P Conversions': 0,
                'Conversion Rate %': 0.0,
                'Notes': ''
            })
        
        impact_df = pd.DataFrame(impact_data)
        
        # 4. Summary
        summary = {
            'total_customers': len(unique_customers),
            'intl_customers': len(intl_customers),
            'intl_not_p2p': len(intl_not_p2p),
            'domestic_not_p2p': len(domestic_not_p2p),
            'p2p_users': len(p2p_customers),
            'total_transactions': len(filtered_df),
            'period_start': start_date.strftime('%Y-%m-%d') if start_date else 'Unknown',
            'period_end': end_date.strftime('%Y-%m-%d') if end_date else 'Unknown',
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M')
        }
        
        return {
            'intl_not_p2p': report1_df,
            'domestic_other_not_p2p': report2_df,
            'impact_template': impact_df,
            'summary': summary
        }
        
    except Exception as e:
        return {
            'error': f"Analysis error: {str(e)}",
            'intl_not_p2p': pd.DataFrame(),
            'domestic_other_not_p2p': pd.DataFrame(),
            'impact_template': pd.DataFrame(),
            'summary': {}
        }
