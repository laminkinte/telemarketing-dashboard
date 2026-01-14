"""
Core analysis functions for telemarketing dashboard
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def analyze_telemarketing_data(df, analysis_period="Last 7 days"):
    """
    Analyze transaction data to generate telemarketing reports
    
    Args:
        df: Pandas DataFrame with transaction data
        analysis_period: Time period for analysis
        
    Returns:
        Dictionary with reports and summary
    """
    
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Ensure required columns exist
    required_columns = ['User Identifier', 'Product Name', 'Service Name', 
                       'Created At', 'Entity Name', 'Full Name']
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in data")
    
    # Clean data
    df['Product Name'] = df['Product Name'].astype(str).str.strip()
    df['Entity Name'] = df['Entity Name'].astype(str).str.strip()
    df['User Identifier'] = pd.to_numeric(df['User Identifier'], errors='coerce')
    
    # Clean date column
    df['Created At'] = pd.to_datetime(df['Created At'], errors='coerce', dayfirst=True)
    
    # Filter data based on analysis period
    if df['Created At'].notna().any():
        end_date = df['Created At'].max()
        
        if analysis_period == "Last 7 days":
            start_date = end_date - timedelta(days=7)
        elif analysis_period == "Last 30 days":
            start_date = end_date - timedelta(days=30)
        elif analysis_period == "Last 90 days":
            start_date = end_date - timedelta(days=90)
        elif analysis_period == "Custom range":
            # Would need custom implementation for date picker
            start_date = end_date - timedelta(days=7)  # Default
        else:  # All available data
            start_date = df['Created At'].min()
        
        filtered_df = df[(df['Created At'] >= start_date) & (df['Created At'] <= end_date)].copy()
    else:
        filtered_df = df.copy()
        start_date = filtered_df['Created At'].min()
        end_date = filtered_df['Created At'].max()
    
    # Clean product names
    filtered_df['Product Name'] = filtered_df['Product Name'].str.strip()
    
    # Define product/service categories
    p2p_products = ['Internal Wallet Transfer (P2P)', 'Internal Wallet Transfer']
    international_remittance = 'International Remittance'
    other_services = [
        'Deposit', 'Scan To Withdraw Agent', 'Scan To Withdraw Customer', 
        'Scan To Send', 'Ticket'
    ]
    bill_payment_services = ['Bill Payment', 'Airtime Topup']
    
    # Filter for customers only
    customer_df = filtered_df[
        (filtered_df['Entity Name'].str.contains('Customer', case=False, na=False)) |
        (filtered_df['Entity Name'].isna()) |
        (filtered_df['Entity Name'] == '')
    ].copy()
    
    # Get unique customers
    unique_customers = customer_df['User Identifier'].dropna().unique()
    
    # REPORT 1: International remittance customers not using P2P
    intl_customers = filtered_df[
        filtered_df['Product Name'].str.contains('International Remittance', case=False, na=False)
    ]['User Identifier'].dropna().unique()
    
    p2p_customers = filtered_df[
        filtered_df['Product Name'].isin(p2p_products)
    ]['User Identifier'].dropna().unique()
    
    intl_not_p2p = [cust for cust in intl_customers if cust not in p2p_customers]
    
    # Create DataFrame for international customers not using P2P
    customer_details = []
    for cust_id in intl_not_p2p:
        cust_records = customer_df[customer_df['User Identifier'] == cust_id]
        
        if not cust_records.empty:
            valid_records = cust_records[cust_records['Full Name'].notna()]
            if not valid_records.empty:
                cust_data = valid_records.iloc[0]
                full_name = cust_data['Full Name']
            else:
                full_name = 'Name not available'
            
            last_date = cust_records['Created At'].max()
            last_date_str = last_date.strftime('%Y-%m-%d') if pd.notna(last_date) else 'Unknown'
            
            customer_details.append({
                'User Identifier': cust_id,
                'Full Name': full_name,
                'Last Transaction Date': last_date_str,
                'Total Transactions': len(cust_records),
                'International Remittance Count': len(cust_records[
                    cust_records['Product Name'].str.contains('International Remittance', case=False, na=False)
                ]),
                'Total Amount': cust_records['Amount'].sum() if 'Amount' in cust_records.columns else 0
            })
    
    if customer_details:
        report1_df = pd.DataFrame(customer_details)
    else:
        report1_df = pd.DataFrame(columns=['User Identifier', 'Full Name', 'Last Transaction Date', 
                                          'Total Transactions', 'International Remittance Count', 'Total Amount'])
    
    # REPORT 2: Non-international customers using other services but not P2P
    other_services_customers = filtered_df[
        (filtered_df['Product Name'].isin(other_services)) |
        (filtered_df['Service Name'].isin(bill_payment_services))
    ]['User Identifier'].dropna().unique()
    
    domestic_other_not_p2p = [
        cust for cust in other_services_customers 
        if (cust not in intl_customers) and (cust not in p2p_customers)
    ]
    
    domestic_details = []
    for cust_id in domestic_other_not_p2p:
        cust_records = customer_df[customer_df['User Identifier'] == cust_id]
        
        if not cust_records.empty:
            valid_records = cust_records[cust_records['Full Name'].notna()]
            if not valid_records.empty:
                cust_data = valid_records.iloc[0]
                full_name = cust_data['Full Name']
            else:
                full_name = 'Name not available'
            
            last_date = cust_records['Created At'].max()
            last_date_str = last_date.strftime('%Y-%m-%d') if pd.notna(last_date) else 'Unknown'
            
            services_used = cust_records['Product Name'].dropna().unique()
            other_services_used = []
            for service in services_used:
                service_str = str(service)
                if service_str in other_services:
                    other_services_used.append(service_str)
                else:
                    service_records = cust_records[cust_records['Product Name'] == service]
                    if not service_records.empty:
                        service_name = service_records.iloc[0]['Service Name']
                        if pd.notna(service_name) and str(service_name) in bill_payment_services:
                            other_services_used.append(service_str)
            
            domestic_details.append({
                'User Identifier': cust_id,
                'Full Name': full_name,
                'Last Transaction Date': last_date_str,
                'Total Transactions': len(cust_records),
                'Other Services Used': ', '.join(other_services_used[:5]),
                'Other Services Count': len(other_services_used),
                'Total Amount': cust_records['Amount'].sum() if 'Amount' in cust_records.columns else 0
            })
    
    if domestic_details:
        report2_df = pd.DataFrame(domestic_details)
    else:
        report2_df = pd.DataFrame(columns=['User Identifier', 'Full Name', 'Last Transaction Date',
                                          'Total Transactions', 'Other Services Used', 
                                          'Other Services Count', 'Total Amount'])
    
    # REPORT 3: Daily Impact Report Template
    if pd.notna(start_date) and pd.notna(end_date):
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
    else:
        dates = pd.date_range(end=datetime.now(), periods=7, freq='D')
    
    impact_template = []
    for date in dates:
        date_str = date.strftime('%Y-%m-%d')
        impact_template.append({
            'Date': date_str,
            'Total Customers Called': 0,
            'Customers Reached': 0,
            'Customers Not Reached': 0,
            'P2P Adoption After Call': 0,
            'Conversion Rate (%)': 0.0,
            'Average Transaction Value': 0.0,
            'Notes': ''
        })
    
    impact_df = pd.DataFrame(impact_template)
    
    # Summary statistics
    summary = {
        'total_customers': len(unique_customers),
        'intl_customers': len(intl_customers),
        'intl_not_p2p': len(intl_not_p2p),
        'domestic_not_p2p': len(domestic_other_not_p2p),
        'p2p_users': len(p2p_customers),
        'other_services_users': len(other_services_customers),
        'report_period_start': start_date.strftime('%Y-%m-%d') if pd.notna(start_date) else 'Unknown',
        'report_period_end': end_date.strftime('%Y-%m-%d') if pd.notna(end_date) else 'Unknown',
        'total_transactions': len(filtered_df),
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return {
        'intl_not_p2p': report1_df,
        'domestic_other_not_p2p': report2_df,
        'impact_template': impact_df,
        'summary': summary
    }