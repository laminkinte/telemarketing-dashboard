"""
Core analysis functions with Excel sheet-wise export
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import warnings
warnings.filterwarnings('ignore')

def clean_and_prepare_data(df):
    """Clean and prepare transaction data"""
    df_clean = df.copy()
    
    # Clean column names
    df_clean.columns = [col.strip() for col in df_clean.columns]
    
    # Convert data types
    if 'User Identifier' in df_clean.columns:
        df_clean['User Identifier'] = pd.to_numeric(df_clean['User Identifier'], errors='coerce')
    
    # Clean text columns
    text_cols = ['Product Name', 'Service Name', 'Entity Name', 'Full Name']
    for col in text_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip()
    
    # Parse dates
    if 'Created At' in df_clean.columns:
        # Try multiple date formats
        for fmt in ['%d/%m/%Y %H:%M', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M', '%Y/%m/%d %H:%M:%S']:
            try:
                df_clean['Created At'] = pd.to_datetime(df_clean['Created At'], format=fmt, errors='coerce')
                if df_clean['Created At'].notna().any():
                    break
            except:
                continue
        # Last resort
        df_clean['Created At'] = pd.to_datetime(df_clean['Created At'], errors='coerce')
    
    return df_clean

def analyze_telemarketing_data(df, analysis_period="Last 7 days"):
    """
    Main analysis function
    Returns dictionary with all reports
    """
    try:
        # Clean data
        df_clean = clean_and_prepare_data(df)
        
        # Check required columns
        required_cols = ['User Identifier', 'Product Name', 'Service Name', 
                        'Created At', 'Entity Name', 'Full Name']
        missing_cols = [col for col in required_cols if col not in df_clean.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
        
        # Filter by date range
        if df_clean['Created At'].notna().any():
            end_date = df_clean['Created At'].max()
            
            if analysis_period == "Last 7 days":
                start_date = end_date - timedelta(days=7)
            elif analysis_period == "Last 30 days":
                start_date = end_date - timedelta(days=30)
            elif analysis_period == "Last 90 days":
                start_date = end_date - timedelta(days=90)
            else:  # All data
                start_date = df_clean['Created At'].min()
            
            date_mask = (df_clean['Created At'] >= start_date) & (df_clean['Created At'] <= end_date)
            df_period = df_clean[date_mask].copy()
        else:
            df_period = df_clean.copy()
            start_date = None
            end_date = None
        
        # Define product categories
        P2P_PRODUCTS = ['Internal Wallet Transfer (P2P)', 'Internal Wallet Transfer']
        INTERNATIONAL_PRODUCTS = ['International Remittance']
        OTHER_SERVICES = ['Deposit', 'Scan To Withdraw Agent', 'Scan To Withdraw Customer', 
                         'Scan To Send', 'Ticket']
        BILL_PAYMENT = ['Bill Payment', 'Airtime Topup']
        
        # Filter for customers only
        customer_mask = (
            df_period['Entity Name'].str.contains('Customer', case=False, na=False) |
            df_period['Entity Name'].isna() |
            (df_period['Entity Name'] == '')
        )
        df_customers = df_period[customer_mask].copy()
        
        # Get unique customers
        unique_customers = df_customers['User Identifier'].dropna().unique()
        
        # REPORT 1: International customers not using P2P
        intl_mask = df_customers['Product Name'].str.contains('International Remittance', case=False, na=False)
        intl_customers = df_customers[intl_mask]['User Identifier'].dropna().unique()
        
        p2p_mask = df_customers['Product Name'].isin(P2P_PRODUCTS)
        p2p_customers = df_customers[p2p_mask]['User Identifier'].dropna().unique()
        
        intl_not_p2p = [cust for cust in intl_customers if cust not in p2p_customers]
        
        # Build detailed report for international customers
        intl_report_data = []
        for cust_id in intl_not_p2p[:10000]:  # Limit to 10,000 for performance
            cust_data = df_customers[df_customers['User Identifier'] == cust_id]
            if not cust_data.empty:
                # Get customer info
                name_data = cust_data[cust_data['Full Name'].notna() & (cust_data['Full Name'] != 'nan')]
                full_name = name_data['Full Name'].iloc[0] if not name_data.empty else 'Unknown'
                
                # Get transaction info
                last_date = cust_data['Created At'].max()
                last_date_str = last_date.strftime('%Y-%m-%d') if pd.notna(last_date) else 'Unknown'
                
                # Count transactions
                total_tx = len(cust_data)
                intl_tx = len(cust_data[cust_data['Product Name'].str.contains('International Remittance', case=False, na=False)])
                
                # Get amounts if available
                avg_amount = cust_data['Amount'].mean() if 'Amount' in cust_data.columns else 0
                total_amount = cust_data['Amount'].sum() if 'Amount' in cust_data.columns else 0
                
                intl_report_data.append({
                    'Customer_ID': cust_id,
                    'Full_Name': full_name,
                    'Last_Transaction_Date': last_date_str,
                    'Total_Transactions': total_tx,
                    'International_Transactions': intl_tx,
                    'Avg_Transaction_Amount': avg_amount,
                    'Total_Transaction_Amount': total_amount,
                    'Days_Since_Last_Tx': (datetime.now() - last_date).days if pd.notna(last_date) else 'Unknown'
                })
        
        intl_report_df = pd.DataFrame(intl_report_data) if intl_report_data else pd.DataFrame(
            columns=['Customer_ID', 'Full_Name', 'Last_Transaction_Date', 'Total_Transactions',
                    'International_Transactions', 'Avg_Transaction_Amount', 
                    'Total_Transaction_Amount', 'Days_Since_Last_Tx']
        )
        
        # REPORT 2: Domestic customers using other services but not P2P
        other_mask = (
            df_customers['Product Name'].isin(OTHER_SERVICES) |
            df_customers['Service Name'].isin(BILL_PAYMENT)
        )
        other_customers = df_customers[other_mask]['User Identifier'].dropna().unique()
        
        domestic_not_p2p = [
            cust for cust in other_customers 
            if (cust not in intl_customers) and (cust not in p2p_customers)
        ]
        
        # Build detailed report for domestic customers
        domestic_report_data = []
        for cust_id in domestic_not_p2p[:10000]:  # Limit to 10,000
            cust_data = df_customers[df_customers['User Identifier'] == cust_id]
            if not cust_data.empty:
                # Get customer info
                name_data = cust_data[cust_data['Full Name'].notna() & (cust_data['Full Name'] != 'nan')]
                full_name = name_data['Full Name'].iloc[0] if not name_data.empty else 'Unknown'
                
                # Get transaction info
                last_date = cust_data['Created At'].max()
                last_date_str = last_date.strftime('%Y-%m-%d') if pd.notna(last_date) else 'Unknown'
                
                # Get services used
                services = cust_data['Product Name'].unique()
                other_services_used = [str(s) for s in services if str(s) in OTHER_SERVICES]
                
                # Get service counts
                service_counts = {}
                for service in OTHER_SERVICES:
                    count = len(cust_data[cust_data['Product Name'] == service])
                    if count > 0:
                        service_counts[service] = count
                
                domestic_report_data.append({
                    'Customer_ID': cust_id,
                    'Full_Name': full_name,
                    'Last_Transaction_Date': last_date_str,
                    'Total_Transactions': len(cust_data),
                    'Services_Used': ', '.join(other_services_used[:3]),
                    'Services_Count': len(other_services_used),
                    'Service_Breakdown': str(service_counts),
                    'Days_Since_Last_Tx': (datetime.now() - last_date).days if pd.notna(last_date) else 'Unknown'
                })
        
        domestic_report_df = pd.DataFrame(domestic_report_data) if domestic_report_data else pd.DataFrame(
            columns=['Customer_ID', 'Full_Name', 'Last_Transaction_Date', 'Total_Transactions',
                    'Services_Used', 'Services_Count', 'Service_Breakdown', 'Days_Since_Last_Tx']
        )
        
        # REPORT 3: Impact template
        if start_date and end_date:
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
        else:
            dates = pd.date_range(end=datetime.now(), periods=7, freq='D')
        
        impact_data = []
        for date in dates:
            impact_data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Customers_Called': 0,
                'Customers_Reached': 0,
                'Customers_Not_Reached': 0,
                'P2P_Conversions': 0,
                'Conversion_Rate_%': 0.0,
                'Avg_Call_Duration_min': 0,
                'Follow_Ups_Required': 0,
                'Notes': ''
            })
        
        impact_df = pd.DataFrame(impact_data)
        
        # REPORT 4: Detailed analysis
        detailed_analysis = create_detailed_analysis(
            df_customers, intl_customers, p2p_customers, 
            intl_not_p2p, domestic_not_p2p
        )
        
        # Summary statistics
        summary = {
            'report_period': f"{start_date.strftime('%Y-%m-%d') if start_date else 'N/A'} to {end_date.strftime('%Y-%m-%d') if end_date else 'N/A'}",
            'total_transactions': len(df_period),
            'total_customers': len(unique_customers),
            'intl_customers': len(intl_customers),
            'intl_not_p2p': len(intl_not_p2p),
            'domestic_not_p2p': len(domestic_not_p2p),
            'p2p_users': len(p2p_customers),
            'other_service_users': len(other_customers),
            'conversion_potential_%': (len(intl_not_p2p) / max(len(intl_customers), 1)) * 100,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_source': 'Transaction CSV Analysis'
        }
        
        return {
            'intl_not_p2p': intl_report_df,
            'domestic_other_not_p2p': domestic_report_df,
            'impact_template': impact_df,
            'detailed_analysis': detailed_analysis,
            'summary': summary
        }
        
    except Exception as e:
        raise Exception(f"Analysis error: {str(e)}")

def create_detailed_analysis(df_customers, intl_customers, p2p_customers, 
                           intl_not_p2p, domestic_not_p2p):
    """Create detailed analysis DataFrame"""
    
    analysis_data = []
    
    # 1. Customer segments
    segments = {
        'International P2P Users': len(set(intl_customers) & set(p2p_customers)),
        'International Non-P2P Users': len(intl_not_p2p),
        'Domestic Non-P2P Users': len(domestic_not_p2p),
        'P2P Only Users': len(set(p2p_customers) - set(intl_customers)),
        'Other Users': len(df_customers['User Identifier'].unique()) - 
                      len(set(intl_customers) | set(p2p_customers) | set(domestic_not_p2p))
    }
    
    for segment, count in segments.items():
        analysis_data.append({
            'Category': 'Customer Segments',
            'Metric': segment,
            'Value': count,
            'Percentage': (count / max(len(df_customers['User Identifier'].unique()), 1)) * 100
        })
    
    # 2. Transaction patterns
    if 'Amount' in df_customers.columns:
        amounts = df_customers['Amount'].dropna()
        if len(amounts) > 0:
            analysis_data.append({
                'Category': 'Transaction Patterns',
                'Metric': 'Avg Transaction Amount',
                'Value': amounts.mean(),
                'Percentage': None
            })
            analysis_data.append({
                'Category': 'Transaction Patterns',
                'Metric': 'Max Transaction Amount',
                'Value': amounts.max(),
                'Percentage': None
            })
    
    # 3. Time-based metrics
    if 'Created At' in df_customers.columns and df_customers['Created At'].notna().any():
        df_customers['Transaction_Date'] = df_customers['Created At'].dt.date
        daily_tx = df_customers.groupby('Transaction_Date').size()
        
        analysis_data.append({
            'Category': 'Time Patterns',
            'Metric': 'Avg Daily Transactions',
            'Value': daily_tx.mean() if len(daily_tx) > 0 else 0,
            'Percentage': None
        })
    
    return pd.DataFrame(analysis_data)

def generate_excel_report(results):
    """
    Generate Excel report with multiple sheets
    Returns: BytesIO object with Excel file
    """
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: International Customers Not Using P2P
        if 'intl_not_p2p' in results and not results['intl_not_p2p'].empty:
            results['intl_not_p2p'].to_excel(
                writer, 
                sheet_name='International_Not_P2P', 
                index=False
            )
            worksheet = writer.sheets['International_Not_P2P']
            worksheet.column_dimensions['A'].width = 15
            worksheet.column_dimensions['B'].width = 25
            worksheet.column_dimensions['C'].width = 20
        
        # Sheet 2: Domestic Customers Not Using P2P
        if 'domestic_other_not_p2p' in results and not results['domestic_other_not_p2p'].empty:
            results['domestic_other_not_p2p'].to_excel(
                writer, 
                sheet_name='Domestic_Not_P2P', 
                index=False
            )
            worksheet = writer.sheets['Domestic_Not_P2P']
            worksheet.column_dimensions['A'].width = 15
            worksheet.column_dimensions['B'].width = 25
        
        # Sheet 3: Detailed Analysis
        if 'detailed_analysis' in results and not results['detailed_analysis'].empty:
            results['detailed_analysis'].to_excel(
                writer, 
                sheet_name='Detailed_Analysis', 
                index=False
            )
        
        # Sheet 4: Summary
        if 'summary' in results:
            summary_df = pd.DataFrame({
                'Metric': list(results['summary'].keys()),
                'Value': list(results['summary'].values())
            })
            summary_df.to_excel(
                writer, 
                sheet_name='Summary', 
                index=False
            )
            worksheet = writer.sheets['Summary']
            worksheet.column_dimensions['A'].width = 30
            worksheet.column_dimensions['B'].width = 30
        
        # Sheet 5: Impact Template
        if 'impact_template' in results:
            results['impact_template'].to_excel(
                writer, 
                sheet_name='Impact_Template', 
                index=False
            )
        
        # Sheet 6: Recommendations
        recommendations = pd.DataFrame({
            'Priority': ['High', 'High', 'Medium', 'Medium', 'Low'],
            'Customer_Group': [
                'Recent International Remittance Users',
                'High Frequency International Users',
                'Active Domestic Service Users',
                'Historical International Users',
                'Inactive Customers'
            ],
            'Action': [
                'Call within 48 hours, highlight P2P benefits',
                'Schedule follow-up calls, offer incentives',
                'Educate on P2P features during next interaction',
                'Re-engagement campaign via email/SMS first',
                'Consider win-back campaigns later'
            ],
            'Expected_Conversion': ['30-40%', '25-35%', '15-25%', '10-20%', '5-15%']
        })
        recommendations.to_excel(
            writer, 
            sheet_name='Recommendations', 
            index=False
        )
        
        # Sheet 7: Key Metrics Dashboard
        metrics_data = []
        if 'summary' in results:
            summary = results['summary']
            key_metrics = [
                ('Total Conversion Opportunity', summary.get('intl_not_p2p', 0) + summary.get('domestic_not_p2p', 0)),
                ('International Conversion Rate %', summary.get('conversion_potential_%', 0)),
                ('P2P Market Penetration', 
                 (summary.get('p2p_users', 0) / max(summary.get('total_customers', 1), 1)) * 100),
                ('Service Diversification Index',
                 len(set(results.get('domestic_other_not_p2p', pd.DataFrame())['Services_Used'].sum().split(', '))) 
                 if not results.get('domestic_other_not_p2p', pd.DataFrame()).empty else 0)
            ]
            
            for metric, value in key_metrics:
                metrics_data.append({
                    'Metric': metric,
                    'Value': value,
                    'Target': 'N/A',
                    'Status': 'On Track' if isinstance(value, (int, float)) and value > 0 else 'Needs Attention'
                })
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_excel(writer, sheet_name='Metrics_Dashboard', index=False)
    
    output.seek(0)
    return output.getvalue()

def create_detailed_reports(results):
    """Create additional detailed reports"""
    reports = {}
    
    if 'intl_not_p2p' in results and not results['intl_not_p2p'].empty:
        # Priority ranking
        df_intl = results['intl_not_p2p'].copy()
        if 'Days_Since_Last_Tx' in df_intl.columns:
            df_intl['Priority'] = df_intl['Days_Since_Last_Tx'].apply(
                lambda x: 'High' if x <= 7 else ('Medium' if x <= 30 else 'Low')
            )
            reports['priority_ranking'] = df_intl[['Customer_ID', 'Full_Name', 'Priority', 'Last_Transaction_Date']]
    
    return reports
